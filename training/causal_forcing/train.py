"""ReactiveGWM Causal Forcing — 统一训练入口.

按 --stage 派发:
  Stage 1 (ar_tf): TF dual-block flow-match teacher forcing
  Stage 2 (cd):    Causal Consistency Distillation
  Stage 3 (dmd):   Asymmetric DMD self-forcing

YAML 是单一来源 (configs/<stage>.yaml; 与 configs/default.yaml 浅合并).
常用路径可通过 BASE_CKPT / WAN_BASE_DIR / TOKENIZER_DIR / DATA_ROOT /
METADATA_PATH / OUT / STUDENT_INIT / TEACHER_CKPT 覆盖。

Checkpoint 每 save_steps 落: model state-dict (via ModelLogger) +
完整 accelerate state (optimizer / LR scheduler / RNG / DataLoader) → 完整 resume.
"""

from __future__ import annotations

import argparse
import os

# 坐实 expandable_segments: 必须在 import torch / CUDA 初始化前进程内设死, 不靠 launch env
# 透传 (实测 accelerate 子进程下 launch 的 export 未生效 → reserved 碎片飙升). 保留已有选项.
_ac = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
if "expandable_segments" not in _ac:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (_ac + "," if _ac else "") + "expandable_segments:True"

import sys
from pathlib import Path
from typing import Any

import yaml

# Make ReactiveGWM_Code importable when launched directly via
# `python training/causal_forcing/train.py` from inside ReactiveGWM_Code.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ReactiveGWM Causal Forcing trainer")
    p.add_argument("--stage", required=True, choices=["ar_tf", "cd", "dmd"])
    p.add_argument("--config", required=True, help="path to stage yaml")
    p.add_argument("--resume_state", default=None, help="state-<N>/ dir to resume from")
    p.add_argument("--use_fsdp", action="store_true", help="hint to use FSDP via launcher")
    return p.parse_args()


def load_cfg(yaml_path: str) -> dict[str, Any]:
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f) or {}
    default_path = Path(yaml_path).parent / "default.yaml"
    if default_path.exists() and default_path.resolve() != Path(yaml_path).resolve():
        with open(default_path) as f:
            base = yaml.safe_load(f) or {}
        base.update(cfg)
        cfg = base
    # Env overrides
    env_to_key = {
        "BASE_CKPT": "base_ckpt",
        "WAN_BASE_DIR": "wan_base_dir",
        "TOKENIZER_DIR": "tokenizer_dir",
        "DATA_ROOT": "dataset_base",
        "METADATA_PATH": "metadata_path",
        "OUT": "output_path",
        "STUDENT_INIT": "student_init",
        "TEACHER_CKPT": "teacher_ckpt",  # Stage 2 CD teacher (默认 = student_init)
    }
    for env_k, cfg_k in env_to_key.items():
        v = os.environ.get(env_k)
        if v:
            cfg[cfg_k] = v
    if (
        cfg.get("dataset_base")
        and (
            not cfg.get("metadata_path")
            or str(cfg.get("metadata_path")).startswith("<path-to-dataset-root>")
        )
    ):
        cfg["metadata_path"] = str(Path(cfg["dataset_base"]) / "metadata.csv")
    return cfg


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config)

    if args.stage == "ar_tf":
        _run_ar_tf(args, cfg)
    elif args.stage == "cd":
        _run_cd(args, cfg)
    elif args.stage == "dmd":
        _run_dmd(args, cfg)


def _run_dmd(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    """Stage 3 Asymmetric DMD — 双优化器交替循环 (写在 modules/runner_dmd.py).

    与 Stage 1/2 的单优化器循环结构不同 (双优化器 / 交替 / 双向 teacher+critic / EMA 仅导出),
    故独立成 runner_dmd.run_dmd; train.py 仅做派发.
    """
    from ReactiveGWM_Code.training.causal_forcing.modules.runner_dmd import run_dmd

    run_dmd(args, cfg)


def _run_ar_tf(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    import accelerate
    import torch
    from tqdm import tqdm

    from diffsynth.diffusion import ModelLogger
    from diffsynth.diffusion.runner import initialize_deepspeed_gradient_checkpointing

    from ReactiveGWM_Code.training.causal_forcing.modules.ar_tf import CFARTrainingModule
    from ReactiveGWM_Code.training.causal_forcing.modules.dataset import build_cf_dataset

    # ---------------- accelerate / DDP
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 1)),
        mixed_precision=cfg.get("mixed_precision", "bf16"),
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(find_unused_parameters=False)
        ],
    )

    # ---------------- dataset
    dataset = build_cf_dataset(
        metadata_path=cfg["metadata_path"],
        dataset_base=cfg["dataset_base"],
        game=cfg.get("game", "sf3"),
        num_frames=int(cfg.get("num_frames", 101)),
        height=int(cfg.get("height", 480)),
        width=int(cfg.get("width", 832)),
        action_hold_window=int(cfg.get("action_hold_window", 10)),
        dataset_repeat=int(cfg.get("dataset_repeat", 100)),
        data_file_keys=tuple(cfg.get("data_file_keys", ("video", "action"))),
    )

    # ---------------- training module
    model = CFARTrainingModule(cfg, device=str(accelerator.device))

    # ---------------- output path + logger
    out_path = cfg["output_path"]
    os.makedirs(out_path, exist_ok=True)
    logger = ModelLogger(out_path, remove_prefix_in_ckpt="pipe.dit.")

    # ---------------- optimizer + scheduler + dataloader
    lr = float(cfg.get("lr", 2.0e-6))
    weight_decay = float(cfg.get("weight_decay", 0.01))   # CF 上游 default_config.yaml
    beta1 = float(cfg.get("ar_beta1", 0.0))               # CF 上游 beta1=0.0
    beta2 = float(cfg.get("ar_beta2", 0.999))
    num_workers = int(cfg.get("dataset_num_workers", 2))
    save_steps = int(cfg.get("save_steps", 2500))
    max_steps = int(cfg.get("max_steps", 50000))
    grad_clip = float(cfg.get("ar_grad_clip", 10.0))

    optimizer = torch.optim.AdamW(
        list(model.trainable_modules()),
        lr=lr,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
    )
    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        collate_fn=lambda x: x[0],
        num_workers=num_workers,
    )

    model.to(device=accelerator.device)
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )
    initialize_deepspeed_gradient_checkpointing(accelerator)

    # ---------------- seamless resume
    step = 0
    if args.resume_state:
        if not os.path.isdir(args.resume_state):
            raise SystemExit(f"--resume_state not a dir: {args.resume_state}")
        accelerator.load_state(args.resume_state)
        # Recover step from state-<N>/ naming convention used by save_state below.
        import re
        m = re.search(r"state-(\d+)/?$", args.resume_state.rstrip("/"))
        if m:
            step = int(m.group(1))
            # Also sync the ModelLogger counter so step-<N>.safetensors aligns
            # with the resumed global step (otherwise it restarts at 0 and
            # overwrites the original step-N from before resume).
            logger.num_steps = step
        if accelerator.is_main_process:
            print(f"[Resume] loaded state from {args.resume_state} (step={step})", flush=True)

    # ---------------- main loop
    num_epochs = int(cfg.get("num_epochs", 100000))
    done = False
    for epoch_id in range(num_epochs):
        if done:
            break
        for data in tqdm(dataloader, disable=not accelerator.is_main_process):
            with accelerator.accumulate(model):
                loss = model(data)
                accelerator.backward(loss)
                if accelerator.sync_gradients and grad_clip > 0:
                    accelerator.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # step 记「优化器更新」次数而非 micro-batch: grad_accum=k 时 sync_gradients
                # 每 k 个 micro-batch 才 True 一次。这样 step/save_steps/max_steps 与
                # grad_accum 解耦 → 4卡grad_accum=2(全局batch8) 与 8卡grad_accum=1 的 step
                # 语义、ckpt 命名、数据预算完全一致。grad_accum=1 时此 guard 恒为 False, 行为不变。
                if not accelerator.sync_gradients:
                    continue
                step += 1
                logger.on_step_end(accelerator, model, save_steps, loss=loss)

                # Save full training state on save_steps boundary.
                if save_steps and step % save_steps == 0:
                    accelerator.wait_for_everyone()
                    state_dir = os.path.join(out_path, f"state-{step}")
                    if accelerator.is_main_process:
                        os.makedirs(state_dir, exist_ok=True)
                    accelerator.save_state(state_dir)
                    if accelerator.is_main_process:
                        print(f"[Save-State] wrote {state_dir}", flush=True)

                if step >= max_steps:
                    done = True
                    break

    logger.on_training_end(accelerator, model, save_steps)
    if accelerator.is_main_process:
        print(f"[Done] reached step={step}", flush=True)


def _run_cd(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    """Stage 2 Causal Consistency Distillation — 单优化器 + EMA 目标网络.

    与 ``_run_ar_tf`` 同构 (单 backward / accelerate state), 增量:
      - 模型为 ``CFCDTrainingModule`` (generator + 冻结 teacher + EMA, 后两者非子模块);
      - 每次 optimizer.step 后 ``update_ema()`` (decay=0.99 从 step 0);
      - 存点除 accelerate state 外, 额外写 ``state-N/ema.safetensors`` (EMA resume) +
        ``step-N.safetensors`` (导出 EMA 权重, 供下阶段使用; step<ema_start 时为 raw);
      - resume 时手动从 ``state-N/ema.safetensors`` 恢复 EMA.
    不复用 Stage 1 的逐帧 timestep / training_weight / ModelLogger save 逻辑.
    """
    import contextlib
    import re

    import accelerate
    import torch
    from safetensors.torch import save_file
    from tqdm import tqdm

    from diffsynth.diffusion.runner import initialize_deepspeed_gradient_checkpointing

    from ReactiveGWM_Code.training.causal_forcing.modules.cd import CFCDTrainingModule
    from ReactiveGWM_Code.training.causal_forcing.modules.dataset import build_cf_dataset

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 1)),
        mixed_precision=cfg.get("mixed_precision", "bf16"),
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(find_unused_parameters=False)
        ],
    )

    dataset = build_cf_dataset(
        metadata_path=cfg["metadata_path"],
        dataset_base=cfg["dataset_base"],
        game=cfg.get("game", "sf3"),
        num_frames=int(cfg.get("num_frames", 101)),
        height=int(cfg.get("height", 480)),
        width=int(cfg.get("width", 832)),
        action_hold_window=int(cfg.get("action_hold_window", 10)),
        dataset_repeat=int(cfg.get("dataset_repeat", 100)),
        data_file_keys=tuple(cfg.get("data_file_keys", ("video", "action"))),
    )

    model = CFCDTrainingModule(cfg, device=str(accelerator.device))
    if accelerator.is_main_process:
        teacher_ckpt = cfg.get("teacher_ckpt") or cfg.get("student_init")
        loss_frames = "1: (exclude frame 0 anchor)" if model.ar_first_frame_anchor else "all"
        print(
            "[CF-CD] settings: "
            f"cd_N={model.cd_N}, "
            f"timestep_shift={cfg.get('timestep_shift', 5.0)}, "
            f"guidance_scale={model.cd_guidance_scale}, "
            f"ema_decay={model.cd_ema_decay}, "
            f"ema_start_step={model.cd_ema_start_step}, "
            f"ar_first_frame_anchor={model.ar_first_frame_anchor}, "
            f"loss_frames={loss_frames}, "
            f"student_init={cfg.get('student_init')}, "
            f"teacher_ckpt={teacher_ckpt}",
            flush=True,
        )

    out_path = cfg["output_path"]
    os.makedirs(out_path, exist_ok=True)

    lr = float(cfg.get("lr", 2.0e-6))
    weight_decay = float(cfg.get("weight_decay", 0.01))
    beta1 = float(cfg.get("ar_beta1", 0.0))
    beta2 = float(cfg.get("ar_beta2", 0.999))
    num_workers = int(cfg.get("dataset_num_workers", 2))
    save_steps = int(cfg.get("save_steps", 2500))
    max_steps = int(cfg.get("max_steps", 50000))
    grad_clip = float(cfg.get("ar_grad_clip", 10.0))
    ema_start_step = int(cfg.get("cd_ema_start_step", 200))

    optimizer = torch.optim.AdamW(
        list(model.trainable_modules()),
        lr=lr,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
    )
    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        collate_fn=lambda x: x[0],
        num_workers=num_workers,
    )

    model.to(device=accelerator.device)
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )
    initialize_deepspeed_gradient_checkpointing(accelerator)

    def _full_params_for_export():
        """Gather FSDP full params before exporting ``step-N.safetensors``.

        Without this, FSDP FULL_SHARD can export rank-local parameter shards as if
        they were complete tensors. All ranks must enter the context; only rank0
        writes the gathered state.
        """
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        except Exception:
            return contextlib.nullcontext()
        if isinstance(model, FSDP):
            return FSDP.summon_full_params(
                model, recurse=True, writeback=False, rank0_only=True, offload_to_cpu=True,
            )
        return contextlib.nullcontext()

    def _save_cd_export(step_now: int, path: str) -> None:
        unwrapped = accelerator.unwrap_model(model)
        with _full_params_for_export():
            if accelerator.is_main_process:
                save_file(unwrapped.cd_export_state_dict(step_now), path)
        accelerator.wait_for_everyone()

    # ---------------- seamless resume (EMA 手动, 因其非子模块不在 accelerate state)
    step = 0
    if args.resume_state:
        if not os.path.isdir(args.resume_state):
            raise SystemExit(f"--resume_state not a dir: {args.resume_state}")
        accelerator.load_state(args.resume_state)
        m = re.search(r"state-(\d+)/?$", args.resume_state.rstrip("/"))
        if m:
            step = int(m.group(1))
        ema_path = os.path.join(args.resume_state, "ema.safetensors")
        if os.path.exists(ema_path):
            accelerator.unwrap_model(model).load_ema_full_state(ema_path)
        elif accelerator.is_main_process:
            print(f"[Resume][warn] no ema.safetensors in {args.resume_state}; "
                  f"EMA stays at student_init", flush=True)
        if accelerator.is_main_process:
            print(f"[Resume] loaded state from {args.resume_state} (step={step})", flush=True)

    # ---------------- main loop
    num_epochs = int(cfg.get("num_epochs", 100000))
    done = False
    for epoch_id in range(num_epochs):
        if done:
            break
        for data in tqdm(dataloader, disable=not accelerator.is_main_process):
            with accelerator.accumulate(model):
                loss = model(data)
                accelerator.backward(loss)
                if accelerator.sync_gradients and grad_clip > 0:
                    accelerator.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                if not accelerator.sync_gradients:
                    continue
                # EMA 从 step 0 更新 (decay=0.99); 只在真正 optimizer step 后.
                accelerator.unwrap_model(model).update_ema()
                step += 1
                if accelerator.is_main_process:
                    print(f"[step {step}] loss={loss.item():.6f}", flush=True)

                if save_steps and step % save_steps == 0:
                    accelerator.wait_for_everyone()
                    state_dir = os.path.join(out_path, f"state-{step}")
                    if accelerator.is_main_process:
                        os.makedirs(state_dir, exist_ok=True)
                    accelerator.save_state(state_dir)
                    ckpt_path = os.path.join(out_path, f"step-{step}.safetensors")
                    kind = "EMA" if step >= ema_start_step else "raw"
                    if accelerator.is_main_process:
                        unwrapped = accelerator.unwrap_model(model)
                        save_file(
                            unwrapped.ema_full_state_dict(),
                            os.path.join(state_dir, "ema.safetensors"),
                        )
                    _save_cd_export(step, ckpt_path)
                    if accelerator.is_main_process:
                        print(
                            f"[Save-State] wrote {state_dir} + step-{step}.safetensors "
                            f"({kind} weights)", flush=True
                        )

                if step >= max_steps:
                    done = True
                    break

    accelerator.wait_for_everyone()
    if not (save_steps and step % save_steps == 0):
        _save_cd_export(step, os.path.join(out_path, f"step-{step}.safetensors"))
    if accelerator.is_main_process:
        print(f"[Done] reached step={step}", flush=True)


if __name__ == "__main__":
    main()
