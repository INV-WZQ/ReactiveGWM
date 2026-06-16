"""Stage 3 DMD 双优化器交替训练循环 (从零写, 对齐 CF 上游 trainer/distillation.py).

与 Stage 1/2 的单优化器 runner (写在 train.py) 不同, Stage 3 需要:
  - 双优化器: generator (lr) + critic (lr_critic), 各自 AdamW、各自 grad clip;
  - 交替更新: generator 每 dfake_gen_update_ratio=5 步训一次、critic 每步训一次
    (上游 trainer/distillation.py:303-331);
  - EMA(generator) 仅导出: step-200 懒创建、只在 generator optimizer step 后更新;
  - 整个 module 仅 DDP 包一次 (find_unused_parameters=True, 因每个 phase 只用 generator
    或 critic 之一); module 每 phase 调一次 → generator rollout 内部 25 次 sub-forward
    不触发多次 DDP-forward 的 reducer 问题;
  - checkpoint: accelerate state (含 generator+critic 子模块 + optimizer + RNG)
    + state-N/ema*.safetensors (EMA fp32 shadow, 非 param 不在 accelerate state)
    + step-N.safetensors (generator EMA 导出, 供下游使用).

DDP / FSDP: 默认 DDP; OOM 由 launch 传 --use_fsdp 切 FSDP (单 unit), 与 Stage 1/2 兜底一致.
"""

from __future__ import annotations

import contextlib
import gc
import json
import os
import re
from typing import Any, Iterator


def _cycle(loader) -> Iterator:
    """无限循环 dataloader (上游 utils/dataset.cycle); 每 epoch 重新 shuffle."""
    while True:
        for x in loader:
            yield x


def _prune_old_ckpts(out_path: str, keep: int) -> None:
    """只保留最近 ``keep`` 个 state-N 目录 + step-N.safetensors (频繁 save 防爆盘).

    state-N 含分片 model+optimizer (大); step-N 是 generator 导出 (~9.4GB).
    keep<=0 → 不修剪. 仅主进程调用.
    """
    import glob
    import shutil
    if keep <= 0:
        return

    def _num(p: str, pat: str) -> int:
        m = re.search(pat, os.path.basename(p))
        return int(m.group(1)) if m else -1

    states = sorted(glob.glob(os.path.join(out_path, "state-*")), key=lambda p: _num(p, r"state-(\d+)"))
    for old in states[:-keep]:
        shutil.rmtree(old, ignore_errors=True)
    steps = sorted(
        glob.glob(os.path.join(out_path, "step-*.safetensors")), key=lambda p: _num(p, r"step-(\d+)")
    )
    for old in steps[:-keep]:
        try:
            os.remove(old)
        except OSError:
            pass


def _none_out_grads(params) -> None:
    """把一组参数的 .grad 显式置 None，让共享同一 optimizer 的 AdamW 跳过它们。

    Stage 3 用单 optimizer + 两 param-group（generator / critic）承载不同 lr/betas，靠
    “非活跃 phase 的参数 grad 为 None → 不更新” 复刻上游“双独立 optimizer 交替更新”。
    该前提在 DDP / 单卡成立（图外参数 grad 天然为 None），但 FSDP 把 generator+critic
    合进同一个 FlatParameter（NO_WRAP 单 unit），非活跃组反而会拿到 grad=0（而非 None）
    → 被 AdamW 每步做 weight-decay + 二阶矩衰减（generator 有效 lr 漂移 ~2.4×）。每个
    phase 在 optimizer.step 前显式置 None，在三种模式下都恢复“只更新活跃组”的语义。
    """
    for p in params:
        p.grad = None


def run_dmd(args, cfg: dict[str, Any]) -> None:
    import accelerate
    import torch
    from safetensors.torch import save_file
    from tqdm import tqdm

    from diffsynth.diffusion.runner import initialize_deepspeed_gradient_checkpointing

    from ReactiveGWM_Code.training.causal_forcing.modules.dmd import CFDMDTrainingModule
    from ReactiveGWM_Code.training.causal_forcing.modules.dataset import build_cf_dataset

    # ---------------- accelerate / DDP (find_unused_parameters=True: 每 phase 只用 gen 或 critic)
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=1,  # Stage 3 用交替更新, 不用 grad-accum
        mixed_precision=cfg.get("mixed_precision", "bf16"),
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(find_unused_parameters=True)
        ],
    )

    # ---------------- dataset (复用 Stage 1/2; Stage 3 只需 frame0 anchor + action + prompt)
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

    model = CFDMDTrainingModule(cfg, device=str(accelerator.device))

    out_path = cfg["output_path"]
    os.makedirs(out_path, exist_ok=True)

    # ---------------- hyperparams (上游 framewise yaml)
    lr = float(cfg.get("lr", 2.0e-6))
    lr_critic = float(cfg.get("lr_critic", 4.0e-7))
    weight_decay = float(cfg.get("weight_decay", 0.01))
    beta1 = float(cfg.get("beta1", 0.0))
    beta2 = float(cfg.get("beta2", 0.999))
    beta1_critic = float(cfg.get("beta1_critic", 0.0))
    beta2_critic = float(cfg.get("beta2_critic", 0.999))
    grad_clip = float(cfg.get("ar_grad_clip", 10.0))
    ratio = int(cfg.get("dfake_gen_update_ratio", 5))
    num_workers = int(cfg.get("dataset_num_workers", 2))
    save_steps = int(cfg.get("save_steps", 2500))
    max_steps = int(cfg.get("max_steps", 50000))
    ema_start_step = int(cfg.get("dmd_ema_start_step", 200))
    empty_cache_each_phase = bool(cfg.get("dmd_empty_cache_each_phase", True))
    gc_interval = int(cfg.get("dmd_gc_interval", 100))

    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers,
    )

    # FSDP: 优化器在 prepare(model) 之后、用 FSDP 管理的 params 建；需配 launch
    # --fsdp_use_orig_params true。用**单 AdamW 两个 param group**承载 generator/critic
    # 的不同 lr/betas；每个 phase 仍各 backward+step 一次，未参与 phase 的参数 grad=None
    # 不会更新。这样保留上游“generator/critic 交替更新”语义，同时避免 FSDP save_state
    # 对多个 partial optimizer 的状态映射报错。
    model.to(device=accelerator.device)
    model = accelerator.prepare(model)
    _m = accelerator.unwrap_model(model)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": _m.generator_params(),
                "lr": lr,
                "betas": (beta1, beta2),
                "weight_decay": weight_decay,
            },
            {
                "params": _m.critic_params(),
                "lr": lr_critic,
                "betas": (beta1_critic, beta2_critic),
                "weight_decay": weight_decay,
            },
        ]
    )
    # factor=1.0 → 真·恒定 lr (PyTorch ConstantLR 默认 factor=1/3,total_iters=5 会先压 lr;
    # 上游是恒定 lr). 双 phase 各 step 一次对恒定 lr 无影响.
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)
    optimizer, scheduler, dataloader = accelerator.prepare(optimizer, scheduler, dataloader)
    initialize_deepspeed_gradient_checkpointing(accelerator)

    # ---------------- per-rank 显存插桩 (env CF_DMD_MEM=1 开; 默认关, 不污染 full run)
    def _mem(tag: str, reset: bool = False):
        if not os.environ.get("CF_DMD_MEM") or not torch.cuda.is_available():
            return
        a = torch.cuda.memory_allocated() / 1e9
        p = torch.cuda.max_memory_allocated() / 1e9
        rsv = torch.cuda.memory_reserved() / 1e9
        print(f"[MEM r{accelerator.process_index}] {tag}: alloc={a:.1f} peak={p:.1f} "
              f"reserved={rsv:.1f} GB", flush=True)
        if reset:
            torch.cuda.reset_peak_memory_stats()
    _mem("baseline (after prepare, sharded-idle)", reset=True)

    def _is_fsdp_wrapped() -> bool:
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        except Exception:
            return False
        return isinstance(model, FSDP)

    def _full_params_for_export():
        """Gather FSDP full params for rank0 export (step-N / critic / final products).

        Under FSDP FULL_SHARD + use_orig_params, rank-local parameter views can be
        zero-sized; exporting outside summon_full_params writes unusable shards.
        All ranks must enter the context; only main process writes files.
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

    def _world_size() -> int:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size()
        return int(accelerator.num_processes)

    def _resume_meta_path(state_dir: str) -> str:
        return os.path.join(state_dir, "cf_resume_meta.json")

    def _current_resume_meta() -> dict[str, Any]:
        return {
            "stage": "dmd",
            "world_size": _world_size(),
            "distributed_type": str(accelerator.distributed_type),
            "fsdp_wrapped": _is_fsdp_wrapped(),
            "fsdp_requires_same_world_size": _is_fsdp_wrapped(),
        }

    def _write_resume_meta(state_dir: str) -> None:
        if not accelerator.is_main_process:
            return
        with open(_resume_meta_path(state_dir), "w") as f:
            json.dump(_current_resume_meta(), f, indent=2, sort_keys=True)

    def _validate_resume_meta(state_dir: str) -> None:
        path = _resume_meta_path(state_dir)
        if not os.path.exists(path):
            if accelerator.is_main_process:
                print(
                    f"[Resume][warn] no cf_resume_meta.json in {state_dir}; "
                    "legacy Stage 3 sharded state cannot verify world-size/FSDP compatibility",
                    flush=True,
                )
            return
        with open(path) as f:
            saved = json.load(f)
        current = _current_resume_meta()
        mismatches = []
        for key in ("stage", "fsdp_wrapped"):
            if saved.get(key) != current.get(key):
                mismatches.append(f"{key}: saved={saved.get(key)!r}, current={current.get(key)!r}")
        if saved.get("fsdp_wrapped") or current.get("fsdp_wrapped"):
            if int(saved.get("world_size", -1)) != int(current["world_size"]):
                mismatches.append(
                    f"world_size: saved={saved.get('world_size')!r}, current={current['world_size']!r}"
                )
        if mismatches and not bool(cfg.get("dmd_allow_sharded_resume_mismatch", False)):
            raise SystemExit(
                "[Resume] Stage 3 sharded state topology mismatch; use the same NUM_PROCESSES/FSDP mode "
                "or set dmd_allow_sharded_resume_mismatch=true only if you know the state format is portable. "
                + "; ".join(mismatches)
            )
        if mismatches and accelerator.is_main_process:
            print("[Resume][warn] ignoring sharded resume metadata mismatch: " + "; ".join(mismatches), flush=True)

    def _ema_rank_path(state_dir: str, rank: int) -> str:
        return os.path.join(state_dir, f"ema-rank{rank:05d}.safetensors")

    def _save_ema_state(state_dir: str) -> None:
        unwrapped = accelerator.unwrap_model(model)
        if unwrapped._ema_shadow is None:
            return
        state = unwrapped.ema_full_state_dict()
        if _is_fsdp_wrapped():
            save_file(state, _ema_rank_path(state_dir, accelerator.process_index))
        elif accelerator.is_main_process:
            save_file(state, os.path.join(state_dir, "ema.safetensors"))

    def _load_ema_state(state_dir: str, step_now: int) -> None:
        unwrapped = accelerator.unwrap_model(model)
        candidates = (
            [_ema_rank_path(state_dir, accelerator.process_index)]
            if _is_fsdp_wrapped()
            else [os.path.join(state_dir, "ema.safetensors"), _ema_rank_path(state_dir, 0)]
        )
        for path in candidates:
            if os.path.exists(path):
                unwrapped.load_ema_full_state(path)
                return
        if step_now >= ema_start_step and accelerator.is_main_process:
            print(f"[Resume][warn] no EMA state found in {state_dir}; EMA will restart from resumed generator", flush=True)

    # ---------------- seamless resume (EMA shadow 非 accelerate-state → 手动)
    step = 0
    if args.resume_state:
        if not os.path.isdir(args.resume_state):
            raise SystemExit(f"--resume_state not a dir: {args.resume_state}")
        _validate_resume_meta(args.resume_state)
        accelerator.load_state(args.resume_state)
        m = re.search(r"state-(\d+)/?$", args.resume_state.rstrip("/"))
        if m:
            step = int(m.group(1))
        _load_ema_state(args.resume_state, step)
        if accelerator.is_main_process:
            print(f"[Resume] loaded state from {args.resume_state} (step={step})", flush=True)

    # ---------------- save helper
    def _save(step_now: int) -> None:
        accelerator.wait_for_everyone()
        state_dir = os.path.join(out_path, f"state-{step_now}")
        if accelerator.is_main_process:
            os.makedirs(state_dir, exist_ok=True)
        accelerator.save_state(state_dir)  # FSDP sharded model+优化器+RNG → 完整 resume
        _write_resume_meta(state_dir)  # 记录 sharded resume 拓扑, 防止不同卡数/FSDP 模式误恢复
        _save_ema_state(state_dir)  # EMA fp32 shadow 不在 accelerate state, 需手动保存
        accelerator.wait_for_everyone()
        ckpt_path = os.path.join(out_path, f"step-{step_now}.safetensors")
        unwrapped = accelerator.unwrap_model(model)
        do_ema = (step_now >= ema_start_step) and (unwrapped._ema_shadow is not None)

        # step-N.safetensors = generator (有 EMA 用 EMA, 否则 raw). 所有 rank 进出 summon (collective);
        # EMA: 先把 shadow shard 换进 live 参数 → summon gather 成完整 → rank0 存 → 换回 (FSDP-safe).
        backup = unwrapped.ema_swap_into_live() if do_ema else None
        try:
            with _full_params_for_export():
                if accelerator.is_main_process:
                    save_file(unwrapped.dmd_export_generator(step_now, prefer_ema=False), ckpt_path)
        finally:
            unwrapped.ema_restore_live(backup)

        # critic.safetensors (live critic under summon)
        with _full_params_for_export():
            if accelerator.is_main_process:
                save_file(unwrapped.dmd_export_critic(), os.path.join(state_dir, "critic.safetensors"))

        if not accelerator.is_main_process:
            return
        kind = "EMA" if do_ema else "raw"
        print(
            f"[Save-State] wrote {state_dir} + step-{step_now}.safetensors "
            f"(generator {kind}) + critic.safetensors", flush=True
        )
        # 默认全部保留；如需防爆盘, 显式设置 dmd_keep_last_ckpts > 0。
        _prune_old_ckpts(out_path, int(cfg.get("dmd_keep_last_ckpts", 0)))

    # ---------------- main alternating loop (上游 trainer/distillation.py:303-331)
    data_iter = _cycle(dataloader)
    pbar = tqdm(total=max_steps, initial=step, disable=not accelerator.is_main_process)
    while step < max_steps:
        train_generator = (step % ratio == 0)

        # ---- generator update (每 ratio 步)
        if train_generator:
            optimizer.zero_grad(set_to_none=True)
            _mem(f"step{step} gen before-fwd", reset=True)
            loss_g = model(next(data_iter), phase="generator")
            _mem(f"step{step} gen after-fwd (rollout+score+loss)")
            accelerator.backward(loss_g)
            _mem(f"step{step} gen after-bwd")
            # 只更新 generator：把 idle(critic) 组的 grad 显式置 None（FSDP 单 flat-param 下
            # 非活跃组会拿到 grad=0 而非 None → 否则被 AdamW 每步 weight-decay + 二阶矩衰减）。
            _none_out_grads(accelerator.unwrap_model(model).critic_params())
            if grad_clip > 0:
                # 必须传 model.parameters() 全集：accelerator.clip_grad_norm_ 只有在参数==整个
                # model.parameters() 时才走 FSDP 感知的跨-rank 范数；传子集（generator_params()）
                # 会回退成普通 clip → 在分片梯度上只算单-rank 局部范数（阈值≈10·√world_size、
                # 各 rank 不一致）。idle 组已置 None、冻结模块 grad 为 None → 不计入范数，等价于
                # 只裁活跃的 generator。
                accelerator.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            # EMA 只在 generator step 后更新 (step-200 懒创建在内部判断)
            accelerator.unwrap_model(model).maybe_update_ema(step)
            g_loss_val = loss_g.item()
            if empty_cache_each_phase and torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            g_loss_val = None

        # ---- critic update (每步)
        optimizer.zero_grad(set_to_none=True)
        _mem(f"step{step} critic before-fwd", reset=True)
        loss_c = model(next(data_iter), phase="critic")
        _mem(f"step{step} critic after-fwd")
        accelerator.backward(loss_c)
        _mem(f"step{step} critic after-bwd")
        # 只更新 critic：把 idle(generator) 组的 grad 置 None（critic phase 里 generator 走
        # no_grad rollout，FSDP 下仍可能拿到 grad=0）。clip 同样传全集走 FSDP 感知的跨-rank 范数。
        _none_out_grads(accelerator.unwrap_model(model).generator_params())
        if grad_clip > 0:
            accelerator.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if empty_cache_each_phase and torch.cuda.is_available():
            torch.cuda.empty_cache()

        step += 1
        pbar.update(1)
        if accelerator.is_main_process:
            gpart = f"gen={g_loss_val:.6f} " if g_loss_val is not None else ""
            print(f"[step {step}] {gpart}critic={loss_c.item():.6f}", flush=True)

        if gc_interval > 0 and step % gc_interval == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if save_steps and step % save_steps == 0:
            _save(step)

    pbar.close()
    # ---------------- final products
    accelerator.wait_for_everyone()
    if cfg.get("dmd_skip_final_save", False):
        if accelerator.is_main_process:
            print(f"[Done] reached step={step}; final save skipped by dmd_skip_final_save", flush=True)
        return
    unwrapped = accelerator.unwrap_model(model)
    do_ema = (step >= ema_start_step) and (unwrapped._ema_shadow is not None)

    # step-N.safetensors (若不在 save_steps 边界): EMA if available else raw
    if not (save_steps and step % save_steps == 0):
        backup = unwrapped.ema_swap_into_live() if do_ema else None
        try:
            with _full_params_for_export():
                if accelerator.is_main_process:
                    save_file(unwrapped.dmd_export_generator(step, prefer_ema=False),
                              os.path.join(out_path, f"step-{step}.safetensors"))
        finally:
            unwrapped.ema_restore_live(backup)

    # raw generator 产物 (live 训练权重, summon gather 成完整)
    with _full_params_for_export():
        if accelerator.is_main_process:
            save_file(unwrapped.dmd_export_generator(step, prefer_ema=False),
                      os.path.join(out_path, "stage3_dmd_generator.safetensors"))

    # EMA generator 产物 (swap shadow→live → summon → save → restore; 无 shadow 时 = raw)
    backup = unwrapped.ema_swap_into_live() if do_ema else None
    try:
        with _full_params_for_export():
            if accelerator.is_main_process:
                save_file(unwrapped.dmd_export_generator(step, prefer_ema=False),
                          os.path.join(out_path, "stage3_dmd_ema.safetensors"))
    finally:
        unwrapped.ema_restore_live(backup)

    # critic 产物
    with _full_params_for_export():
        if accelerator.is_main_process:
            save_file(unwrapped.dmd_export_critic(),
                      os.path.join(out_path, "stage3_dmd_critic.safetensors"))
    if accelerator.is_main_process:
        print(f"[Done] reached step={step}; final products written to {out_path}", flush=True)
