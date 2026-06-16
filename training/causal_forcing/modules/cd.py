"""CFCDTrainingModule — Stage 2 Causal Consistency Distillation (Causal Forcing++).

对齐 CF 上游 ``model/naive_consistency.py`` + ``trainer/naive_cd.py`` + I2V 适配.
详见 PLAN.md §2 Stage 2 与 implement.md "Stage 2 实施前置约束".

三个网络 (都从 Stage 1 产物 step-N 初始化, 同结构 ``CausalForcingReactiveGWMModel``):
  - generator / student  = ``self.pipe.dit``   (可训练; 注册子模块 → DDP / optimizer / state)
  - teacher              = ``self._aux["teacher"]``  (冻结 Stage 1 副本; **非子模块**, 不进 ckpt)
  - EMA-student (目标网络) = ``self._aux["ema"]``      (EMA-of-student; **非子模块**, 手动 save/load)

每个 train step (uniform t, **≠** Stage 1 逐帧独立):
  1. pipe units → x0 (26 帧 GT clean latent) + pos context + action; frame 0 = I2V anchor.
  2. 随机 idx ∈ [0, N-1) → t=timesteps[idx], t_next=timesteps[idx+1], 各 broadcast 全 26 帧;
     frame 0 的 t / σ = 0 (anchor).
  3. latent_t = (1-σ_t)·x0 + σ_t·ε (frame 0 仍 = x0).
  4. teacher 单步 ODE + neg-prompt CFG(g=3) → latent_t_next (frame 0 anchor).
  5. student (grad) TF forward → flow → cm_pred_t = latent_t − σ_t·flow.
  6. EMA (no_grad) TF forward @ (latent_t_next, t_next) → cm_pred_t_next = latent_t_next − σ_next·flow.
  7. loss = MSE(cm_pred_t[:,1:], cm_pred_t_next[:,1:])  (纯 MSE, 无 training_weight; frame 0 排除).

保护规则: 不改 ``model_fn_causal_forcing`` / ``forward`` 默认返回 (flow); x0 转换在 ``flow_to_x0``
helper 内做. 不动 Stage 1 ``ar_tf.py`` 任何逻辑.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from diffsynth.core import ModelConfig
from diffsynth.diffusion import DiffusionTrainingModule
from diffsynth.models.reactive_gwm_casual_forcing_dit import (
    CausalForcingReactiveGWMModel,
)
from diffsynth.pipelines.reactive_gwm import ReactiveGWMPipeline
from diffsynth.pipelines.reactive_gwm_casual_forcing import (
    DEFAULT_NEGATIVE_PROMPT,
    build_cd_scheduler,
    cd_teacher_one_step,
    flow_to_x0,
    model_fn_causal_forcing,
)

# Reuse Stage 1 的 arch / model_config 构造 (单一来源, Stage 2 必须同 arch).
from ReactiveGWM_Code.training.causal_forcing.modules.ar_tf import (
    WAN_MODEL_KWARGS,
    _build_model_configs,
)
from ReactiveGWM_Code.training.causal_forcing.modules.ema import (
    ema_copy_to_model,
    ema_init_shadow,
    ema_update_shadow,
)

from ReactiveGWM_Code.training.data.profiles import get_profile  # noqa: E402
from ReactiveGWM_Code.training.data.prompt_utils import resolve_prompt  # noqa: E402

# DEFAULT_NEGATIVE_PROMPT 从 pipeline 模块 import，与 CF 上游
# causal_cd_framewise.yaml negative_prompt 逐字一致 (Wan 默认负面串).


def _build_cf_dit(cfg: dict[str, Any], num_buttons: int, ckpt_paths: list[str], device) -> CausalForcingReactiveGWMModel:
    """Cold-start CausalForcingReactiveGWMModel + 依次 strict=False 叠加 ckpt_paths.

    ckpt_paths 一般 = [base_ckpt, stage1_product]; 后者覆盖前者. 任一不存在则跳过.
    """
    dit = CausalForcingReactiveGWMModel(
        num_buttons=num_buttons,
        kv_window_size=int(cfg.get("kv_window_size", 16)),
        sink_size=int(cfg.get("kv_sink_size", 2)),
        num_frame_per_block=int(cfg.get("num_frame_per_block", 1)),
        text_len=int(cfg.get("text_len", 512)),
        **WAN_MODEL_KWARGS,
    ).to(torch.bfloat16)
    for p in ckpt_paths:
        if p and os.path.exists(p):
            state = load_file(p)
            missing, unexpected = dit.load_state_dict(state, strict=False)
            print(
                f"[CF-CD] loaded {os.path.basename(p)}: {len(state)} keys, "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
            del state
    return dit.to(device)


class CFCDTrainingModule(DiffusionTrainingModule):
    """Stage 2 CD: generator(student) + frozen teacher + EMA target network."""

    def __init__(self, cfg: dict[str, Any], device: str = "cpu") -> None:
        super().__init__()
        self.cfg = cfg
        self.profile = get_profile(cfg.get("game", "sf3"))
        nb = self.profile.num_buttons

        # 1) Base pipeline (VAE + T5 + a base DiT we will replace).
        model_configs = _build_model_configs(cfg)
        tokenizer_config = ModelConfig(path=cfg["tokenizer_dir"])
        self.pipe = ReactiveGWMPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
        )
        dev = self.pipe.device

        student_init = cfg.get("student_init")
        teacher_ckpt = cfg.get("teacher_ckpt") or student_init  # 默认 teacher 同 student-init
        if not student_init or not os.path.exists(student_init):
            raise SystemExit(f"[CF-CD] student_init 不存在: {student_init!r} (Stage 1 产物必填)")

        # 2) generator (student) = pipe.dit, 可训练.
        #    step-N.safetensors 是完整 dit 导出 (Stage 1 全部参数可训练) → fresh-init 提供
        #    buffers(freqs) + 这些 param = 正确模型; 不必先叠 base_ckpt (省 ~一半启动 IO,
        #    4 rank × 3 模型). _build_cf_dit 打印的 missing 应只是 buffers; 若缺 param 才需 base.
        generator = _build_cf_dit(cfg, nb, [student_init], dev)
        self.pipe.dit = generator
        self.pipe.freeze_except(["dit"])
        self.pipe.dit.train()
        self.pipe.dit.requires_grad_(True)

        # 3) teacher (冻结 Stage 1 副本) + EMA-student (目标网络), 都 **非子模块** (放 _aux dict)
        #    → 不进 self.parameters() / DDP / accelerate save_state, 避免每 ckpt 多存 20 GB.
        teacher = _build_cf_dit(cfg, nb, [teacher_ckpt], dev)
        teacher.eval()
        teacher.requires_grad_(False)
        ema = _build_cf_dit(cfg, nb, [student_init], dev)  # EMA 起点 = student-init
        ema.eval()
        ema.requires_grad_(False)
        # plain-dict 容器: nn.Module.__setattr__ 不追踪 dict 内的 Module → 不注册为 child.
        self._aux: dict[str, CausalForcingReactiveGWMModel] = {"teacher": teacher, "ema": ema}

        # fp32 master shadow (对齐官方 EMA_FSDP): bf16 `ema` model 仅作 forward 载体,
        # 真正的 EMA 累加在 fp32 shadow 上做, 每步 copy_to 同步到 `ema` model.
        # shadow 默认放 CPU (不占 GPU 显存; 3×5B 同驻时关键), 可用 cd_ema_shadow_device=cuda
        # 提速 (省每步 CPU↔GPU 传输, 但 +约 2× param 字节 VRAM). 起点 = student_init (同 ema model).
        self.cd_ema_shadow_device = cfg.get("cd_ema_shadow_device", "cpu")
        self._ema_shadow = ema_init_shadow(self.pipe.dit, device=self.cd_ema_shadow_device)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 3.5) pipe 自带 scheduler 进 training 模式 (1000-step): pipe.units 的
        #      InputVideoEmbedder 会查 pipe.scheduler.training 决定返回 clean input_latents
        #      (训练路径) 而非加噪 latents. 与 ar_tf.py 一致. 与下方独立 48-node cd_sched 无关.
        self.pipe.scheduler.set_timesteps(
            1000, training=True, shift=float(cfg.get("timestep_shift", 5.0))
        )

        # 4) CD 离散调度 (48 节点, 丢 σ=0) + 缓存 sigmas / timesteps.
        self.cd_N = int(cfg.get("cd_discrete_N", 48))
        self.cd_sched = build_cd_scheduler(self.cd_N, float(cfg.get("timestep_shift", 5.0)))
        self.cd_sigmas = self.cd_sched.sigmas.to(dev)
        self.cd_timesteps = self.cd_sched.timesteps.to(dev)

        # 5) CD 超参.
        self.cd_guidance_scale = float(cfg.get("cd_guidance_scale", 3.0))
        self.cd_ema_decay = float(cfg.get("cd_ema_decay", 0.99))
        self.cd_ema_start_step = int(cfg.get("cd_ema_start_step", 200))
        self.ar_first_frame_anchor = bool(cfg.get("ar_first_frame_anchor", True))
        self.use_gradient_checkpointing = bool(cfg.get("gradient_checkpointing", True))
        self.use_csv_prompt = bool(cfg.get("use_csv_prompt", True))
        self.prompt_column = cfg.get("prompt_column", "prompt")

        # 6) negative-prompt context 一次性编码并缓存 (CFG 用).
        neg_prompt = cfg.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT
        self._neg_context = self._encode_neg_context(neg_prompt).detach()

    # ------------------------------------------------------------------ neg context
    @torch.no_grad()
    def _encode_neg_context(self, neg_prompt: str) -> torch.Tensor:
        """用 pipe 的 PromptEmbedder unit 编码 negative prompt → [1, L, dim] (on device)."""
        embedder = next(u for u in self.pipe.units if hasattr(u, "encode_prompt"))
        self.pipe.load_models_to_device(("text_encoder",))
        ctx = embedder.encode_prompt(self.pipe, neg_prompt)
        return ctx.to(self.pipe.torch_dtype)

    # ------------------------------------------------------------------ EMA / export helpers
    @torch.no_grad()
    def update_ema(self) -> None:
        """EMA = decay·EMA + (1-decay)·generator (fp32 shadow 累加 → 同步 bf16 ema model).

        由 train loop 在每次 optimizer.step 后调. shadow 在 fp32 下累加 (避免 bf16 舍入
        吃掉 0.01·(p−e) 小增量 → 长程跟不动), 再 copy_to ema model 供下一步 forward.
        对齐官方 EMA_FSDP.update + copy_to.
        """
        ema_update_shadow(self._ema_shadow, self.pipe.dit, self.cd_ema_decay)
        ema_copy_to_model(self._ema_shadow, self._aux["ema"])

    @torch.no_grad()
    def cd_export_state_dict(self, step: int) -> dict[str, torch.Tensor]:
        """导出 bare-DiT 权重 (供下阶段使用). step≥ema_start → EMA (fp32 shadow→bf16),
        否则 raw generator.

        只取 generator 的可训练参数名 (与 Stage 1 step-N.safetensors 同格式; 无 buffers,
        下阶段加载时 strict=False, buffers 由 fresh init 提供, 恒等).
        """
        names = {n for n, p in self.pipe.dit.named_parameters() if p.requires_grad}
        if step >= self.cd_ema_start_step:
            # 从 fp32 master shadow 导出 (cast bf16); 比从 bf16 ema model 取更精确.
            return {
                n: self._ema_shadow[n].detach().to(torch.bfloat16).cpu().contiguous()
                for n in names
                if n in self._ema_shadow
            }
        src_sd = self.pipe.dit.state_dict()
        return {n: src_sd[n].detach().cpu().contiguous() for n in names if n in src_sd}

    @torch.no_grad()
    def ema_full_state_dict(self) -> dict[str, torch.Tensor]:
        """EMA fp32 master shadow (params only), 供 resume 用 (写 state-N/ema.safetensors).

        存 fp32 shadow 而非 bf16 ema model: shadow 才是累加 master, resume 必须无损恢复.
        buffers (RoPE freqs) 是常量, resume 时由 fresh init 提供, 不需存.
        """
        return {n: s.detach().float().cpu().contiguous() for n, s in self._ema_shadow.items()}

    @torch.no_grad()
    def load_ema_full_state(self, path: str) -> None:
        """从 ema.safetensors 恢复 fp32 master shadow + 同步 bf16 ema model.

        按 shadow key 取交集 (params only), cast fp32. 兼容旧格式 (多余 buffers key 忽略).
        """
        state = load_file(path)
        loaded = 0
        for n, s in self._ema_shadow.items():
            if n in state:
                s.copy_(state[n].to(device=s.device, dtype=torch.float32))
                loaded += 1
        ema_copy_to_model(self._ema_shadow, self._aux["ema"])
        print(f"[CF-CD] EMA shadow resumed from {os.path.basename(path)}: "
              f"{loaded}/{len(self._ema_shadow)} params", flush=True)
        del state

    # ------------------------------------------------------------------ forward
    def forward(self, data: dict[str, Any]) -> torch.Tensor:
        # 1) prompt + pipe units → input_latents (GT clean) / pos context / action.
        prompt = resolve_prompt(data, self.profile, self.use_csv_prompt, self.prompt_column)
        inputs_shared = {
            "input_video": data["video"],
            "input_image": data["video"][0],
            "keyboard_action": data["action"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
        }
        inputs = (inputs_shared, {"prompt": prompt}, {})
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        inputs_shared, inputs_posi, _ = inputs

        input_latents = inputs_shared["input_latents"].to(self.pipe.torch_dtype)  # [B,C,F,H,W]
        context_pos = inputs_posi["context"].to(self.pipe.torch_dtype)
        action = inputs_shared["keyboard_action"].to(self.pipe.torch_dtype)

        x0 = input_latents.permute(0, 2, 1, 3, 4).contiguous()  # [B,F,C,H,W]
        B, F_lat = x0.shape[:2]
        dev, dt = x0.device, x0.dtype
        context_neg = self._neg_context.to(device=dev, dtype=dt).expand(B, -1, -1)

        # 2) uniform t: 单 idx ∈ [0, N-1) broadcast 全 26 帧 (≠ Stage 1 逐帧).
        idx = int(torch.randint(0, self.cd_N - 1, (1,), device=dev).item())
        t_curr = self.cd_timesteps[idx]
        t_next = self.cd_timesteps[idx + 1]
        sig_curr = self.cd_sigmas[idx]
        sig_next = self.cd_sigmas[idx + 1]

        def _bcast(val: torch.Tensor) -> torch.Tensor:
            v = torch.full((B, F_lat), float(val), device=dev, dtype=dt)
            if self.ar_first_frame_anchor and F_lat > 0:
                v[:, 0] = 0.0  # frame 0 anchor
            return v

        timestep_pf = _bcast(t_curr)
        timestep_next_pf = _bcast(t_next)
        sigma_pf = _bcast(sig_curr)
        sigma_next_pf = _bcast(sig_next)

        # 3) latent_t = (1-σ_t)·x0 + σ_t·ε (frame 0: σ=0 → 保持 x0).
        noise = torch.randn_like(x0)
        s = sigma_pf[:, :, None, None, None]
        latent_t = (1.0 - s) * x0 + s * noise

        # 4) teacher 单步 ODE + neg-CFG → latent_t_next (no_grad; frame 0 anchor).
        latent_t_next = cd_teacher_one_step(
            self._aux["teacher"],
            latent_t=latent_t,
            timestep_per_frame=timestep_pf,
            t_idx=idx,
            scheduler=self.cd_sched,
            clean_x=x0,
            action=action,
            context_pos=context_pos,
            context_neg=context_neg,
            guidance_scale=self.cd_guidance_scale,
            first_frame_x0=x0 if self.ar_first_frame_anchor else None,
        )

        # 5) student (grad) TF forward → flow → cm_pred_t = latent_t − σ_t·flow.
        flow_s = model_fn_causal_forcing(
            self.pipe.dit,
            tf_mode=True,
            noisy_x=latent_t,
            timestep=timestep_pf,
            clean_x=x0,
            action=action,
            context=context_pos,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
        )
        cm_pred_t = flow_to_x0(latent_t, flow_s, sigma_pf)

        # 6) EMA (no_grad) TF forward @ (latent_t_next, t_next) → cm_pred_t_next.
        with torch.no_grad():
            flow_e = model_fn_causal_forcing(
                self._aux["ema"],
                tf_mode=True,
                noisy_x=latent_t_next,
                timestep=timestep_next_pf,
                clean_x=x0,
                action=action,
                context=context_pos,
            )
        cm_pred_t_next = flow_to_x0(latent_t_next, flow_e, sigma_next_pf)

        # 7) CD loss = MSE(cm_t, cm_t_next), frame 0 (I2V anchor) 排除; 纯 MSE 无 weight.
        if self.ar_first_frame_anchor and F_lat > 1:
            loss = F.mse_loss(cm_pred_t[:, 1:].float(), cm_pred_t_next[:, 1:].float())
        else:
            loss = F.mse_loss(cm_pred_t.float(), cm_pred_t_next.float())
        return loss
