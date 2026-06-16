"""CFDMDTrainingModule — Stage 3 Asymmetric DMD self-forcing.

对齐 CF 上游 ``model/dmd.py`` + ``model/base.py`` + ``pipeline/self_forcing_training.py``
+ ``trainer/distillation.py`` + ``configs/causal_forcing_dmd_framewise.yaml``;自包含实现.
详见 PLAN.md §2 Stage 3、stage3_cf_alignment_review.md、implement.md Stage 3.

四个网络:
  - generator / student = ``self.pipe.dit``  —— **因果** CausalForcingReactiveGWMModel
    (从 Stage 2 CD 产物起步, 可训练; 注册子模块 → DDP / gen_optimizer / state)
  - critic / fake_score = ``self.critic``     —— **双向** ReactiveGWMModel (父类, base_ckpt
    起步, 可训练; 注册子模块 → DDP / critic_optimizer / state)
  - teacher / real_score = ``self._aux["teacher"]`` —— **双向** ReactiveGWMModel (冻结;
    **非子模块**, 不进 DDP / optimizer / state)
  - EMA(generator) = ``self._ema_shadow`` —— fp32 shadow, **仅导出**, step-200 懒创建,
    只在 generator optimizer step 后更新 (上游 trainer/distillation.py:316-336)

两个 phase (runner 交替驱动, 整个 module 仅 DDP 包一次→module 每 phase 调一次,
generator 内部 rollout 的 25 次 sub-forward 不触发多次 DDP-forward):
  - "generator": rollout(keep_grad=True) → DMD x0-space loss (teacher/critic no_grad score).
  - "critic":    no_grad rollout → critic flow-matching denoising loss (critic grad).
每个 phase 只有 {generator, critic} 之一拿梯度 → DDP 用 find_unused_parameters=True.

保护规则: 不改 model_fn / forward 默认返回 (flow); rollout / DMD 数学全在 pipeline helpers.
不动 Stage 1/2 (ar_tf.py / cd.py) 任何逻辑.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.distributed as dist

from diffsynth.core import ModelConfig
from diffsynth.diffusion import DiffusionTrainingModule
from diffsynth.models.reactive_gwm_casual_forcing_dit import CausalForcingReactiveGWMModel
from diffsynth.models.reactive_gwm_dit import ReactiveGWMModel
from diffsynth.pipelines.reactive_gwm import ReactiveGWMPipeline
from diffsynth.pipelines.reactive_gwm_casual_forcing import (
    DEFAULT_NEGATIVE_PROMPT,
    build_cf_scheduler,
    dmd_critic_flow_loss,
    dmd_distribution_matching_loss,
    dmd_long_tail_self_rollout,
    dmd_self_rollout,
    dmd_slice_tail_action,
    warp_denoising_steps,
)
from safetensors.torch import load_file

# Reuse Stage 1 的 arch / model_config 构造 (单一来源).
from ReactiveGWM_Code.training.causal_forcing.modules.ar_tf import (
    WAN_MODEL_KWARGS,
    _build_model_configs,
)
from ReactiveGWM_Code.training.causal_forcing.modules.ema import (
    ema_init_shadow,
    ema_update_shadow,
)

from ReactiveGWM_Code.training.data.profiles import get_profile  # noqa: E402
from ReactiveGWM_Code.training.data.prompt_utils import resolve_prompt  # noqa: E402


def _load_into(
    model: torch.nn.Module,
    ckpt_paths: list[str],
    tag: str,
    *,
    required: bool = False,
) -> torch.nn.Module:
    """依次 strict=False 叠加 ckpt_paths (后者覆盖前者). required=True 时缺失即失败."""
    loaded_any = False
    for p in ckpt_paths:
        if not p or not os.path.exists(p):
            if required:
                raise SystemExit(f"[CF-DMD] required {tag} checkpoint 不存在: {p!r}")
            continue
        state = load_file(p)
        if required and len(state) == 0:
            raise SystemExit(f"[CF-DMD] required {tag} checkpoint 为空: {p!r}")
        if required and not (set(model.state_dict().keys()) & set(state.keys())):
            raise SystemExit(f"[CF-DMD] required {tag} checkpoint 没有可匹配 key: {p!r}")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"[CF-DMD] {tag} loaded {os.path.basename(p)}: {len(state)} keys, "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
        loaded_any = True
        del state
    if required and not loaded_any:
        raise SystemExit(f"[CF-DMD] required {tag} checkpoint 未加载: {ckpt_paths!r}")
    return model


def _build_cf_generator(cfg: dict[str, Any], nb: int, ckpt_paths: list[str], device) -> CausalForcingReactiveGWMModel:
    """因果 generator (Stage 2 产物起步)."""
    dit = CausalForcingReactiveGWMModel(
        num_buttons=nb,
        kv_window_size=int(cfg.get("kv_window_size", 16)),
        sink_size=int(cfg.get("kv_sink_size", 0)),
        num_frame_per_block=int(cfg.get("num_frame_per_block", 1)),
        text_len=int(cfg.get("text_len", 512)),
        **WAN_MODEL_KWARGS,
    ).to(torch.bfloat16)
    return _load_into(dit, ckpt_paths, "generator", required=True).to(device)


def _build_bidirectional(cfg: dict[str, Any], nb: int, ckpt_paths: list[str], device) -> ReactiveGWMModel:
    """双向 teacher / critic (父类 ReactiveGWMModel, base_ckpt 起步)."""
    dit = ReactiveGWMModel(num_buttons=nb, **WAN_MODEL_KWARGS).to(torch.bfloat16)
    return _load_into(dit, ckpt_paths, "bidir", required=True).to(device)


class CFDMDTrainingModule(DiffusionTrainingModule):
    """Stage 3 DMD: generator(causal) + critic(bidir) trainable + teacher(bidir) frozen + EMA."""

    def __init__(self, cfg: dict[str, Any], device: str = "cpu") -> None:
        super().__init__()
        self.cfg = cfg
        self.profile = get_profile(cfg.get("game", "sf3"))
        nb = self.profile.num_buttons

        # 1) Base pipeline (VAE + T5 + a base DiT we replace with the causal generator).
        model_configs = _build_model_configs(cfg)
        tokenizer_config = ModelConfig(path=cfg["tokenizer_dir"])
        self.pipe = ReactiveGWMPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
        )
        dev = self.pipe.device

        base_ckpt = cfg.get("base_ckpt")
        student_init = cfg.get("student_init")  # Stage 2 CD product (EMA weights)
        if not student_init or not os.path.exists(student_init):
            raise SystemExit(f"[CF-DMD] student_init 不存在: {student_init!r} (Stage 2 产物必填)")

        # 2) generator (causal, Stage 2 起步) → pipe.dit, 可训练.
        #    step-N.safetensors 是完整 dit 导出 → fresh-init 提供 buffers + 这些 param;
        #    若缺 param 才需先叠 base_ckpt (一般不需要, 与 cd.py 一致).
        generator = _build_cf_generator(cfg, nb, [student_init], dev)
        self.pipe.dit = generator
        self.pipe.freeze_except(["dit"])  # 冻结 VAE / T5; dit 可训练
        self.pipe.dit.train()
        self.pipe.dit.requires_grad_(True)

        # 3) critic (双向, base_ckpt 起步) —— 子模块, 可训练 (单独 optimizer, 见 runner).
        self.critic = _build_bidirectional(cfg, nb, [base_ckpt], dev)
        self.critic.train()
        self.critic.requires_grad_(True)

        # 4) teacher (双向, base_ckpt 起步) —— **非子模块** (放 _aux), 冻结.
        self.dmd_teacher_offload_cpu = bool(cfg.get("dmd_teacher_offload_cpu", True))
        self.dmd_empty_cache_each_phase = bool(cfg.get("dmd_empty_cache_each_phase", True))
        teacher_device = "cpu" if self.dmd_teacher_offload_cpu else dev
        teacher = _build_bidirectional(cfg, nb, [base_ckpt], teacher_device)
        teacher.eval()
        teacher.requires_grad_(False)
        self._aux: dict[str, ReactiveGWMModel] = {"teacher": teacher}

        if self.dmd_empty_cache_each_phase and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 5) pipe.scheduler training 模式 (InputVideoEmbedder 查 pipe.scheduler.training
        #    决定返回 clean input_latents). 与 ar_tf / cd 一致.
        self.pipe.scheduler.set_timesteps(
            1000, training=True, shift=float(cfg.get("timestep_shift", 5.0))
        )

        # 6) warped 4-step (上游 warp; 用一个独立 1000-step 调度算 warped timestep/sigma).
        sched1000 = build_cf_scheduler(
            num_train_timestep=int(cfg.get("num_train_timestep", 1000)),
            sigma_shift=float(cfg.get("timestep_shift", 5.0)),
            training=False,
        )
        warped_t, warped_s = warp_denoising_steps(
            sched1000, cfg.get("denoising_step_list", [1000, 750, 500, 250]), dev
        )
        self.warped_timesteps = warped_t
        self.warped_sigmas = warped_s

        # 7) DMD 超参 (上游 framewise yaml; 见 stage3_cf_alignment_review.md §2.3/§4).
        self.dmd_score_frames = int(cfg.get("dmd_score_frames", 26))
        self.real_guidance_scale = float(cfg.get("real_guidance_scale", 3.0))
        self.fake_guidance_scale = float(cfg.get("fake_guidance_scale", 0.0))
        self.num_train_timestep = int(cfg.get("num_train_timestep", 1000))
        self.timestep_shift = float(cfg.get("timestep_shift", 5.0))
        clamp = cfg.get("dmd_timestep_clamp", [20, 980])
        self.clamp_min, self.clamp_max = float(clamp[0]), float(clamp[1])
        self.stochastic_exit_step = bool(cfg.get("stochastic_exit_step", True))
        self.last_step_only = bool(cfg.get("last_step_only", False)) or not self.stochastic_exit_step
        self.ar_first_frame_anchor = bool(cfg.get("ar_first_frame_anchor", True))
        self.use_gradient_checkpointing = bool(cfg.get("gradient_checkpointing", True))
        # 训练 rollout: sink+recent cache；stage3 yaml 覆盖为 sink=2/recent=16。
        self.dmd_train_kv_window = int(cfg.get("dmd_train_kv_window", self.dmd_score_frames))
        self.dmd_train_sink = int(cfg.get("dmd_train_sink", 0))

        # CF++ long-tail DMD: 长程 rollout 后只取末尾 dmd_score_frames 训练。
        self.dmd_long_rollout = bool(cfg.get("dmd_long_rollout", False))
        self.dmd_rollout_min_frames = max(
            self.dmd_score_frames, int(cfg.get("dmd_rollout_min_frames", self.dmd_score_frames))
        )
        self.dmd_rollout_max_frames = max(
            self.dmd_rollout_min_frames, int(cfg.get("dmd_rollout_max_frames", self.dmd_score_frames))
        )
        self.dmd_rollout_force_max_prob = float(cfg.get("dmd_rollout_force_max_prob", 0.0))
        self.dmd_random_action_extend = bool(cfg.get("dmd_random_action_extend", True))
        self.dmd_action_extend_mode = str(cfg.get("dmd_action_extend_mode", "preset_chunks"))
        self.dmd_action_chunk_frames = max(1, int(cfg.get("dmd_action_chunk_frames", 10)))
        self.dmd_random_action_neutral_prob = float(cfg.get("dmd_random_action_neutral_prob", 0.3))
        self.dmd_random_action_direction_prob = float(cfg.get("dmd_random_action_direction_prob", 0.4))
        self.dmd_tail_reanchor = str(cfg.get("dmd_tail_reanchor", "none")).lower()
        self.dmd_reanchor_decode_latents = max(1, int(cfg.get("dmd_reanchor_decode_latents", 1)))
        self._action_preset_groups = self._build_action_preset_groups()

        # 8) EMA(generator) —— fp32 shadow, 仅导出, step-200 懒创建 (review §4.1).
        self.dmd_ema_decay = float(cfg.get("dmd_ema_decay", 0.99))
        self.dmd_ema_start_step = int(cfg.get("dmd_ema_start_step", 200))
        self.dmd_ema_shadow_device = cfg.get("dmd_ema_shadow_device", "cpu")
        self._ema_shadow: dict[str, torch.Tensor] | None = None  # 懒创建

        self.use_csv_prompt = bool(cfg.get("use_csv_prompt", True))
        self.prompt_column = cfg.get("prompt_column", "prompt")

        # 9) negative-prompt context (CFG 用), 一次性编码缓存.
        neg_prompt = cfg.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT
        self._neg_context = self._encode_neg_context(neg_prompt).detach()

    # ------------------------------------------------------------------ neg context
    @torch.no_grad()
    def _encode_neg_context(self, neg_prompt: str) -> torch.Tensor:
        embedder = next(u for u in self.pipe.units if hasattr(u, "encode_prompt"))
        self.pipe.load_models_to_device(("text_encoder",))
        ctx = embedder.encode_prompt(self.pipe, neg_prompt)
        return ctx.to(self.pipe.torch_dtype)

    # ------------------------------------------------------------------ optimizer param groups
    def generator_params(self):
        return [p for p in self.pipe.dit.parameters() if p.requires_grad]

    def critic_params(self):
        return [p for p in self.critic.parameters() if p.requires_grad]

    # ------------------------------------------------------------------ EMA (export only)
    @torch.no_grad()
    def maybe_update_ema(self, step: int) -> None:
        """上游式懒创建 + 只在 generator step 后调 (runner 保证). step<start 不跟踪."""
        if step < self.dmd_ema_start_step:
            return
        gen = self.pipe.dit
        if self._ema_shadow is None:
            self._ema_shadow = ema_init_shadow(gen, device=self.dmd_ema_shadow_device)
            return  # 创建步 = 当前权重, 本步不再 update (与上游 EMA_FSDP 懒创建一致)
        ema_update_shadow(self._ema_shadow, gen, self.dmd_ema_decay)

    @torch.no_grad()
    def dmd_export_generator(self, step: int, prefer_ema: bool = True) -> dict[str, torch.Tensor]:
        """导出 generator 权重 (bare CF DiT).

        ``prefer_ema=True`` (默认, 用于 step-N / 最终 EMA 产物): step>=ema_start 且
        shadow 存在 → EMA, 否则 raw. ``prefer_ema=False`` (最终 raw generator 产物): 总是 raw.
        """
        names = {n for n, p in self.pipe.dit.named_parameters() if p.requires_grad}
        if prefer_ema and step >= self.dmd_ema_start_step and self._ema_shadow is not None:
            return {
                n: self._ema_shadow[n].detach().to(torch.bfloat16).cpu().contiguous()
                for n in names if n in self._ema_shadow
            }
        sd = self.pipe.dit.state_dict()
        return {n: sd[n].detach().to(torch.bfloat16).cpu().contiguous() for n in names if n in sd}

    # ---------------- EMA export under FSDP (swap shadow shards into live params → summon → save → restore)
    @torch.no_grad()
    def ema_swap_into_live(self) -> dict[str, torch.Tensor] | None:
        """把 EMA shadow (per-rank shard) 拷进 live generator 参数; 返回 backup 供 restore.

        FSDP FULL_SHARD+use_orig_params 下,shadow[n] 与 live ``p.data`` 是同形的 rank-local
        shard,直接 copy 合法。调用方必须在 ``summon_full_params`` 内导出后调 ``ema_restore_live``。
        无 shadow (step<200) → 返回 None,调用方退回导出 raw live 参数。
        """
        if self._ema_shadow is None:
            return None
        backup: dict[str, torch.Tensor] = {}
        for n, p in self.pipe.dit.named_parameters():
            s = self._ema_shadow.get(n)
            if s is not None:
                backup[n] = p.detach().clone()
                p.data.copy_(s.to(device=p.device, dtype=p.dtype))
        return backup

    @torch.no_grad()
    def ema_restore_live(self, backup: dict[str, torch.Tensor] | None) -> None:
        """还原 ``ema_swap_into_live`` 换出的 live 参数。"""
        if not backup:
            return
        for n, p in self.pipe.dit.named_parameters():
            b = backup.get(n)
            if b is not None:
                p.data.copy_(b)

    @torch.no_grad()
    def dmd_export_critic(self) -> dict[str, torch.Tensor]:
        names = {n for n, p in self.critic.named_parameters() if p.requires_grad}
        sd = self.critic.state_dict()
        return {n: sd[n].detach().to(torch.bfloat16).cpu().contiguous() for n in names if n in sd}

    @torch.no_grad()
    def ema_full_state_dict(self) -> dict[str, torch.Tensor]:
        """EMA fp32 shadow (供 resume); shadow=None (step<200) 时返回空 dict."""
        if self._ema_shadow is None:
            return {}
        return {n: s.detach().float().cpu().contiguous() for n, s in self._ema_shadow.items()}

    @torch.no_grad()
    def load_ema_full_state(self, path: str) -> None:
        """从 ema.safetensors 恢复 fp32 shadow (resume). 空文件 → shadow 保持 None."""
        state = load_file(path)
        if len(state) == 0:
            return
        if self._ema_shadow is None:
            self._ema_shadow = ema_init_shadow(self.pipe.dit, device=self.dmd_ema_shadow_device)
        loaded = 0
        for n, s in self._ema_shadow.items():
            if n in state:
                s.copy_(state[n].to(device=s.device, dtype=torch.float32))
                loaded += 1
        print(f"[CF-DMD] EMA shadow resumed from {os.path.basename(path)}: "
              f"{loaded}/{len(self._ema_shadow)} params", flush=True)
        del state

    # ------------------------------------------------------------------ inputs
    @torch.no_grad()
    def _resolve_inputs(self, data: dict[str, Any]):
        """pipe units → (anchor_latent [B,1,C,H,W], context_pos, context_neg, action).

        只 encode 首帧 (input_video[:1] + num_frames=1): anchor 只用首帧 latent; encode
        整段 101 帧是 ~50GB window-independent 浪费 (dry-run Bug 1). VAE/T5 冻结 → no_grad.
        keyboard_action 不被 input unit 处理 (model_fn 内才 bin), 故截 input_video 安全、action 完整.
        """
        prompt = resolve_prompt(data, self.profile, self.use_csv_prompt, self.prompt_column)
        inputs_shared = {
            "input_video": data["video"][:1],
            "input_image": data["video"][0],
            "keyboard_action": data["action"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": 1,
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
        B = x0.shape[0]
        anchor = x0[:, 0:1].contiguous()  # [B,1,C,H,W] GT 干净首帧 (I2V anchor)
        context_neg = self._neg_context.to(device=x0.device, dtype=x0.dtype).expand(B, -1, -1)
        return anchor, context_pos, context_neg, action

    # ------------------------------------------------------------------ long-tail rollout helpers
    def _build_action_preset_groups(self) -> dict[str, list[tuple[int, ...]]]:
        dirs = {"UP", "DOWN", "LEFT", "RIGHT"}
        groups: dict[str, list[tuple[int, ...]]] = {"neutral": [], "direction": [], "other": []}
        for _slug, buttons in self.profile.action_presets:
            idxs = tuple(self.profile.button_index(b) for b in buttons)
            if not idxs:
                groups["neutral"].append(idxs)
            elif len(buttons) == 1 and buttons[0] in dirs:
                groups["direction"].append(idxs)
            else:
                groups["other"].append(idxs)
        if not groups["neutral"]:
            groups["neutral"].append(())
        if not groups["direction"]:
            groups["direction"] = groups["neutral"]
        if not groups["other"]:
            groups["other"] = groups["direction"]
        return groups

    def _sample_rollout_frames(self, device: torch.device) -> int:
        if not self.dmd_long_rollout:
            return self.dmd_score_frames
        if self.dmd_rollout_min_frames == self.dmd_rollout_max_frames:
            L = torch.tensor([self.dmd_rollout_max_frames], device=device, dtype=torch.long)
        elif torch.rand((), device=device) < self.dmd_rollout_force_max_prob:
            L = torch.tensor([self.dmd_rollout_max_frames], device=device, dtype=torch.long)
        else:
            L = torch.randint(
                self.dmd_rollout_min_frames,
                self.dmd_rollout_max_frames + 1,
                (1,),
                device=device,
            )
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(L, src=0)
        return int(L.item())

    def _sample_action_group(self, device: torch.device) -> str:
        neutral_p = min(max(self.dmd_random_action_neutral_prob, 0.0), 1.0)
        direction_p = min(max(self.dmd_random_action_direction_prob, 0.0), 1.0 - neutral_p)
        r = float(torch.rand((), device=device).item())
        if r < neutral_p:
            return "neutral"
        if r < neutral_p + direction_p:
            return "direction"
        return "other"

    def _sample_preset_action_suffix(
        self,
        batch_size: int,
        extra_len: int,
        num_buttons: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        suffix = torch.zeros(batch_size, extra_len, num_buttons, device=device, dtype=dtype)
        for b in range(batch_size):
            for start in range(0, extra_len, self.dmd_action_chunk_frames):
                end = min(start + self.dmd_action_chunk_frames, extra_len)
                group = self._sample_action_group(device)
                presets = self._action_preset_groups[group]
                preset_idx = int(torch.randint(len(presets), (1,), device=device).item())
                buttons = presets[preset_idx]
                if buttons:
                    suffix[b, start:end, list(buttons)] = 1.0
        return suffix

    def _extend_action_for_rollout(self, action: torch.Tensor, rollout_frames: int) -> torch.Tensor:
        target_len = 4 * (int(rollout_frames) - 1) + 1
        if int(action.shape[1]) >= target_len:
            return action[:, :target_len].contiguous()
        extra_len = target_len - int(action.shape[1])
        if self.dmd_random_action_extend and self.dmd_action_extend_mode == "preset_chunks":
            suffix = self._sample_preset_action_suffix(
                action.shape[0], extra_len, action.shape[2], device=action.device, dtype=action.dtype,
            )
        elif int(action.shape[1]) > 0:
            suffix = action[:, -1:].expand(-1, extra_len, -1).clone()
        else:
            suffix = torch.zeros(action.shape[0], extra_len, action.shape[2], device=action.device, dtype=action.dtype)
        return torch.cat([action, suffix], dim=1).contiguous()

    @torch.no_grad()
    def _vae_reanchor_tail_anchor(self, reanchor_latents: torch.Tensor) -> torch.Tensor:
        latent_wan = reanchor_latents.detach().permute(0, 2, 1, 3, 4).contiguous()
        was_training = self.pipe.vae.training
        self.pipe.vae.eval()
        video = self.pipe.vae.decode(latent_wan, device=self.pipe.device, tiled=False)
        last_frame = video[:, :, -1:].contiguous()
        anchor = self.pipe.vae.encode(last_frame, device=self.pipe.device, tiled=False)
        if was_training:
            self.pipe.vae.train()
        return anchor.to(device=reanchor_latents.device, dtype=self.pipe.torch_dtype).permute(0, 2, 1, 3, 4).contiguous()

    def _rollout_dmd_window(
        self,
        anchor: torch.Tensor,
        action: torch.Tensor,
        context_pos: torch.Tensor,
        *,
        keep_grad: bool,
        use_gradient_checkpointing: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.dmd_long_rollout:
            pred_video, _exit = dmd_self_rollout(
                self.pipe.dit,
                first_frame_latent=anchor,
                action_per_frame=action,
                context=context_pos,
                num_frames=self.dmd_score_frames,
                warped_timesteps=self.warped_timesteps,
                warped_sigmas=self.warped_sigmas,
                kv_window_size=self.dmd_train_kv_window,
                sink_size=self.dmd_train_sink,
                keep_grad=keep_grad,
                last_step_only=self.last_step_only,
                use_gradient_checkpointing=use_gradient_checkpointing,
                dtype=self.pipe.torch_dtype,
            )
            return pred_video, action

        rollout_frames = self._sample_rollout_frames(anchor.device)
        action_full = self._extend_action_for_rollout(action, rollout_frames)
        tail_start = rollout_frames - self.dmd_score_frames
        pred_video, reanchor_latents, _exit = dmd_long_tail_self_rollout(
            self.pipe.dit,
            first_frame_latent=anchor,
            action_per_frame=action_full,
            context=context_pos,
            num_frames=rollout_frames,
            score_frames=self.dmd_score_frames,
            warped_timesteps=self.warped_timesteps,
            warped_sigmas=self.warped_sigmas,
            kv_window_size=self.dmd_train_kv_window,
            sink_size=self.dmd_train_sink,
            keep_grad=keep_grad,
            last_step_only=self.last_step_only,
            use_gradient_checkpointing=use_gradient_checkpointing,
            reanchor_decode_latents=self.dmd_reanchor_decode_latents,
            dtype=self.pipe.torch_dtype,
        )
        if tail_start > 0 and self.dmd_tail_reanchor == "vae":
            tail_anchor = self._vae_reanchor_tail_anchor(reanchor_latents)
            pred_video = torch.cat([tail_anchor, pred_video[:, 1:]], dim=1).contiguous()
        tail_action = dmd_slice_tail_action(action_full, tail_start, rollout_frames)
        return pred_video, tail_action

    # ------------------------------------------------------------------ forward (phase dispatch)
    def forward(self, data: dict[str, Any], phase: str = "generator") -> torch.Tensor:
        anchor, context_pos, context_neg, action = self._resolve_inputs(data)

        if phase == "generator":
            # rollout (keep grad on tail exit-step forwards) → DMD x0-space loss.
            pred_video, score_action = self._rollout_dmd_window(
                anchor,
                action,
                context_pos,
                keep_grad=True,
                use_gradient_checkpointing=self.use_gradient_checkpointing,
            )
            teacher = self._aux["teacher"]
            if self.dmd_teacher_offload_cpu:
                teacher.to(device=pred_video.device)
                teacher.eval()
            try:
                loss = dmd_distribution_matching_loss(
                    pred_video,
                    teacher=teacher,
                    critic=self.critic,
                    context_pos=context_pos,
                    context_neg=context_neg,
                    action=score_action,
                    real_guidance_scale=self.real_guidance_scale,
                    fake_guidance_scale=self.fake_guidance_scale,
                    num_train_timestep=self.num_train_timestep,
                    timestep_shift=self.timestep_shift,
                    clamp_min=self.clamp_min,
                    clamp_max=self.clamp_max,
                    first_frame_anchor=self.ar_first_frame_anchor,
                )
            finally:
                if self.dmd_teacher_offload_cpu:
                    teacher.to(device="cpu")
                    if self.dmd_empty_cache_each_phase and torch.cuda.is_available():
                        torch.cuda.empty_cache()
            return loss

        if phase == "critic":
            # no_grad rollout (generator 不进图) → critic flow-matching denoising loss.
            with torch.no_grad():
                pred_video, score_action = self._rollout_dmd_window(
                    anchor,
                    action,
                    context_pos,
                    keep_grad=False,
                    use_gradient_checkpointing=False,
                )
            return dmd_critic_flow_loss(
                pred_video.detach(),
                critic=self.critic,
                context_pos=context_pos,
                action=score_action,
                num_train_timestep=self.num_train_timestep,
                timestep_shift=self.timestep_shift,
                clamp_min=self.clamp_min,
                clamp_max=self.clamp_max,
                first_frame_anchor=self.ar_first_frame_anchor,
                use_gradient_checkpointing=self.use_gradient_checkpointing,
            )

        raise ValueError(f"[CF-DMD] unknown phase: {phase!r} (expect 'generator' or 'critic')")
