"""CFARTrainingModule — Stage 1 AR Diffusion Teacher Forcing.

对每个训练 step:
  1. Resolve prompt + run pipe units → input_latents (GT clean) + context + first_frame_latents
  2. Permute [B, C, F, H, W] → [B, F, C, H, W] (CF API layout)
  3. Sample PER-FRAME independent t over full [0, num_steps) range
     (AR diffusion; 对齐 CF 上游 model/diffusion.py:74-87 + base.py:87-110,
      _get_timestep(0, 1000, uniform_timestep=False))
  4. First-frame anchor (I2V): frame 0 = (t=0, sigma=0)
  5. noisy_x = (1-σ)·x0 + σ·ε; frame 0 stays anchor
  6. model_fn_causal_forcing(tf_mode=True, clean_x=x0, noisy_x=..., timestep=...)
  7. target = noise - x0; 乘 per-frame training_weight; loss = weighted MSE on frames 1..F-1
     (对齐 CF 上游 model/diffusion.py:120-124)
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from ReactiveGWM_Code.training.data.profiles import get_profile  # noqa: E402
from ReactiveGWM_Code.training.data.prompt_utils import resolve_prompt  # noqa: E402

from diffsynth.core import ModelConfig  # noqa: E402
from diffsynth.diffusion import DiffusionTrainingModule  # noqa: E402
from diffsynth.models.reactive_gwm_casual_forcing_dit import (  # noqa: E402
    CausalForcingReactiveGWMModel,
)
from diffsynth.pipelines.reactive_gwm import ReactiveGWMPipeline  # noqa: E402
from diffsynth.pipelines.reactive_gwm_casual_forcing import model_fn_causal_forcing  # noqa: E402


# Wan2.2-TI2V-5B arch — mirror examples/ReactiveGWM/model_training/train.py.
WAN_MODEL_KWARGS = {
    "has_image_input": False,
    "patch_size": [1, 2, 2],
    "in_dim": 48,
    "dim": 3072,
    "ffn_dim": 14336,
    "freq_dim": 256,
    "text_dim": 4096,
    "out_dim": 48,
    "num_heads": 24,
    "num_layers": 30,
    "eps": 1e-06,
    "seperated_timestep": True,
    "require_clip_embedding": False,
    "require_vae_embedding": False,
    "fuse_vae_embedding_in_latents": True,
}


def _build_model_configs(cfg: dict[str, Any]) -> list[ModelConfig]:
    """Construct ReactiveGWMPipeline model_configs from cfg.

    Accepts (in priority order):
      1. cfg["model_paths"]: JSON-string OR list (same shape as SFT)
         e.g. [["dit_shard1.safetensors", ...], "t5.pth", "vae.pth"]
      2. cfg["wan_base_dir"]: dir containing default Wan2.2-TI2V-5B filenames
    """
    raw = cfg.get("model_paths")
    if raw is None:
        base = cfg["wan_base_dir"].rstrip("/")
        raw = [
            [
                f"{base}/diffusion_pytorch_model-00001-of-00003.safetensors",
                f"{base}/diffusion_pytorch_model-00002-of-00003.safetensors",
                f"{base}/diffusion_pytorch_model-00003-of-00003.safetensors",
            ],
            f"{base}/models_t5_umt5-xxl-enc-bf16.pth",
            f"{base}/Wan2.2_VAE.pth",
        ]
    if isinstance(raw, str):
        raw = json.loads(raw)
    return [ModelConfig(path=p) for p in raw]


class CFARTrainingModule(DiffusionTrainingModule):
    """Stage 1 AR-TF generator-only training module."""

    def __init__(self, cfg: dict[str, Any], device: str = "cpu") -> None:
        super().__init__()
        self.cfg = cfg
        self.profile = get_profile(cfg.get("game", "sf3"))

        # 1) Build base pipeline (VAE + T5 + base Wan DiT).
        model_configs = _build_model_configs(cfg)
        tokenizer_config = ModelConfig(path=cfg["tokenizer_dir"])
        self.pipe = ReactiveGWMPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
        )

        # 2) Cold-start CausalForcingReactiveGWMModel + load base_ckpt (strict).
        custom_dit = CausalForcingReactiveGWMModel(
            num_buttons=self.profile.num_buttons,
            kv_window_size=int(cfg.get("kv_window_size", 16)),
            sink_size=int(cfg.get("kv_sink_size", 2)),
            num_frame_per_block=int(cfg.get("num_frame_per_block", 1)),
            text_len=int(cfg.get("text_len", 512)),
            **WAN_MODEL_KWARGS,
        ).to(torch.bfloat16)

        base_ckpt = cfg.get("base_ckpt")
        if base_ckpt and os.path.exists(base_ckpt):
            state = load_file(base_ckpt)
            missing, unexpected = custom_dit.load_state_dict(state, strict=False)
            print(
                f"[CF-AR] base_ckpt loaded: {len(state)} keys, "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
            del state

        student_init = cfg.get("student_init")
        if student_init and os.path.exists(student_init):
            state = load_file(student_init)
            missing, unexpected = custom_dit.load_state_dict(state, strict=False)
            print(
                f"[CF-AR] student_init overlaid: {len(state)} keys, "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
            del state

        self.pipe.dit = custom_dit.to(self.pipe.device)
        del custom_dit
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 3) Freeze pipeline; only DiT trainable.
        self.pipe.freeze_except(["dit"])
        self.pipe.dit.train()
        self.pipe.dit.requires_grad_(True)

        # 4) FlowMatchScheduler training mode (1000-step).
        self.pipe.scheduler.set_timesteps(
            1000, training=True, shift=float(cfg.get("timestep_shift", 5.0))
        )

        # Cache fields used in forward
        self.use_csv_prompt = bool(cfg.get("use_csv_prompt", True))
        self.prompt_column = cfg.get("prompt_column", "prompt")
        # ar_min_step / ar_max_step 已弃用: CF 上游 generator_loss 用全 [0,1000) 范围
        # 逐帧采样 (model/diffusion.py:74-81), 不做截断. 只保留 first-frame anchor (I2V).
        self.ar_first_frame_anchor = bool(cfg.get("ar_first_frame_anchor", True))
        self.use_gradient_checkpointing = bool(cfg.get("gradient_checkpointing", True))

    # ------------------------------------------------------------------ forward
    def forward(self, data: dict[str, Any]) -> torch.Tensor:
        # 1. Prompt
        prompt = resolve_prompt(data, self.profile, self.use_csv_prompt, self.prompt_column)

        # 2. Run pipeline units (ShapeChecker / NoiseInitializer / PromptEmbedder
        #    / InputVideoEmbedder / ImageEmbedderFused) to obtain input_latents,
        #    context, first_frame_latents.
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
        inputs_posi = {"prompt": prompt}
        inputs_nega: dict[str, Any] = {}
        inputs = (inputs_shared, inputs_posi, inputs_nega)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        inputs_shared, inputs_posi, _ = inputs

        # 3. Pull GT latents (clean) + text context + first-frame anchor.
        input_latents = inputs_shared["input_latents"].to(self.pipe.torch_dtype)  # [B, C, F, H, W]
        context = inputs_posi["context"].to(self.pipe.torch_dtype)
        keyboard_action = inputs_shared["keyboard_action"].to(self.pipe.torch_dtype)

        # 4. Permute to CF API layout [B, F, C, H, W].
        x0 = input_latents.permute(0, 2, 1, 3, 4).contiguous()
        B, F_lat = x0.shape[:2]

        # 5. Per-frame INDEPENDENT timestep over the full [0, num_steps) range
        #    (AR diffusion; CF 上游 model/diffusion.py:74-87 + base.py:87-110,
        #    _get_timestep(0, 1000, uniform_timestep=False)). 直接采 index, 用同一
        #    index gather timestep / sigma / training_weight, 与上游 argmin 反查等价.
        #    num_frame_per_block=1 (framewise): per-frame == per-block; 若 npfb>1
        #    需按 block 复制 timestep (见 CF base.py:106-109).
        sigmas = self.pipe.scheduler.sigmas.to(x0.device)
        timesteps = self.pipe.scheduler.timesteps.to(x0.device)
        weights = self.pipe.scheduler.linear_timesteps_weights.to(x0.device)
        num_steps = timesteps.shape[0]
        idx = torch.randint(0, num_steps, (B, F_lat), device=x0.device)  # [B, F]
        timestep_per_frame = timesteps[idx].to(x0.dtype)  # [B, F]
        sigma_per_frame = sigmas[idx].to(x0.dtype)         # [B, F]
        weight_per_frame = weights[idx].to(torch.float32)  # [B, F]

        # 6. First-frame anchor (I2V): frame 0 stays clean (t=0, sigma=0).
        if self.ar_first_frame_anchor:
            timestep_per_frame = timestep_per_frame.clone()
            sigma_per_frame = sigma_per_frame.clone()
            timestep_per_frame[:, 0] = 0.0
            sigma_per_frame[:, 0] = 0.0

        # 7. noisy_x = (1-σ)·x0 + σ·ε  (frame 0 has σ=0 so it stays anchored).
        noise = torch.randn_like(x0)
        s = sigma_per_frame[:, :, None, None, None]
        noisy_x = (1.0 - s) * x0 + s * noise

        # 8. Forward TF dual-block.
        flow_pred = model_fn_causal_forcing(
            self.pipe.dit,
            tf_mode=True,
            noisy_x=noisy_x,
            timestep=timestep_per_frame,
            clean_x=x0,
            action=keyboard_action,
            context=context,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
        )

        # 9. Wan flow-match target = noise - x0.
        target = noise - x0

        # 10. Per-frame training-weighted MSE (CF 上游 model/diffusion.py:120-124):
        #     mse_per_frame · training_weight(t_frame), 再 mean. frame 0 anchor (I2V)
        #     时跳过首帧 (首帧永远给定真实图像, 不训练).
        mse_per_frame = F.mse_loss(
            flow_pred.float(), target.float(), reduction="none"
        ).mean(dim=(2, 3, 4))  # [B, F]
        loss_per_frame = mse_per_frame * weight_per_frame  # [B, F]
        if self.ar_first_frame_anchor and F_lat > 1:
            loss = loss_per_frame[:, 1:].mean()
        else:
            loss = loss_per_frame.mean()
        return loss
