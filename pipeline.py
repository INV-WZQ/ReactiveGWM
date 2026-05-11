"""SFPipeline: diffusers-style inference for SF2/SF3 game-action video DiT.

    from ReactiveGWM import SFPipeline
    pipe = SFPipeline.from_pretrained(base_model_dir, ckpt_path, variant="sf3").to("cuda")
    out  = pipe(image=PIL_image, actions_parquet="actions.parquet", num_frames=101, ...)
    frames = out.frames[0]   # list[PIL.Image]
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import torch
from PIL import Image
from safetensors.torch import load_file
from tqdm import tqdm

from .constants import NEG_PROMPT, SF_BUTTON_COLS, VARIANT_DEFAULTS
from .models import WanModelAction, WanTextEncoder, WanTokenizer, WanVideoVAE38
from .utils import (
    FlowMatchScheduler,
    load_actions,
    preprocess_image,
    round_up,
    to_pil_video,
)


@dataclass
class SFPipelineOutput:
    frames: List[List[Image.Image]] = field(default_factory=list)


class SFPipeline(torch.nn.Module):
    """SF2/SF3 video-DiT inference. Diffusers-style API."""

    def __init__(
        self,
        dit: WanModelAction,
        vae: WanVideoVAE38,
        text_encoder: WanTextEncoder,
        tokenizer: WanTokenizer,
        scheduler: FlowMatchScheduler,
        variant: str,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.dit = dit
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.variant = variant
        self.torch_dtype = torch_dtype
        self._device = torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device, *args, **kwargs):
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        self.dit = self.dit.to(self._device, dtype=self.torch_dtype)
        self.vae = self.vae.to(self._device, dtype=self.torch_dtype)
        self.text_encoder = self.text_encoder.to(self._device, dtype=self.torch_dtype)
        return self

    # ---------------- loading ----------------

    @classmethod
    def from_pretrained(
        cls,
        base_model_dir: str,
        checkpoint_path: str,
        variant: str,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "SFPipeline":
        """Build the pipeline. Loads:
          - `<base_model_dir>/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth`        → VAE
          - `<base_model_dir>/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth`  → T5
          - `<base_model_dir>/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/`     → tokenizer
          - `<checkpoint_path>` (full DiT state dict, .safetensors)        → DiT
        """
        if variant not in VARIANT_DEFAULTS:
            raise ValueError(f"variant must be 'sf2' or 'sf3', got {variant!r}")
        base = Path(base_model_dir)
        ti2v_dir = base / "Wan-AI" / "Wan2.2-TI2V-5B"
        tok_dir = base / "Wan-AI" / "Wan2.1-T2V-1.3B" / "google" / "umt5-xxl"

        # --- VAE ---
        vae = WanVideoVAE38()
        vae_state = torch.load(ti2v_dir / "Wan2.2_VAE.pth",
                               map_location="cpu", weights_only=True)
        vae_state = {f"model.{k}": v for k, v in vae_state.items()}
        miss, unexp = vae.load_state_dict(vae_state, strict=False)
        if unexp:
            raise RuntimeError(f"Unexpected VAE keys: {unexp[:5]}")
        vae = vae.to(torch_dtype).eval().requires_grad_(False)

        # --- T5 ---
        t5 = WanTextEncoder()
        t5_state = torch.load(ti2v_dir / "models_t5_umt5-xxl-enc-bf16.pth",
                              map_location="cpu", weights_only=True)
        miss, unexp = t5.load_state_dict(t5_state, strict=False)
        if unexp:
            raise RuntimeError(f"Unexpected T5 keys: {unexp[:5]}")
        t5 = t5.to(torch_dtype).eval().requires_grad_(False)
        tokenizer = WanTokenizer(str(tok_dir))

        # --- DiT (custom checkpoint) ---
        dit = WanModelAction(num_buttons=len(SF_BUTTON_COLS))
        ckpt_state = load_file(checkpoint_path)
        miss, unexp = dit.load_state_dict(ckpt_state, strict=False)
        n_action = sum(1 for k in ckpt_state if k.startswith("action_embedders"))
        print(f"[SFPipeline] DiT loaded: {len(ckpt_state)} keys "
              f"({n_action} action_embedders), missing={len(miss)}, unexpected={len(unexp)}")
        if unexp:
            print(f"  unexpected (first 5): {unexp[:5]}")
        dit = dit.to(torch_dtype).eval().requires_grad_(False)

        scheduler = FlowMatchScheduler()
        return cls(dit, vae, t5, tokenizer, scheduler,
                   variant=variant, torch_dtype=torch_dtype)

    # ---------------- inference ----------------

    @torch.no_grad()
    def encode_prompt(self, prompt: str) -> torch.Tensor:
        ids, mask = self.tokenizer(prompt)
        ids, mask = ids.to(self.device), mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        emb = self.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            emb[i, v:] = 0   # zero pad positions per-sample (no-op vs `emb[:, v:]` while B=1)
        return emb.to(self.torch_dtype)

    @torch.no_grad()
    def __call__(
        self,
        image: Image.Image,
        actions_parquet: str,
        prompt: Optional[str] = None,
        negative_prompt: str = NEG_PROMPT,
        num_frames: int = 101,
        num_inference_steps: int = 30,
        cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        seed: int = 0,
        sigma_shift: float = 5.0,
        action_hold_window: int = 10,
        # SF2 (480x608) and SF3 (480x832) latents (30x38 / 30x52) are ≤ the
        # default tile_size, so tiling degenerates to a single-tile encode plus
        # a CPU↔GPU round-trip with no VRAM benefit. Set tiled=True only when
        # rendering at resolutions where the whole image can't fit on GPU.
        tiled: bool = False,
        tile_size: Sequence[int] = (30, 52),
        tile_stride: Sequence[int] = (15, 26),
    ) -> SFPipelineOutput:
        defaults = VARIANT_DEFAULTS[self.variant]
        prompt = prompt if prompt is not None else defaults["prompt"]
        height = height if height is not None else defaults["height"]
        width = width if width is not None else defaults["width"]

        # Shape constraints: H,W % 32 == 0 (Wan2.2 patch×VAE), num_frames ≡ 1 (mod 4).
        # Round H,W,num_frames UP to the next valid value (matches the training-time
        # `base_pipeline.check_resize_height_width` behavior; rounding down would
        # silently shrink an off-grid request, e.g. 102 → 101 vs 102 → 105).
        height, width = round_up(height, 32), round_up(width, 32)
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1

        # --- Encode prompts (positive + negative for CFG) ---
        ctx_pos = self.encode_prompt(prompt)
        ctx_neg = self.encode_prompt(negative_prompt) if cfg_scale != 1.0 else None

        # --- Load actions ---
        keyboard = load_actions(actions_parquet, num_frames,
                                hold_window=action_hold_window,
                                device=str(self.device), dtype=self.torch_dtype)

        # --- First-frame VAE embedding (TI2V-5B fuse-VAE convention) ---
        img_t = preprocess_image(image, height, width,
                                 device=str(self.device), dtype=self.torch_dtype)
        # When tiled=True the encode accumulators live on CPU so the returned
        # tensor is on CPU; tiled=False returns on `device`. Either way, an
        # explicit `.to(device, dtype)` here pins the device/dtype contract and
        # is a no-op when nothing needs moving. (PyTorch's __setitem__ below
        # would also handle the cross-device case implicitly, but being
        # explicit keeps the contract obvious.)
        first_frame_latent = self.vae.encode(
            [img_t], device=self.device, tiled=tiled,
            tile_size=tile_size, tile_stride=tile_stride,
        ).to(device=self.device, dtype=self.torch_dtype)   # [1, 48, 1, h, w]

        # --- Init noise + pin first-frame latent ---
        latent_T = (num_frames - 1) // 4 + 1
        latent_H, latent_W = height // 16, width // 16
        gen = torch.Generator("cpu").manual_seed(seed)
        latents = torch.randn(
            (1, 48, latent_T, latent_H, latent_W),
            generator=gen, dtype=torch.float32,
        ).to(self.device, self.torch_dtype)
        latents[:, :, 0:1] = first_frame_latent

        # --- Denoising loop ---
        self.scheduler.set_timesteps(num_inference_steps, shift=sigma_shift)

        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="Denoising")):
            t_dev = t.unsqueeze(0).to(self.device, self.torch_dtype)

            eps_pos = self.dit(
                latents=latents, timestep=t_dev,
                context=ctx_pos, keyboard_action=keyboard,
                fuse_vae_embedding_in_latents=True,
            )

            if action_cfg_scale != 1.0:
                eps_pos_noact = self.dit(
                    latents=latents, timestep=t_dev,
                    context=ctx_pos, keyboard_action=torch.zeros_like(keyboard),
                    fuse_vae_embedding_in_latents=True,
                )
                eps_pos = eps_pos_noact + action_cfg_scale * (eps_pos - eps_pos_noact)

            if cfg_scale != 1.0:
                eps_neg = self.dit(
                    latents=latents, timestep=t_dev,
                    context=ctx_neg, keyboard_action=keyboard,
                    fuse_vae_embedding_in_latents=True,
                )
                eps = eps_neg + cfg_scale * (eps_pos - eps_neg)
            else:
                eps = eps_pos

            latents = self.scheduler.step(eps, self.scheduler.timesteps[i], latents)
            latents[:, :, 0:1] = first_frame_latent   # keep first-frame condition pinned

        # --- VAE decode ---
        video = self.vae.decode(latents, device=self.device, tiled=tiled,
                                tile_size=tile_size, tile_stride=tile_stride)
        return SFPipelineOutput(frames=[to_pil_video(video)])
