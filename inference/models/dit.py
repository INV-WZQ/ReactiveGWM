"""SF2/SF3 custom DiT: Wan2.2-TI2V-5B + 30 per-block action embedders.

Direct port of `diffsynth/models/our_sf2_dit.py` with training-only paths
(gradient checkpointing, sequence-parallel) removed. Matches the .safetensors
key layout of `step-37000` / `step-42000`.
"""
import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


try:
    import flash_attn
    _FLASH_ATTN_2 = True
except ImportError:
    _FLASH_ATTN_2 = False


def _attention(q, k, v, num_heads):
    """Multi-head scaled-dot-product attention. Uses Flash-Attn 2 if installed."""
    if _FLASH_ATTN_2:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        return rearrange(x, "b s n d -> b s (n d)")
    q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
    k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
    v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
    x = F.scaled_dot_product_attention(q, k, v)
    return rearrange(x, "b n s d -> b s (n d)")


def _modulate(x, shift, scale):
    return x * (1 + scale) + shift


def sinusoidal_embedding_1d(dim, position):
    sin = torch.outer(
        position.type(torch.float64),
        torch.pow(10000, -torch.arange(dim // 2, dtype=torch.float64,
                                       device=position.device).div(dim // 2))
    )
    return torch.cat([torch.cos(sin), torch.sin(sin)], dim=1).to(position.dtype)


def _precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].double() / dim))
    freqs = torch.outer(torch.arange(end), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def _precompute_freqs_cis_3d(dim: int, end: int = 1024):
    return (
        _precompute_freqs_cis(dim - 2 * (dim // 3), end),
        _precompute_freqs_cis(dim // 3, end),
        _precompute_freqs_cis(dim // 3, end),
    )


def _rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_c = torch.view_as_complex(x.to(torch.float64).reshape(*x.shape[:-1], -1, 2))
    x_out = torch.view_as_real(x_c * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return y.to(dtype) * self.weight


class _SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, eps=1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

    def forward(self, x, freqs):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = _rope_apply(q, freqs, self.num_heads)
        k = _rope_apply(k, freqs, self.num_heads)
        return self.o(_attention(q, k, v, self.num_heads))


class _CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, eps=1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

    def forward(self, x, ctx):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        return self.o(_attention(q, k, v, self.num_heads))


class _DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_dim, eps=1e-6):
        super().__init__()
        self.self_attn = _SelfAttention(dim, num_heads, eps)
        self.cross_attn = _CrossAttention(dim, num_heads, eps)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

    def forward(self, x, context, t_mod, freqs):
        # t_mod: [B, 6, C] (separated_timestep=False) or [B, T, 6, C]
        has_seq = t_mod.dim() == 4
        chunk_dim = 2 if has_seq else 1
        mods = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
                ).chunk(6, dim=chunk_dim)
        if has_seq:
            mods = [m.squeeze(2) for m in mods]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods

        x = x + gate_msa * self.self_attn(_modulate(self.norm1(x), shift_msa, scale_msa), freqs)
        x = x + self.cross_attn(self.norm3(x), context)
        x = x + gate_mlp * self.ffn(_modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class _Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)

    def forward(self, x, t_mod):
        # t_mod: [B, C] (separated=False) or [B, T, C]
        if t_mod.dim() == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype,
                                                            device=t_mod.device)
                            + t_mod.unsqueeze(2)).chunk(2, dim=2)
            return self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2))
        shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
                        + t_mod).chunk(2, dim=1)
        return self.head(self.norm(x) * (1 + scale) + shift)


class WanModelAction(nn.Module):
    """Wan2.2-TI2V-5B DiT with 30 per-block action embedders.

    Action conditioning: each `nn.Linear(num_buttons, dim)` projects the
    temporally binned keyboard signal and adds the result before its DiTBlock.
    No gates — see `our_sf2_dit.WanModelAction` docstring for v3→v4 rationale.
    """
    def __init__(
        self,
        num_buttons: int = 10,
        dim: int = 3072,
        in_dim: int = 48,
        ffn_dim: int = 14336,
        out_dim: int = 48,
        text_dim: int = 4096,
        freq_dim: int = 256,
        num_heads: int = 24,
        num_layers: int = 30,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_buttons = num_buttons

        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            _DiTBlock(dim, num_heads, ffn_dim, eps) for _ in range(num_layers)
        ])
        self.head = _Head(dim, out_dim, patch_size, eps)
        self.action_embedders = nn.ModuleList([
            nn.Linear(num_buttons, dim, bias=False) for _ in range(num_layers)
        ])

        head_dim = dim // num_heads
        self.freqs = _precompute_freqs_cis_3d(head_dim)

    def _bin_action(self, keyboard: Optional[torch.Tensor], f: int):
        if keyboard is None:
            return None
        # keyboard: [B, num_raw, K] dense. Adaptive max-pool → [B, f, K].
        return F.adaptive_max_pool1d(keyboard.transpose(1, 2), output_size=f).transpose(1, 2)

    def _inject_action(self, x, action_binned, block_id, f, h, w):
        if action_binned is None:
            return x
        B, _, K = action_binned.shape
        emb = self.action_embedders[block_id](action_binned.to(x.dtype))   # [B, f, C]
        bias = emb.unsqueeze(2).expand(-1, -1, h * w, -1).reshape(B, f * h * w, self.dim)
        return x + bias

    def forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        keyboard_action: Optional[torch.Tensor] = None,
        fuse_vae_embedding_in_latents: bool = True,
    ):
        """
        latents: [B, C_in, F, H, W] noisy latents (B, 48, f, h, w for Wan2.2-TI2V-5B)
        timestep: [1] tensor (single scalar)
        context: [B, L_text, text_dim] T5 embeddings
        keyboard_action: [B, T_raw, 10] dense one-hot (hold-window upsampled), or None
        fuse_vae_embedding_in_latents: True for TI2V-5B (separated timestep with t=0
            on first frame for the image-conditioning path).
        """
        # Per-token timestep on the first latent frame (TI2V-5B convention).
        if fuse_vae_embedding_in_latents:
            B, _, F_, H_, W_ = latents.shape
            tokens_per_frame = (H_ * W_) // 4   # patch_size = (1,2,2)
            ts = torch.cat([
                torch.zeros((1, tokens_per_frame), dtype=latents.dtype, device=latents.device),
                torch.ones((F_ - 1, tokens_per_frame), dtype=latents.dtype, device=latents.device) * timestep,
            ]).flatten()
            t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, ts).unsqueeze(0))
            t_mod = self.time_projection(t).unflatten(2, (6, self.dim))
        else:
            t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
            t_mod = self.time_projection(t).unflatten(1, (6, self.dim))

        ctx = self.text_embedding(context)

        x = latents
        if x.shape[0] != ctx.shape[0]:
            x = torch.cat([x] * ctx.shape[0], dim=0)

        x = self.patch_embedding(x)
        f, h, w = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()

        action_binned = self._bin_action(keyboard_action, f)

        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

        for i, block in enumerate(self.blocks):
            x = self._inject_action(x, action_binned, i, f, h, w)
            x = block(x, ctx, t_mod, freqs)

        x = self.head(x, t)
        x = rearrange(
            x, "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=f, h=h, w=w,
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2],
        )
        return x
