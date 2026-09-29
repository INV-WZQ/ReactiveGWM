"""Native Wan self-attention with frame-prefix or full-sequence visibility."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from diffsynth.models.wan_video_dit import rope_apply


def prefix_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    temporal_tokens: int,
    spatial_tokens: int,
) -> torch.Tensor:
    """Attend each query frame to all tokens in its history and current frame."""

    if query.shape != key.shape or query.shape != value.shape or query.ndim != 4:
        raise ValueError("q/k/v must share shape [B,H,T*P,D]")
    if query.shape[2] != temporal_tokens * spatial_tokens:
        raise ValueError("q/k/v sequence disagrees with temporal/spatial grid")
    outputs = []
    for frame in range(temporal_tokens):
        query_start = frame * spatial_tokens
        query_end = query_start + spatial_tokens
        key_end = query_end
        outputs.append(
            F.scaled_dot_product_attention(
                query[:, :, query_start:query_end],
                key[:, :, :key_end],
                value[:, :, :key_end],
                dropout_p=0.0,
            )
        )
    return torch.cat(outputs, dim=2)


def _native_self_attention(
    self_attention: nn.Module,
    hidden_states: torch.Tensor,
    freqs: torch.Tensor,
    *,
    temporal_tokens: int,
    spatial_tokens: int,
    bidirectional: bool,
) -> torch.Tensor:
    """Share the original projections and RoPE between visibility modes."""

    dim = int(self_attention.dim)
    num_heads = int(self_attention.num_heads)
    if dim <= 0 or num_heads <= 0 or dim % num_heads:
        raise ValueError("native self-attention dimensions are invalid")
    if hidden_states.ndim != 3 or hidden_states.shape[-1] != dim:
        raise ValueError("native self-attention input must be [B,T*P,D]")
    if hidden_states.shape[1] != temporal_tokens * spatial_tokens:
        raise ValueError("native self-attention input disagrees with temporal/spatial grid")

    query = rope_apply(self_attention.norm_q(self_attention.q(hidden_states)), freqs, num_heads)
    key = rope_apply(self_attention.norm_k(self_attention.k(hidden_states)), freqs, num_heads)
    value = self_attention.v(hidden_states)
    batch, sequence, _ = query.shape
    head_dim = dim // num_heads
    query = query.reshape(batch, sequence, num_heads, head_dim).transpose(1, 2)
    key = key.reshape(batch, sequence, num_heads, head_dim).transpose(1, 2)
    value = value.reshape(batch, sequence, num_heads, head_dim).transpose(1, 2)
    if bidirectional:
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
    else:
        attended = prefix_scaled_dot_product_attention(
            query,
            key,
            value,
            temporal_tokens=temporal_tokens,
            spatial_tokens=spatial_tokens,
        )
    attended = attended.transpose(1, 2).reshape(batch, sequence, dim)
    return self_attention.o(attended)


def native_frame_causal_self_attention(
    self_attention: nn.Module,
    hidden_states: torch.Tensor,
    freqs: torch.Tensor,
    *,
    temporal_tokens: int,
    spatial_tokens: int,
) -> torch.Tensor:
    """Run native Wan attention with exact frame-prefix visibility."""
    return _native_self_attention(
        self_attention,
        hidden_states,
        freqs,
        temporal_tokens=temporal_tokens,
        spatial_tokens=spatial_tokens,
        bidirectional=False,
    )


def native_bidirectional_self_attention(
    self_attention: nn.Module,
    hidden_states: torch.Tensor,
    freqs: torch.Tensor,
    *,
    temporal_tokens: int,
    spatial_tokens: int,
) -> torch.Tensor:
    """Let every video token, including the anchor, read the full sequence."""
    return _native_self_attention(
        self_attention,
        hidden_states,
        freqs,
        temporal_tokens=temporal_tokens,
        spatial_tokens=spatial_tokens,
        bidirectional=True,
    )
