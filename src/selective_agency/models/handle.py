"""Episode-local learned handles for anchor injection, key binding, and routing."""

from __future__ import annotations

import torch
from torch import nn

from selective_agency.constants import HANDLE_RMS_EPS


def rms_normalize_fp32(value: torch.Tensor, eps: float = HANDLE_RMS_EPS) -> torch.Tensor:
    """Normalize the final dimension with an FP32 reduction."""

    value_fp32 = value.float()
    inverse_rms = torch.rsqrt(value_fp32.square().mean(dim=-1, keepdim=True) + eps)
    return value_fp32 * inverse_rms


class HandleCodebook(nn.Module):
    """One background row followed by the six subject-handle rows."""

    def __init__(self, dim: int, max_subjects: int, eps: float = HANDLE_RMS_EPS):
        super().__init__()
        if dim <= 0 or max_subjects <= 0 or eps <= 0:
            raise ValueError("handle codebook dimensions and epsilon must be positive")
        self.dim = int(dim)
        self.max_subjects = int(max_subjects)
        self.eps = float(eps)
        self.embedding = nn.Embedding(max_subjects + 1, dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, subject_valid: torch.Tensor) -> torch.Tensor:
        if subject_valid.ndim != 2 or subject_valid.shape[1] != self.max_subjects:
            raise ValueError("subject validity must be [B,max_subjects]")
        normalized = rms_normalize_fp32(self.embedding.weight, self.eps).to(
            self.embedding.weight.dtype
        )
        handles = normalized.unsqueeze(0).expand(subject_valid.shape[0], -1, -1)
        slot_valid = torch.cat(
            (
                torch.ones(
                    subject_valid.shape[0],
                    1,
                    dtype=torch.bool,
                    device=subject_valid.device,
                ),
                subject_valid.bool(),
            ),
            dim=1,
        )
        return handles * slot_valid.unsqueeze(-1).to(handles.dtype)


def inject_first_frame_handles(
    video_tokens: torch.Tensor,
    subject_handles: torch.Tensor,
    subject_coverage: torch.Tensor,
    *,
    temporal_tokens: int,
    spatial_tokens: int,
    alpha: float,
) -> torch.Tensor:
    """Add soft-coverage-weighted subject handles to the anchor frame only."""

    if video_tokens.ndim != 3:
        raise ValueError("video tokens must be [B,T*P,D]")
    batch, sequence, dim = video_tokens.shape
    if sequence != temporal_tokens * spatial_tokens:
        raise ValueError("video token sequence disagrees with temporal/spatial grid")
    if subject_handles.ndim != 3 or subject_handles.shape[0] != batch:
        raise ValueError("subject handles must be [B,N,D]")
    if subject_handles.shape[-1] != dim:
        raise ValueError("subject handle width differs from video token width")
    if subject_coverage.shape != (
        batch,
        subject_handles.shape[1],
        spatial_tokens,
    ):
        raise ValueError("subject coverage must be [B,N,P]")
    if alpha < 0:
        raise ValueError("handle injection alpha must be non-negative")
    if float(alpha) == 0.0:
        # Disabling anchor injection must leave the tokens unchanged.
        return video_tokens
    binding = torch.einsum(
        "bnp,bnd->bpd",
        subject_coverage.to(video_tokens.dtype),
        subject_handles.to(video_tokens.dtype),
    )
    frames = video_tokens.reshape(batch, temporal_tokens, spatial_tokens, dim)
    anchor = frames[:, 0] + float(alpha) * binding
    return torch.cat((anchor[:, None], frames[:, 1:]), dim=1).reshape_as(video_tokens)
