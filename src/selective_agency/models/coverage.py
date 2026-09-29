"""Pure geometry for first-frame subject-handle coverage."""

from __future__ import annotations

from typing import NamedTuple

import torch


class VisibleCoverage(NamedTuple):
    subjects: torch.Tensor
    background: torch.Tensor


def prepare_visible_coverage(
    subject_masks: torch.Tensor, subject_valid: torch.Tensor
) -> VisibleCoverage:
    """Return the subject/background partition on the patch grid."""

    if subject_masks.ndim != 4 or subject_valid.shape != subject_masks.shape[:2]:
        raise ValueError("coverage expects masks[B,N,H,W] and valid[B,N]")
    masks = subject_masks.float()
    if not bool(torch.isfinite(masks).all()):
        raise ValueError("subject coverage must be finite")
    if bool(((masks < 0) | (masks > 1)).any()):
        raise ValueError("subject coverage must lie in [0,1]")
    masks = masks * subject_valid[:, :, None, None].to(masks.dtype)
    total = masks.sum(dim=1, keepdim=True)
    masks = masks / total.clamp_min(1.0)
    background = (1.0 - masks.sum(dim=1)).clamp(0.0, 1.0)
    return VisibleCoverage(subjects=masks, background=background)
