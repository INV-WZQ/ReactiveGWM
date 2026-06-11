"""ReactiveGWM action utilities — profile-driven parquet -> tensor pipeline.

Generic replacement for the per-game `data/action_utils.py` files in the
`sf2_final` and `sf3` forks. The button schema and fixed-prompt fallback live
on a `GameProfile` (see `profiles.py`); this module only owns the shape /
upsampling behavior, which is identical across every fighting-game profile we
have seen so far.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch

from diffsynth.core.data.operators import RouteByType

from .profiles import GameProfile


def hold_last_upsample(btn: torch.Tensor, window: int = 10) -> torch.Tensor:
    """Forward-broadcast each window-sized block's first row across the block.

    Parquets are sampled at 10 Hz but stored 1:1 with the 20 fps video, so 9/10
    rows are zero-filled. window=10 collapses that back to a dense per-frame
    signal. window=1 disables the upsample.

    Args:
      btn: [T, num_buttons] float tensor.
      window: block size in frames.
    Returns: [T, num_buttons] tensor with the same dtype/device as `btn`.
    """
    if window <= 1:
        return btn
    T = btn.shape[0]
    idx = (torch.arange(T, device=btn.device) // window) * window
    idx = idx.clamp(max=T - 1)
    return btn[idx]


def get_action_op(
    profile: GameProfile,
    base_path: str,
    num_frames: int,
    hold_window: int | None = None,
) -> RouteByType:
    """UnifiedDataset operator: parquet path -> [1, T, num_buttons] action tensor.

    Strips the parquet to `profile.button_cols`, truncates to `num_frames`,
    casts to float32, then hold-last upsamples.
    """
    cols = list(profile.button_cols)
    win = profile.default_action_hold_window if hold_window is None else hold_window

    def process_parquet(rel_path: str) -> torch.Tensor:
        full = os.path.join(base_path, rel_path)
        df = pd.read_parquet(full)
        arr = df[cols].values[:num_frames].astype(np.float32)
        keyboard = torch.tensor(arr)
        keyboard = hold_last_upsample(keyboard, window=win)
        return keyboard.unsqueeze(0)

    return RouteByType(operator_map=[(str, process_parquet)])
