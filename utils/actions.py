"""Action loading: parquet → keyboard tensor, with hold-window densification."""
import numpy as np
import pandas as pd
import torch

from ..constants import SF_BUTTON_COLS


def hold_last_upsample(btn: torch.Tensor, window: int) -> torch.Tensor:
    """Forward-broadcast each `window`-sized bucket's start value over its siblings.

    Mirrors training-time densification: parquet sampled at 10Hz, video at 20fps,
    so half the rows would be zeros — this fills them by holding the bucket's
    first value (including legitimate zeros).
    """
    if window <= 1:
        return btn
    T = btn.shape[0]
    idx = (torch.arange(T) // window) * window
    idx = idx.clamp(max=T - 1)
    return btn[idx]


def load_actions(parquet_path: str, num_frames: int, hold_window: int = 10,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Read a parquet, slice to `num_frames`, hold-upsample, return [1, T, 10] tensor."""
    df = pd.read_parquet(parquet_path)
    arr = df[SF_BUTTON_COLS].values[:num_frames].astype(np.float32)
    btn = torch.tensor(arr)
    btn = hold_last_upsample(btn, hold_window)
    return btn.unsqueeze(0).to(device, dtype)
