"""ReactiveGWM prompt resolver — shared by training, cache precompute, infer.

Any change to the NaN / empty-string fallback logic MUST be mirrored everywhere
or cache hash keys silently drift and training-time T5 lookups go to the wrong
row. The fixed-prompt fallback string lives on the `GameProfile`.
"""
from __future__ import annotations

import pandas as pd

from .profiles import GameProfile


def resolve_prompt(
    row,
    profile: GameProfile,
    use_csv_prompt: bool = True,
    prompt_column: str = "prompt",
) -> str:
    """Return the exact prompt string T5 will see for this dataset row.

      use_csv_prompt=False               -> profile.fixed_prompt
      raw is None / NaN / blank          -> profile.fixed_prompt
      otherwise                          -> str(raw)

    `row` is anything supporting `.get(key, default)` (pandas Series, dict,
    UnifiedDataset row dict).
    """
    if not use_csv_prompt:
        return profile.fixed_prompt
    raw = row.get(prompt_column, None)
    if raw is None:
        return profile.fixed_prompt
    if isinstance(raw, float) and pd.isna(raw):
        return profile.fixed_prompt
    if isinstance(raw, str) and not raw.strip():
        return profile.fixed_prompt
    return str(raw)
