"""ReactiveGWM game profiles — registry that swaps SF2 / SF3 / future games.

A `GameProfile` carries everything that differs between titles built on the
shared ReactiveGWM training/inference stack:

  - `button_cols`      parquet column order (also the dim order the model sees)
  - `fixed_prompt`     fallback prompt used when use_csv_prompt=False or the
                       per-clip CSV prompt is missing/blank
  - `action_presets`   ordered list of (slug, button-name tuple) for the 13
                       single-action evaluation grid (see eval_action.py).
                       NOTE: SF2 and SF3 share the same 10 button names but the
                       light/medium/heavy punch+kick mapping is permuted, so
                       presets 5-10 differ between the two titles.
  - `default_*`        sensible defaults for video / prompt flags so launch
                       shell scripts can elide them. CLI args still override.

Add a new profile by appending to PROFILES below. No code changes elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


# Canonical 10-button fighting game layout. Both SF2 (Genesis) and SF3 (CPS3)
# parquets store buttons in this exact column order; if a future game permutes
# the layout, redefine this tuple in that game's profile.
_FIGHT_BUTTONS: Tuple[str, ...] = (
    "UP", "DOWN", "LEFT", "RIGHT",
    "Y", "X", "Z", "A", "B", "C",
)


# action_id -> (slug, button-list)  — the canonical 13-action evaluation grid.
# id 0..4 are direction-only and identical across all fighting titles.
# id 5..10 vary because each game maps light/medium/heavy punch+kick to a
# different physical button (see SF2 / SF3 README 拳脚映射 tables).

_DIR_PRESETS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("00_noop",  ()),
    ("01_LEFT",  ("LEFT",)),
    ("02_RIGHT", ("RIGHT",)),
    ("03_UP",    ("UP",)),
    ("04_DOWN",  ("DOWN",)),
)

_JUMP_DIR_PRESETS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("11_UP_RIGHT", ("UP", "RIGHT")),
    ("12_UP_LEFT",  ("UP", "LEFT")),
)

# SF2 (Genesis): X=LP, Y=MP, Z=HP; A=LK, B=MK, C=HK.
_SF2_ATTACK_PRESETS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("05_Z_heavyP",  ("Z",)),
    ("06_X_lightP",  ("X",)),
    ("07_Y_mediumP", ("Y",)),
    ("08_A_lightK",  ("A",)),
    ("09_B_mediumK", ("B",)),
    ("10_C_heavyK",  ("C",)),
)

# SF3 (CPS3 6-panel): A=LP, Y=MP, X=HP; B=LK, C=MK, Z=HK.
_SF3_ATTACK_PRESETS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("05_A_lightP",  ("A",)),
    ("06_Y_mediumP", ("Y",)),
    ("07_X_heavyP",  ("X",)),
    ("08_B_lightK",  ("B",)),
    ("09_C_mediumK", ("C",)),
    ("10_Z_heavyK",  ("Z",)),
)


@dataclass(frozen=True)
class GameProfile:
    """Per-game configuration consumed by training, caching, and inference.

    Frozen so accidental mutation of a registered profile is a hard error.
    """
    name: str
    description: str
    button_cols: Tuple[str, ...]
    fixed_prompt: str
    # Each entry: (slug, tuple of button names). The button names MUST be a
    # subset of `button_cols`; eval_action.py and _mp_runner.py validate this.
    action_presets: Tuple[Tuple[str, Tuple[str, ...]], ...]
    default_height: int = 480
    default_width: int = 608
    default_num_frames: int = 101
    default_use_csv_prompt: bool = True
    # Defaults for hold-last upsample window (10Hz parquet -> 20fps video).
    default_action_hold_window: int = 10

    @property
    def num_buttons(self) -> int:
        return len(self.button_cols)

    def button_index(self, name: str) -> int:
        """Resolve a button name to its column index (raises if unknown)."""
        try:
            return self.button_cols.index(name)
        except ValueError as e:
            raise ValueError(
                f"button {name!r} not in profile {self.name!r}; "
                f"valid: {list(self.button_cols)}"
            ) from e


# ---- Registered profiles --------------------------------------------------

SF2 = GameProfile(
    name="sf2",
    description="Street Fighter II (Genesis): Ryu vs Guile @ Air Force Base",
    button_cols=_FIGHT_BUTTONS,
    fixed_prompt="Street Fighter 2 arcade fighting game gameplay",
    action_presets=tuple(_DIR_PRESETS + _SF2_ATTACK_PRESETS + _JUMP_DIR_PRESETS),
    default_height=480,
    default_width=608,
    default_num_frames=101,
    default_use_csv_prompt=True,
    default_action_hold_window=10,
)

SF3 = GameProfile(
    name="sf3",
    description="Street Fighter III (CPS3): Ryu vs Ibuki @ Ibuki's stage",
    button_cols=_FIGHT_BUTTONS,
    fixed_prompt="SF3 Game.",
    action_presets=tuple(_DIR_PRESETS + _SF3_ATTACK_PRESETS + _JUMP_DIR_PRESETS),
    default_height=480,
    default_width=832,
    default_num_frames=101,
    default_use_csv_prompt=False,
    default_action_hold_window=10,
)


PROFILES: dict[str, GameProfile] = {
    "sf2": SF2,
    "sf3": SF3,
}


def get_profile(name: str) -> GameProfile:
    """Return the registered profile or raise with the list of valid names."""
    if name not in PROFILES:
        raise ValueError(
            f"Unknown ReactiveGWM game profile {name!r}; "
            f"registered: {sorted(PROFILES.keys())}"
        )
    return PROFILES[name]


def profile_names() -> list[str]:
    return sorted(PROFILES.keys())
