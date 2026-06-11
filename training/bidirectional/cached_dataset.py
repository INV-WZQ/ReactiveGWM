"""CachedReactiveGWMDataset — UnifiedDataset wrapper that overlays VAE+T5 cache.

Pairs with `ReactiveGWM_Code.training.bidirectional.precompute_cache`. On `__getitem__`,
the underlying UnifiedDataset still runs (so action parquet + input_image are
fresh), but four `__cached_*` keys are attached so the training pipeline skips
the PromptEmbedder / InputVideoEmbedder / ImageEmbedderFused / NoiseInitializer
units entirely.

KEEP IN SYNC with `precompute_cache.py` — `video_cache_key` and
`t5_cache_key` MUST stay byte-identical or trained models silently consume
mismatched embeddings.

`failed_rows` (manifest field) is honored opportunistically: if precompute
recorded rows that failed (e.g., corrupt source video), `__getitem__` on those
rows raises rather than serving zero tensors. Caches written without that
field (older SF2 caches) work transparently — `failed_rows` is just empty.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from diffsynth.core import UnifiedDataset

from ReactiveGWM_Code.training.data.action_utils import get_action_op
from ReactiveGWM_Code.training.data.profiles import GameProfile
from ReactiveGWM_Code.training.data.prompt_utils import resolve_prompt


# Hash helpers — MUST match precompute_cache.py.

def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def video_cache_key(rel_path: str, H: int, W: int, nf: int, hdf: int, wdf: int,
                    first_frame: bool = False) -> str:
    base = f"{rel_path}|h={H}|w={W}|nf={nf}|hdf={hdf}|wdf={wdf}|fps=20"
    if first_frame:
        base += "|first_frame=1"
    return sha256_str(base)


def t5_cache_key(resolved_prompt: str) -> str:
    return sha256_str(f"t5|v1|{resolved_prompt}")


def _shard_path(root: Path, kind: str, h: str) -> Path:
    return root / kind / h[:2] / f"{h}.pt"


class CachedReactiveGWMDataset(Dataset):
    """ReactiveGWM cached dataset (game-profile driven).

    Args:
      profile: GameProfile (drives action operator + fixed_prompt fallback).
      base_path / metadata_path: same as UnifiedDataset.
      cache_root: directory written by precompute_cache.py.
      num_frames / height / width: must match cache manifest.
      action_hold_window: same value training uses (default 10).
      use_csv_prompt / prompt_column: prompt resolution config — must match cache.
      repeat: dataset repeat factor (forwarded to UnifiedDataset).
      hdf / wdf: image division factors (default 16, must match cache).
      strict: if True, validates every shard exists at __init__.
    """

    def __init__(
        self,
        profile: GameProfile,
        base_path: str,
        metadata_path: str,
        cache_root: str,
        num_frames: int,
        height: int,
        width: int,
        action_hold_window: int = 10,
        use_csv_prompt: bool = True,
        prompt_column: str = "prompt",
        repeat: int = 1,
        hdf: int = 16,
        wdf: int = 16,
        strict: bool = True,
    ):
        super().__init__()
        self.profile = profile
        self.cache_root = Path(cache_root)
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.hdf = hdf
        self.wdf = wdf
        self.use_csv_prompt = use_csv_prompt
        self.prompt_column = prompt_column

        manifest_path = self.cache_root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"[CachedReactiveGWMDataset] manifest.json missing under {self.cache_root}"
            )
        self.manifest: dict[str, Any] = json.loads(manifest_path.read_text())
        self._assert_config_match()

        # Underlying UnifiedDataset — reads parquet + first frame from disk
        # exactly the way the non-cached training path does.
        self.base = UnifiedDataset(
            base_path=base_path,
            metadata_path=metadata_path,
            repeat=repeat,
            data_file_keys=["video", "action"],
            main_data_operator=UnifiedDataset.default_video_operator(
                base_path=base_path,
                max_pixels=None,
                height=height, width=width,
                height_division_factor=hdf, width_division_factor=wdf,
                num_frames=num_frames,
                time_division_factor=4, time_division_remainder=1,
            ),
            special_operator_map={
                "action": get_action_op(profile, base_path, num_frames,
                                        hold_window=action_hold_window),
            },
        )

        df = pd.read_csv(metadata_path)
        self.num_rows = len(df)
        self.video_hash: list[str] = []
        self.ff_hash: list[str] = []
        self.prompt_hash: list[str] = []
        for i in range(self.num_rows):
            row = df.iloc[i].to_dict()
            rel = row["video"]
            self.video_hash.append(video_cache_key(rel, height, width, num_frames, hdf, wdf, False))
            self.ff_hash.append(video_cache_key(rel, height, width, num_frames, hdf, wdf, True))
            self.prompt_hash.append(t5_cache_key(
                resolve_prompt(row, profile, use_csv_prompt, prompt_column)))

        self.empty_prompt_hash = self.manifest.get("empty_prompt_hash") or t5_cache_key("")
        self.failed_rows = {
            meta["csv_index"]
            for meta in (self.manifest.get("failed_rows") or [])
        }
        if self.failed_rows:
            print(f"[CachedReactiveGWMDataset] manifest records {len(self.failed_rows)} "
                  f"failed rows from precompute; __getitem__ on these will raise.")

        if strict:
            self._assert_all_pt_exist()
        self.empty_context = torch.load(
            _shard_path(self.cache_root, "t5", self.empty_prompt_hash),
            map_location="cpu",
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> dict:
        row_i = i % self.num_rows
        if row_i in self.failed_rows:
            raise RuntimeError(
                f"[CachedReactiveGWMDataset] row {row_i} was marked failed during precompute "
                f"— rerun precompute with --skip_existing after fixing source file."
            )
        data = self.base[i]
        data["__cached_input_latents"] = torch.load(
            _shard_path(self.cache_root, "video", self.video_hash[row_i]),
            map_location="cpu",
        )
        data["__cached_first_frame_latents"] = torch.load(
            _shard_path(self.cache_root, "first_frame", self.ff_hash[row_i]),
            map_location="cpu",
        )
        data["__cached_context_posi"] = torch.load(
            _shard_path(self.cache_root, "t5", self.prompt_hash[row_i]),
            map_location="cpu",
        )
        data["__cached_context_nega"] = self.empty_context
        return data

    @property
    def load_from_cache(self) -> bool:
        return getattr(self.base, "load_from_cache", False)

    def _assert_config_match(self) -> None:
        cfg = self.manifest.get("config") or {}
        want = {
            "height": self.height, "width": self.width, "num_frames": self.num_frames,
            "height_division_factor": self.hdf, "width_division_factor": self.wdf,
            "use_csv_prompt": bool(self.use_csv_prompt),
            "prompt_column": self.prompt_column,
        }
        bad = [(k, cfg.get(k), v) for k, v in want.items() if cfg.get(k) != v]
        if bad:
            lines = "\n  ".join(f"{k}: manifest={a!r} shell={b!r}" for k, a, b in bad)
            raise RuntimeError(
                f"[CachedReactiveGWMDataset] manifest != training flags:\n  {lines}\n"
                f"  cache_root={self.cache_root}"
            )

    def _assert_all_pt_exist(self) -> None:
        missing: list[str] = []
        for h in set(self.video_hash):
            if not _shard_path(self.cache_root, "video", h).exists():
                missing.append(f"video/{h[:2]}/{h}.pt")
        for h in set(self.ff_hash):
            if not _shard_path(self.cache_root, "first_frame", h).exists():
                missing.append(f"first_frame/{h[:2]}/{h}.pt")
        for h in set(self.prompt_hash) | {self.empty_prompt_hash}:
            if not _shard_path(self.cache_root, "t5", h).exists():
                missing.append(f"t5/{h[:2]}/{h}.pt")
        if missing:
            preview = "\n  ".join(missing[:10])
            raise RuntimeError(
                f"[CachedReactiveGWMDataset] {len(missing)} cache files missing; first 10:\n"
                f"  {preview}"
            )
