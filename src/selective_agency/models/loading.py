"""Streaming shape-matched loading of the three Wan2.2 DiT shards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open


@dataclass(frozen=True)
class LoadReport:
    loaded_tensors: int
    loaded_scalars: int
    skipped_tensors: tuple[str, ...]
    missing_pretrained_tensors: tuple[str, ...]


def _copy_shards(
    destinations: dict[str, torch.Tensor], shard_paths: list[str | Path]
) -> tuple[set[str], list[str], int]:
    loaded: set[str] = set()
    skipped: list[str] = []
    scalars = 0
    with torch.no_grad():
        for shard_path in shard_paths:
            path = Path(shard_path)
            if not path.is_file():
                raise FileNotFoundError(path)
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    source = handle.get_tensor(name)
                    destination = destinations.get(name)
                    if destination is None or destination.shape != source.shape:
                        skipped.append(name)
                        continue
                    destination.copy_(source.to(device=destination.device, dtype=destination.dtype))
                    loaded.add(name)
                    scalars += source.numel()
                    del source
    return loaded, skipped, scalars


def load_shape_matched_wan(model: torch.nn.Module, shard_paths: list[str | Path]) -> LoadReport:
    destinations = model.state_dict()
    custom_prefixes = tuple(getattr(model, "_custom_state_prefixes", ()))
    expected_base = {
        name
        for name in destinations
        if not any(name.startswith(prefix) for prefix in custom_prefixes)
    }
    loaded, skipped, scalars = _copy_shards(destinations, shard_paths)
    missing = tuple(sorted(expected_base - loaded))
    if missing:
        raise RuntimeError(
            f"pretrained shards did not initialize {len(missing)} base tensors; first={missing[:8]}"
        )
    return LoadReport(len(loaded), scalars, tuple(sorted(skipped)), missing)


def load_exported_checkpoint(
    model: torch.nn.Module, checkpoint_directory: str | Path
) -> LoadReport:
    directory = Path(checkpoint_directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file():
        import json

        index = json.loads(index_path.read_text())
        shard_paths = [directory / name for name in sorted(set(index["weight_map"].values()))]
    else:
        shard_paths = sorted(directory.glob("*.safetensors"))
    if not shard_paths:
        raise FileNotFoundError(f"no safetensors model shards in {directory}")
    destinations = model.state_dict()
    loaded, skipped, scalars = _copy_shards(destinations, shard_paths)
    missing = tuple(sorted(set(destinations) - loaded))
    if missing or skipped:
        raise RuntimeError(
            f"checkpoint/model mismatch: missing={missing[:8]}, skipped={tuple(skipped[:8])}"
        )
    return LoadReport(len(loaded), scalars, tuple(), tuple())
