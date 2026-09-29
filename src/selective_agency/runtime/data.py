"""Read cached tensors and controls at the locations listed in a CSV."""

from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from safetensors import safe_open

from .io import resolve_path
from .csv_data import CSVIndex
from .geometry import DEFAULT_VIDEO


def validate_latents(tensors, *, control_steps=DEFAULT_VIDEO.control_steps, geometry=None, source="cache"):
    """Apply the same shape, dtype and finite-value checks to every cached input."""
    first = tensors["first_frame_latents"]
    if (
        first.ndim != 4 or first.shape[0] < 1 or first.shape[1] != 1
        or any(size < 2 or size % 2 for size in first.shape[-2:])
        or first.dtype not in (torch.bfloat16, torch.float32)
        or not torch.isfinite(first).all()
    ):
        raise ValueError(f"{source}: first-frame latents must be finite FP32/BF16 [C,1,H,W] with even H/W")
    if geometry is not None and first.shape[-2:] != geometry.latent_size:
        raise ValueError(f"{source}: latent size {tuple(first.shape[-2:])} differs from video configuration {geometry.latent_size}")
    video = tensors.get("input_latents")
    if video is not None and (
        video.shape != (first.shape[0], control_steps + 1, *first.shape[-2:])
        or video.dtype not in (torch.bfloat16, torch.float32)
        or not torch.isfinite(video).all()
    ):
        raise ValueError(f"{source}: video latents must be finite FP32/BF16 with {control_steps + 1} frames matching the first frame")


def read_mask(path, shape, *, allow_empty=False):
    with Image.open(path) as image:
        grayscale = image.convert("L")
        original = np.asarray(grayscale, dtype=np.uint8).copy() > 0
        nearest = grayscale.resize((shape[1], shape[0]), resample=Image.Resampling.NEAREST)
        hard = torch.from_numpy(np.asarray(nearest, dtype=np.uint8).copy()) > 0
    if not original.any() or not hard.any():
        if not allow_empty:
            raise ValueError(f"empty subject mask: {path}")
        if original.any():
            raise ValueError(f"invisible mask is no longer empty: {path}")
        return hard, torch.zeros(shape, dtype=torch.float32), torch.zeros(4)
    soft = F.interpolate(
        torch.from_numpy(original.astype(np.float32))[None, None], size=shape, mode="area"
    )[0, 0].contiguous()
    ys, xs = np.nonzero(original)
    height, width = original.shape
    box = torch.tensor(
        [xs.min() / width, ys.min() / height, (xs.max() + 1) / width, (ys.max() + 1) / height],
        dtype=torch.float32,
    )
    return hard, soft, box


class CachedDataset(Dataset):
    def __init__(self, data, split, *, seed=20260808, permutation="identity", text_cache=None, geometry=None, allow_uncached=False):
        self.geometry = geometry
        steps = geometry.control_steps if geometry is not None else DEFAULT_VIDEO.control_steps
        self.index = data if isinstance(data, CSVIndex) else CSVIndex(
            data, text_cache=text_cache, control_steps=steps
        )
        self.control_steps = self.index.control_steps
        if geometry is not None and self.control_steps != geometry.control_steps:
            raise ValueError("CSV action length and video configuration differ")
        self.root = self.index.root
        self.records = [row for row in self.index.records if row["split"] == split]
        if not self.records:
            raise ValueError(f"{self.index.path}: no samples in split {split!r}")
        if not allow_uncached and any(not row["latents"] for row in self.records):
            raise ValueError("CSV has unencoded samples; run prepare_cache.py and use its samples.csv")
        if permutation not in {"identity", "train_random"}:
            raise ValueError("unknown permutation policy")
        self.seed, self.permutation = int(seed), permutation
        self.actions = self.index.actions
        self.prompts = self.index.prompts
        self.prompt_to_id = {text: row for row, text in enumerate(self.prompts)}

    def load_text(self, *, expected_dim=None):
        from .text import load_text_asset

        if self.index.text_path is None:
            raise ValueError("set text_cache in the CSV, or encode the text with prepare_cache.py")
        tensors, metadata = load_text_asset(self.index.text_path, expected_dim=expected_dim)
        if metadata["spec"]["actions"] != self.actions or metadata["spec"]["npc_prompts"] != self.prompts:
            raise ValueError("CSV contains text absent from its T5 table; encode text with prepare_cache.py")
        return tensors, metadata

    def __len__(self):
        return len(self.records)

    def _item(self, key, *, forced_permutation=None, load_video=True, first_frame_latents=None):
        from selective_agency.data.permutation import deterministic_train_permutation

        index, epoch, access = key if isinstance(key, tuple) else (key, 0, 0)
        row = self.records[index]
        count = len(row["subjects"])
        if not 2 <= count <= 6:
            raise ValueError("subject count must be 2..6")
        names = row.get("tensor_names", {"video": "input_latents", "first": "first_frame_latents"})
        path = row["latents"] or row["first_frame"]
        if first_frame_latents is not None:
            if load_video:
                raise ValueError("first-frame override is inference-only")
            first, video = first_frame_latents, None
        else:
            if not row["latents"]:
                raise ValueError("encode the first frame before loading an uncached inference item")
            with safe_open(row["latents"], framework="pt", device="cpu") as handle:
                first = handle.get_tensor(names["first"])
                video = handle.get_tensor(names["video"]) if load_video else None
        validate_latents(
            {"first_frame_latents": first, "input_latents": video},
            control_steps=self.control_steps, geometry=self.geometry, source=path,
        )
        shape = (first.shape[-2] // 2, first.shape[-1] // 2)
        visible = row["subject_visible"]
        masks = [
            read_mask(resolve_path(path, self.root), shape, allow_empty=not is_visible)
            for path, is_visible in zip(row["x0_masks"], visible)
        ]
        if len(masks) != count:
            raise ValueError("mask count differs from subject count")
        if forced_permutation is not None:
            permutation = tuple(forced_permutation)
        elif self.permutation == "train_random":
            permutation = deterministic_train_permutation(
                count,
                master_seed=self.seed,
                epoch=epoch,
                sample_index=row["sample_index"],
                access=access,
            )
        else:
            permutation = tuple(range(count))
        if sorted(permutation) != list(range(count)):
            raise ValueError("invalid role permutation")
        actions = torch.tensor(row["action_ids"], dtype=torch.long)
        if actions.shape != (self.control_steps, count):
            raise ValueError(f"Action conditions must have {self.control_steps} steps per subject")
        prompts = [-1 if p is None else self.prompt_to_id[p] for p in row["npc_prompts"]]
        result = {
            "record_id": row["record_id"],
            "family_id": row["family_id"],
            "branch": row["branch"],
            "cardinality": count,
            "permutation": permutation,
            "subject_masks": torch.stack([masks[i][0] for i in permutation]),
            "subject_roi_masks": torch.stack([masks[i][1] for i in permutation]),
            "subject_boxes": torch.stack([masks[i][2] for i in permutation]),
            "control_kind": torch.tensor(row["control_kind"], dtype=torch.long)[list(permutation)],
            "action_ids": actions[:, list(permutation)],
            "npc_prompt_ids": torch.tensor(prompts, dtype=torch.long)[list(permutation)],
            "source_subjects": tuple(row["subjects"][i] for i in permutation),
            "source_identity": tuple(row["source_identity"]),
            "source_group_identity": tuple(row["source_group_identity"]),
            "first_frame_latents": first,
            "subject_visible": torch.tensor(visible, dtype=torch.bool)[list(permutation)],
        }
        if video is not None:
            result["input_latents"] = video
        return result

    def __getitem__(self, key):
        return self._item(key)

    def item_with_permutation(self, index, permutation):
        return self._item(index, forced_permutation=permutation)

    def inference_item(self, index, *, first_frame_latents=None):
        return self._item(index, load_video=False, first_frame_latents=first_frame_latents)
