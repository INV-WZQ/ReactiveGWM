"""Padding and stacking for records with variable subject counts."""

from __future__ import annotations

from typing import Any

import torch

from selective_agency.constants import (
    ID_OCCUPANCY_CHANNELS,
    MAX_SUBJECTS,
)


def collate_samples(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty batch")
    result: dict[str, Any] = {
        "record_id": [item["record_id"] for item in batch],
        "family_id": [item["family_id"] for item in batch],
        "branch": [item["branch"] for item in batch],
        "permutation": [item["permutation"] for item in batch],
    }
    for key in ("source_identity", "source_group_identity", "source_subjects"):
        if any(key in item for item in batch):
            result[key] = [item.get(key) for item in batch]
    for key in ("input_latents", "first_frame_latents"):
        result[key] = torch.stack([item[key] for item in batch], dim=0)
    occupancy_presence = ["occupancy_counts" in item for item in batch]
    if any(occupancy_presence) and not all(occupancy_presence):
        raise ValueError("batch mixes records with and without occupancy sidecars")
    batch_size = len(batch)
    control_steps = batch[0]["action_ids"].shape[0]
    mask_h, mask_w = batch[0]["subject_masks"].shape[-2:]
    masks = torch.zeros((batch_size, MAX_SUBJECTS, mask_h, mask_w), dtype=torch.bool)
    roi_masks = torch.zeros((batch_size, MAX_SUBJECTS, mask_h, mask_w), dtype=torch.float32)
    action_ids = torch.full((batch_size, control_steps, MAX_SUBJECTS), -1, dtype=torch.long)
    control_kind = torch.zeros((batch_size, MAX_SUBJECTS), dtype=torch.long)
    npc_prompt_ids = torch.full((batch_size, MAX_SUBJECTS), -1, dtype=torch.long)
    valid = torch.zeros((batch_size, MAX_SUBJECTS), dtype=torch.bool)
    cardinality = torch.empty((batch_size,), dtype=torch.long)
    occupancy_counts = None
    if all(occupancy_presence):
        occupancy_counts = torch.zeros(
            (
                batch_size,
                control_steps,
                ID_OCCUPANCY_CHANNELS,
                mask_h,
                mask_w,
            ),
            dtype=torch.int16,
        )
    for row, item in enumerate(batch):
        count = int(item["cardinality"])
        if not 2 <= count <= MAX_SUBJECTS:
            raise ValueError(f"invalid cardinality {count}")
        if item["subject_masks"].shape != (count, mask_h, mask_w):
            raise ValueError("subject mask shape/cardinality mismatch")
        if item["subject_roi_masks"].shape != (count, mask_h, mask_w):
            raise ValueError("subject ROI mask shape/cardinality mismatch")
        if item["subject_boxes"].shape != (count, 4):
            raise ValueError("subject box shape/cardinality mismatch")
        if item["subject_roi_masks"].dtype != torch.float32:
            raise ValueError("subject ROI masks must be float32")
        if item["subject_boxes"].dtype != torch.float32:
            raise ValueError("subject boxes must be float32")
        if not bool(torch.isfinite(item["subject_roi_masks"]).all()):
            raise ValueError("subject ROI masks must be finite")
        if not bool(torch.isfinite(item["subject_boxes"]).all()):
            raise ValueError("subject boxes must be finite")
        if bool(((item["subject_roi_masks"] < 0) | (item["subject_roi_masks"] > 1)).any()):
            raise ValueError("subject ROI masks must lie in [0,1]")
        item_boxes = item["subject_boxes"]
        if bool(((item_boxes < 0) | (item_boxes > 1)).any()):
            raise ValueError("subject boxes must lie in [0,1]")
        visible = item.get("subject_visible", torch.ones(count, dtype=torch.bool))
        if visible.shape != (count,) or visible.dtype != torch.bool:
            raise ValueError("subject visibility must be a boolean per subject")
        invisible = ~visible
        if (
            bool(item["subject_masks"][invisible].any())
            or bool(item["subject_roi_masks"][invisible].any())
            or bool(item_boxes[invisible].any())
        ):
            raise ValueError("subjects marked invisible must have zero masks and boxes")
        if bool(
            (((item_boxes[:, 2] <= item_boxes[:, 0]) | (item_boxes[:, 3] <= item_boxes[:, 1])) & visible).any()
        ):
            raise ValueError("subject boxes must have positive extent")
        if bool(((item["subject_roi_masks"].sum(dim=(-2, -1)) <= 0) & visible).any()):
            raise ValueError("valid subject ROI masks must be non-empty")
        if item["action_ids"].shape != (control_steps, count):
            raise ValueError("action id shape/cardinality mismatch")
        kinds = item.get("control_kind", torch.ones(count, dtype=torch.long))
        prompts = item.get("npc_prompt_ids", torch.full((count,), -1, dtype=torch.long))
        if any(value.shape != (count,) for value in (kinds, prompts)):
            raise ValueError("mixed-role condition shape/cardinality mismatch")
        if bool(((kinds != 1) & (kinds != 2)).any()):
            raise ValueError("valid subjects must be external_action or npc_prompt")
        external, npc = kinds == 1, kinds == 2
        if (
            bool((item["action_ids"][:, external] < 0).any())
            or bool((item["action_ids"][:, npc] != -1).any())
            or bool((prompts[external] != -1).any())
            or bool((prompts[npc] < 0).any())
        ):
            raise ValueError("Action/prompt IDs must be valid only for their declared control kind")
        masks[row, :count] = item["subject_masks"]
        roi_masks[row, :count] = item["subject_roi_masks"]
        action_ids[row, :, :count] = item["action_ids"]
        control_kind[row, :count] = kinds
        npc_prompt_ids[row, :count] = prompts
        valid[row, :count] = True
        cardinality[row] = count
        if occupancy_counts is not None:
            expected = (control_steps, count + 1, mask_h, mask_w)
            if item["occupancy_counts"].shape != expected:
                raise ValueError("occupancy count shape/cardinality mismatch")
            occupancy_counts[row, :, : count + 1] = item["occupancy_counts"]
    result.update(
        subject_masks=masks,
        subject_roi_masks=roi_masks,
        action_ids=action_ids,
        control_kind=control_kind,
        npc_prompt_ids=npc_prompt_ids,
        subject_valid=valid,
        cardinality=cardinality,
    )
    if occupancy_counts is not None:
        result["occupancy_counts"] = occupancy_counts
    return result
