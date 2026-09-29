"""HNM source records with explicit prompt or all-Action conditions.

The default path keeps NPC execution audit-only. The all-Action view supplies
verified controller-cache sequences through the explicit npc_action_sequences
argument, while preserving the original records and physical subject roster.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Mapping, Sequence
from selective_agency.constants import CONTROL_STEPS, MAX_SUBJECTS
from .hnm_prompts import PROMPT_PROTOCOL, compile_hnm_prompt
from .records import (
    SourceViewFile,
    ViewContractError,
    ViewRecord,
    _identity_label,
    _resolve_model_path,
)


def parse_hnm_record(
    raw_line: bytes | str,
    *,
    asset_root: Path,
    action_vocabulary: Sequence[str],
    expected_split: str | None = None,
    require_assets: bool = True,
    prompt_protocol: str = PROMPT_PROTOCOL,
    npc_action_sequences: Mapping[str, Sequence[str]] | None = None,
) -> ViewRecord:
    raw_bytes = raw_line if isinstance(raw_line, bytes) else raw_line.encode("utf-8")
    item = json.loads(raw_bytes)
    if item.get("schema_version") != 4 or item.get("sample_type") not in {"hnm", "reactivity"}:
        raise ViewContractError("HNM requires schema4 hnm/reactivity records")
    split = item.get("split")
    if split not in {"train", "val", "test"} or (expected_split and split != expected_split):
        raise ViewContractError("HNM split differs from requested source")
    index = item.get("sample_index")
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ViewContractError("HNM sample_index must be a nonnegative integer")
    vocabulary = tuple(action_vocabulary)
    if not vocabulary or len(set(vocabulary)) != len(vocabulary):
        raise ViewContractError("HNM Action vocabulary must be nonempty and unique")
    lookup = {name: row for row, name in enumerate(vocabulary)}
    model = item["model_input"]
    subjects = model["subjects"]
    if not 2 <= len(subjects) <= MAX_SUBJECTS:
        raise ViewContractError("HNM subject cardinality must be 2..6")
    handles = tuple((_identity_label(subject["handle"], "subject handle") for subject in subjects))
    if len(set(handles)) != len(handles):
        raise ViewContractError("HNM physical subject handles must be unique")
    handle_to_slot = {handle: slot for slot, handle in enumerate(handles)}
    npc_handles = {s["handle"] for s in subjects if s.get("role") == "npc"}
    all_actions = npc_action_sequences is not None
    if all_actions and (
        set(npc_action_sequences) != npc_handles or prompt_protocol != "no-npc-prompts"
    ):
        raise ViewContractError(
            "all-Action HNM requires exactly the source NPC roster and no prompt protocol"
        )
    mask_entries = model["x0_masks"]
    masks = {
        mask["handle"]: _resolve_model_path(mask["path"], asset_root) for mask in mask_entries
    }
    if len(mask_entries) != len(handles) or set(masks) != set(handles):
        raise ViewContractError("HNM first-frame masks do not cover physical subjects")
    mask_paths = tuple((masks[handle] for handle in handles))
    try:
        engine_ids = tuple((int(path.stem) for path in mask_paths))
    except ValueError as exc:
        raise ViewContractError("HNM masks must retain source engine IDs") from exc
    if len(set(engine_ids)) != len(engine_ids):
        raise ViewContractError("HNM engine IDs must be distinct")
    kinds, prompts, per_subject = ([], [], [])
    for subject in subjects:
        role = subject.get("role")
        if role == "external_player":
            actions = subject.get("actions")
            if (
                not isinstance(actions, list)
                or len(actions) != CONTROL_STEPS
                or any((a not in lookup for a in actions))
            ):
                raise ViewContractError("HNM external Action vocabulary/time alignment differs")
            kinds.append(1)
            prompts.append(None)
            per_subject.append(tuple(actions))
        elif role == "npc":
            prompt = (
                None
                if all_actions
                else compile_hnm_prompt(subject, prompt_protocol=prompt_protocol)
            )
            source_target = subject.get("target")
            if (
                not isinstance(source_target, dict)
                or source_target.get("role") != "external_player"
                or source_target.get("handle") not in handle_to_slot
            ):
                raise ViewContractError(
                    "HNM source NPC target must identify an external-player instance"
                )
            source_target_slot = handle_to_slot[source_target["handle"]]
            if subjects[source_target_slot].get("role") != "external_player":
                raise ViewContractError("HNM source NPC target resolves to a non-player subject")
            npc_actions = (
                tuple(npc_action_sequences[subject["handle"]])
                if all_actions
                else (None,) * CONTROL_STEPS
            )
            if all_actions and (
                len(npc_actions) != CONTROL_STEPS or any((a not in lookup for a in npc_actions))
            ):
                raise ViewContractError("HNM NPC Action vocabulary/time alignment differs")
            kinds.append(1 if all_actions else 2)
            prompts.append(prompt)
            per_subject.append(npc_actions)
        else:
            raise ViewContractError(f"unsupported original HNM role: {role!r}")
    if (item["sample_type"] == "hnm") != (not npc_handles):
        raise ViewContractError("HNM sample type and original NPC roster differ")
    actions = tuple(
        (tuple((column[step] for column in per_subject)) for step in range(CONTROL_STEPS))
    )
    ids = tuple((tuple((-1 if name is None else lookup[name] for name in row)) for row in actions))
    x0_rgb = _resolve_model_path(model["x0_rgb"], asset_root)
    video = _resolve_model_path(item["training_target"]["rgb_video"], asset_root)
    if require_assets:
        for path in (x0_rgb, video, *mask_paths):
            if not path.is_file():
                raise ViewContractError(f"HNM model-visible asset missing: {path}")
    audit = item.get("audit", {})
    id_video = (
        _resolve_model_path(audit["id_video"], asset_root) if audit.get("id_video") else None
    )
    return ViewRecord(
        schema_version=4,
        split=split,
        sample_index=index,
        family_id=_identity_label(item["family_id"], "family_id"),
        branch=_identity_label(item["branch_id"], "branch_id"),
        subjects=handles,
        x0_rgb=x0_rgb,
        x0_masks=mask_paths,
        actions=actions,
        rgb_video=video,
        id_video=id_video,
        engine_subject_ids=engine_ids,
        data_format="hnm_all_actions" if all_actions else "hnm",
        control_kind=tuple(kinds),
        action_ids=ids,
        npc_prompts=tuple(prompts),
        origin_dataset="hnm_reactivity_v3",
        sample_id=_identity_label(item["sample_id"], "sample_id"),
        source_episode_id=item["family_id"],
        source_metadata={
            "sample_type": item["sample_type"],
            "source": item.get("source"),
            "subjects": subjects,
            "audit": audit,
        },
        prompt_compiler_version=prompt_protocol,
    )


class HNMViewFile(SourceViewFile):

    def __init__(self, path, split, *, action_vocabulary, allow_test=False, prompt_protocol=None):
        super().__init__(path, split, action_vocabulary=action_vocabulary, allow_test=allow_test)
        self.data_format = "hnm"
        self.raw_id_to_row = {row: row for row in range(len(self.action_vocabulary))}
        self.action_protocol = "hnm-canonical-actions"
        self.prompt_compiler_version = prompt_protocol or PROMPT_PROTOCOL

    def record(self, index: int, *, require_assets: bool = True) -> ViewRecord:
        return parse_hnm_record(
            self.raw_line(index),
            asset_root=self.asset_root,
            action_vocabulary=self.action_vocabulary,
            expected_split=self.split,
            require_assets=require_assets,
            prompt_protocol=self.prompt_compiler_version,
        )
