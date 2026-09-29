"""CSV sample locations and explicit per-subject controls, without dataset versions."""

import csv
import json
from collections import Counter
from pathlib import Path

from .io import read_json, write_json, resolve_path, relative_path
from .geometry import DEFAULT_VIDEO

COLUMNS = (
    "sample_id", "split", "video", "first_frame", "masks", "controls", "latents", "text_cache"
)
TENSOR_NAMES = {"video": "input_latents", "first": "first_frame_latents"}


def text_asset_path(value, base):
    path = resolve_path(value, base)
    return path / "control_text.safetensors" if path.is_dir() else path


class CSVIndex:
    def __init__(self, path, *, text_cache=None, control_steps=DEFAULT_VIDEO.control_steps):
        self.control_steps = control_steps
        self.path = Path(path).expanduser().resolve()
        if self.path.is_dir():
            self.path /= "samples.csv"
        self.root = self.path.parent
        with self.path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            missing = {"sample_id", "split", "masks", "controls"} - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{self.path}: missing CSV columns {sorted(missing)}")
            raw_rows = list(reader)
        if not raw_rows:
            raise ValueError(f"empty CSV: {self.path}")
        if text_cache is not None:
            text_paths = {text_asset_path(text_cache, Path.cwd())}
        else:
            text_paths = {
                text_asset_path(row["text_cache"], self.root)
                for row in raw_rows if row.get("text_cache")
            }
        if len(text_paths) > 1:
            raise ValueError("one CSV uses one shared Action/NPC text table; use --text-cache to select it")
        self.text_path = next(iter(text_paths), None)
        self.text_metadata = (
            read_json(self.text_path.with_suffix(".json")) if self.text_path is not None else None
        )
        source_spec = self.text_metadata["spec"] if self.text_metadata else {}
        self.actions = list(source_spec.get("actions", ()))
        if len(self.actions) != len(set(self.actions)):
            raise ValueError("text cache contains duplicate action names")
        prompts = list(source_spec.get("npc_prompts", ()))
        if len(prompts) != len(set(prompts)):
            raise ValueError("text cache contains duplicate NPC prompts")
        self.records = []
        identities = set()
        positions = Counter()
        for line, raw in enumerate(raw_rows, 2):
            try:
                sample_id, split = raw["sample_id"], raw["split"]
                if not sample_id or not split:
                    raise ValueError("sample_id and split must be nonempty")
                identity = (split, sample_id)
                if identity in identities:
                    raise ValueError(f"duplicate sample_id within split: {identity}")
                identities.add(identity)
                masks = json.loads(raw["masks"])
                payload = read_json(resolve_path(raw["controls"], self.root))
                subjects = payload["subjects"]
                if not isinstance(masks, list) or not 2 <= len(subjects) <= 6 or len(masks) != len(subjects):
                    raise ValueError("provide 2..6 subjects and one mask path for each, in the same order")
                if any(not isinstance(mask, str) or not mask for mask in masks):
                    raise ValueError("mask paths must be nonempty strings")
                names, kinds, subject_actions, subject_prompts, visible = [], [], [], [], []
                for slot, subject in enumerate(subjects):
                    name = subject.get("name", f"P{slot + 1}")
                    if not isinstance(name, str) or not name or name in names:
                        raise ValueError("subject names must be nonempty and unique within a sample")
                    names.append(name)
                    visibility = subject.get("visible", True)
                    if not isinstance(visibility, bool):
                        raise ValueError("subject.visible must be a boolean")
                    visible.append(visibility)
                    if ("actions" in subject) == ("prompt" in subject):
                        raise ValueError("each subject supplies either actions or prompt")
                    if "actions" in subject:
                        actions = subject["actions"]
                        if not isinstance(actions, list) or len(actions) != self.control_steps or any(
                            not isinstance(action, str) or not action.strip() for action in actions
                        ):
                            raise ValueError(f"an Action subject supplies {self.control_steps} nonempty action names")
                        for action in actions:
                            if action not in self.actions:
                                self.actions.append(action)
                        kinds.append(1)
                        subject_actions.append(actions)
                        subject_prompts.append(None)
                    else:
                        prompt = subject["prompt"]
                        if not isinstance(prompt, str) or not prompt.strip():
                            raise ValueError("an NPC subject supplies a nonempty prompt")
                        if prompt not in prompts:
                            prompts.append(prompt)
                        kinds.append(2)
                        subject_actions.append([None] * self.control_steps)
                        subject_prompts.append(prompt)
                sample_index = payload.get("sample_index", positions[split])
                if isinstance(sample_index, bool) or not isinstance(sample_index, int) or sample_index < 0:
                    raise ValueError("sample_index must be a nonnegative integer")
                family = payload.get("family_id", sample_id)
                tensor_names = payload.get("tensor_names", TENSOR_NAMES)
                if set(tensor_names) != {"video", "first"} or any(
                    not isinstance(value, str) or not value for value in tensor_names.values()
                ):
                    raise ValueError("tensor_names must name the video and first-frame tensors")
                self.records.append({
                    "record_id": sample_id,
                    "split": split,
                    "sample_index": sample_index,
                    "family_id": family,
                    "branch": payload.get("branch", "main"),
                    "subjects": names,
                    "control_kind": kinds,
                    "action_names": subject_actions,
                    "npc_prompts": subject_prompts,
                    "source_identity": payload.get("source_identity", ["csv", split, sample_id]),
                    "source_group_identity": payload.get("source_group_identity", ["csv", family]),
                    "x0_masks": [str(resolve_path(mask, self.root)) for mask in masks],
                    "subject_visible": visible,
                    "tensor_names": dict(tensor_names),
                    **{
                        key: str(resolve_path(raw[key], self.root)) if raw.get(key) else None
                        for key in ("video", "first_frame", "latents")
                    },
                })
                positions[split] += 1
            except (ValueError, KeyError, TypeError, OSError) as error:
                raise ValueError(f"{self.path}:{line}: {error}") from error
        self.prompts = prompts
        action_rows = {text: row for row, text in enumerate(self.actions)}
        for record in self.records:
            subject_actions = record.pop("action_names")
            record["action_ids"] = [
                [-1 if actions[step] is None else action_rows[actions[step]] for actions in subject_actions]
                for step in range(self.control_steps)
            ]

    def text_spec(self):
        from .text import TEXT_PROTOCOL, validate_text_spec

        original = self.text_metadata["spec"] if self.text_metadata else {}
        raw_map = original.get("raw_id_to_row")
        if original.get("actions") != self.actions:
            raw_map = None
        spec = {
            "protocol": TEXT_PROTOCOL,
            "prompt_protocol": "csv",
            "action_protocol": "action-names",
            "actions": self.actions,
            "action_texts": [f"ACTION={name}" for name in self.actions],
            "npc_prompts": self.prompts,
            "raw_id_to_row": raw_map or {str(i): i for i in range(len(self.actions))},
        }
        validate_text_spec(spec)
        return spec


def write_dataset_csv(path, records, actions, *, text_cache=None, assets_dir=None):
    """Write locations into CSV and controls into readable JSON files."""
    path = Path(path).expanduser().resolve()
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assets = Path(assets_dir).resolve() if assets_dir else path.parent
    counts = Counter()
    output = []
    for record in records:
        split = record["split"]
        # Use ordinal directories so split/sample labels are never filesystem paths.
        split_number = list(counts).index(split) if split in counts else len(counts)
        control_path = assets / "controls" / f"split-{split_number}" / f"{counts[split]:06d}.json"
        counts[split] += 1
        if control_path.exists():
            raise FileExistsError(control_path)
        subjects = []
        visibility = record.get("subject_visible", [True] * len(record["subjects"]))
        for slot, (name, kind) in enumerate(zip(record["subjects"], record["control_kind"])):
            subject = {"name": name}
            if not visibility[slot]:
                subject["visible"] = False
            if kind == 1:
                subject["actions"] = [actions[step[slot]] for step in record["action_ids"]]
            else:
                subject["prompt"] = record["npc_prompts"][slot]
            subjects.append(subject)
        controls = {"subjects": subjects}
        for key in ("sample_index", "family_id", "branch", "source_identity", "source_group_identity"):
            if key in record:
                controls[key] = record[key]
        if record.get("tensor_names", TENSOR_NAMES) != TENSOR_NAMES:
            controls["tensor_names"] = record["tensor_names"]
        write_json(control_path, controls)
        output.append({
            "sample_id": record["record_id"],
            "split": split,
            "masks": json.dumps([relative_path(mask, path.parent) for mask in record["x0_masks"]]),
            "controls": relative_path(control_path, path.parent),
            "text_cache": relative_path(text_cache, path.parent) if text_cache else "",
            **{
                key: relative_path(record[key], path.parent) if record.get(key) else ""
                for key in ("video", "first_frame", "latents")
            },
        })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(output)
