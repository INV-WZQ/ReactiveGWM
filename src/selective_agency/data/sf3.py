"""SF schema1 NPC0 / schema2 NPC1 adapters for shared FM-only pretraining.

The two source namespaces and their asset roots are explicit. Native NPC
decisions are reconstructed solely by the audit path and never become model
conditions. Formal training requires the accepted combined publication.
"""

from __future__ import annotations
from collections import Counter
import copy
import json
from pathlib import Path, PurePosixPath
from typing import Sequence
import numpy as np
from selective_agency.paths import PROJECT_ROOT, require_within
from .records import ViewContractError, ViewRecord, _identity_label
from .sf_prompts import PROMPT_PROTOCOL, compile_sf_prompt

EXPORT_VERSION = "sf3_v2_pretraining_export_v1"
DATASET_VERSION = "sf3_v2_roles_block12_v1"
PLAYER_VERSION = "sf3_v2_player20_block12_v1"
NPC_VERSION = "sf3_v2_npc18_block12_v1"
ENGINE_IDS = (33608076, 33609060)
BODY_MASK_POLICY = "sf3_primary_character_body_only_v1"
ORIGINS = {"sf3_legacy_npc0": "data/sf3/ready", "sf3_v2_npc1": "data/sf3_v2/npc1_dataset"}
COMBAT_BUTTONS = ("UP", "DOWN", "LEFT", "RIGHT", "A", "Y", "X", "B", "C", "Z")
PORT_BUTTONS = ("B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z")
PLAYER_IDS = (*range(14), 19, 27, 28, 30, 31, 32)


class SF3ActionSpaces:
    """Canonical text rows and independently namespaced exact input templates."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or Path(__file__).with_name("action_spaces.json"))
        payload = self.path.read_bytes()
        self.spec = json.loads(payload)
        if (
            self.spec.get("version") != DATASET_VERSION
            or self.spec.get("block_ticks") != 12
            or self.spec.get("press_ticks") != 3
            or (self.spec.get("native_fps") != 60)
            or (tuple(self.spec.get("combat_buttons", ())) != COMBAT_BUTTONS)
        ):
            raise ViewContractError("SF action_spaces timing/version differs from block12 protocol")
        self.actions, self.templates = ({}, {})
        for namespace, version, ids in (
            ("player", PLAYER_VERSION, set(PLAYER_IDS)),
            ("npc", NPC_VERSION, set(range(18))),
        ):
            space = self.spec["spaces"][namespace]
            actions = space["actions"]
            if (
                space["version"] != version
                or {action["id"] for action in actions} != ids
                or len(actions) != len(ids)
            ):
                raise ViewContractError(f"SF {namespace} namespace/version differs")
            self.actions[namespace] = {action["id"]: action for action in actions}
            self.templates[namespace] = {}
            for action in actions:
                buttons = action["held"] + action["pulsed"]
                if (
                    not set(buttons) <= set(COMBAT_BUTTONS)
                    or len(buttons) != len(set(buttons))
                    or len(buttons) > 2
                    or (len(set(buttons) & set(COMBAT_BUTTONS[4:])) > 1)
                    or ({"LEFT", "RIGHT"} <= set(buttons))
                    or ({"UP", "DOWN"} <= set(buttons))
                ):
                    raise ViewContractError(
                        "SF action contains an invalid simultaneous combination"
                    )
                template = np.zeros((12, 10), dtype=np.uint8)
                for key in action["held"]:
                    template[:, COMBAT_BUTTONS.index(key)] = 1
                for key in action["pulsed"]:
                    template[:3, COMBAT_BUTTONS.index(key)] = 1
                self.templates[namespace][action["id"]] = template
        player_actions = self.spec["spaces"]["player"]["actions"]
        self.player_actions = tuple((action["name"] for action in player_actions))
        if len(set(self.player_actions)) != 20:
            raise ViewContractError("SF Player20 text rows must be unique")
        self.player_raw_id_to_row = {action["id"]: row for row, action in enumerate(player_actions)}

    def template(self, namespace: str, raw_id: int) -> np.ndarray:
        if (
            isinstance(raw_id, (bool, np.bool_))
            or not isinstance(raw_id, (int, np.integer))
            or namespace not in self.templates
            or (int(raw_id) not in self.templates[namespace])
        ):
            raise ViewContractError(f"unknown SF {namespace} raw action ID {raw_id!r}")
        return self.templates[namespace][int(raw_id)].copy()

    def native_from_decisions(self, decisions: np.ndarray, namespaces: Sequence[str]) -> np.ndarray:
        decisions = np.asarray(decisions)
        if decisions.shape != (25, 2) or decisions.dtype.kind not in "iu" or len(namespaces) != 2:
            raise ViewContractError("SF requires exactly 25 integer decisions per physical slot")
        native = np.empty((300, 2, 10), dtype=np.uint8)
        for slot, namespace in enumerate(namespaces):
            for step, raw_id in enumerate(decisions[:, slot]):
                native[12 * step : 12 * (step + 1), slot] = self.template(namespace, raw_id)
        return native


def resolve_sf3_asset(repo_root: Path, row: dict, relative: str, *, nested: bool = False) -> Path:
    origin = row.get("origin_dataset")
    if origin not in ORIGINS or row.get("asset_root") != ORIGINS[origin]:
        raise ViewContractError("SF asset_root differs from its explicit origin_dataset")
    if not isinstance(relative, str) or not relative:
        raise ViewContractError("SF asset path must be nonempty")
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ViewContractError("SF asset path must stay relative to its declared root")
    split = row["split"]
    if split not in {"train", "val", "test"}:
        raise ViewContractError("invalid SF split")
    if nested:
        if path.parts[0] in {"train", "val", "test"}:
            raise ViewContractError("nested SF record path already contains a split")
        path = PurePosixPath(split) / path
    elif path.parts[0] != split:
        raise ViewContractError("top-level SF asset path must already contain its split")
    asset_root = (
        Path(repo_root[origin])
        if isinstance(repo_root, dict)
        else Path(repo_root) / row["asset_root"]
    )
    return require_within(asset_root / str(path), asset_root)


def _valid_slot(value) -> bool:
    return isinstance(value, int) and (not isinstance(value, bool)) and (value in (0, 1))


def audit_sf3_native(
    row: dict, record: dict, *, repo_root: Path, action_spaces: SF3ActionSpaces
) -> dict:
    """Check both raw namespaces against native inputs, ports and native clock.

    The returned report contains no arrays and is never passed to the model.
    """
    path = resolve_sf3_asset(repo_root, row, row["native_actions"])
    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    decisions = arrays["decisions"]
    legacy = row["origin_dataset"] == "sf3_legacy_npc0"
    namespaces = (
        ("player", "player")
        if legacy
        else tuple(("player" if slot == record["player_slot"] else "npc" for slot in range(2)))
    )
    if legacy and (
        decisions.dtype.kind not in "iu" or np.any(decisions < 0) or np.any(decisions > 12)
    ):
        raise ViewContractError("legacy NPC0 raw IDs must remain the original 0..12")
    reconstructed = action_spaces.native_from_decisions(decisions, namespaces)
    native, ports = (arrays["native_inputs"], arrays["applied_ports"])
    if native.shape != (300, 2, 10) or not np.array_equal(reconstructed, native):
        raise ViewContractError(
            "SF native inputs do not round-trip through their exact role-specific templates"
        )
    if ports.shape != (300, 24) or not np.isin(ports, (0, 1)).all():
        raise ViewContractError("SF requires all 300 complete binary physical controller ports")
    shaped = ports.reshape(300, 2, 12)
    columns = [PORT_BUTTONS.index(button) for button in COMBAT_BUTTONS]
    if not np.array_equal(shaped[:, :, columns], native) or shaped[:, :, [2, 3]].any():
        raise ViewContractError(
            "SF applied controller ports differ from native inputs or contain menu keys"
        )
    for name, expected in (
        ("native_tick", np.arange(300)),
        ("source_native_tick", np.arange(row["native_start"], row["native_end"])),
        ("video_native_tick", np.arange(0, 301, 3)),
    ):
        if name not in arrays or not np.array_equal(arrays[name], expected):
            raise ViewContractError(f"SF {name} clock differs from the 25-block/101-frame mapping")
    if legacy:
        protocol_path = resolve_sf3_asset(
            repo_root, row, record["audit"]["action_protocol"], nested=True
        )
        original = json.loads(protocol_path.read_text())
        if (
            tuple(original["combat_buttons"]) != COMBAT_BUTTONS
            or tuple(original["port_buttons_per_player"]) != PORT_BUTTONS
        ):
            raise ViewContractError("legacy NPC0 port order differs")
        for action in original["actions"]:
            if not np.array_equal(
                action_spaces.template("player", action["id"]), action["template"]
            ):
                raise ViewContractError("Player20 changed an original NPC0 action template")
    for slot, subject in enumerate(record["model_input"]["subjects"]):
        if subject["role"] == "external_player":
            if subject["decision_ids"] != decisions[:, slot].tolist():
                raise ViewContractError(
                    "SF external Player decision IDs differ from the submitted per-slot controls"
                )
            if legacy and subject.get("actions") != [
                original["actions"][int(i)]["text"] for i in decisions[:, slot]
            ]:
                raise ViewContractError(
                    "legacy NPC0 execution descriptions differ from raw decisions"
                )
    from .sf3_native import audit_consumption

    audit_consumption(ports, arrays)
    return {"exact": True, "native_ticks": 300, "decisions_per_slot": 25, "namespaces": namespaces}


def parse_sf3_record(
    row: dict,
    *,
    repo_root: Path,
    action_spaces: SF3ActionSpaces,
    sample_index: int = 0,
    expected_split: str | None = None,
    require_assets: bool = True,
    data_format: str = "sf3_ready",
) -> ViewRecord:
    """Parse one explicit export row; synthetic callers use this pure function."""
    if row.get("export_version") != EXPORT_VERSION:
        raise ViewContractError("SF wrapper export_version differs")
    origin = row.get("origin_dataset")
    if origin not in ORIGINS or row.get("asset_root") != ORIGINS[origin]:
        raise ViewContractError("SF has an excluded origin or mismatched asset root")
    split = row.get("split")
    if split not in {"train", "val", "test"} or (expected_split and split != expected_split):
        raise ViewContractError("SF split differs from requested source")
    raw = row["record_json"]
    if not isinstance(raw, str):
        raise ViewContractError("SF record_json must contain the original JSON string")
    record = json.loads(raw)
    if record.get("sample_type") != "sf3":
        raise ViewContractError("SF requires its own sample_type, never portable schema parsing")
    for field in (
        "schema_version",
        "dataset_version",
        "sample_id",
        "split",
        "source_episode_id",
        "window_index",
        "npc_count",
        "p1_character",
        "p2_character",
        "stage_id",
        "stage_round_variant",
        "rule_id",
        "trigger_count",
        "negative_type",
        "branch_id",
        "release",
        "source_mode",
        "mask_policy",
    ):
        if field not in row or row[field] != record.get(field):
            raise ViewContractError(f"SF manifest/record identity differs: {field}")
    legacy = origin == "sf3_legacy_npc0"
    if legacy:
        if (
            record["schema_version"] != 1
            or record["npc_count"] != 0
            or record.get("branch_id") != "external_random"
            or (record.get("source_mode") != "independent_uniform_random")
        ):
            raise ViewContractError(
                "legacy SF source must be unchanged schema1 NPC0; old NPC1 is excluded"
            )
        kinds, namespaces = ((1, 1), (PLAYER_VERSION, PLAYER_VERSION))
    else:
        if (
            record["schema_version"] != 2
            or record["npc_count"] != 1
            or record["dataset_version"] != DATASET_VERSION
        ):
            raise ViewContractError("new SF source must be schema2 block12 NPC1")
        player, npc = (record.get("player_slot"), record.get("npc_slot"))
        if not _valid_slot(player) or not _valid_slot(npc) or npc != 1 - player:
            raise ViewContractError("SF Player/NPC physical slot allocation differs")
        kinds = tuple((1 if slot == player else 2 for slot in range(2)))
        namespaces = tuple((PLAYER_VERSION if slot == player else NPC_VERSION for slot in range(2)))
        roles = ["external_player" if kind == 1 else "reactive_npc" for kind in kinds]
        if record.get("roles_by_slot") != roles or record.get("action_space_by_slot") != list(
            namespaces
        ):
            raise ViewContractError("SF physical roles or role-specific action namespace differs")
    if record.get("mask_policy") != BODY_MASK_POLICY:
        raise ViewContractError("SF masks must bind primary character body instances")
    geometry = {
        "frames": 101,
        "fps": 20,
        "native_transitions": 300,
        "native_fps": 60,
        "width": 832,
        "height": 480,
    }
    if any((row.get(key) != value for key, value in geometry.items())):
        raise ViewContractError("SF geometry differs from 101 frames / 25 future latent controls")
    start, end = (row.get("native_start"), row.get("native_end"))
    if not isinstance(start, int) or start < 0 or start % 12 or (end != start + 300):
        raise ViewContractError("SF native source window must start on a common 12-tick boundary")
    if record["audit"].get("native_start") != start or record["audit"].get("native_end") != end:
        raise ViewContractError("SF audit and manifest source clocks differ")
    model = record["model_input"]
    if set(model) != {"subjects", "x0_rgb", "x0_masks"}:
        raise ViewContractError("SF model_input contains undeclared conditioning")
    subjects = model["subjects"]
    if len(subjects) != 2 or [subject.get("native_slot") for subject in subjects] != [0, 1]:
        raise ViewContractError("SF subjects must remain in physical P1/P2 order")
    handles = ("P1", "P2")
    prompt_slots, target_slots, per_subject = ([], [], [])
    for slot, subject in enumerate(subjects):
        if (
            subject.get("engine_id") != ENGINE_IDS[slot]
            or subject.get("handle") != handles[slot]
            or subject.get("character") != record["p1_character" if slot == 0 else "p2_character"]
        ):
            raise ViewContractError("SF mask/subject native instance identity differs")
        basic = {"native_slot", "handle", "engine_id", "character", "role"}
        if kinds[slot] == 1:
            if subject.get("role") != "external_player":
                raise ViewContractError("SF external Action slot has the wrong role")
            if not legacy and set(subject) != basic | {"decision_ids"}:
                raise ViewContractError("new SF Player condition contains undeclared fields")
            decisions = subject.get("decision_ids")
            if not isinstance(decisions, list) or len(decisions) != 25:
                raise ViewContractError("SF Player requires exactly 25 future decisions")
            dense = []
            for raw_id in decisions:
                action_spaces.template("player", raw_id)
                if legacy and raw_id not in range(13):
                    raise ViewContractError("legacy NPC0 must retain the original 13 raw IDs")
                dense.append(action_spaces.player_raw_id_to_row[raw_id])
            per_subject.append(dense)
            prompt_slots.append(None)
            target_slots.append(-1)
        else:
            if subject.get("role") != "reactive_npc" or set(subject) != basic | {
                "rule_id",
                "target_slot",
                "reactivity",
            }:
                raise ViewContractError(
                    "SF NPC must contain only static rule/binding fields, never future execution IDs"
                )
            if (
                subject["target_slot"] != record["player_slot"]
                or not _valid_slot(subject["target_slot"])
                or subject["rule_id"] != record.get("rule_id")
            ):
                raise ViewContractError("SF NPC Actor/Target or rule identity differs")
            per_subject.append([-1] * 25)
            prompt_slots.append(compile_sf_prompt(subject["reactivity"]))
            target_slots.append(
                -1
                if subject["reactivity"] == {"policy": "all_controls_released"}
                else subject["target_slot"]
            )
    masks = model["x0_masks"]
    if len(masks) != 2 or [mask.get("handle") for mask in masks] != list(handles):
        raise ViewContractError("SF first-frame masks must retain both physical instance handles")
    mask_paths = tuple(
        (resolve_sf3_asset(repo_root, row, mask["path"], nested=True) for mask in masks)
    )
    if any((path.stem != str(engine) for path, engine in zip(mask_paths, ENGINE_IDS))):
        raise ViewContractError("SF first-frame mask does not bind its source engine ID")
    x0_rgb = resolve_sf3_asset(repo_root, row, model["x0_rgb"], nested=True)
    rgb_video = resolve_sf3_asset(repo_root, row, row["rgb_video"])
    id_video = resolve_sf3_asset(repo_root, row, row["id_video"])
    for field, nested in (
        ("rgb_video", record["training_target"]["rgb_video"]),
        ("id_video", record["audit"]["id_video"]),
        ("native_actions", record["audit"]["native_actions"]),
    ):
        if resolve_sf3_asset(repo_root, row, row[field]) != resolve_sf3_asset(
            repo_root, row, nested, nested=True
        ):
            raise ViewContractError(f"SF top-level and nested asset bases differ: {field}")
    if require_assets:
        for path in (x0_rgb, rgb_video, *mask_paths):
            if not path.is_file():
                raise ViewContractError(f"SF model-visible asset missing: {path}")
        audit_sf3_native(row, record, repo_root=repo_root, action_spaces=action_spaces)
    action_ids = tuple((tuple((column[step] for column in per_subject)) for step in range(25)))
    actions = tuple(
        (
            tuple((None if i < 0 else action_spaces.player_actions[i] for i in values))
            for values in action_ids
        )
    )
    return ViewRecord(
        schema_version=record["schema_version"],
        split=split,
        sample_index=sample_index,
        family_id=f"{origin}:{_identity_label(record['source_episode_id'], 'source_episode_id')}",
        branch=_identity_label(record.get("branch_id", "main"), "branch_id"),
        subjects=handles,
        x0_rgb=x0_rgb,
        x0_masks=mask_paths,
        actions=actions,
        rgb_video=rgb_video,
        id_video=id_video,
        engine_subject_ids=ENGINE_IDS,
        data_format=data_format,
        control_kind=kinds,
        action_ids=action_ids,
        npc_prompts=tuple(prompt_slots),
        audit_target_slot=tuple(target_slots),
        origin_dataset=origin,
        sample_id=_identity_label(record["sample_id"], "sample_id"),
        source_episode_id=record["source_episode_id"],
        prompt_compiler_version=PROMPT_PROTOCOL,
        source_metadata={
            "origin_dataset": origin,
            "asset_root": row["asset_root"],
            "window_index": record["window_index"],
            "native_slots": (0, 1),
            "action_space_by_slot": namespaces,
            "subjects": subjects,
            "player_slot": record.get("player_slot"),
            "npc_slot": record.get("npc_slot"),
            "audit": record["audit"],
        },
    )


class SF3ViewFile:

    def __init__(
        self,
        path,
        split,
        *,
        data_format="sf3_ready",
        repo_root=None,
        action_spaces_path=None,
        action_vocabulary=(),
        allow_test=False,
    ):
        if split == "test" and (not allow_test):
            raise ViewContractError("pass allow_test=True to open the test split")
        if split not in {"train", "val", "test"}:
            raise ViewContractError("invalid SF split")
        self.repo_root = (
            {key: Path(value).expanduser().resolve() for key, value in repo_root.items()}
            if isinstance(repo_root, dict)
            else Path(repo_root or PROJECT_ROOT).expanduser().resolve()
        )
        source = Path(path).expanduser()
        self.data_format, self.split = (data_format, split)
        self.action_spaces = SF3ActionSpaces(action_spaces_path)
        self.action_vocabulary = self.action_spaces.player_actions
        if action_vocabulary and tuple(action_vocabulary) != self.action_vocabulary:
            raise ViewContractError(
                "configured SF Player text rows differ from authoritative action_spaces.json order"
            )
        self.raw_id_to_row = self.action_spaces.player_raw_id_to_row
        self.action_protocol = "sf-player20-block12"
        self.prompt_compiler_version = PROMPT_PROTOCOL
        if data_format == "sf3_ready":
            self.path = source if source.suffix == ".parquet" else source / "manifest.parquet"
        else:
            raise ViewContractError("SF adapter never falls back to a different source format")
        self.path = self.path.resolve(strict=True)
        self.asset_root = self.repo_root
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError("SF Parquet adapters require pyarrow") from exc
        rows = pq.read_table(self.path).to_pylist()
        counts = Counter()
        source_splits = {}
        for row in rows:
            origin, row_split = (row.get("origin_dataset"), row.get("split"))
            if (
                row.get("export_version") != EXPORT_VERSION
                or origin not in ORIGINS
                or row.get("asset_root") != ORIGINS[origin]
                or (row.get("release") != "formal")
                or (row.get("npc_count") != (0 if origin == "sf3_legacy_npc0" else 1))
            ):
                raise ViewContractError(
                    "formal SF manifest contains an excluded origin or mismatched NPC provenance"
                )
            counts[origin, row_split] += 1
            group = (origin, row["source_episode_id"])
            if group in source_splits and source_splits[group] != row_split:
                raise ViewContractError("SF source episode crosses data splits")
            source_splits[group] = row_split
        identities = [(row["origin_dataset"], row["split"], row["sample_id"]) for row in rows]
        if len(set(identities)) != len(identities):
            raise ViewContractError("SF manifest contains duplicate original sample identities")
        self.rows = sorted(
            (row for row in rows if row["split"] == split),
            key=lambda row: (row["origin_dataset"], row["split"], row["sample_id"]),
        )

    def __len__(self):
        return len(self.rows)

    def record(self, index, *, require_assets=True):
        return parse_sf3_record(
            self.rows[index],
            repo_root=self.repo_root,
            action_spaces=self.action_spaces,
            sample_index=index,
            expected_split=self.split,
            require_assets=require_assets,
            data_format=self.data_format,
        )

    def iter_records(self, *, require_assets=True):
        for index in range(len(self)):
            yield self.record(index, require_assets=require_assets)
