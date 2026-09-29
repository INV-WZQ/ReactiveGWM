"""Strict parser for portable SelectiveAgency model-visible JSONL views."""

from __future__ import annotations
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable, Sequence
from selective_agency.constants import CONTROL_STEPS, MAX_SUBJECTS
from selective_agency.paths import require_within


class ViewContractError(ValueError):
    pass


@dataclass(frozen=True)
class ViewRecord:
    schema_version: int
    split: str
    sample_index: int
    family_id: str
    branch: str
    subjects: tuple[str, ...]
    x0_rgb: Path
    x0_masks: tuple[Path, ...]
    actions: tuple[tuple[str | None, ...], ...]
    rgb_video: Path
    id_video: Path | None
    engine_subject_ids: tuple[int, ...] | None
    data_format: str = "portable"
    control_kind: tuple[int, ...] = ()
    action_ids: tuple[tuple[int, ...], ...] = ()
    npc_prompts: tuple[str | None, ...] = ()
    audit_target_slot: tuple[int, ...] = ()
    origin_dataset: str | None = None
    sample_id: str | None = None
    source_episode_id: str | None = None
    source_metadata: dict[str, Any] | None = None
    prompt_compiler_version: str | None = None

    @property
    def cardinality(self) -> int:
        return len(self.subjects)

    @property
    def record_id(self) -> str:
        if self.origin_dataset is not None and self.sample_id is not None:
            return f"{self.origin_dataset}:{self.split}:{self.sample_id}"
        return f"{self.split}:{self.sample_index}"

    @property
    def source_identity(self) -> tuple[str, str, str]:
        return (
            self.origin_dataset or "portable",
            self.split,
            self.sample_id or str(self.sample_index),
        )

    @property
    def source_group_identity(self) -> tuple[str, str]:
        return (self.origin_dataset or "portable", self.source_episode_id or self.family_id)

    @property
    def video_path(self) -> Path:
        return self.rgb_video

    @property
    def first_frame_path(self) -> Path:
        return self.x0_rgb


def _resolve_model_path(value: object, asset_root: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ViewContractError(f"invalid model-visible path: {value!r}")
    path = Path(value)
    if path.is_absolute():
        raise ViewContractError("public metadata paths must be split-relative")
    if ".." in path.parts:
        raise ViewContractError("public metadata path escapes its split root")
    return require_within(asset_root / path, asset_root)


def _identity_label(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or (value in {".", ".."})
        or ("/" in value)
        or ("\\" in value)
    ):
        raise ViewContractError(f"{field} must be a non-empty path-safe label")
    return value


class SourceViewFile:
    """Byte-offset indexed, fork-safe reader for one immutable JSONL view."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        split: str,
        *,
        action_vocabulary: Sequence[str],
        allow_test: bool = False,
    ):
        self.path = Path(path).expanduser().resolve(strict=True)
        if split == "test" and (not allow_test):
            raise ViewContractError("pass allow_test=True to open the test split")
        self.asset_root = self.path.parent
        self.split = split
        self.action_vocabulary = tuple(action_vocabulary)
        if not self.action_vocabulary or len(self.action_vocabulary) != len(
            set(self.action_vocabulary)
        ):
            raise ViewContractError("action vocabulary must be non-empty and unique")
        self.offsets: list[int] = []
        offset = 0
        with self.path.open("rb") as handle:
            for line in handle:
                if not line.strip():
                    raise ViewContractError(f"blank line in immutable view at byte {offset}")
                self.offsets.append(offset)
                offset += len(line)
        self._handle: BinaryIO | None = None
        self._handle_pid: int | None = None

    def __len__(self) -> int:
        return len(self.offsets)

    def _open_for_process(self) -> BinaryIO:
        pid = os.getpid()
        if self._handle is None or self._handle_pid != pid or self._handle.closed:
            if self._handle is not None and (not self._handle.closed):
                self._handle.close()
            self._handle = self.path.open("rb")
            self._handle_pid = pid
        return self._handle

    def raw_line(self, index: int) -> bytes:
        if not 0 <= index < len(self):
            raise IndexError(index)
        handle = self._open_for_process()
        handle.seek(self.offsets[index])
        line = handle.readline()
        if not line:
            raise ViewContractError(f"immutable view truncated at record {index}")
        return line

    def iter_records(self, *, require_assets: bool = True) -> Iterable[ViewRecord]:
        for index in range(len(self)):
            yield self.record(index, require_assets=require_assets)
