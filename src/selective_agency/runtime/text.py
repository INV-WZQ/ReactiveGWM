"""Complete, untruncated Wan T5 encoding and plain text-table metadata."""

from __future__ import annotations
import html
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import torch
from safetensors.torch import load_file, save_file
from .io import read_json, write_json

TEXT_PROTOCOL = "selective-agency-full-condition-text"
TOKENIZATION_PROTOCOL = "wan-umt5-whitespace-full-text"
TENSOR_KEYS = {
    "action_hidden_states",
    "action_attention_mask",
    "npc_hidden_states",
    "npc_attention_mask",
}


class TextCacheError(ValueError):
    pass


def validate_text_spec(spec):
    actions, prompts = spec["actions"], spec["npc_prompts"]
    if (
        not actions
        or len(set(actions)) != len(actions)
        or spec["action_texts"] != [f"ACTION={x}" for x in actions]
    ):
        raise TextCacheError("Action text rows are inconsistent")
    if len(prompts) != len(set(prompts)) or any(not x.strip() for x in prompts):
        raise TextCacheError("NPC text rows must be unique and nonempty")
    if "raw_id_to_row" in spec and sorted(spec["raw_id_to_row"].values()) != list(range(len(actions))):
        raise TextCacheError("Action row map is inconsistent")


def normalized_text(text: str) -> str:
    import ftfy

    return re.sub("\\s+", " ", html.unescape(html.unescape(ftfy.fix_text(text)))).strip()


def tokenize_complete_texts(tokenizer: Any, texts: Sequence[str]) -> list[list[int]]:
    """Return full unpadded IDs, including special tokens, regardless of defaults."""
    if not texts:
        return []
    actual = getattr(tokenizer, "tokenizer", tokenizer)
    encoded = actual(
        [normalized_text(text) for text in texts],
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_attention_mask=False,
        return_tensors=None,
    )
    rows = encoded["input_ids"]
    if len(rows) != len(texts):
        raise TextCacheError("tokenizer returned a different row count")
    result: list[list[int]] = []
    for row in rows:
        ids = [int(value) for value in row]
        if not ids or any((value < 0 for value in ids)):
            raise TextCacheError("tokenizer produced empty or invalid full-text IDs")
        result.append(ids)
    return result


def load_full_text_tokenizer(path: str | Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(Path(path).expanduser().resolve(strict=True)), local_files_only=True
    )


def load_frozen_wan_text_encoder(path: str | Path, device: str | torch.device) -> torch.nn.Module:
    from diffsynth.models.wan_video_text_encoder import WanTextEncoder

    with torch.device("meta"):
        encoder = WanTextEncoder()
    encoder.load_state_dict(
        torch.load(str(path), map_location="cpu", weights_only=True, mmap=True), assign=True
    )
    return encoder.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)


def _padded_ids(
    rows: Sequence[Sequence[int]], pad_token_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(map(len, rows), default=0)
    ids = torch.full((len(rows), width), pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.bool)
    for index, row in enumerate(rows):
        ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
        mask[index, : len(row)] = True
    return (ids, mask)


@torch.no_grad()
def encode_text_spec(
    spec: dict[str, Any],
    *,
    tokenizer: Any,
    text_encoder: torch.nn.Module,
    device: str | torch.device = "cpu",
    batch_size: int = 8,
    encoder_identity: Mapping[str, Any] | None = None,
    tokenizer_info: Mapping[str, Any] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Encode complete prompts together, never concatenate old trigger features."""
    validate_text_spec(spec)
    if batch_size < 1:
        raise ValueError("text batch size must be positive")
    text_encoder.eval().requires_grad_(False)
    actual_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    pad_id = getattr(actual_tokenizer, "pad_token_id", None)
    if pad_id is None:
        raise TextCacheError("tokenizer has no pad token")
    texts = [*spec["action_texts"], *spec["npc_prompts"]]
    rows = tokenize_complete_texts(tokenizer, texts)
    configuration = getattr(text_encoder, "config", None)
    position_limit = getattr(configuration, "max_position_embeddings", None)
    if isinstance(position_limit, int) and max(map(len, rows)) > position_limit:
        raise TextCacheError(
            f"complete text requires {max(map(len, rows))} tokens but encoder supports {position_limit}; text was not truncated"
        )
    encoded_rows: list[torch.Tensor] = []
    for start in range(0, len(rows), batch_size):
        ids, mask = _padded_ids(rows[start : start + batch_size], int(pad_id))
        try:
            hidden = text_encoder(ids.to(device), mask.to(device))
        except RuntimeError as error:
            raise TextCacheError(
                f"full-text encoding failed for rows {start}:{start + len(ids)}, padded length {ids.shape[1]}; no truncation or shortened retry was used: {error}"
            ) from error
        if hasattr(hidden, "last_hidden_state"):
            hidden = hidden.last_hidden_state
        if hidden.ndim != 3 or hidden.shape[:2] != mask.shape or (not torch.isfinite(hidden).all()):
            raise TextCacheError("encoder output must preserve every input token and be finite")
        hidden = hidden.detach().to(device="cpu", dtype=torch.bfloat16)
        encoded_rows.extend(
            (
                hidden[index, : len(row)].clone()
                for index, row in enumerate(rows[start : start + batch_size])
            )
        )
    dim = encoded_rows[0].shape[-1]
    action_count = len(spec["actions"])
    tensors: dict[str, torch.Tensor] = {}
    for prefix, selected in (
        ("action", encoded_rows[:action_count]),
        ("npc", encoded_rows[action_count:]),
    ):
        width = max((len(row) for row in selected), default=0)
        hidden = torch.zeros(len(selected), width, dim, dtype=torch.bfloat16)
        mask = torch.zeros(len(selected), width, dtype=torch.bool)
        for index, row in enumerate(selected):
            hidden[index, : len(row)] = row
            mask[index, : len(row)] = True
        tensors[f"{prefix}_hidden_states"] = hidden
        tensors[f"{prefix}_attention_mask"] = mask
    metadata = {
        "status": "FROZEN",
        "protocol": TEXT_PROTOCOL,
        "spec": spec,
        "encoder": dict(encoder_identity or {"kind": type(text_encoder).__name__, "text_dim": dim}),
        "tokenizer": dict(tokenizer_info or {"kind": type(actual_tokenizer).__name__}),
        "tokenization": {
            "protocol": TOKENIZATION_PROTOCOL,
            "truncation": False,
            "padding": "dynamic-right",
            "add_special_tokens": True,
            "tokenizer_model_max_length": getattr(actual_tokenizer, "model_max_length", None),
            "encoder_absolute_position_limit": position_limit,
        },
        "token_ids": {"action": rows[:action_count], "npc": rows[action_count:]},
        "lengths": {
            "action": list(map(len, rows[:action_count])),
            "npc": list(map(len, rows[action_count:])),
        },
        "text_dim": dim,
    }
    return (tensors, metadata)


def _validate_tensors(
    tensors: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any],
    *,
    expected_dim: int | None = None,
) -> None:
    if set(tensors) != TENSOR_KEYS:
        raise TextCacheError("full-condition tensor names differ")
    spec = metadata["spec"]
    dim = metadata.get("text_dim")
    if expected_dim is not None and dim != expected_dim:
        raise TextCacheError(f"text hidden width {dim} differs from {expected_dim}")
    for prefix, count in (("action", len(spec["actions"])), ("npc", len(spec["npc_prompts"]))):
        hidden, mask = (tensors[f"{prefix}_hidden_states"], tensors[f"{prefix}_attention_mask"])
        lengths = metadata["lengths"][prefix]
        token_ids = metadata["token_ids"][prefix]
        if (
            hidden.ndim != 3
            or hidden.shape[:2] != mask.shape
            or hidden.shape[0] != count
            or (hidden.shape[2] != dim)
            or (hidden.dtype != torch.bfloat16)
            or (mask.dtype != torch.bool)
            or (len(lengths) != count)
            or (lengths != [len(row) for row in token_ids])
            or (mask.sum(1).tolist() != lengths)
            or any((length < 1 for length in lengths))
        ):
            raise TextCacheError(f"{prefix} text tensor/complete token contract differs")
        expected_mask = torch.arange(mask.shape[1])[None] < torch.tensor(lengths)[:, None]
        if not torch.equal(mask.cpu(), expected_mask) or not torch.isfinite(hidden).all():
            raise TextCacheError(
                f"{prefix} text contains invalid padding or non-finite hidden values"
            )
        if torch.count_nonzero(hidden.masked_select(~mask[..., None].expand_as(hidden))):
            raise TextCacheError(f"{prefix} padding hidden states must be zero")


def load_text_asset(path, *, expected_spec=None, expected_dim=None):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / "control_text.safetensors"
    metadata = read_json(path.with_suffix(".json"))
    validate_text_spec(metadata["spec"])
    if expected_spec is not None and metadata["spec"] != expected_spec:
        raise TextCacheError("text table differs from the selected recipe")
    tensors = load_file(str(path), device="cpu")
    _validate_tensors(tensors, metadata, expected_dim=expected_dim)
    return tensors, metadata


def save_text_asset(root, tensors, metadata):
    """Write only the actual text table, token lengths and encoding settings."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    keep = ("protocol", "spec", "tokenization", "token_ids", "lengths", "text_dim")
    clean = {key: metadata[key] for key in keep if key in metadata}
    # Normalize our earlier labels on save without requiring them on input.
    if clean.get("protocol") == TEXT_PROTOCOL + "-v1":
        clean["protocol"] = TEXT_PROTOCOL
    for key, protocol in (("spec", TEXT_PROTOCOL), ("tokenization", TOKENIZATION_PROTOCOL)):
        if key in clean:
            clean[key] = dict(clean[key])
            if clean[key].get("protocol") == protocol + "-v1":
                clean[key]["protocol"] = protocol
    validate_text_spec(clean["spec"])
    _validate_tensors(tensors, clean)
    save_file(
        {k: v.contiguous() for k, v in tensors.items()}, str(root / "control_text.safetensors")
    )
    write_json(root / "control_text.json", clean)


def import_text_asset(path, spec):
    """Reuse old T5 values and rebuild metadata from readable fields only."""
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / "control_text.safetensors"
    metadata = read_json(path.with_suffix(".json"))
    original = metadata["spec"]
    # Reuse by the actual text rows. Source protocol names and raw game IDs
    # do not determine which text embedding is used.
    validate_text_spec(spec)
    for field in ("actions", "action_texts"):
        if original[field] != spec[field]:
            raise TextCacheError(f"source T5 table differs in {field}")
    tensors = load_file(str(path), device="cpu")
    _validate_tensors(tensors, metadata)
    if spec["npc_prompts"]:
        if original["npc_prompts"] != spec["npc_prompts"]:
            raise TextCacheError("source T5 table uses different NPC sentences")
    else:
        dim = metadata["text_dim"]
        tensors["npc_hidden_states"] = torch.empty((0, 0, dim), dtype=torch.bfloat16)
        tensors["npc_attention_mask"] = torch.empty((0, 0), dtype=torch.bool)
        metadata["token_ids"]["npc"], metadata["lengths"]["npc"] = [], []
    metadata["spec"] = spec
    return tensors, metadata
