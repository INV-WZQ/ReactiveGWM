"""Resolve public main checkpoints without changing their keys or text tables."""
from pathlib import Path
from .io import read_json

MODEL_REPO = "INV-WZQ/ReactiveGWM-v2-Models"
DATASET_REPO = "INV-WZQ/ReactiveGWM-v2-Datasets"


def resolve_checkpoint(source, *, subfolder=None, revision=None, cache_dir=None):
    path = Path(source).expanduser()
    if path.is_dir():
        path = path / subfolder if subfolder else path
        return path.resolve(strict=True), {"source": str(path.resolve())}
    if subfolder not in {"HNM/main", "SF3/main"}:
        raise ValueError("HF checkpoints require --model-subfolder HNM/main or SF3/main")
    from huggingface_hub import HfApi, snapshot_download

    commit = HfApi().model_info(source, revision=revision).sha
    root = snapshot_download(
        source, revision=commit, allow_patterns=[f"{subfolder}/*"], cache_dir=cache_dir,
    )
    return Path(root) / subfolder, {"repo_id": source, "subfolder": subfolder, "revision": commit}


def validate_checkpoint(directory, config, model):
    path = Path(directory) / "metadata.json"
    metadata = read_json(path) if path.is_file() else {}
    for key, expected in (
        ("game", config["game"]), ("method", "pretraining"),
        ("architecture_version", model.architecture_version),
        ("prompt_protocol", config["prompt_protocol"]),
    ):
        if metadata.get(key, expected) != expected:
            raise ValueError(f"checkpoint {key} differs from the selected main model")
    alpha = metadata.get("architecture_contract", {}).get("handle_injection_alpha", 0.25)
    if alpha != 0.25:
        raise ValueError("checkpoint does not use main-model spatial role binding")
    return metadata
