"""Readable recipes and explicit command-line overrides."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import yaml

from .geometry import VideoGeometry

WAN_MODEL = {
    "has_image_input": False,
    "patch_size": (1, 2, 2),
    "in_dim": 48,
    "dim": 3072,
    "ffn_dim": 14336,
    "freq_dim": 256,
    "text_dim": 4096,
    "out_dim": 48,
    "num_heads": 24,
    "num_layers": 30,
    "eps": 1e-6,
    "seperated_timestep": True,
    "require_clip_embedding": False,
    "require_vae_embedding": False,
    "fuse_vae_embedding_in_latents": True,
    "max_subjects": 6,
}
DEFAULT_TRAINING = {
    "seed": 20260808,
    "precision": "bf16",
    "batch_size": 1,
    "gradient_accumulation_steps": None,
    "effective_batch_size": 8,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "betas": [0.9, 0.999],
    "epsilon": 1e-8,
    "warmup_factor": 1 / 3,
    "warmup_steps": 5,
    "max_steps": 20000,
    "save_every": 1000,
    "validate_every": 1000,
    "num_workers": 4,
    "gradient_checkpointing": True,
    "train_timesteps": 1000,
    "sigma_shift": 5.0,
    "deterministic": True,
}


def load_config(path):
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if value.get("game") == "sf3":
        value["game"] = "sf"
    if value.get("game") not in {"hnm", "sf"}:
        raise ValueError("game must be hnm or sf")
    value.setdefault("method", "pretraining")
    if value["method"] != "pretraining":
        raise ValueError("this release supports HNM/SF3 main models only")
    protocol = "hnm-if-then" if value["game"] == "hnm" else "sf-if-then"
    if value.get("prompt_protocol", protocol) != protocol:
        raise ValueError(f"this recipe requires {protocol}")
    value["prompt_protocol"] = protocol
    value["training"] = {**DEFAULT_TRAINING, **value.get("training", {})}
    video = VideoGeometry(**value.get("video", {}))
    value["video"] = asdict(video)
    value["model"] = {**WAN_MODEL, **value.get("model", {})}
    if value["model"].get("control_steps", video.control_steps) != video.control_steps:
        raise ValueError("model.control_steps must match (video.frames - 1) / 4")
    value["model"]["control_steps"] = video.control_steps
    value["inference"] = {"denoising_steps": 30, "sigma_shift": 5.0, **value.get("inference", {})}
    return value


def resolve_batch_size(settings, world_size):
    """Derive accumulation from a target batch unless accumulation is explicit."""
    batch = settings["batch_size"]
    target = settings["effective_batch_size"]
    accumulation = settings.get("gradient_accumulation_steps")
    if world_size < 1 or batch < 1 or target < 1:
        raise ValueError("world size and batch sizes must be positive")
    if accumulation is None:
        if target % (world_size * batch):
            raise ValueError("effective batch must be divisible by world size * batch size; adjust the batch arguments")
        accumulation = target // (world_size * batch)
    if accumulation < 1:
        raise ValueError("gradient accumulation must be positive")
    settings["gradient_accumulation_steps"] = accumulation
    settings["effective_batch_size"] = world_size * batch * accumulation


def select_core(game=None):
    """The shared core is imported as an installed package."""
    if game not in {None, "hnm", "sf"}:
        raise ValueError("game must be hnm or sf")


def original_optimizer_config(config):
    """Use the original Adam parameter groups and five-update ConstantLR."""
    settings = config["training"]
    learning_rates = {
        role: settings["learning_rate"] for role in ("backbone", "cross_attention", "selective")
    }
    learning_rates.update(settings.get("learning_rates", {}))
    return {
        "training": {"learning_rates": learning_rates},
        "optimization": {
            "optimizer": {
                "name": "AdamW",
                "weight_decay": settings["weight_decay"],
                "betas": settings["betas"],
                "epsilon": settings["epsilon"],
                "amsgrad": False,
            },
            "lr_scheduler": {
                "name": "ConstantLR",
                "factor": settings["warmup_factor"],
                "total_iters": settings["warmup_steps"],
            },
        },
    }


def make_model(config, device, *, trainable=False):
    import torch
    from selective_agency.models.dit import SelectiveAgencyWanDiT

    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[config["training"]["precision"]]
    kwargs = deepcopy(config["model"])
    kwargs["self_attention_mode"] = "frame_causal"
    kwargs["game"] = config["game"]
    kwargs["handle_injection_alpha"] = 0.25
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            model = SelectiveAgencyWanDiT(**kwargs)
    finally:
        torch.set_default_dtype(previous)
    model.requires_grad_(trainable)
    if trainable:
        training_parameters(model)
    return model


def training_parameters(model):
    """The original weight-only cross-output projection leaves its bias frozen."""
    model.requires_grad_(True)
    for name, parameter in model.named_parameters():
        if name.endswith(".cross_attn.o.bias"):
            parameter.requires_grad_(False)


def bind_text(model, tensors):
    model.set_control_text_cache(
        tensors["action_hidden_states"],
        tensors["action_attention_mask"],
        tensors["npc_hidden_states"],
        tensors["npc_attention_mask"],
    )
