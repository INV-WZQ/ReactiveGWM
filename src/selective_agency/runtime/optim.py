"""Original parameter groups, AdamW and learning-rate schedule."""

from typing import Any
import torch

_CUSTOM_PREFIXES = (
    "control_token_builder.",
    "handle_codebook.",
    "handle_key_projections.",
    "background_slot_attention.",
    "background_value_scales.",
)


def _training_config(config):
    return config["training"]


def parameter_role(name: str) -> str:
    name = name.removeprefix("module.")
    if any(name.startswith(prefix) for prefix in _CUSTOM_PREFIXES):
        return "selective"
    if name.startswith("blocks.") and ".cross_attn." in name:
        return "cross_attention"
    return "backbone"


def use_weight_decay(name: str, parameter: torch.Tensor) -> bool:
    normalized = name.removeprefix("module.")
    if parameter.ndim < 2 or normalized.endswith(".bias"):
        return False
    no_decay_tokens = (
        "norm",
        "embedding",
        "modulation",
        "q_ref",
    )
    return not any(token in normalized for token in no_decay_tokens)


def build_training_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.AdamW:
    training_config = _training_config(config)
    optimizer_config = config["optimization"]["optimizer"]
    learning_rates = {
        name: float(value) for name, value in training_config["learning_rates"].items()
    }
    expected_roles = {"backbone", "cross_attention", "selective"}
    if set(learning_rates) != expected_roles:
        raise RuntimeError("training learning-rate roles are incomplete")
    time_learning_rate = training_config.get("time_conditioning_learning_rate")
    if time_learning_rate is not None:
        learning_rates["time_conditioning"] = float(time_learning_rate)
    grouped: dict[tuple[str, bool], list[torch.nn.Parameter]] = {}
    grouped_names: dict[tuple[str, bool], list[str]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        role = parameter_role(name)
        if time_learning_rate is not None and name.removeprefix("module.").startswith(
            ("time_embedding.", "time_projection.")
        ):
            role = "time_conditioning"
        decay = use_weight_decay(name, parameter)
        grouped.setdefault((role, decay), []).append(parameter)
        grouped_names.setdefault((role, decay), []).append(name)
    if not grouped:
        raise RuntimeError("training optimizer has no trainable parameters")
    parameter_groups = []
    for role, decay in sorted(grouped):
        names = grouped_names[(role, decay)]
        parameter_groups.append(
            {
                "params": grouped[(role, decay)],
                "lr": learning_rates[role],
                "weight_decay": (float(optimizer_config["weight_decay"]) if decay else 0.0),
                "group_name": f"{role}.{'decay' if decay else 'no_decay'}",
                "parameter_names": names,
            }
        )
    return torch.optim.AdamW(
        parameter_groups,
        betas=tuple(float(value) for value in optimizer_config["betas"]),
        eps=float(optimizer_config["epsilon"]),
        amsgrad=bool(optimizer_config["amsgrad"]),
        foreach=optimizer_config.get("foreach"),
    )


def build_training_scheduler(optimizer, config: dict[str, Any]):
    scheduler = config["optimization"]["lr_scheduler"]
    if scheduler["name"] == "ConstantLR":
        return torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=float(scheduler["factor"]),
            total_iters=int(scheduler["total_iters"]),
        )
    if scheduler["name"] == "LinearLR":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=float(scheduler["start_factor"]),
            end_factor=float(scheduler["end_factor"]),
            total_iters=int(scheduler["total_iters"]),
        )
    raise ValueError(f"unsupported training scheduler: {scheduler['name']}")
