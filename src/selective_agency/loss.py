"""Video Flow Matching; detached routing diagnostics are explicitly optional."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from diffsynth.diffusion import FlowMatchScheduler

from selective_agency.constants import ID_OCCUPANCY_DENOMINATOR
from selective_agency.models.dit import SelectiveAgencyOutput


@dataclass(frozen=True)
class RoutingTargets:
    probability: torch.Tensor
    slot_valid: torch.Tensor


@dataclass(frozen=True)
class RoutingLossOutput:
    foreground_loss: torch.Tensor
    background_loss: torch.Tensor
    routing_loss: torch.Tensor
    top1_accuracy: torch.Tensor
    correct_mass: torch.Tensor
    background_accuracy: torch.Tensor
    foreground_mass: torch.Tensor
    conditional_correct_subject_mass: torch.Tensor
    random_subject_mass: torch.Tensor
    padding_probability_max: torch.Tensor


@dataclass(frozen=True)
class FlowLossOutput:
    loss: torch.Tensor
    flow_loss: torch.Tensor
    unweighted_mse: torch.Tensor
    handle_routing: RoutingLossOutput | None
    group_routing: RoutingLossOutput | None
    routing_top1_agreement: torch.Tensor | None
    routing_correct_mass_gap: torch.Tensor | None
    routing_total_variation: torch.Tensor | None
    background_value_norm: torch.Tensor | None
    background_residual_norm: torch.Tensor | None
    projected_handle_rms: torch.Tensor | None
    handle_rms: torch.Tensor | None
    timestep_id: int
    timestep: float
    sigma: float
    weight: float


def flow_loss_metrics(result: FlowLossOutput) -> dict[str, torch.Tensor]:
    """Expose only metrics actually measured by this forward."""
    metrics = {
        "loss": result.loss.detach(),
        "flow_loss": result.flow_loss.detach(),
        "unweighted_mse": result.unweighted_mse,
    }
    for name in (
        "routing_top1_agreement",
        "routing_correct_mass_gap",
        "routing_total_variation",
        "handle_rms",
    ):
        value = getattr(result, name)
        if value is not None:
            metrics[name] = value.detach()
    for prefix in ("handle", "group"):
        routing = getattr(result, f"{prefix}_routing")
        if routing is not None:
            for name, value in vars(routing).items():
                metrics[f"{prefix}_{name}"] = value.detach()
    return metrics


def prepare_routing_targets(
    occupancy_counts: torch.Tensor,
    subject_valid: torch.Tensor,
    *,
    occupancy_denominator: int = ID_OCCUPANCY_DENOMINATOR,
) -> RoutingTargets:
    """Convert exact per-patch occupancy counts to a soft slot distribution."""

    if occupancy_counts.ndim != 5:
        raise ValueError("occupancy counts must be [B,T,N+1,H,W]")
    batch, _, slots, _, _ = occupancy_counts.shape
    if subject_valid.shape != (batch, slots - 1):
        raise ValueError("subject validity shape disagrees with occupancy channels")
    if occupancy_denominator <= 0:
        raise ValueError("occupancy denominator must be positive")
    counts = occupancy_counts.to(torch.int32)
    if bool((counts < 0).any()):
        raise ValueError("occupancy counts must be non-negative")
    if not bool((counts.sum(dim=2) == occupancy_denominator).all()):
        raise ValueError("occupancy channels must exactly partition every patch")

    target = counts.float().permute(0, 1, 3, 4, 2).flatten(2, 3)
    target = target / float(occupancy_denominator)
    slot_valid = torch.cat(
        (
            torch.ones(batch, 1, dtype=torch.bool, device=subject_valid.device),
            subject_valid.bool(),
        ),
        dim=1,
    )
    invalid_target = target.masked_select(~slot_valid[:, None, None].expand_as(target))
    if invalid_target.numel() and bool((invalid_target != 0).any()):
        raise ValueError("invalid slot has non-zero occupancy target")
    return RoutingTargets(probability=target, slot_valid=slot_valid)


def balanced_soft_routing_loss(
    log_probabilities: torch.Tensor,
    targets: RoutingTargets,
) -> RoutingLossOutput:
    """Balance foreground and background masses for any N+1 slot softmax."""

    if log_probabilities.ndim != 5:
        raise ValueError("routing log probabilities must be [L,B,T,P,N+1]")
    layers, batch, steps, spatial, slots = log_probabilities.shape
    if targets.probability.shape != (batch, steps, spatial, slots):
        raise ValueError("routing target shape disagrees with log probabilities")
    if targets.slot_valid.shape != (batch, slots):
        raise ValueError("routing slot validity shape disagrees")

    log_probability = log_probabilities.float()
    expanded_valid = targets.slot_valid[None, :, None, None].expand_as(log_probability)
    valid_values = log_probability.masked_select(expanded_valid)
    if not bool(torch.isfinite(valid_values).all()):
        raise FloatingPointError("valid routing log probabilities are not finite")
    safe_log_probability = torch.where(
        expanded_valid, log_probability, torch.zeros_like(log_probability)
    )
    target = targets.probability
    subject_target = target[..., 1:]
    background_target = target[..., 0]
    foreground = subject_target.sum(dim=-1)

    foreground_per_sample = foreground.sum(dim=(1, 2))
    has_foreground = foreground_per_sample > 0
    foreground_numerator = -(safe_log_probability[..., 1:] * subject_target[None]).sum(
        dim=(2, 3, 4)
    )
    foreground_per_layer_sample = foreground_numerator / foreground_per_sample.clamp_min(1e-6)[None]
    foreground_loss = (foreground_per_layer_sample * has_foreground.float()[None]).sum() / (
        layers * has_foreground.float().sum().clamp_min(1.0)
    )

    background_per_sample = background_target.sum(dim=(1, 2))
    has_background = background_per_sample > 0
    background_numerator = -(safe_log_probability[..., 0] * background_target[None]).sum(dim=(2, 3))
    background_per_layer_sample = background_numerator / background_per_sample.clamp_min(1e-6)[None]
    background_loss = (background_per_layer_sample * has_background.float()[None]).sum() / (
        layers * has_background.float().sum().clamp_min(1.0)
    )
    routing_loss = 0.5 * (foreground_loss + background_loss)

    probability = log_probability.exp()
    padding_probability = probability.masked_select(~expanded_valid)
    padding_probability_max = (
        padding_probability.max() if padding_probability.numel() else probability.new_zeros(())
    )
    background_probability = probability[..., 0]
    subject_probability = probability[..., 1:]
    top1_accuracy = (probability.argmax(dim=-1) == target.argmax(dim=-1)[None]).float().mean()
    correct_mass = (probability * target[None]).sum() / (layers * batch * steps * spatial)
    background_accuracy = (background_probability * background_target[None]).sum() / (
        layers * background_target.sum()
    ).clamp_min(1e-6)
    foreground_mass = (subject_probability.sum(dim=-1) * foreground[None]).sum() / (
        layers * foreground.sum()
    ).clamp_min(1e-6)

    conditional_subject_probability = subject_probability / subject_probability.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-6)
    correct_numerator = (conditional_subject_probability * subject_target[None]).sum(dim=(2, 3, 4))
    conditional_per_layer_sample = correct_numerator / foreground_per_sample.clamp_min(1e-6)[None]
    conditional_correct_subject_mass = (
        conditional_per_layer_sample * has_foreground.float()[None]
    ).sum() / (layers * has_foreground.float().sum().clamp_min(1.0))
    cardinality = targets.slot_valid[:, 1:].sum(dim=-1).float()
    random_subject_mass = (
        cardinality.clamp_min(1).reciprocal() * has_foreground.float()
    ).sum() / has_foreground.float().sum().clamp_min(1.0)

    return RoutingLossOutput(
        foreground_loss=foreground_loss,
        background_loss=background_loss,
        routing_loss=routing_loss,
        top1_accuracy=top1_accuracy.detach(),
        correct_mass=correct_mass.detach(),
        background_accuracy=background_accuracy.detach(),
        foreground_mass=foreground_mass.detach(),
        conditional_correct_subject_mass=conditional_correct_subject_mass.detach(),
        random_subject_mass=random_subject_mass.detach(),
        padding_probability_max=padding_probability_max.detach(),
    )


def compare_routing(
    handle_logs: torch.Tensor,
    group_logs: torch.Tensor,
    targets: RoutingTargets,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return top-1 agreement, signed correct-mass gap, and mean TV distance."""

    if handle_logs.shape != group_logs.shape:
        raise ValueError("handle and group routing shapes differ")
    handle_probability = handle_logs.float().exp()
    group_probability = group_logs.float().exp()
    agreement = (
        (handle_probability.argmax(dim=-1) == group_probability.argmax(dim=-1)).float().mean()
    )
    target = targets.probability[None]
    handle_correct = (handle_probability * target).sum(dim=-1).mean()
    group_correct = (group_probability * target).sum(dim=-1).mean()
    total_variation = 0.5 * (handle_probability - group_probability).abs().sum(dim=-1).mean()
    return (
        agreement.detach(),
        (handle_correct - group_correct).detach(),
        total_variation.detach(),
    )


class WanFlowMatchingObjective:
    def __init__(
        self,
        train_timesteps: int = 1000,
        sigma_shift: float = 5.0,
        *,
        mu_handle: float = 0.0,
        routing_diagnostics: bool = False,
    ):
        if float(mu_handle) != 0.0:
            raise ValueError(
                "the flow-only contract requires handle routing loss weight to be exactly zero"
            )
        self.scheduler = FlowMatchScheduler("Wan")
        self.scheduler.set_timesteps(train_timesteps, training=True, shift=sigma_shift)
        self.train_timesteps = int(train_timesteps)
        self.mu_handle = float(mu_handle)
        self.routing_diagnostics = bool(routing_diagnostics)

    def __call__(
        self,
        model: torch.nn.Module,
        batch: dict[str, torch.Tensor],
        *,
        use_gradient_checkpointing: bool,
        timestep_id: int | None = None,
        noise: torch.Tensor | None = None,
    ) -> FlowLossOutput:
        clean = batch["input_latents"]
        first_frame = batch["first_frame_latents"]
        if self.routing_diagnostics and "occupancy_counts" not in batch:
            raise ValueError("explicit routing diagnostics require occupancy_counts")
        if clean.ndim != 5 or clean.shape[2] < 2 or first_frame.shape != clean[:, :, :1].shape:
            raise ValueError("FM requires video latents and one matching clean first frame")
        if timestep_id is None:
            timestep_id = int(torch.randint(0, self.train_timesteps, (1,), device="cpu"))
        if not 0 <= timestep_id < self.train_timesteps:
            raise ValueError("timestep_id outside training schedule")
        timestep_cpu = self.scheduler.timesteps[timestep_id]
        sigma_cpu = self.scheduler.sigmas[timestep_id]
        weight_cpu = self.scheduler.linear_timesteps_weights[timestep_id]
        sigma = sigma_cpu.to(device=clean.device, dtype=clean.dtype)
        if noise is None:
            noise = torch.randn_like(clean)
        elif noise.shape != clean.shape:
            raise ValueError("noise shape must match video latents")
        noisy = (1 - sigma) * clean + sigma * noise
        noisy[:, :, 0:1] = first_frame
        target = noise - clean
        timestep = timestep_cpu.to(device=clean.device, dtype=clean.dtype).expand(clean.shape[0])
        output = model(
            noisy,
            timestep,
            batch["subject_roi_masks"],
            batch["action_ids"],
            batch["subject_valid"],
            **{key: batch[key] for key in ("control_kind", "npc_prompt_ids") if key in batch},
            use_gradient_checkpointing=use_gradient_checkpointing,
            return_routing_aux=self.routing_diagnostics,
        )
        prediction = output.prediction if isinstance(output, SelectiveAgencyOutput) else output
        if not isinstance(prediction, torch.Tensor) or prediction.shape != clean.shape:
            raise ValueError("FM prediction shape must match video latents")
        mse = F.mse_loss(prediction[:, :, 1:].float(), target[:, :, 1:].float())
        flow_loss = mse * weight_cpu.to(device=mse.device, dtype=mse.dtype)

        # The model retains routing measurements for mechanism analysis, but keeps
        # the entire diagnostic path outside autograd and optimization.
        handle_routing = group_routing = None
        agreement = correct_gap = total_variation = None
        if self.routing_diagnostics:
            if not isinstance(output, SelectiveAgencyOutput):
                raise TypeError("model did not return requested routing diagnostics")
            with torch.no_grad():
                routing_targets = prepare_routing_targets(
                    batch["occupancy_counts"], batch["subject_valid"]
                )
                handle_routing = balanced_soft_routing_loss(
                    output.handle_log_probabilities.detach(), routing_targets
                )
                group_routing = balanced_soft_routing_loss(
                    output.group_log_probabilities.detach(), routing_targets
                )
                agreement, correct_gap, total_variation = compare_routing(
                    output.handle_log_probabilities.detach(),
                    output.group_log_probabilities.detach(),
                    routing_targets,
                )
        loss = flow_loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("flow-only loss is not finite")
        return FlowLossOutput(
            loss=loss,
            flow_loss=flow_loss,
            unweighted_mse=mse.detach(),
            handle_routing=handle_routing,
            group_routing=group_routing,
            routing_top1_agreement=agreement,
            routing_correct_mass_gap=correct_gap,
            routing_total_variation=total_variation,
            background_value_norm=(
                output.background_value_norms.detach() if self.routing_diagnostics else None
            ),
            background_residual_norm=(
                output.background_residual_norms.detach() if self.routing_diagnostics else None
            ),
            projected_handle_rms=(
                output.projected_handle_rms.detach() if self.routing_diagnostics else None
            ),
            handle_rms=(output.handle_rms.detach() if self.routing_diagnostics else None),
            timestep_id=timestep_id,
            timestep=float(timestep_cpu),
            sigma=float(sigma_cpu),
            weight=float(weight_cpu),
        )
