"""Handle-bound cross-attention over complete length-normalized control groups."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class CausalHandleAttentionOutput:
    hidden_states: torch.Tensor
    handle_log_probabilities: torch.Tensor | None
    group_log_probabilities: torch.Tensor | None
    background_value_norm: torch.Tensor | None
    background_residual_norm: torch.Tensor | None
    projected_handle_rms: torch.Tensor | None


class JointBackgroundSubjectKVAttention(nn.Module):
    """Native Wan cross-attention with one projected Actor handle per group."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = int(dim)

    def project_kv(
        self,
        cross_attention: nn.Module,
        control_tokens: torch.Tensor,
        control_token_valid: torch.Tensor,
        handles: torch.Tensor,
        handle_key_projection: nn.Module,
        background_value_scale: torch.Tensor,
        *,
        ablate_subject_handle_keys: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, steps, slot_count, per_slot, dim = control_tokens.shape
        if dim != self.dim or per_slot <= 0:
            raise ValueError("control token shape disagrees with attention contract")
        if control_token_valid.shape != control_tokens.shape[:4]:
            raise ValueError("control token validity shape disagrees with tokens")
        if handles.shape != (batch, slot_count, dim):
            raise ValueError("handle shape disagrees with control slots")

        kv_count = slot_count * per_slot
        flat_tokens = control_tokens.reshape(batch * steps, kv_count, dim)
        base_keys = cross_attention.k(flat_tokens).reshape(batch, steps, slot_count, per_slot, dim)
        if not isinstance(ablate_subject_handle_keys, bool):
            raise TypeError("subject handle-key ablation flag must be boolean")
        projected_handles = handle_key_projection(handles)
        if ablate_subject_handle_keys:
            handle_mask = torch.ones(
                (1, slot_count, 1),
                dtype=projected_handles.dtype,
                device=projected_handles.device,
            )
            handle_mask[:, 1:] = 0
            projected_handles = projected_handles * handle_mask
        keys = cross_attention.norm_k(base_keys + projected_handles[:, None, :, None]).reshape(
            batch, steps, kv_count, dim
        )

        values = cross_attention.v(flat_tokens)
        gamma_mask = torch.ones(
            (batch * steps, kv_count, 1), dtype=values.dtype, device=values.device
        )
        gamma_mask[:, :per_slot] = background_value_scale.to(values.dtype)
        values = (values * gamma_mask).reshape(batch, steps, kv_count, dim)
        valid = control_token_valid.reshape(batch, steps, kv_count)
        return keys, values, valid, projected_handles

    @staticmethod
    def _attention_bias(
        valid: torch.Tensor,
        background_logit_prior: torch.Tensor,
        *,
        per_slot: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        batch, steps, kv_count = valid.shape
        if per_slot <= 0 or kv_count % per_slot:
            raise ValueError("flattened control width is not divisible by group width")
        expanded_valid = valid.reshape(batch * steps, kv_count)
        grouped_valid = expanded_valid.reshape(batch * steps, -1, per_slot)
        valid_counts = grouped_valid.sum(dim=-1)
        if bool((valid_counts[:, 0] != 1).any()):
            raise ValueError("background group must contain exactly one valid sentinel")
        # Turn each group's sum(exp(logit)) into a mean(exp(logit)) baseline.
        # This prevents differently tokenized action strings from receiving
        # different group priors solely because of their token counts.
        length_bias = -valid_counts.clamp_min(1).float().log()
        bias = (
            length_bias[..., None].expand(-1, -1, per_slot).reshape(batch * steps, 1, 1, kv_count)
        )
        bias = bias.to(dtype=dtype)
        bias.masked_fill_(~expanded_valid[:, None, None], float("-inf"))
        prior = background_logit_prior[:, None].expand(batch, steps).reshape(batch * steps, 1, 1, 1)
        bias[:, :, :, :per_slot] += prior.to(dtype)
        return bias

    @staticmethod
    def _randomize_subject_group_probabilities(
        head_probabilities: torch.Tensor,
        biased_logits: torch.Tensor,
        valid: torch.Tensor,
        *,
        batch: int,
        steps: int,
        spatial_tokens: int,
        slot_count: int,
        per_slot: int,
        seed: int,
    ) -> torch.Tensor:
        """Keep background/foreground mass but choose a random subject per query."""

        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("random subject-selection seed must be non-negative")
        grouped_valid = valid.reshape(batch, steps, slot_count, per_slot).any(dim=-1)
        foreground_valid = grouped_valid[..., 1:]
        if not torch.equal(
            foreground_valid,
            foreground_valid[:, :1].expand_as(foreground_valid),
        ):
            raise ValueError("valid subject groups must be constant across control steps")

        selections = torch.empty((batch, steps, spatial_tokens), dtype=torch.long, device="cpu")
        for batch_index in range(batch):
            valid_subjects = torch.nonzero(
                foreground_valid[batch_index, 0].detach().cpu(), as_tuple=False
            ).flatten()
            if valid_subjects.numel() == 0:
                raise ValueError("random selection requires a valid foreground subject")
            generator = torch.Generator(device="cpu").manual_seed(
                seed ^ ((batch_index + 1) * 0x45D9F3B)
            )
            sampled = torch.randint(
                int(valid_subjects.numel()),
                (steps, spatial_tokens),
                generator=generator,
            )
            selections[batch_index] = valid_subjects[sampled]

        batch_steps, heads = head_probabilities.shape[:2]
        grouped_probabilities = head_probabilities.reshape(
            batch_steps, heads, spatial_tokens, slot_count, per_slot
        )
        grouped_logits = biased_logits.reshape(
            batch_steps, heads, spatial_tokens, slot_count, per_slot
        )
        selection_index = (
            selections.reshape(batch_steps, 1, spatial_tokens, 1, 1)
            .to(head_probabilities.device)
            .expand(batch_steps, heads, spatial_tokens, 1, per_slot)
        )
        selected_logits = grouped_logits[..., 1:, :].gather(dim=3, index=selection_index).squeeze(3)
        selected_token_probabilities = torch.softmax(selected_logits, dim=-1)
        foreground_mass = grouped_probabilities[..., 1:, :].sum(dim=(-2, -1))

        randomized = torch.zeros_like(grouped_probabilities)
        randomized[..., 0, :] = grouped_probabilities[..., 0, :]
        randomized[..., 1:, :].scatter_(
            dim=3,
            index=selection_index,
            src=(foreground_mass[..., None] * selected_token_probabilities).unsqueeze(3),
        )
        return randomized.reshape_as(head_probabilities)

    def forward(
        self,
        cross_attention: nn.Module,
        queries: torch.Tensor,
        control_tokens: torch.Tensor,
        control_token_valid: torch.Tensor,
        handles: torch.Tensor,
        handle_key_projection: nn.Module,
        background_value_scale: torch.Tensor,
        background_logit_prior: torch.Tensor,
        *,
        temporal_tokens: int,
        spatial_tokens: int,
        return_routing_aux: bool = False,
        ablate_subject_control_values: bool = False,
        ablate_subject_handle_keys: bool = False,
        random_subject_selection_seed: int | None = None,
    ) -> torch.Tensor | CausalHandleAttentionOutput:
        batch, sequence, dim = queries.shape
        if dim != self.dim or sequence != temporal_tokens * spatial_tokens:
            raise ValueError("query grid disagrees with attention contract")
        if control_tokens.shape[1] != temporal_tokens - 1:
            raise ValueError("control step k must align to temporal latent k+1")
        if background_logit_prior.shape != (batch,):
            raise ValueError("background logit prior must be per-sample")

        heads = int(cross_attention.num_heads)
        head_dim = dim // heads
        normalized_queries = cross_attention.norm_q(cross_attention.q(queries))
        future_queries = normalized_queries.reshape(batch, temporal_tokens, spatial_tokens, dim)[
            :, 1:
        ]
        keys, values, valid, projected_handles = self.project_kv(
            cross_attention,
            control_tokens,
            control_token_valid,
            handles,
            handle_key_projection,
            background_value_scale,
            ablate_subject_handle_keys=ablate_subject_handle_keys,
        )
        steps = temporal_tokens - 1
        kv_count = keys.shape[2]
        per_slot = control_tokens.shape[3]
        slot_count = kv_count // per_slot
        if not isinstance(ablate_subject_control_values, bool):
            raise TypeError("subject-control value ablation flag must be boolean")
        if ablate_subject_control_values:
            value_mask = torch.ones((1, 1, kv_count, 1), dtype=values.dtype, device=values.device)
            value_mask[:, :, per_slot:] = 0
            attention_values = values * value_mask
        else:
            attention_values = values
        query_heads = future_queries.reshape(
            batch * steps, spatial_tokens, heads, head_dim
        ).transpose(1, 2)
        key_heads = keys.reshape(batch * steps, kv_count, heads, head_dim).transpose(1, 2)
        value_heads = attention_values.reshape(batch * steps, kv_count, heads, head_dim).transpose(
            1, 2
        )
        bias = self._attention_bias(
            valid,
            background_logit_prior,
            per_slot=per_slot,
            dtype=query_heads.dtype,
        )
        head_probabilities: torch.Tensor | None = None
        if random_subject_selection_seed is None:
            attended = F.scaled_dot_product_attention(
                query_heads, key_heads, value_heads, attn_mask=bias, dropout_p=0.0
            )
        else:
            attention_logits = torch.matmul(
                query_heads.float(), key_heads.float().transpose(-2, -1)
            )
            attention_logits.mul_(head_dim**-0.5)
            biased_logits = attention_logits + bias.float()
            normal_probabilities = torch.softmax(biased_logits, dim=-1)
            head_probabilities = self._randomize_subject_group_probabilities(
                normal_probabilities,
                biased_logits,
                valid,
                batch=batch,
                steps=steps,
                spatial_tokens=spatial_tokens,
                slot_count=slot_count,
                per_slot=per_slot,
                seed=random_subject_selection_seed,
            )
            attended = torch.matmul(head_probabilities.to(value_heads.dtype), value_heads)
        attended = attended.transpose(1, 2).reshape(batch, steps, spatial_tokens, dim)
        attended = F.linear(attended, cross_attention.o.weight, bias=None)
        anchor = attended.new_zeros((batch, 1, spatial_tokens, dim))
        hidden_states = torch.cat((anchor, attended), dim=1).reshape(batch, sequence, dim)
        if not return_routing_aux:
            return hidden_states

        slot_valid = valid.reshape(batch, steps, slot_count, per_slot).any(dim=-1)[:, 0]
        handle_logits = torch.einsum(
            "btpd,bgd->btpg", future_queries.float(), projected_handles.float()
        ) * (dim**-0.5)
        handle_logits.masked_fill_(~slot_valid[:, None, None], float("-inf"))
        handle_log_probabilities = torch.log_softmax(handle_logits, dim=-1)

        if head_probabilities is None:
            logits = torch.matmul(query_heads.float(), key_heads.float().transpose(-2, -1))
            logits.mul_(head_dim**-0.5)
            head_probabilities = torch.softmax(logits + bias.float(), dim=-1)
        probabilities = head_probabilities.mean(dim=1)
        grouped = probabilities.reshape(batch * steps, spatial_tokens, slot_count, per_slot)
        group_probabilities = grouped.sum(dim=-1)
        group_log_probabilities = torch.where(
            group_probabilities > 0,
            group_probabilities.clamp_min(torch.finfo(torch.float32).tiny).log(),
            group_probabilities.new_full((), float("-inf")),
        ).reshape(batch, steps, spatial_tokens, slot_count)

        background_values = value_heads.detach().float()
        background_probabilities = head_probabilities.detach()[..., :per_slot]
        background_attended = (
            background_probabilities[..., None] * background_values[:, :, None, :per_slot]
        ).sum(dim=3)
        background_attended = background_attended.transpose(1, 2).reshape(
            batch, steps, spatial_tokens, dim
        )
        background_residual = F.linear(
            background_attended,
            cross_attention.o.weight.detach().float(),
            bias=None,
        )
        background_valid = valid[:, 0, :per_slot].float()
        values0 = values[:, 0, :per_slot].detach().float()
        background_value_norm = values0.norm(dim=-1).mul(
            background_valid
        ).sum() / background_valid.sum().clamp_min(1.0)
        projected_handle_rms = projected_handles.float().square().mean(dim=-1).sqrt()
        projected_handle_rms = projected_handle_rms.mul(
            slot_valid
        ).sum() / slot_valid.sum().clamp_min(1)
        return CausalHandleAttentionOutput(
            hidden_states=hidden_states,
            handle_log_probabilities=handle_log_probabilities,
            group_log_probabilities=group_log_probabilities.detach(),
            background_value_norm=background_value_norm,
            background_residual_norm=background_residual.norm(dim=-1).mean(),
            projected_handle_rms=projected_handle_rms,
        )
