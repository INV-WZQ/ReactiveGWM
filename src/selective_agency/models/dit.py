"""Wan DiT with episode handles and explicit video attention visibility."""

from __future__ import annotations

from typing import NamedTuple, Tuple

import torch
import torch.utils.checkpoint
from einops import rearrange

from diffsynth.models.wan_video_dit import WanModel, modulate, sinusoidal_embedding_1d

from selective_agency.constants import (
    architecture_for_attention,
    BACKGROUND_LOGIT_PRIOR_OFFSET,
    CONTROL_STEPS,
    HANDLE_INJECTION_ALPHA,
    MAX_SUBJECTS,
)
from selective_agency.models.attention import (
    CausalHandleAttentionOutput,
    JointBackgroundSubjectKVAttention,
)
from selective_agency.models.causal_attention import (
    native_bidirectional_self_attention,
    native_frame_causal_self_attention,
)
from selective_agency.models.control import ControlTokenBuilder
from selective_agency.models.coverage import prepare_visible_coverage
from selective_agency.models.handle import HandleCodebook, inject_first_frame_handles


def _advance_initialization_rng(dim: int) -> None:
    """Preserve the RNG offset used to initialize the control modules."""

    projection = torch.nn.Linear(dim, dim, bias=False)
    torch.nn.init.xavier_normal_(projection.weight)


class SelectiveAgencyOutput(NamedTuple):
    prediction: torch.Tensor
    handle_log_probabilities: torch.Tensor
    group_log_probabilities: torch.Tensor
    background_value_norms: torch.Tensor
    background_residual_norms: torch.Tensor
    projected_handle_rms: torch.Tensor
    handle_rms: torch.Tensor


class SelectiveAgencyWanDiT(WanModel):
    """Wan2.2 TI2V backbone with causal or fully bidirectional self-attention."""

    _custom_state_prefixes = (
        "control_token_builder.",
        "handle_codebook.",
        "handle_key_projections.",
        "background_slot_attention.",
        "background_value_scales.",
    )

    def __init__(
        self,
        *,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool = False,
        seperated_timestep: bool = True,
        require_vae_embedding: bool = False,
        require_clip_embedding: bool = False,
        fuse_vae_embedding_in_latents: bool = True,
        max_subjects: int = MAX_SUBJECTS,
        control_steps: int = CONTROL_STEPS,
        self_attention_mode: str = "frame_causal",
        game: str = "hnm",
        handle_injection_alpha: float = HANDLE_INJECTION_ALPHA,
    ):
        architecture_version, architecture_contract = architecture_for_attention(
            self_attention_mode, game=game
        )
        if len(patch_size) != 3 or int(patch_size[0]) != 1:
            raise ValueError("per-frame control alignment requires temporal patch size 1")
        super().__init__(
            dim=dim,
            in_dim=in_dim,
            ffn_dim=ffn_dim,
            out_dim=out_dim,
            text_dim=text_dim,
            freq_dim=freq_dim,
            eps=eps,
            patch_size=patch_size,
            num_heads=num_heads,
            num_layers=num_layers,
            has_image_input=has_image_input,
            seperated_timestep=seperated_timestep,
            require_vae_embedding=require_vae_embedding,
            require_clip_embedding=require_clip_embedding,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        if has_image_input or require_vae_embedding or require_clip_embedding:
            raise ValueError("this model disables Wan global image/text context branches")
        self.max_subjects = int(max_subjects)
        self.control_steps = int(control_steps)
        self.self_attention_mode = self_attention_mode
        self.handle_injection_alpha = handle_injection_alpha
        self.architecture_version = architecture_version
        self.architecture_contract = architecture_contract

        _advance_initialization_rng(dim)
        self.control_token_builder = ControlTokenBuilder(dim, eps)
        self.handle_codebook = HandleCodebook(dim, max_subjects)
        self.handle_key_projections = torch.nn.ModuleList(
            [torch.nn.Linear(dim, dim, bias=False) for _ in range(num_layers)]
        )
        for projection in self.handle_key_projections:
            torch.nn.init.xavier_normal_(projection.weight)
        self.background_slot_attention = JointBackgroundSubjectKVAttention(dim, eps)
        self.background_value_scales = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros(1)) for _ in range(num_layers)]
        )
        self.subject_control_value_ablation_blocks: frozenset[int] = frozenset()
        self.subject_handle_key_ablation_blocks: frozenset[int] = frozenset()
        self.random_subject_query_selection_seeds: dict[int, int] = {}
        self.register_buffer("control_text_hidden", torch.empty(0), persistent=False)
        self.register_buffer(
            "control_text_mask", torch.empty(0, dtype=torch.bool), persistent=False
        )
        self.register_buffer("npc_text_hidden", torch.empty(0), persistent=False)
        self.register_buffer("npc_text_mask", torch.empty(0, dtype=torch.bool), persistent=False)

    def set_control_text_cache(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor,
        npc_hidden: torch.Tensor | None = None,
        npc_mask: torch.Tensor | None = None,
    ) -> None:
        """Attach complete frozen T5 tables; trainable projections stay uncached."""

        if hidden.ndim != 3 or hidden.shape[0] <= 0:
            raise ValueError("control hidden must contain one or more action rows")
        if mask.shape != hidden.shape[:2]:
            raise ValueError("control attention mask shape mismatch")
        if not bool(mask.bool().any(dim=-1).all()):
            raise ValueError("Action text rows must contain at least one valid token")
        if (npc_hidden is None) != (npc_mask is None):
            raise ValueError("NPC hidden and mask must be supplied together")
        if npc_hidden is None:
            npc_hidden = hidden.new_empty((0, 0, hidden.shape[-1]))
            npc_mask = mask.new_empty((0, 0), dtype=torch.bool)
        if npc_hidden.ndim != 3 or npc_hidden.shape[-1] != hidden.shape[-1]:
            raise ValueError("NPC hidden must be [N,L,text_dim]")
        if npc_mask.shape != npc_hidden.shape[:2]:
            raise ValueError("NPC attention mask shape mismatch")
        if not bool(npc_mask.bool().any(dim=-1).all()):
            raise ValueError("NPC text rows must contain at least one valid token")
        device = self.patch_embedding.weight.device
        self.control_text_hidden = hidden.detach().to(device=device, dtype=hidden.dtype)
        self.control_text_mask = mask.detach().to(device=device, dtype=torch.bool)
        self.npc_text_hidden = npc_hidden.detach().to(device=device, dtype=npc_hidden.dtype)
        self.npc_text_mask = npc_mask.detach().to(device=device, dtype=torch.bool)

    def set_subject_control_value_ablation(self, block_indices: tuple[int, ...]) -> None:
        """Remove subject value contributions in selected cross-attention blocks."""

        values = tuple(block_indices)
        if any(not isinstance(index, int) or isinstance(index, bool) for index in values):
            raise TypeError("subject-control ablation blocks must be integer indices")
        if len(values) != len(set(values)):
            raise ValueError("subject-control ablation blocks must be unique")
        if any(index < 0 or index >= len(self.blocks) for index in values):
            raise ValueError("subject-control ablation block index is out of range")
        self.subject_control_value_ablation_blocks = frozenset(values)

    def set_subject_handle_key_ablation(self, block_indices: tuple[int, ...]) -> None:
        """Remove subject handle contributions from keys in selected blocks."""

        values = tuple(block_indices)
        if any(not isinstance(index, int) or isinstance(index, bool) for index in values):
            raise TypeError("subject handle-key ablation blocks must be integer indices")
        if len(values) != len(set(values)):
            raise ValueError("subject handle-key ablation blocks must be unique")
        if any(index < 0 or index >= len(self.blocks) for index in values):
            raise ValueError("subject handle-key ablation block index is out of range")
        self.subject_handle_key_ablation_blocks = frozenset(values)

    def set_random_subject_query_selection(
        self,
        block_indices: tuple[int, ...],
        *,
        seed: int,
    ) -> dict[int, int]:
        """Assign deterministic per-block seeds for random Q-to-subject selection."""

        values = tuple(block_indices)
        if any(not isinstance(index, int) or isinstance(index, bool) for index in values):
            raise TypeError("random Q-selection blocks must be integer indices")
        if len(values) != len(set(values)):
            raise ValueError("random Q-selection blocks must be unique")
        if any(index < 0 or index >= len(self.blocks) for index in values):
            raise ValueError("random Q-selection block index is out of range")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("random Q-selection seed must be a non-negative integer")

        generator = torch.Generator(device="cpu").manual_seed(seed)
        seeds = {
            block_index: int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item())
            for block_index in values
        }
        self.random_subject_query_selection_seeds = seeds
        return dict(seeds)

    def _selective_block(
        self,
        block: torch.nn.Module,
        handle_key_projection: torch.nn.Module,
        x: torch.Tensor,
        t_mod: torch.Tensor,
        freqs: torch.Tensor,
        control_tokens: torch.Tensor,
        control_token_valid: torch.Tensor,
        handles: torch.Tensor,
        background_value_scale: torch.nn.Parameter,
        background_logit_prior: torch.Tensor,
        temporal_tokens: int,
        spatial_tokens: int,
        return_routing_aux: bool,
        ablate_subject_control_values: bool = False,
        ablate_subject_handle_keys: bool = False,
        random_subject_selection_seed: int | None = None,
    ) -> (
        torch.Tensor
        | tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ):
        has_seq = t_mod.ndim == 4
        chunk_dim = 2 if has_seq else 1
        values = (block.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(
            6, dim=chunk_dim
        )
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = values
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                item.squeeze(2) for item in values
            )

        self_input = modulate(block.norm1(x), shift_msa, scale_msa)
        self_attention = (
            native_bidirectional_self_attention
            if self.self_attention_mode == "full_bidirectional"
            else native_frame_causal_self_attention
        )
        self_output = self_attention(
            block.self_attn,
            self_input,
            freqs,
            temporal_tokens=temporal_tokens,
            spatial_tokens=spatial_tokens,
        )
        x = block.gate(x, gate_msa, self_output)
        cross_result = self.background_slot_attention(
            block.cross_attn,
            block.norm3(x),
            control_tokens,
            control_token_valid,
            handles,
            handle_key_projection,
            background_value_scale,
            background_logit_prior,
            temporal_tokens=temporal_tokens,
            spatial_tokens=spatial_tokens,
            return_routing_aux=return_routing_aux,
            ablate_subject_control_values=ablate_subject_control_values,
            ablate_subject_handle_keys=ablate_subject_handle_keys,
            random_subject_selection_seed=random_subject_selection_seed,
        )
        if return_routing_aux:
            if not isinstance(cross_result, CausalHandleAttentionOutput):
                raise TypeError("routing auxiliary output was not returned")
            cross_output = cross_result.hidden_states
        else:
            if not isinstance(cross_result, torch.Tensor):
                raise TypeError("plain attention output was not returned")
            cross_output = cross_result

        x = x + cross_output
        mlp_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
        x = block.gate(x, gate_mlp, block.ffn(mlp_input))
        if not return_routing_aux:
            return x
        if any(
            item is None
            for item in (
                cross_result.handle_log_probabilities,
                cross_result.group_log_probabilities,
                cross_result.background_value_norm,
                cross_result.background_residual_norm,
                cross_result.projected_handle_rms,
            )
        ):
            raise RuntimeError("incomplete routing auxiliary output")
        return (
            x,
            cross_result.handle_log_probabilities,
            cross_result.group_log_probabilities,
            cross_result.background_value_norm,
            cross_result.background_residual_norm,
            cross_result.projected_handle_rms,
        )

    def forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        subject_roi_masks: torch.Tensor,
        action_ids: torch.Tensor,
        subject_valid: torch.Tensor,
        *,
        control_kind: torch.Tensor | None = None,
        npc_prompt_ids: torch.Tensor | None = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        return_routing_aux: bool = False,
    ) -> torch.Tensor | SelectiveAgencyOutput:
        if use_gradient_checkpointing_offload:
            raise ValueError("checkpoint CPU offload is disabled")
        if self.control_text_hidden.numel() == 0:
            raise RuntimeError("control T5 cache has not been attached")
        if latents.ndim != 5:
            raise ValueError("latents must be [B,C,F,H,W]")
        batch = latents.shape[0]
        if timestep.ndim == 0:
            timestep = timestep.expand(batch)
        timestep = timestep.reshape(batch).to(device=latents.device, dtype=latents.dtype)

        clean_x_grid = self.patch_embedding(latents)
        _, _, temporal, height, width = clean_x_grid.shape
        if temporal != self.control_steps + 1:
            raise ValueError(
                f"expected {self.control_steps + 1} temporal latents, received {temporal}"
            )
        spatial = height * width
        if subject_roi_masks.shape != (
            batch,
            self.max_subjects,
            height,
            width,
        ):
            raise ValueError(
                "subject ROI masks must match the patch grid "
                f"{(batch, self.max_subjects, height, width)}"
            )
        if action_ids.shape != (batch, self.control_steps, self.max_subjects):
            raise ValueError("action ids must be [B,control_steps,max_subjects]")
        if subject_valid.shape != (batch, self.max_subjects):
            raise ValueError("subject_valid must be [B,max_subjects]")

        handles = self.handle_codebook(subject_valid)
        control_tokens, control_token_valid = self.control_token_builder(
            text_embedding=self.text_embedding,
            control_text_hidden=self.control_text_hidden,
            control_text_mask=self.control_text_mask,
            action_ids=action_ids,
            subject_valid=subject_valid,
            control_kind=control_kind,
            npc_prompt_ids=npc_prompt_ids,
            npc_text_hidden=self.npc_text_hidden,
            npc_text_mask=self.npc_text_mask,
        )
        x = rearrange(clean_x_grid, "b c f h w -> b (f h w) c").contiguous()
        coverage = prepare_visible_coverage(subject_roi_masks, subject_valid)
        x = inject_first_frame_handles(
            x,
            handles[:, 1:],
            coverage.subjects.flatten(2),
            temporal_tokens=temporal,
            spatial_tokens=spatial,
            alpha=self.handle_injection_alpha,
        )

        valid_count = subject_valid.bool().sum(dim=-1).to(torch.float32)
        if bool((valid_count <= 0).any()):
            raise ValueError("each sample must contain at least one valid subject")
        background_logit_prior = valid_count.log() + BACKGROUND_LOGIT_PRIOR_OFFSET
        token_timestep = timestep[:, None].expand(batch, temporal * spatial).clone()
        token_timestep[:, :spatial] = 0
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, token_timestep.reshape(-1)).reshape(
                batch, temporal * spatial, self.freq_dim
            )
        )
        t_mod = self.time_projection(t).unflatten(-1, (6, self.dim))
        freqs = (
            torch.cat(
                [
                    self.freqs[0][:temporal]
                    .view(temporal, 1, 1, -1)
                    .expand(temporal, height, width, -1),
                    self.freqs[1][:height]
                    .view(1, height, 1, -1)
                    .expand(temporal, height, width, -1),
                    self.freqs[2][:width].view(1, 1, width, -1).expand(temporal, height, width, -1),
                ],
                dim=-1,
            )
            .reshape(temporal * spatial, 1, -1)
            .to(x.device)
        )

        handle_logs: list[torch.Tensor] = []
        group_logs: list[torch.Tensor] = []
        background_value_norms: list[torch.Tensor] = []
        background_residual_norms: list[torch.Tensor] = []
        projected_handle_rms: list[torch.Tensor] = []
        for block_index, block in enumerate(self.blocks):
            handle_key_projection = self.handle_key_projections[block_index]
            background_value_scale = self.background_value_scales[block_index]
            ablate_subject_control_values = (
                block_index in self.subject_control_value_ablation_blocks
            )
            ablate_subject_handle_keys = block_index in self.subject_handle_key_ablation_blocks
            random_subject_selection_seed = self.random_subject_query_selection_seeds.get(
                block_index
            )
            if use_gradient_checkpointing and self.training:

                def forward_block(
                    current: torch.Tensor,
                    *,
                    _block=block,
                    _handle_key_projection=handle_key_projection,
                    _background_value_scale=background_value_scale,
                    _ablate_subject_control_values=ablate_subject_control_values,
                    _ablate_subject_handle_keys=ablate_subject_handle_keys,
                    _random_subject_selection_seed=random_subject_selection_seed,
                ):
                    return self._selective_block(
                        _block,
                        _handle_key_projection,
                        current,
                        t_mod,
                        freqs,
                        control_tokens,
                        control_token_valid,
                        handles,
                        _background_value_scale,
                        background_logit_prior,
                        temporal,
                        spatial,
                        return_routing_aux,
                        _ablate_subject_control_values,
                        _ablate_subject_handle_keys,
                        _random_subject_selection_seed,
                    )

                block_result = torch.utils.checkpoint.checkpoint(
                    forward_block, x, use_reentrant=False
                )
            else:
                block_result = self._selective_block(
                    block,
                    handle_key_projection,
                    x,
                    t_mod,
                    freqs,
                    control_tokens,
                    control_token_valid,
                    handles,
                    background_value_scale,
                    background_logit_prior,
                    temporal,
                    spatial,
                    return_routing_aux,
                    ablate_subject_control_values,
                    ablate_subject_handle_keys,
                    random_subject_selection_seed,
                )
            if not return_routing_aux:
                if not isinstance(block_result, torch.Tensor):
                    raise TypeError("selective block did not return hidden states")
                x = block_result
                continue
            if not isinstance(block_result, tuple) or len(block_result) != 6:
                raise TypeError("selective block did not return routing tensors")
            (
                x,
                block_handle_log,
                block_group_log,
                block_background_value_norm,
                block_background_residual_norm,
                block_projected_handle_rms,
            ) = block_result
            handle_logs.append(block_handle_log)
            group_logs.append(block_group_log)
            background_value_norms.append(block_background_value_norm)
            background_residual_norms.append(block_background_residual_norm)
            projected_handle_rms.append(block_projected_handle_rms)

        x = self.head(x, t)
        prediction = self.unpatchify(x, (temporal, height, width))
        if not return_routing_aux:
            return prediction

        slot_valid = torch.cat(
            (
                torch.ones(batch, 1, dtype=torch.bool, device=subject_valid.device),
                subject_valid.bool(),
            ),
            dim=1,
        )
        handle_norms = handles.float().square().mean(dim=-1).sqrt()
        handle_rms = handle_norms.mul(slot_valid).sum() / slot_valid.sum().clamp_min(1)
        return SelectiveAgencyOutput(
            prediction=prediction,
            handle_log_probabilities=torch.stack(handle_logs),
            group_log_probabilities=torch.stack(group_logs),
            background_value_norms=torch.stack(background_value_norms),
            background_residual_norms=torch.stack(background_residual_norms),
            projected_handle_rms=torch.stack(projected_handle_rms),
            handle_rms=handle_rms,
        )
