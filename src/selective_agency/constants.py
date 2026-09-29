"""First-frame-handle architectures with explicit video attention visibility."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping
from selective_agency.runtime.geometry import DEFAULT_VIDEO

ARCHITECTURE_VERSION = "native_causal_handle_full_condition_fm"
BIDIRECTIONAL_ARCHITECTURE_VERSION = "native_bidirectional_handle_full_condition_fm"
SELF_ATTENTION_MODES = ("frame_causal", "full_bidirectional")
CONDITION_PROTOCOL = "selective-agency-full-condition-text"
HANDLE_RMS_EPS = 1e-6
HANDLE_INJECTION_ALPHA = 0.25
HANDLE_ROUTING_DIM = 3072
# Each token group is normalized by its valid-token count before groups compete.
# A background prior of log(N_valid) therefore gives neutral 50/50 mass between
# the one background group and all valid subject groups when their mean logits
# are equal.
BACKGROUND_LOGIT_PRIOR_OFFSET = 0.0
ARCHITECTURE_CONTRACT = {
    "handle_codebook": "learned_background_plus_six_subject_rows_rms_norm_fp32",
    "handle_count": 7,
    "handle_injection": "visible_patch_coverage_anchor_only",
    "handle_injection_alpha": HANDLE_INJECTION_ALPHA,
    "background_handle_injection": False,
    "native_self_attention": "frame_prefix_with_same_frame_bidirectional",
    "causal_position_encoding": "wan_3d_rope",
    "native_self_attention_parameters": "direct_original_wan_qkvo_and_qk_rmsnorm",
    "additional_self_attention": False,
    "temporal_patch_size": 1,
    "handle_key_projection": "per_block_bias_free_full_rank_xavier",
    "handle_routing": "full_3072d_query_projected_handle_softmax",
    "handle_routing_background_prior": False,
    "cross_key_binding": "norm_k(w_k_context_plus_projected_handle)",
    "cross_value_binding": "unchanged_w_v_context",
    "control_token_segments": {"external_action": "full_text", "npc_prompt": "full_text"},
    "control_tokens_per_slot": "batch_dynamic_valid_length",
    "condition_protocol": CONDITION_PROTOCOL,
    "npc_actor_binding": "actor_handle_key_only",
    "npc_trigger_gate": False,
    "text_truncation": False,
    "control_vocabulary": "ordered_execution_config_and_asset_manifest",
    "explicit_visual_control_tokens": False,
    "appearance_source": "wan_first_frame_latent_no_explicit_control_roi",
    "handle_anchor_source": "first_frame_soft_patch_coverage",
    "subject_masks_model_visible": True,
    "subject_mask_usage": "soft_patch_coverage_anchor_only",
    "subject_boxes_model_visible": False,
    "control_padding_mask": "dynamic_per_group_bias_neg_inf",
    "control_group_length_normalization": "subtract_log_valid_token_count",
    "background_slot": "learned_zero_init_single_token_sentinel",
    "background_value": "per_block_zero_init_gamma_scaled",
    "background_logit_prior": "log_valid_subject_count",
    "background_output_projection": "weight_only_zero_bias",
    "prediction_path": "native_causal_first_frame_actor_handle_bound_full_condition",
    "training_objective": "future_latent_flow_matching_mse_only",
    "training_auxiliary": "none",
    "handle_routing_loss_weight": 0.0,
    "routing_diagnostics": "optional_detached_evaluation_only",
    "retained_ablation_modules": [
        "first_frame_handle_injection",
        "native_frame_prefix_causal_self_attention",
        "handle_augmented_cross_attention_key",
    ],
    "new_trainable_parameters": True,
}


def architecture_for_attention(mode: str = "frame_causal", *, game="hnm") -> tuple[str, dict[str, Any]]:
    """Return the architecture identifier and contract for the selected visibility."""
    if mode not in SELF_ATTENTION_MODES:
        raise ValueError(f"self_attention_mode must be one of {SELF_ATTENTION_MODES}, got {mode!r}")
    if game not in {"hnm", "sf"}:
        raise ValueError("game must be hnm or sf")
    contract = deepcopy(ARCHITECTURE_CONTRACT)
    suffix = "full_condition" if game == "hnm" else "actor_prompt"
    prediction = "actor_handle_bound_full_condition" if game == "hnm" else "handle_bound_actor_prompt"
    if game == "sf":
        del contract["npc_actor_binding"]
        contract["npc_condition_binding"] = "own_actor_handle_and_complete_if_then_prompt"
    contract["prediction_path"] = f"native_causal_first_frame_{prediction}"
    if mode == "frame_causal":
        return f"native_causal_handle_{suffix}_fm", contract
    contract.update(
        native_self_attention="full_sequence_bidirectional_including_anchor",
        prediction_path=f"native_bidirectional_first_frame_{prediction}",
        retained_ablation_modules=[
            "first_frame_handle_injection",
            "native_full_bidirectional_self_attention",
            "handle_augmented_cross_attention_key",
        ],
    )
    return f"native_bidirectional_handle_{suffix}_fm", contract


def checkpoint_has_attention(metadata: Mapping[str, Any], mode: str) -> bool:
    """Causal exports may omit mode; bidirectional exports must name it."""
    game = metadata.get("game", "hnm")
    version, contract = architecture_for_attention(mode, game=game)
    recorded_mode = metadata.get("self_attention_mode")
    if (
        "self_attention_mode" not in metadata
        and metadata.get("architecture_version") == architecture_for_attention(game=game)[0]
    ):
        recorded_mode = "frame_causal"
    return (
        recorded_mode == mode
        and metadata.get("architecture_version") == version
        and metadata.get("architecture_contract") == contract
    )


MAX_SUBJECTS = 6
CONTROL_STEPS = DEFAULT_VIDEO.control_steps
VIDEO_FRAMES = DEFAULT_VIDEO.frames
FPS = DEFAULT_VIDEO.fps
HEIGHT = DEFAULT_VIDEO.height
WIDTH = DEFAULT_VIDEO.width
LATENT_FRAMES = DEFAULT_VIDEO.latent_frames
LATENT_HEIGHT, LATENT_WIDTH = DEFAULT_VIDEO.latent_size
PATCH_HEIGHT, PATCH_WIDTH = LATENT_HEIGHT // 2, LATENT_WIDTH // 2
ID_OCCUPANCY_DENOMINATOR = 4 * 32 * 32
ID_OCCUPANCY_CHANNELS = MAX_SUBJECTS + 1
