"""SF model text and validation of static response semantics."""

from __future__ import annotations

import json

from .records import ViewContractError

PROMPT_PROTOCOL = "sf-if-then"
PROMPTS = {
    "R01": "If the opponent retreats, then approach until close.",
    "R02": "If the opponent approaches, then retreat until safely out of reach.",
    "R03": "If the opponent whiffs an attack, then approach if needed and punish once before that attack recovers.",
    "R04": "If the opponent launches a threatening projectile, then jump once to evade it when it gets close.",
    "R05": "If this NPC takes an unblocked hit from the opponent, then recover, approach if needed, and counterattack once.",
    "R06": "If the opponent starts a threatening attack, then stand-block high, overhead, or mid attacks, crouch-block lows, and release when the threat ends.",
    "R07": "If the opponent jumps toward this NPC, then anti-air once as they descend within reach, before landing.",
    "R08": "If the opponent's projectile approaches, then tap toward the opponent to parry once when it gets close, and release.",
    "R09": "If the opponent walks into normal throw range while both fighters are grounded and throwable, then perform one normal throw and release.",
    "R10": "If this NPC is assigned no-op, then keep all controls released.",
}
IMMEDIATE_PROMPTS = {
    "R04": "If the opponent launches a threatening projectile, then jump once to evade it as soon as able.",
    "R07": "If the opponent jumps toward this NPC, then anti-air once as soon as able, before they land.",
}
_RESPONSE_RULES = {
    "move_toward_bound_target": "R01",
    "move_away_from_bound_target": "R02",
    "perform_one_qualified_punish_attack": "R03",
    "jump_to_evade_bound_projectile": "R04",
    "perform_one_qualified_counterattack": "R05",
    "apply_guard_branch_for_bound_attack_height": "R06",
    "perform_one_qualified_anti_air_attack": "R07",
    "perform_qualified_new_generation_parry_for_bound_projectile": "R08",
    "perform_one_qualified_new_generation_normal_throw": "R09",
}
EXECUTION_PROTOCOL = {
    "version": "sf3_v2_causal_block12_rules_v1",
    "block_ticks": 12,
    "press_ticks": 3,
    "decisions": "at_common_boundaries_from_already_observed_events",
    "completion": "finish_selected_template_then_release_at_common_boundary",
}
_EXECUTION_PARAMETERS = {
    "R01": {"near_exit", "close_gap"},
    "R02": {"near_entry", "safe_gap"},
    "R03": {"punish_gap"},
    "R04": {"jump_gap"},
    "R05": {"response_gap"},
    "R06": {"threat_gap", "threat_policy"},
    "R07": {"anti_air_height"},
    "R08": {"parry_gap"},
    "R09": {"throw_box"},
}

# Supported static clauses; these are validated without rendering paragraphs.
_CLAUSES = {
    "target_exits_near_band_while_moving_away",
    "previously_inside_near_exit",
    "target_actively_moves_away_from_npc",
    "gap_increases_continuously",
    "gap_crosses_calibrated_near_exit",
    "move_toward_bound_target",
    "stop_at_calibrated_close_gap",
    "gap_at_or_below_calibrated_response_stop",
    "target_reenters_near_band",
    "target_enters_near_band_while_approaching",
    "previously_outside_near_entry",
    "target_actively_moves_toward_npc",
    "gap_decreases",
    "gap_crosses_calibrated_near_entry",
    "move_away_from_bound_target",
    "stop_at_calibrated_safe_gap",
    "gap_at_or_above_calibrated_response_stop",
    "target_exits_near_band",
    "target_whiff_enters_recovery",
    "target_attack_active_interval_finished",
    "bound_attack_has_no_hit_block_or_parry_contact",
    "bound_target_attack_still_in_recovery",
    "npc_actionable",
    "approach_if_needed",
    "perform_one_qualified_punish_attack",
    "punish_hits_bound_target_within_bound_recovery",
    "bound_target_attack_recovery_closed",
    "target_projectile_spawned_toward_npc",
    "actual_projectile_spawn_observed",
    "projectile_source_is_target",
    "projectile_trajectory_threatens_npc",
    "npc_grounded_and_actionable",
    "jump_to_evade_bound_projectile",
    "npc_was_airborne",
    "bound_projectile_cleared_without_hitting_npc",
    "npc_landed",
    "bound_projectile_cleared",
    "npc_grounded",
    "target_deals_unblocked_damage_to_npc",
    "damage_source_is_target",
    "damage_victim_is_npc",
    "npc_hp_decreased",
    "damage_is_unblocked_not_guard_chip",
    "source_is_qualified_strike_or_projectile",
    "keep_controls_released_until_npc_hit_recovery_ends",
    "wait_until_npc_actionable",
    "perform_one_qualified_counterattack",
    "counterattack_hits_bound_target_after_npc_recovery",
    "npc_hit_stun_and_knockdown_recovery_cleared",
    "target_height_classified_attack_threatens_npc",
    "target_attack_faces_npc",
    "target_attack_in_qualified_threat_range",
    "bound_attack_guard_height_authoritatively_known",
    "npc_can_guard_before_contact",
    "apply_guard_branch_for_bound_attack_height",
    "release_all_controls_when_bound_threat_ends",
    "bound_attack_blocked_with_matching_guard_height",
    "no_unblocked_damage",
    "high_or_overhead",
    "stand_guard",
    "low",
    "crouch_guard",
    "mid",
    "qualified_mid_guard",
    "guard_released",
    "target_jumps_toward_npc",
    "target_transitions_ground_to_air",
    "authoritative_target_displacement_toward_npc",
    "perform_one_qualified_anti_air_attack",
    "anti_air_hits_bound_target_while_airborne",
    "target_landed",
    "bound_projectile_approaches_npc",
    "npc_can_parry",
    "track_bound_projectile_with_controls_released_until_qualified_parry_window",
    "perform_qualified_new_generation_parry_for_bound_projectile",
    "bound_projectile_parried",
    "npc_hp_loss_zero",
    "parry_input_released",
    "native_parry_rearmed",
    "target_walks_from_outside_into_true_normal_throw_range",
    "previously_outside_calibrated_normal_throw_range",
    "target_actively_walks_toward_npc",
    "target_crosses_into_calibrated_normal_throw_range",
    "npc_can_execute_normal_throw",
    "target_is_normally_throwable",
    "both_fighters_grounded_and_throw_eligible",
    "qualified_throw_facing_and_geometry",
    "perform_one_qualified_new_generation_normal_throw",
    "native_normal_throw_by_npc_captures_bound_target",
    "throw_sequence_complete",
    "native_throw_sequence_and_recovery_complete",
    "target_exits_true_normal_throw_range",
    "both_fighters_throw_eligible_again",
    "release_all_controls",
    "all_npc_controls_released",
    "response_complete",
    "natural_recovery_complete",
    "previous_bound_evidence_closed",
    "trigger_condition_cleared_before_rearm",
    "new_event_instance_required",
    "target_identity",
    "npc_identity",
    "event_instance",
    "distance_crossing_instance",
    "target_attack_instance",
    "target_recovery_interval",
    "projectile_instance",
    "npc_damage_event",
    "npc_hit_recovery_interval",
    "attack_guard_height",
    "target_jump_instance",
    "normal_throw_range_entry_instance",
}

def _validate_clause(value: str) -> None:
    try:
        supported = value in _CLAUSES
    except TypeError:
        supported = False
    if not supported:
        raise ViewContractError(f"unsupported SF static rule clause {value!r}")


def _validate_clauses(value, field: str) -> None:
    if not isinstance(value, (list, tuple)) or not value:
        raise ViewContractError(f"SF rule {field} must be a nonempty list")
    for clause in value:
        _validate_clause(clause)


def _check_keys(value, required, field, *, optional=()):
    if not isinstance(value, dict) or not set(required).issubset(value):
        raise ViewContractError(f"SF rule {field} is missing complete static semantics")
    if set(value) - set(required) - {"parameters", "static_parameters"} - set(optional):
        raise ViewContractError(
            f"SF rule {field} contains unsupported fields: {sorted(set(value)-set(required))}"
        )


def _validate_execution(rule_id: str, execution: dict) -> None:
    if not isinstance(execution, dict) or set(execution) != set(EXECUTION_PROTOCOL) | {
        "static_parameters"
    }:
        raise ViewContractError(
            "SF execution fields differ from the published static block12 protocol"
        )
    for key, expected in EXECUTION_PROTOCOL.items():
        if type(execution[key]) is not type(expected) or execution[key] != expected:
            raise ViewContractError(
                f"SF execution {key} differs from the published static block12 protocol"
            )
    parameters = execution["static_parameters"]
    if not isinstance(parameters, dict) or set(parameters) != _EXECUTION_PARAMETERS[rule_id]:
        raise ViewContractError(
            f"SF {rule_id} execution static parameter fields differ from the published rule"
        )
    for key, value in parameters.items():
        if key == "threat_policy":
            valid = value == "qualified_attack_geometry_v1" and type(value) is str
        elif key == "throw_box":
            valid = (
                isinstance(value, list)
                and len(value) == 4
                and all(type(item) is int for item in value)
            )
        else:
            valid = type(value) is int
        if not valid:
            raise ViewContractError(
                f"SF {rule_id} execution static parameter {key} has an unsupported type or policy"
            )


def compile_sf_prompt(reactivity: dict) -> str:
    """Select model text from static semantics, independently of source rule labels."""
    if reactivity == {"policy": "all_controls_released"}:
        return PROMPTS["R10"]
    _check_keys(reactivity, {"trigger", "response", "reset"}, "reactivity", optional={"execution"})
    trigger, response = reactivity["trigger"], reactivity["response"]
    _check_keys(trigger, {"event", "all_of", "bind", "count_policy"}, "trigger")
    _check_keys(
        response,
        {"steps", "branches", "complete_when", "outside_response", "start_policy"},
        "response",
    )
    if (
        trigger["count_policy"] != "once_per_bound_event_after_rearm"
        or response["outside_response"] != "all_npc_controls_released"
        or response["start_policy"] != "only_after_bound_trigger"
    ):
        raise ViewContractError("SF event counting, released-wait, or trigger-first policy differs")
    _validate_clause(trigger["event"])
    _validate_clauses(trigger["all_of"], "all_of")
    _validate_clauses(response["steps"], "steps")
    _validate_clauses(response["complete_when"], "complete_when")
    _validate_clauses(trigger["bind"], "bind")
    _validate_clauses(reactivity["reset"], "reset")
    rules = {_RESPONSE_RULES[step] for step in response["steps"] if step in _RESPONSE_RULES}
    if len(rules) != 1:
        raise ViewContractError("SF response must identify exactly one supported rule")
    rule_id = rules.pop()
    if "execution" in reactivity:
        _validate_execution(rule_id, reactivity["execution"])
    branches = response["branches"]
    if not isinstance(branches, (list, tuple)):
        raise ViewContractError("SF response branches must be a list")
    for branch in branches:
        _check_keys(branch, {"when", "action"}, "response branch")
        if set(branch) != {"when", "action"}:
            raise ViewContractError("SF response branch contains unsupported fields")
        _validate_clause(branch["when"])
        _validate_clause(branch["action"])
    for section in (reactivity, trigger, response):
        for key in ("parameters", "static_parameters"):
            if key in section:
                value = section[key]
                if not isinstance(value, dict):
                    raise ViewContractError(
                        "SF static parameters must be an explicitly named object"
                    )
                json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    parameters = reactivity.get("execution", {}).get("static_parameters", {})
    immediate_key = {"R04": "jump_gap", "R07": "anti_air_height"}.get(rule_id)
    if immediate_key is not None and parameters.get(immediate_key) == 0:
        return IMMEDIATE_PROMPTS[rule_id]
    return PROMPTS[rule_id]
