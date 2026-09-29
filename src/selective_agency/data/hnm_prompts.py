"""HNM response policies, rule labels, and model text."""

from functools import lru_cache
import json
from pathlib import Path

PROMPT_PROTOCOL = "hnm-if-then"
MAPPING_PATH = Path(__file__).with_name("prompt_assets") / "hnm_prompts.json"


@lru_cache(maxsize=1)
def prompt_mapping():
    mapping = json.loads(MAPPING_PATH.read_text())
    if mapping.get("protocol") != PROMPT_PROTOCOL:
        raise ValueError("HNM prompt mapping protocol differs")
    entries = mapping["entries"]
    if (
        len(entries) != 12
        or len({row["policy_id"] for row in entries}) != 12
        or len({row["text"] for row in entries}) != 12
    ):
        raise ValueError("HNM text table requires eleven active policies and no-op")
    return {row["policy_id"]: row for row in entries}


def compile_hnm_prompt(subject: dict, *, prompt_protocol: str = PROMPT_PROTOCOL) -> str:
    from .records import ViewContractError

    if prompt_protocol != PROMPT_PROTOCOL:
        raise ViewContractError(f"unsupported HNM prompt protocol: {prompt_protocol!r}")
    trigger, response = subject.get("trigger"), subject.get("reactivity")
    if subject.get("npc_kind") == "passive":
        if trigger is not None or response is not None:
            raise ViewContractError("passive HNM NPC must not carry an active rule")
        return prompt_mapping()["no_op"]["text"]
    if not isinstance(trigger, dict) or not isinstance(response, dict):
        raise ViewContractError("active HNM NPC requires Trigger and Reactivity")

    policy_id = response.get("policy_id")
    entry = prompt_mapping().get(policy_id)
    if entry is None or entry["trigger"] is None:
        raise ViewContractError(f"unsupported HNM response policy: {policy_id!r}")

    # Numeric rule labels may differ in source records; the policy and complete
    # trigger/response semantics determine the model condition.
    source_rule = trigger.get("rule_id")
    expected_trigger = {
        "clause_id": f"{source_rule}.{entry['trigger']['clause']}",
        "rule_id": source_rule,
        "text": entry["trigger"]["text"],
    }
    if (
        not isinstance(source_rule, str)
        or not source_rule
        or trigger != expected_trigger
        or response != {"policy_id": policy_id, "text": entry["response"]}
    ):
        raise ViewContractError(f"HNM source semantics differ for response policy: {policy_id}")
    return entry["text"]
