"""Handle-renaming permutations that preserve physical conditioning."""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Sequence, TypeVar

T = TypeVar("T")


class PermutationError(ValueError):
    pass


def validate_permutation(old_slot_for_new_slot: Sequence[int], cardinality: int) -> tuple[int, ...]:
    value = tuple(int(item) for item in old_slot_for_new_slot)
    if len(value) != cardinality or set(value) != set(range(cardinality)):
        raise PermutationError(f"not a permutation of range({cardinality}): {value}")
    return value


def deterministic_train_permutation(
    cardinality: int, *, master_seed: int, epoch: int, sample_index: int, access: int = 0
) -> tuple[int, ...]:
    # Concatenate four integer fields; the training seed remains reproducible.
    fields = (master_seed, epoch, sample_index, access)
    if any(not 0 <= int(value) < 2**64 for value in fields):
        raise ValueError("permutation seed fields must be unsigned 64-bit integers")
    seed = 0
    for field in fields:
        seed = (seed << 64) | int(field)
    value = list(range(cardinality))
    random.Random(seed).shuffle(value)
    return tuple(value)


def validation_permutations(cardinality: int) -> tuple[tuple[int, ...], ...]:
    if cardinality not in range(2, 7):
        raise PermutationError("validation cardinality must be 2..6")
    if cardinality <= 3:
        return tuple(itertools.permutations(range(cardinality)))
    cyclic = [
        tuple((new + shift) % cardinality for new in range(cardinality))
        for shift in range(cardinality)
    ]
    reversed_cyclic = [
        tuple((shift - new) % cardinality for new in range(cardinality))
        for shift in range(cardinality)
    ]
    result = tuple(cyclic + reversed_cyclic)
    if len(set(result)) != 2 * cardinality:
        raise AssertionError("cyclic validation mapping construction produced duplicates")
    return result


def reorder_slots(values: Sequence[T], old_slot_for_new_slot: Sequence[int]) -> tuple[T, ...]:
    permutation = validate_permutation(old_slot_for_new_slot, len(values))
    return tuple(values[old] for old in permutation)


def reorder_action_steps(
    actions: Sequence[Sequence[T]], old_slot_for_new_slot: Sequence[int]
) -> tuple[tuple[T, ...], ...]:
    return tuple(reorder_slots(row, old_slot_for_new_slot) for row in actions)


@dataclass(frozen=True)
class PermutedCondition:
    subjects: tuple[str, ...]
    masks: tuple[T, ...]
    actions: tuple[tuple[str | None, ...], ...]
    old_slot_for_new_slot: tuple[int, ...]
    control_kind: tuple[int, ...] = ()
    action_ids: tuple[tuple[int, ...], ...] = ()
    npc_prompts: tuple[str | None, ...] = ()


def permute_condition(
    subjects: Sequence[str],
    masks: Sequence[T],
    actions: Sequence[Sequence[str | None]],
    old_slot_for_new_slot: Sequence[int],
    *,
    control_kind: Sequence[int] = (),
    action_ids: Sequence[Sequence[int]] = (),
    npc_prompts: Sequence[str | None] = (),
) -> PermutedCondition:
    cardinality = len(subjects)
    if len(masks) != cardinality or any(len(row) != cardinality for row in actions):
        raise PermutationError("condition fields have inconsistent cardinality")
    permutation = validate_permutation(old_slot_for_new_slot, cardinality)
    kinds = tuple(control_kind) or (1,) * cardinality
    prompts = tuple(npc_prompts) or (None,) * cardinality
    if (
        len(kinds) != cardinality
        or len(prompts) != cardinality
        or any(kind not in (1, 2) for kind in kinds)
        or (action_ids and any(len(row) != cardinality for row in action_ids))
    ):
        raise PermutationError("mixed-role condition fields have inconsistent cardinality or kinds")
    return PermutedCondition(
        subjects=tuple(f"SUBJECT{i+1}" for i in range(cardinality)),
        masks=reorder_slots(masks, permutation),
        actions=reorder_action_steps(actions, permutation),
        old_slot_for_new_slot=permutation,
        control_kind=reorder_slots(kinds, permutation),
        action_ids=reorder_action_steps(action_ids, permutation) if action_ids else (),
        npc_prompts=reorder_slots(prompts, permutation),
    )
