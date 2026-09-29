"""Read-only audit of SF's recorded CPS3 input consumption.

The mapping and ClearOpposite default-mode algorithm follow the published
sf_v2/engine_observer.py audit. Nothing here loads a game/core or issues input.
"""

from __future__ import annotations

import numpy as np

from .records import ViewContractError


def words_from_ports(ports: np.ndarray) -> np.ndarray:
    # Port: B,A,MODE,START,UP,DOWN,LEFT,RIGHT,C,Y,X,Z.
    native = (
        np.asarray(ports)
        .reshape(-1, 2, 12)[:, :, [4, 5, 6, 7, 1, 9, 10, 0, 8, 11]]
        .astype(np.uint16)
    )
    words = np.zeros((len(native), 4), dtype=np.uint16)
    for slot in range(2):
        for button in range(7):
            words[:, 0] |= native[:, slot, button] << (button + slot * 8)
    for button, bit in ((7, 3), (8, 2), (9, 1)):
        words[:, 3] |= native[:, 0, button] << bit
    words[:, 3] |= (native[:, 1, 7] << 4) | (native[:, 1, 8] << 5)
    words[:, 1] |= native[:, 1, 9] << 10
    return words


def filter_socd(word: int, state: np.ndarray) -> tuple[int, np.ndarray]:
    """Apply the qualified default SOCD mode 3 to the prior recorded state."""
    state = np.asarray(state, dtype=np.uint16).copy()
    for slot in range(2):
        up, down, left, right = (1 << (bit + slot * 8) for bit in range(4))
        ud, lr = up | down, left | right
        prev = int(state[slot])
        inp_lr, inp_ud = word & lr, word & ud
        if inp_lr == lr:
            word &= ~int(state[4 + slot])
        elif inp_lr:
            state[4 + slot] = inp_lr
        if inp_ud == ud:
            word &= ~int(state[2 + slot])
        elif inp_ud:
            state[2 + slot] = inp_ud
        if (word & ud) == ud:
            word &= ~ud
        if (word & lr) == lr:
            word &= ~lr
        if ((word | prev) & ud) == ud:
            word &= ~ud
        if ((word | prev) & lr) == lr:
            word &= ~lr
        inp_e, prev_e = word & (ud | lr), prev & (ud | lr)
        if (inp_e, prev_e) in (
            (down | left, down | right),
            (down | right, down | left),
            (up | left, up | right),
            (up | right, up | left),
        ):
            word &= ~lr
        if (inp_e, prev_e) in (
            (down | left, up | left),
            (up | left, down | left),
            (down | right, up | right),
            (up | right, down | right),
        ):
            word &= ~ud
        state[slot] = word
    return word, state


def audit_consumption(ports: np.ndarray, arrays: dict) -> None:
    count = len(ports)
    for name, shape in (
        ("game_input_words", (count + 1, 4)),
        ("input_words_before_socd", (count + 1, 4)),
        ("socd_state", (count + 1, 6)),
        ("socd_config", (count + 1, 2)),
    ):
        if name not in arrays or arrays[name].shape != shape:
            raise ViewContractError(f"SF native consumption evidence has wrong boundaries: {name}")
    if not np.all(arrays["socd_config"] == 3):
        raise ViewContractError("SF actual SOCD mode differs from the qualified default 3")
    expected = words_from_ports(ports)
    before = arrays["input_words_before_socd"][1:]
    consumed = arrays["game_input_words"][1:]
    columns = [0, 1, 3]
    if not np.array_equal(expected[:, columns], before[:, columns]):
        raise ViewContractError("SF submitted ports differ from observed pre-SOCD input words")
    predicted = before.copy()
    states = arrays["socd_state"]
    state = states[0].copy()
    for tick in range(count):
        predicted[tick, 0], state = filter_socd(int(before[tick, 0]), state)
        if not np.array_equal(state, states[tick + 1]):
            raise ViewContractError(
                "SF recorded SOCD state differs from actual submitted input filtering"
            )
    if not np.array_equal(predicted[:, columns], consumed[:, columns]):
        raise ViewContractError("SF consumed input words differ from submitted ports after SOCD")
