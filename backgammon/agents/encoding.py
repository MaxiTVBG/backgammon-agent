"""Tesauro (TD-Gammon) input encoding: 198 inputs.

For each player (white first, then black):
  * 24 points x 4 units: n>=1, n>=2, n>=3, (n-3)/2 if n>3   -> 96 units
  * checkers on the bar / 2                                 -> 1 unit
  * checkers borne off / 15                                 -> 1 unit
Finally 2 units for the side to move: [white to move, black to move].

Each player's points are given in that player's own numbering
(see :mod:`backgammon.engine.board`).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from backgammon.engine.board import State

NUM_INPUTS = 198

# Lookup table: number of checkers (0..15) -> the 4 point units.
_POINT_TABLE = np.zeros((16, 4), dtype=np.float32)
for _n in range(16):
    _POINT_TABLE[_n] = [_n >= 1, _n >= 2, _n >= 3, max(_n - 3, 0) / 2.0]


def encode_positions(pos: np.ndarray, turns: np.ndarray) -> np.ndarray:
    """Encode an integer array of shape (N, 2, 26) and turns (N,) -> (N, 198)."""
    n = pos.shape[0]
    out = np.empty((n, NUM_INPUTS), dtype=np.float32)
    pts = _POINT_TABLE[pos[:, :, 1:25]]          # (N, 2, 24, 4)
    out[:, 0:96] = pts[:, 0].reshape(n, 96)
    out[:, 96] = pos[:, 0, 25] * 0.5
    out[:, 97] = pos[:, 0, 0] * (1.0 / 15.0)
    out[:, 98:194] = pts[:, 1].reshape(n, 96)
    out[:, 194] = pos[:, 1, 25] * 0.5
    out[:, 195] = pos[:, 1, 0] * (1.0 / 15.0)
    out[:, 196] = turns == 0
    out[:, 197] = turns == 1
    return out


def encode_states(states: Sequence[State]) -> np.ndarray:
    """Encode a list of states into a float32 array of shape (N, 198)."""
    pos = np.array([s.pos for s in states], dtype=np.int64)
    turns = np.fromiter((s.turn for s in states), dtype=np.int64, count=len(states))
    return encode_positions(pos, turns)


def encode_state(state: State) -> np.ndarray:
    return encode_states([state])[0]
