"""TD(lambda) self-play learner with hand-written eligibility traces.

For every game the agent plays against itself (both sides use the same
network and pick the greedy move). With the afterstate sequence
s_0, s_1, ..., s_T and V = P(White wins):

    e_0     = grad V(s_0)
    delta_t = V(s_{t+1}) - V(s_t)          (V(s_T) := 1 if White won else 0)
    theta  += alpha * delta_t * e_t
    e_{t+1} = lambda * e_t + grad V(s_{t+1})

No discounting (gamma = 1): the only reward is the final result. The
gradients come from :meth:`TDNetwork.value_and_grad` (manual backprop) and
the traces are plain tensors updated in place, so no autograd is involved.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from backgammon.agents.encoding import encode_positions, encode_states
from backgammon.agents.network import TDNetwork
from backgammon.engine.board import NUM_CHECKERS, OFF, WHITE, State, generate_moves
from backgammon.engine.game import opening_roll, roll_dice


class TDLambdaLearner:
    def __init__(self, net: TDNetwork, lam: float = 0.7, device: str = "cpu"):
        self.net = net.to(device)
        self.lam = lam
        self.device = device
        self.params = [net.fc1.weight, net.fc1.bias, net.fc2.weight, net.fc2.bias]
        self.traces = [torch.zeros_like(p) for p in self.params]

    def _reset_traces(self) -> None:
        for e in self.traces:
            e.zero_()

    def _accumulate(self, grads) -> None:
        lam = self.lam
        for e, g in zip(self.traces, grads):
            e.mul_(lam).add_(g)

    def _update(self, scale: torch.Tensor) -> None:
        """theta += scale * e   (scale = alpha * delta, a 0-dim tensor)."""
        for p, e in zip(self.params, self.traces):
            p.add_(e * scale)

    @torch.no_grad()
    def play_game(self, rng: random.Random, lr: float) -> tuple[int, int, int]:
        """Play one self-play game while learning. Returns (winner, points, turns)."""
        net = self.net
        dev = self.device
        first, dice = opening_roll(rng)
        state = State.initial(first)

        x = torch.from_numpy(encode_states([state])[0]).to(dev)
        v, grads = net.value_and_grad(x)
        self._reset_traces()
        self._accumulate(grads)
        lr_t = torch.tensor(lr, device=dev)

        turns = 0
        while True:
            moves = generate_moves(state, dice)
            mover = state.turn
            afters = [a for _, a in moves]
            turns += 1

            # A winning move is always taken (its exact value is 1).
            win_idx = -1
            for i, a in enumerate(afters):
                if a.pos[mover][OFF] == NUM_CHECKERS:
                    win_idx = i
                    break
            if win_idx >= 0:
                final = afters[win_idx]
                target = 1.0 if mover == WHITE else 0.0
                self._update(lr_t * (target - v))
                return mover, final.result_points(), turns

            pos = np.array([a.pos for a in afters], dtype=np.int64)
            turns_arr = np.full(len(afters), 1 - mover, dtype=np.int64)
            xb = torch.from_numpy(encode_positions(pos, turns_arr)).to(dev)
            vb = net(xb)
            idx = int(torch.argmax(vb if mover == WHITE else -vb))
            v_next = vb[idx]

            self._update(lr_t * (v_next - v))
            # value and gradient of the new state with the updated weights
            v, grads = net.value_and_grad(xb[idx])
            self._accumulate(grads)

            state = afters[idx]
            dice = roll_dice(rng)
