"""2-ply expectimax search on top of the TD value network.

For each of our candidate moves we look at all 21 distinct opponent rolls
(weights 1/36 for doubles, 2/36 otherwise). For each roll the opponent is
assumed to play the reply that is best for them according to the network
(min node), and the candidate is valued by the expectation over rolls
(chance node):

    value(a) = sum_r P(r) * min_{b in replies(a, r)} V_me(b)

To keep it fast only the ``top_k`` candidates by the 1-ply evaluation are
searched; all opponent replies are evaluated with the network in a single
batch.
"""

from __future__ import annotations

import numpy as np

from backgammon.engine.board import ALL_ROLLS, generate_moves

from .network import TDNetwork
from .td_agent import TDAgent


class ExpectimaxAgent(TDAgent):
    name = "expectimax"

    def __init__(self, net: TDNetwork, device: str = "cpu", top_k: int = 6):
        super().__init__(net, device)
        self.top_k = top_k

    def search(self, state, moves) -> list[tuple[int, float]]:
        me = state.turn
        afters = [a for _, a in moves]
        one_ply = self.state_scores(afters, me)
        order = list(np.argsort(-one_ply, kind="stable"))
        cands = order[: self.top_k]
        rest = order[self.top_k:]

        replies = []       # all opponent afterstates (flattened)
        segments = []      # (candidate index, roll weight, start, end)
        exact = {}
        for ci in cands:
            a = afters[ci]
            if a.is_over():
                exact[ci] = 1.0 if a.winner() == me else 0.0
                continue
            for dice, weight in ALL_ROLLS:
                res = generate_moves(a, dice)
                start = len(replies)
                replies.extend(b for _, b in res)
                segments.append((ci, weight, start, len(replies)))

        values = {ci: 0.0 for ci in cands if ci not in exact}
        if replies:
            scores = self.state_scores(replies, me)
            for ci, weight, s, e in segments:
                values[ci] += weight * float(scores[s:e].min()) / 36.0
        values.update(exact)
        ranked = sorted(((int(ci), values[ci]) for ci in cands), key=lambda t: -t[1])
        ranked += [(int(i), float(one_ply[i])) for i in rest]
        return ranked

    def rank_moves(self, state, dice, moves):
        return self.search(state, moves)

    def choose(self, state, dice, moves):
        if len(moves) == 1:
            return 0
        return self.search(state, moves)[0][0]
