"""Greedy 1-ply agent driven by the TD value network."""

from __future__ import annotations

import numpy as np
import torch

from backgammon.engine.board import WHITE, State

from .base import Agent
from .encoding import encode_states
from .network import TDNetwork


class TDAgent(Agent):
    """Evaluates all afterstates in one batch and plays the best one."""

    name = "td"

    def __init__(self, net: TDNetwork, device: str = "cpu"):
        self.net = net.to(device)
        self.net.eval()
        self.device = device

    @torch.no_grad()
    def values(self, states: list[State]) -> np.ndarray:
        """P(White wins) for each state (batched)."""
        x = torch.from_numpy(encode_states(states)).to(self.device)
        return self.net(x).cpu().numpy()

    def state_scores(self, states: list[State], player: int) -> np.ndarray:
        """P(player wins) for each state, using exact values for finished games."""
        v = self.values(states)
        scores = v if player == WHITE else 1.0 - v
        for i, s in enumerate(states):
            w = s.winner()
            if w is not None:
                scores[i] = 1.0 if w == player else 0.0
        return scores

    def rank_moves(self, state, dice, moves):
        scores = self.state_scores([a for _, a in moves], state.turn)
        order = np.argsort(-scores, kind="stable")
        return [(int(i), float(scores[i])) for i in order]

    def choose(self, state, dice, moves):
        if len(moves) == 1:
            return 0
        scores = self.state_scores([a for _, a in moves], state.turn)
        return int(np.argmax(scores))

    def win_probability(self, state: State) -> float:
        w = state.winner()
        if w is not None:
            return 1.0 if w == state.turn else 0.0
        v = float(self.values([state])[0])
        return v if state.turn == WHITE else 1.0 - v
