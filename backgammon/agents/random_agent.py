"""Agent that plays a uniformly random legal move."""

from __future__ import annotations

import random

from .base import Agent


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def choose(self, state, dice, moves):
        return self.rng.randrange(len(moves))

    def rank_moves(self, state, dice, moves):
        idx = list(range(len(moves)))
        self.rng.shuffle(idx)
        return [(i, 0.0) for i in idx]
