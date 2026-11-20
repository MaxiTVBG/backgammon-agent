"""Matches between agents, optionally in parallel worker processes."""

from __future__ import annotations

import multiprocessing as mp
import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

from backgammon.agents.factory import make_agent
from backgammon.engine.board import WHITE
from backgammon.engine.game import play_game


@dataclass
class MatchResult:
    games: int = 0
    wins_a: int = 0
    points_a: int = 0       # points won by A
    points_b: int = 0       # points won by B
    gammons_a: int = 0      # gammons + backgammons won by A
    backgammons_a: int = 0
    gammons_b: int = 0
    backgammons_b: int = 0
    turns: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        return self.wins_a / self.games if self.games else 0.0

    @property
    def ppg(self) -> float:
        """Average net points per game for A (points won minus points lost)."""
        return (self.points_a - self.points_b) / self.games if self.games else 0.0

    def merge(self, other: "MatchResult") -> None:
        for f in ("games", "wins_a", "points_a", "points_b", "gammons_a",
                  "backgammons_a", "gammons_b", "backgammons_b", "turns"):
            setattr(self, f, getattr(self, f) + getattr(other, f))

    def summary(self, name_a: str = "A", name_b: str = "B") -> str:
        return (f"{name_a} vs {name_b}: {self.games} games, "
                f"{name_a} wins {self.wins_a} ({100 * self.win_rate:.1f}%), "
                f"avg points/game {self.ppg:+.3f}, "
                f"gammons {self.gammons_a}:{self.gammons_b}, "
                f"backgammons {self.backgammons_a}:{self.backgammons_b}")


def _init_worker() -> None:
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:  # pragma: no cover
        pass


def _play_chunk(args) -> MatchResult:
    spec_a, spec_b, start, count, seed = args
    rng = random.Random(seed)
    a = make_agent(dict(spec_a, seed=seed))
    b = make_agent(dict(spec_b, seed=seed + 1))
    res = MatchResult()
    for g in range(start, start + count):
        a_white = g % 2 == 0
        white, black = (a, b) if a_white else (b, a)
        r = play_game(white, black, rng)
        a_won = (r.winner == WHITE) == a_white
        res.games += 1
        res.turns += r.turns
        if a_won:
            res.wins_a += 1
            res.points_a += r.points
            res.gammons_a += r.points >= 2
            res.backgammons_a += r.points == 3
        else:
            res.points_b += r.points
            res.gammons_b += r.points >= 2
            res.backgammons_b += r.points == 3
    return res


def make_pool(workers: int) -> ProcessPoolExecutor | None:
    """Create a process pool (spawn context, works on Windows) or None for 1 worker."""
    if workers <= 1:
        return None
    return ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                               initializer=_init_worker)


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def play_match(spec_a: dict, spec_b: dict, n_games: int, seed: int = 0,
               workers: int = 1, pool: ProcessPoolExecutor | None = None,
               chunk: int = 25) -> MatchResult:
    """Play ``n_games`` between A and B, alternating colours every game."""
    tasks = []
    start = 0
    while start < n_games:
        cnt = min(chunk, n_games - start)
        tasks.append((spec_a, spec_b, start, cnt, seed * 1_000_003 + start))
        start += cnt
    total = MatchResult()
    own_pool = None
    if pool is None and workers > 1:
        own_pool = pool = make_pool(workers)
    try:
        if pool is None:
            for t in tasks:
                total.merge(_play_chunk(t))
        else:
            for r in pool.map(_play_chunk, tasks):
                total.merge(r)
    finally:
        if own_pool is not None:
            own_pool.shutdown()
    return total
