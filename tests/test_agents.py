"""Tests for the baseline agents and the match runner."""

from __future__ import annotations

import random

from backgammon.agents import HeuristicAgent, RandomAgent, make_agent
from backgammon.engine import State, generate_moves
from backgammon.training.evaluate import play_match


def test_random_agent_returns_valid_index():
    agent = RandomAgent(seed=0)
    s = State.initial()
    for dice in [(3, 1), (6, 6), (5, 2)]:
        moves = generate_moves(s, dice)
        for _ in range(20):
            assert 0 <= agent.choose(s, dice, moves) < len(moves)


def test_heuristic_prefers_making_the_five_point_with_31():
    s = State.initial()
    moves = generate_moves(s, (3, 1))
    idx = HeuristicAgent().choose(s, (3, 1), moves)
    after = moves[idx][1]
    assert after.pos[0][5] == 2  # 8/5 6/5


def test_heuristic_hits_when_safe():
    # Black blot on white's 10 point, white can hit from 13 with a 3.
    s = State.from_points({13: 5, 8: 3, 6: 7}, {15: 1, 1: 14}, 0)
    moves = generate_moves(s, (3, 3))
    idx = HeuristicAgent().choose(s, (3, 3), moves)
    assert moves[idx][1].pos[1][25] == 1


def test_rank_moves_sorted():
    s = State.initial()
    moves = generate_moves(s, (6, 5))
    ranked = HeuristicAgent().rank_moves(s, (6, 5), moves)
    scores = [sc for _, sc in ranked]
    assert scores == sorted(scores, reverse=True)
    assert sorted(i for i, _ in ranked) == list(range(len(moves)))


def test_factory_kinds():
    assert isinstance(make_agent("random"), RandomAgent)
    assert isinstance(make_agent({"kind": "heuristic"}), HeuristicAgent)


def test_heuristic_beats_random():
    res = play_match({"kind": "heuristic"}, {"kind": "random"}, 200, seed=1)
    assert res.games == 200
    assert res.win_rate > 0.9
    assert res.ppg > 1.0


def test_match_result_consistency():
    res = play_match({"kind": "random"}, {"kind": "random"}, 100, seed=2)
    assert res.games == 100
    assert res.points_a + res.points_b >= 100
    assert 0.3 < res.win_rate < 0.7


def test_parallel_match_equals_serial():
    a = play_match({"kind": "heuristic"}, {"kind": "random"}, 60, seed=5, workers=1)
    b = play_match({"kind": "heuristic"}, {"kind": "random"}, 60, seed=5, workers=2)
    assert (a.wins_a, a.points_a, a.points_b) == (b.wins_a, b.points_a, b.points_b)
