"""Tests for the step-by-step human move input used by the GUI."""

from __future__ import annotations

import random

from backgammon.engine import BAR, WHITE, State, generate_moves, single_steps
from backgammon.gui.move_input import TurnInput
from tests.test_engine import random_position


def test_single_steps_bar_first():
    s = State.from_points({13: 5, 8: 3, 6: 6}, {13: 15}, WHITE, white_bar=1)
    assert single_steps(s, 3) == [(BAR, 22)]


def test_every_legal_move_can_be_entered_step_by_step():
    rng = random.Random(4)
    for _ in range(60):
        s = random_position(rng)
        dice = (rng.randint(1, 6), rng.randint(1, 6))
        legal = {a.pos for _, a in generate_moves(s, dice)}
        reached = set()

        def explore(ti: TurnInput):
            if ti.is_complete():
                reached.add(ti.current.pos)
                return
            opts = ti.options()
            assert opts, "dead end: input offered a step that cannot be completed"
            for step, _ in opts:
                ti.play(step)
                explore(ti)
                ti.undo()

        explore(TurnInput(s, dice))
        assert reached == legal


def test_forced_larger_die_is_the_only_option():
    s = State.from_points({3: 14, 20: 1}, {24: 2, 23: 2, 12: 2, 1: 9}, WHITE)
    ti = TurnInput(s, (6, 1))
    assert ti.options() == [((20, 14), 6)]
    ti.play((20, 14))
    assert ti.is_complete()
    idx, move = ti.result()
    assert ti.legal[idx][0] == ((20, 14),)


def test_undo_restores_dice():
    ti = TurnInput(State.initial(), (6, 5))
    ti.play((24, 18))
    assert ti.remaining == (5,)
    ti.undo()
    assert ti.remaining == (5, 6) and ti.current == State.initial()


def test_no_move_is_complete_immediately():
    black = {1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 13: 3}
    s = State.from_points({13: 5, 8: 3, 6: 6}, black, WHITE, white_bar=1)
    ti = TurnInput(s, (3, 4))
    assert ti.no_moves() and ti.is_complete()
