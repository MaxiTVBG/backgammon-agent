"""Exhaustive tests for the backgammon rules engine."""

from __future__ import annotations

import random

import pytest

from backgammon.engine import (
    BAR,
    BLACK,
    OFF,
    WHITE,
    State,
    apply_move,
    format_move,
    generate_moves,
    opening_roll,
    play_game,
    render_board,
)
from tests.reference_movegen import reference_afterstates

ALL_DICE = [(a, b) for a in range(1, 7) for b in range(1, 7)]


def moves_of(state, dice):
    return generate_moves(state, dice)


def after_white(state, dice):
    """Return list of white's own 26-tuples of all afterstates."""
    return [a.pos[WHITE] for _, a in moves_of(state, dice)]


# Frozen white stack: 14 checkers on white's point 3 that can never move
# (point 2 blocked, bearing off impossible while a checker is outside home).
def frozen_position(extra_white: dict, black: dict) -> State:
    white = {3: 14}
    white.update(extra_white)
    return State.from_points(white, black, WHITE)


# ---------------------------------------------------------------- basics

def test_initial_position():
    s = State.initial()
    s.validate()
    assert s.pip_count(WHITE) == 167
    assert s.pip_count(BLACK) == 167
    assert not s.is_over()
    assert s.winner() is None


def test_absolute_board_mirrors_black():
    s = State.initial()
    b = s.absolute_board()
    assert b[24] == 2 and b[13] == 5 and b[8] == 3 and b[6] == 5
    assert b[1] == -2 and b[12] == -5 and b[17] == -3 and b[19] == -5


def test_opening_31_has_16_moves_including_point_making():
    s = State.initial()
    res = moves_of(s, (3, 1))
    assert len(res) == 16
    expected = State.from_points({24: 2, 13: 5, 8: 2, 6: 4, 5: 2},
                                 {24: 2, 13: 5, 8: 3, 6: 5}, BLACK)
    assert expected in [a for _, a in res]


def test_afterstate_passes_turn_and_conserves_checkers():
    s = State.initial()
    for dice in ALL_DICE:
        for _, a in moves_of(s, dice):
            assert a.turn == BLACK
            a.validate()


def test_dice_order_does_not_matter():
    rng = random.Random(1)
    for _ in range(30):
        s = random_position(rng)
        for a in range(1, 7):
            for b in range(1, 7):
                x = {st for _, st in moves_of(s, (a, b))}
                y = {st for _, st in moves_of(s, (b, a))}
                assert x == y


def test_no_duplicate_afterstates():
    rng = random.Random(2)
    for _ in range(200):
        s = random_position(rng)
        for dice in ALL_DICE:
            res = [st for _, st in moves_of(s, dice)]
            assert len(res) == len(set(res))


# ---------------------------------------------------------------- doubles

def test_doubles_play_four_times():
    s = State.initial()
    for d in range(1, 7):
        for move, _ in moves_of(s, (d, d)):
            assert len(move) == 4


def test_doubles_partial_when_blocked():
    # White single checker on 20 can move 20->16->12 with 4s, 8 is blocked.
    s = frozen_position({20: 1}, {23: 2, 24: 2, 17: 2})  # black own 17 = white 8
    res = moves_of(s, (4, 4))
    assert len(res) == 1
    move, after = res[0]
    assert move == ((20, 16), (16, 12))
    assert after.pos[WHITE][12] == 1


def test_doubles_opening_66():
    s = State.initial()
    res = moves_of(s, (6, 6))
    for move, a in res:
        assert len(move) == 4
    # The classic 24/18(2) 13/7(2) must be there.
    target = State.from_points({18: 2, 13: 3, 7: 2, 8: 3, 6: 5},
                               {24: 2, 13: 5, 8: 3, 6: 5}, BLACK)
    assert target in [a for _, a in res]


# ---------------------------------------------------------------- bar

def test_bar_checker_must_enter_first():
    s = State.from_points({13: 5, 8: 3, 6: 5, 24: 1}, {24: 2, 13: 5, 8: 3, 6: 5},
                          WHITE, white_bar=1)
    for move, _ in moves_of(s, (6, 3)):
        assert move[0][0] == BAR


def test_bar_all_entry_points_blocked_means_no_move():
    # Black holds its whole home board = white's entry points 19..24.
    black = {1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 13: 3}
    s = State.from_points({13: 5, 8: 3, 6: 6}, black, WHITE, white_bar=1)
    for dice in ALL_DICE:
        res = moves_of(s, dice)
        assert len(res) == 1
        move, after = res[0]
        assert move == ()
        assert after.pos == s.pos and after.turn == BLACK


def test_bar_enter_with_open_die_then_play_other():
    # Entry with 6 (white point 19 = black own 6) blocked, entry with 2 open.
    black = {6: 2, 13: 13}
    s = State.from_points({13: 5, 8: 3, 6: 6}, black, WHITE, white_bar=1)
    res = moves_of(s, (6, 2))
    assert res
    for move, _ in res:
        assert move[0] == (BAR, 23)
        assert len(move) == 2


def test_two_on_bar_only_one_can_enter():
    # Entry with 5 (white 20 = black 5) blocked, with 6 open: enter one with
    # the 6, the 5 cannot be played because a checker is still on the bar.
    black = {5: 2, 13: 13}
    s = State.from_points({13: 5, 8: 3, 6: 5}, black, WHITE, white_bar=2)
    res = moves_of(s, (6, 5))
    assert len(res) == 1
    move, after = res[0]
    assert move == ((BAR, 19),)
    assert after.pos[WHITE][BAR] == 1


def test_two_on_bar_doubles_enter_both_then_continue():
    s = State.from_points({13: 5, 8: 3, 6: 5}, {13: 15}, WHITE, white_bar=2)
    res = moves_of(s, (4, 4))
    for move, after in res:
        assert move[0] == (BAR, 21) and move[1] == (BAR, 21)
        assert len(move) == 4
        assert after.pos[WHITE][BAR] == 0


def test_entering_can_hit():
    s = State.from_points({13: 5, 8: 3, 6: 6}, {6: 1, 13: 14}, WHITE, white_bar=1)
    res = moves_of(s, (6, 1))
    hits = [a for m, a in res if m[0] == (BAR, 19)]
    assert hits and all(a.pos[BLACK][BAR] == 1 for a in hits)


# ---------------------------------------------------------------- blocking / hitting

def test_cannot_land_on_point_with_two_opponents():
    # Black owns white's point 5 (= black own point 20): 8/5 is impossible.
    s = State.from_points({8: 1, 6: 14}, {20: 2, 13: 13}, WHITE)
    res = moves_of(s, (3, 1))
    assert res
    for move, a in res:
        assert (8, 5) not in move and (6, 5) not in move
        assert a.pos[WHITE][5] == 0


def test_hit_blot_sends_it_to_bar():
    # Black blot on white's point 10 (= black own 15).
    s = State.from_points({13: 15}, {15: 1, 1: 14}, WHITE)
    res = moves_of(s, (3, 1))
    hit_states = [a for m, a in res if (13, 10) in m]
    assert hit_states
    for a in hit_states:
        assert a.pos[BLACK][BAR] == 1
        assert a.pos[BLACK][15] == 0


def test_hit_and_continue_through():
    # 13/10* then 10/9 - checker hits and continues.
    s = State.from_points({13: 15}, {15: 1, 1: 14}, WHITE)
    res = dict((a, m) for m, a in moves_of(s, (3, 1)))
    target = [a for a in res if a.pos[WHITE][9] == 1 and a.pos[BLACK][BAR] == 1]
    assert len(target) == 1


def test_black_moves_in_its_own_direction():
    s = State.initial(BLACK)
    res = moves_of(s, (6, 5))
    # 24/13 for black: black own point 13 gains a checker
    found = [a for m, a in res if m == ((24, 18), (18, 13)) or m == ((24, 19), (19, 13))]
    assert found
    a = found[0]
    assert a.pos[BLACK][13] == 6 and a.pos[BLACK][24] == 1
    assert a.absolute_board()[12] == -6


# ---------------------------------------------------------------- forced dice rules

def test_must_use_both_dice_when_possible():
    # Only the order 1-then-6 lets both dice play: 20/19/13.
    # Black blocks white 14 (black 11) and white 1,2 (black 24,23).
    s = frozen_position({20: 1}, {24: 2, 23: 2, 11: 2, 1: 9})
    res = moves_of(s, (6, 1))
    assert len(res) == 1
    move, after = res[0]
    assert move == ((20, 19), (19, 13))


def test_must_play_larger_die_when_only_one_can_be_played():
    # 20/14 or 20/19 can be played alone, but neither allows the other die
    # (white 13 blocked by black 12). The larger die (6) is forced.
    s = frozen_position({20: 1}, {24: 2, 23: 2, 12: 2, 1: 9})
    res = moves_of(s, (6, 1))
    assert len(res) == 1
    move, _ = res[0]
    assert move == ((20, 14),)


def test_play_smaller_die_if_larger_impossible():
    # 6 is blocked completely (white 14 = black 11), 1 can be played once.
    s = frozen_position({20: 1}, {24: 2, 23: 2, 12: 2, 11: 2, 1: 7})
    res = moves_of(s, (6, 1))
    assert len(res) == 1
    move, _ = res[0]
    assert move == ((20, 19),)


def test_no_legal_move_at_all():
    s = frozen_position({20: 1}, {24: 2, 23: 2, 11: 2, 6: 2, 1: 7})
    # 20-6 = 14 blocked (black 11), 20-1 = 19 blocked (black 6)
    res = moves_of(s, (6, 1))
    assert res == [((), s.swap_turn())]


# ---------------------------------------------------------------- bearing off

def test_no_bear_off_with_checker_outside_home():
    s = State.from_points({7: 1, 6: 4, 2: 10}, {12: 15}, WHITE)
    for dice in ALL_DICE:
        for move, a in moves_of(s, dice):
            # The checker on 7 must have come home before any bear-off step.
            seen_home = False
            for src, dst in move:
                if src == 7:
                    seen_home = True
                if dst == OFF:
                    assert seen_home


def test_enter_home_then_bear_off_same_turn():
    s = State.from_points({7: 1, 6: 4, 2: 10}, {12: 15}, WHITE)
    res = moves_of(s, (1, 6))
    offs = [a for m, a in res if a.pos[WHITE][OFF] > 0]
    assert offs
    # 7/6 6/off and 7/1 1/off reach the same afterstate (listed only once).
    target = State.from_points({6: 4, 2: 10}, {12: 15}, BLACK, white_off=1)
    assert target in [a for _, a in res]


def test_bear_off_exact():
    s = State.from_points({6: 1, 1: 1}, {12: 15}, WHITE, white_off=13)
    res = moves_of(s, (6, 1))
    # Either 6/off 1/off (win) or 6/5 5/off (higher die from farthest point).
    assert len(res) == 2
    assert any(a.is_over() for _, a in res)
    assert ((6, OFF), (1, OFF)) in [m for m, _ in res]


def test_bear_off_higher_die_only_from_farthest_point():
    s = State.from_points({5: 1, 2: 1}, {12: 15}, WHITE, white_off=13)
    res = moves_of(s, (6, 1))
    results = {a.pos[WHITE] for _, a in res}
    a_state = State.from_points({1: 1}, {12: 15}, WHITE, white_off=14).pos[WHITE]
    b_state = State.from_points({2: 1}, {12: 15}, WHITE, white_off=14).pos[WHITE]
    assert results == {a_state, b_state}
    for move, _ in res:
        assert move[0] != (2, OFF)  # 2/off with the 6 while 5 is occupied is illegal


def test_higher_die_bears_off_farthest_with_doubles():
    s = State.from_points({5: 1, 2: 1}, {12: 15}, WHITE, white_off=13)
    res = moves_of(s, (6, 6))
    assert len(res) == 1
    move, after = res[0]
    assert move[0] == (5, OFF)
    assert after.is_over()


def test_smaller_die_must_move_inside_home():
    s = State.from_points({6: 2}, {12: 15}, WHITE, white_off=13)
    res = moves_of(s, (2, 1))
    for move, a in res:
        assert a.pos[WHITE][OFF] == 13  # neither die can bear off a checker from 6
    assert {a.pos[WHITE][3] + a.pos[WHITE][4] + a.pos[WHITE][5] for _, a in res}


def test_bear_off_blocked_inside_home_by_opponent():
    # White checker on 6 with dice 1-1 - point 5 blocked by black anchor.
    s = State.from_points({6: 1, 1: 1}, {20: 2, 12: 13}, WHITE, white_off=13)
    res = moves_of(s, (1, 1))
    # 6/5 blocked; 1/off with 1 is exact; then 6 cannot move (5 blocked),
    # cannot bear off with 1 because it is not exact and not farthest move.
    assert len(res) == 1
    move, a = res[0]
    assert move == ((1, OFF),)
    assert a.pos[WHITE][6] == 1


def test_win_in_middle_of_move_counts_as_full():
    s = State.from_points({1: 1}, {12: 15}, WHITE, white_off=14)
    for dice in ALL_DICE:
        res = moves_of(s, dice)
        assert len(res) == 1
        assert res[0][1].is_over()


# ---------------------------------------------------------------- game end / scoring

def test_single_game_win():
    s = State.from_points({}, {6: 5}, BLACK, white_off=15, black_off=10)
    assert s.winner() == WHITE and s.result_points() == 1


def test_gammon():
    s = State.from_points({}, {6: 15}, BLACK, white_off=15, black_off=0)
    assert s.result_points() == 2


def test_backgammon_with_checker_in_winners_home():
    s = State.from_points({}, {6: 14, 20: 1}, BLACK, white_off=15, black_off=0)
    assert s.result_points() == 3


def test_backgammon_with_checker_on_bar():
    s = State.from_points({}, {6: 14}, BLACK, white_off=15, black_off=0, black_bar=1)
    assert s.result_points() == 3


def test_black_gammon():
    s = State.from_points({8: 15}, {}, WHITE, black_off=15)
    assert s.winner() == BLACK and s.result_points() == 2


def test_opening_roll_never_double():
    rng = random.Random(0)
    for _ in range(500):
        first, (w, b) = opening_roll(rng)
        assert w != b
        assert first == (WHITE if w > b else BLACK)


# ---------------------------------------------------------------- apply / notation

def test_apply_move_reproduces_afterstate():
    rng = random.Random(3)
    for _ in range(300):
        s = random_position(rng)
        dice = rng.choice(ALL_DICE)
        for move, a in moves_of(s, dice):
            assert apply_move(s, move) == a


def test_format_move():
    assert format_move(((24, 18), (13, 8))) == "24/18 13/8"
    assert format_move(((BAR, 22), (6, OFF))) == "bar/22 6/off"
    assert format_move(((24, 18),), BLACK, absolute=True) == "1/7"
    assert format_move(()) == "(no move)"


def test_render_board_contains_info():
    txt = render_board(State.initial(), (6, 5))
    assert "Pips" in txt and "6-5" in txt


# ---------------------------------------------------------------- reference comparison

def random_position(rng: random.Random) -> State:
    """Generate a random legal position (sometimes bear-off/bar heavy)."""
    mode = rng.random()
    while True:
        board = [0] * 25
        pos = []
        for player in (WHITE, BLACK):
            arr = [0] * 26
            for _ in range(15):
                r = rng.random()
                if mode < 0.3:  # bear-off style
                    if r < 0.3:
                        arr[OFF] += 1
                    elif r < 0.95:
                        arr[rng.randint(1, 6)] += 1
                    else:
                        arr[rng.randint(7, 24)] += 1
                else:
                    if r < 0.06:
                        arr[BAR] += 1
                    elif r < 0.1:
                        arr[OFF] += 1
                    else:
                        arr[rng.randint(1, 24)] += 1
            pos.append(arr)
        w, b = pos
        ok = all(not (w[q] and b[25 - q]) for q in range(1, 25))
        if ok and w[OFF] < 15 and b[OFF] < 15:
            return State((tuple(w), tuple(b)), rng.randint(0, 1))


def positions_from_games(rng: random.Random, n_games: int) -> list[State]:
    from backgammon.engine.game import roll_dice
    out = []
    for _ in range(n_games):
        first, dice = opening_roll(rng)
        s = State.initial(first)
        while not s.is_over():
            out.append(s)
            res = generate_moves(s, dice)
            s = rng.choice(res)[1]
            dice = roll_dice(rng)
    return out


@pytest.mark.parametrize("seed", range(4))
def test_matches_reference_on_random_positions(seed):
    rng = random.Random(100 + seed)
    for _ in range(150):
        s = random_position(rng)
        for dice in [(a, b) for a in range(1, 7) for b in range(a, 7)]:
            fast = {a.pos for _, a in generate_moves(s, dice)}
            ref = reference_afterstates(s, dice)
            assert fast == ref, (s, dice)


def test_matches_reference_on_game_positions():
    rng = random.Random(7)
    states = positions_from_games(rng, 15)
    for s in states[::2]:
        dice = (rng.randint(1, 6), rng.randint(1, 6))
        fast = {a.pos for _, a in generate_moves(s, dice)}
        assert fast == reference_afterstates(s, dice)


def test_random_games_finish_and_score():
    class Rnd:
        def __init__(self, rng):
            self.rng = rng

        def choose(self, state, dice, moves):
            return self.rng.randrange(len(moves))

    rng = random.Random(11)
    for _ in range(50):
        r = play_game(Rnd(rng), Rnd(rng), rng)
        assert r.winner in (WHITE, BLACK)
        assert r.points in (1, 2, 3)
