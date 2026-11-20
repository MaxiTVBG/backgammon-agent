"""A deliberately simple, slow, independent move generator used only in tests.

It works on a *signed absolute board* (positive = white, negative = black)
and enumerates every permutation of the dice without any pruning, then
applies the "use as many dice as possible" and "larger die" rules on the
complete set of sequences. The fast engine must agree with it exactly.
"""

from __future__ import annotations

from itertools import permutations

from backgammon.engine.board import BLACK, WHITE, State


def _to_abs(state: State):
    board = [0] * 25  # absolute points 1..24
    w, b = state.pos
    for q in range(1, 25):
        board[q] = w[q] - b[25 - q]
    bars = [w[25], b[25]]
    offs = [w[0], b[0]]
    return board, bars, offs


def _from_abs(board, bars, offs, turn) -> tuple:
    w = [0] * 26
    b = [0] * 26
    for q in range(1, 25):
        if board[q] > 0:
            w[q] = board[q]
        elif board[q] < 0:
            b[25 - q] = -board[q]
    w[25], b[25] = bars
    w[0], b[0] = offs
    return (tuple(w), tuple(b))


def _own(player, q):
    """Absolute point -> player's own point number."""
    return q if player == WHITE else 25 - q


def _abs(player, p):
    return p if player == WHITE else 25 - p


def _single_steps(board, bars, offs, player, die):
    sign = 1 if player == WHITE else -1
    res = []
    if bars[player] > 0:
        own_dst = 25 - die
        q = _abs(player, own_dst)
        if board[q] * sign >= 0 or abs(board[q]) == 1:
            res.append(("bar", q))
        return res
    mine = [q for q in range(1, 25) if board[q] * sign > 0]
    all_home = all(1 <= _own(player, q) <= 6 for q in mine)
    for q in mine:
        own_src = _own(player, q)
        own_dst = own_src - die
        if own_dst >= 1:
            t = _abs(player, own_dst)
            if board[t] * sign >= 0 or abs(board[t]) == 1:
                res.append((q, t))
        elif all_home:
            if own_dst == 0:
                res.append((q, "off"))
            else:
                farthest = max(_own(player, x) for x in mine)
                if own_src == farthest:
                    res.append((q, "off"))
    return res


def _apply(board, bars, offs, player, step):
    board = board[:]
    bars = bars[:]
    offs = offs[:]
    sign = 1 if player == WHITE else -1
    src, dst = step
    if src == "bar":
        bars[player] -= 1
    else:
        board[src] -= sign
    if dst == "off":
        offs[player] += 1
    else:
        if board[dst] == -sign:  # hit
            board[dst] = 0
            bars[1 - player] += 1
        board[dst] += sign
    return board, bars, offs


def reference_afterstates(state: State, dice) -> set:
    """Return the set of legal afterstate positions (tuple pairs)."""
    player = state.turn
    board, bars, offs = _to_abs(state)
    d1, d2 = dice
    orders = [(d1,) * 4] if d1 == d2 else list(set(permutations((d1, d2))))

    finals = []  # (num_dice_used, dice_used_tuple, position)

    def rec(board, bars, offs, seq, used):
        if offs[player] == 15:
            finals.append((len(seq), used + seq[len(used):], _from_abs(board, bars, offs, player)))
            return
        if len(used) == len(seq):
            finals.append((len(used), used, _from_abs(board, bars, offs, player)))
            return
        die = seq[len(used)]
        steps = _single_steps(board, bars, offs, player, die)
        if not steps:
            finals.append((len(used), used, _from_abs(board, bars, offs, player)))
            return
        for st in steps:
            nb, nbar, noff = _apply(board, bars, offs, player, st)
            rec(nb, nbar, noff, seq, used + (die,))

    for seq in orders:
        rec(board, bars, offs, seq, ())

    mx = max(f[0] for f in finals)
    finals = [f for f in finals if f[0] == mx]
    if d1 != d2 and mx == 1:
        hi = max(d1, d2)
        with_hi = [f for f in finals if f[1][0] == hi]
        if with_hi:
            finals = with_hi
    return {f[2] for f in finals}
