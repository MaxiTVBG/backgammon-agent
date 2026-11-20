"""Core backgammon rules: state representation and legal move generation.

Representation
--------------
Each player's checkers are stored in a 26-element tuple written from that
player's *own* point of view:

    index 0       -> checkers already borne off
    index 1..24   -> checkers on points 1..24 (the player moves 24 -> 1,
                     points 1..6 are the player's home board)
    index 25      -> checkers on the bar

A checker of the opponent standing on the opponent's point ``k`` stands on
point ``25 - k`` from our own point of view.

White's own numbering is used as the "absolute" numbering of the board
(for display and notation); black's point ``k`` is absolute point ``25 - k``.

A :class:`State` is immutable and hashable, so it can be used as a dict key.
"""

from __future__ import annotations

from typing import Iterable, NamedTuple

WHITE = 0
BLACK = 1
PLAYER_NAMES = ("White", "Black")

NUM_CHECKERS = 15
OFF = 0
BAR = 25

# Standard starting position from a player's own point of view.
START_POSITION: tuple[int, ...] = (
    0,              # off
    0, 0, 0, 0, 0, 5,   # points 1-6
    0, 3, 0, 0, 0, 0,   # points 7-12
    5, 0, 0, 0, 0, 0,   # points 13-18
    0, 0, 0, 0, 0, 2,   # points 19-24
    0,              # bar
)

# All 21 distinct dice rolls with their probabilities (out of 36).
ALL_ROLLS: tuple[tuple[tuple[int, int], int], ...] = tuple(
    ((a, b), 1 if a == b else 2) for a in range(1, 7) for b in range(a, 7)
)

# A single checker step: (from_point, to_point) in the mover's own numbering.
# from_point == 25 means entering from the bar, to_point == 0 means bearing off.
Step = tuple[int, int]
Move = tuple[Step, ...]


class State:
    """Immutable game state: checker layout for both players and side to move."""

    __slots__ = ("pos", "turn", "_hash")

    def __init__(self, pos: tuple[tuple[int, ...], tuple[int, ...]], turn: int):
        self.pos = pos
        self.turn = turn
        self._hash = hash((pos, turn))

    # ----------------------------------------------------------------- basics
    @classmethod
    def initial(cls, turn: int = WHITE) -> "State":
        return cls((START_POSITION, START_POSITION), turn)

    @classmethod
    def from_points(
        cls,
        white: dict[int, int],
        black: dict[int, int],
        turn: int = WHITE,
        white_bar: int = 0,
        black_bar: int = 0,
        white_off: int | None = None,
        black_off: int | None = None,
    ) -> "State":
        """Build a state from {own_point: count} dictionaries.

        Borne-off counts default to whatever is missing up to 15 checkers.
        """
        def build(points: dict[int, int], bar: int, off: int | None) -> tuple[int, ...]:
            arr = [0] * 26
            for p, n in points.items():
                if not 1 <= p <= 24:
                    raise ValueError(f"invalid point {p}")
                arr[p] += n
            arr[BAR] = bar
            on_board = sum(arr)
            arr[OFF] = NUM_CHECKERS - on_board if off is None else off
            if sum(arr) != NUM_CHECKERS or min(arr) < 0:
                raise ValueError("a player must have exactly 15 checkers")
            return tuple(arr)

        st = cls((build(white, white_bar, white_off), build(black, black_bar, black_off)), turn)
        st.validate()
        return st

    def validate(self) -> None:
        """Raise ValueError if the state is not a legal backgammon position."""
        for p in (WHITE, BLACK):
            arr = self.pos[p]
            if len(arr) != 26 or sum(arr) != NUM_CHECKERS or min(arr) < 0:
                raise ValueError(f"bad checker counts for {PLAYER_NAMES[p]}")
        w, b = self.pos
        for q in range(1, 25):
            if w[q] and b[25 - q]:
                raise ValueError(f"both players on absolute point {q}")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, State) and self.turn == other.turn and self.pos == other.pos

    def __hash__(self) -> int:
        return self._hash

    def __repr__(self) -> str:
        return f"State(pos={self.pos!r}, turn={self.turn})"

    # ------------------------------------------------------------- accessors
    @property
    def me(self) -> tuple[int, ...]:
        return self.pos[self.turn]

    @property
    def opp(self) -> tuple[int, ...]:
        return self.pos[1 - self.turn]

    def pip_count(self, player: int) -> int:
        arr = self.pos[player]
        return sum(i * arr[i] for i in range(1, 26))

    def is_over(self) -> bool:
        return self.pos[WHITE][OFF] == NUM_CHECKERS or self.pos[BLACK][OFF] == NUM_CHECKERS

    def winner(self) -> int | None:
        if self.pos[WHITE][OFF] == NUM_CHECKERS:
            return WHITE
        if self.pos[BLACK][OFF] == NUM_CHECKERS:
            return BLACK
        return None

    def result_points(self) -> int:
        """Points won by the winner: 1 single, 2 gammon, 3 backgammon (0 if not over)."""
        w = self.winner()
        if w is None:
            return 0
        loser = self.pos[1 - w]
        if loser[OFF] > 0:
            return 1
        # Backgammon: loser still has a checker on the bar or in the winner's
        # home board (loser's own points 19..24).
        if loser[BAR] > 0 or any(loser[19:25]):
            return 3
        return 2

    def absolute_board(self) -> list[int]:
        """Signed board indexed by absolute point 1..24 (index 0 unused).

        Positive values are white checkers, negative values black checkers.
        """
        w, b = self.pos
        board = [0] * 25
        for q in range(1, 25):
            board[q] = w[q] - b[25 - q]
        return board

    def swap_turn(self) -> "State":
        return State(self.pos, 1 - self.turn)


class GameResult(NamedTuple):
    winner: int
    points: int  # 1 = single, 2 = gammon, 3 = backgammon
    turns: int


# --------------------------------------------------------------------------
# Move generation
# --------------------------------------------------------------------------

def _dfs(me: list[int], op: list[int], seq: tuple[int, ...], i: int, max_src: int,
         doubles: bool, path: list[Step], out: dict, best: list[int]) -> None:
    """Depth-first search over the dice in ``seq`` starting at index ``i``.

    ``out`` maps afterstate keys to (move, first_die_used, dice_used); ``best``
    holds the maximum number of dice used so far.
    """
    n = len(seq)
    if i < n and me[OFF] != NUM_CHECKERS:
        d = seq[i]
        moved = False
        if me[BAR]:
            sources: Iterable[int] = (BAR,)
        else:
            top = max_src if doubles else 24
            sources = [s for s in range(top, 0, -1) if me[s]]
        all_home = None
        for src in sources:
            dst = src - d
            if dst >= 1:
                opp_pt = 25 - dst
                cnt = op[opp_pt]
                if cnt >= 2:
                    continue  # blocked
                # play the step
                me[src] -= 1
                me[dst] += 1
                if cnt == 1:  # hit a blot
                    op[opp_pt] = 0
                    op[BAR] += 1
                path.append((src, dst))
                _dfs(me, op, seq, i + 1, src, doubles, path, out, best)
                path.pop()
                if cnt == 1:
                    op[BAR] -= 1
                    op[opp_pt] = 1
                me[dst] -= 1
                me[src] += 1
                moved = True
            else:
                # bearing off
                if all_home is None:
                    all_home = not any(me[7:26])
                if not all_home:
                    continue
                if dst < 0 and any(me[src + 1:7]):
                    continue  # a higher die may only bear off the farthest checker
                me[src] -= 1
                me[OFF] += 1
                path.append((src, OFF))
                _dfs(me, op, seq, i + 1, src, doubles, path, out, best)
                path.pop()
                me[OFF] -= 1
                me[src] += 1
                moved = True
        if moved:
            return
        used = i
    else:
        # all dice played, or game already won (counts as fully played)
        used = n
    if used < best[0]:
        return
    if used > best[0]:
        best[0] = used
        out.clear()
    key = (tuple(me), tuple(op))
    first = seq[0] if used else 0
    prev = out.get(key)
    if prev is None:
        out[key] = (tuple(path), first)
    elif first > prev[1]:
        # Same position reachable with a larger first die: remember that, so
        # the "play the larger die" rule keeps this afterstate.
        out[key] = (tuple(path), first)


def generate_moves(state: State, dice: tuple[int, int]) -> list[tuple[Move, State]]:
    """Return every legal move for ``dice`` as a list of (move, afterstate).

    Afterstates are unique: different move orders that lead to the same
    position are returned only once. The afterstate has the turn passed to
    the opponent. If no move is possible a single entry with an empty move
    and the unchanged position (opponent to move) is returned.

    Rules enforced: entering from the bar first, blocked points, hitting
    blots, doubles play four times, as many dice as possible must be used,
    if only one die can be used it must be the larger one when possible,
    bearing off only with all checkers home and with a higher die only from
    the farthest point.
    """
    d1, d2 = dice
    turn = state.turn
    me = list(state.pos[turn])
    op = list(state.pos[1 - turn])
    out: dict = {}
    best = [0]
    if d1 == d2:
        _dfs(me, op, (d1, d1, d1, d1), 0, BAR, True, [], out, best)
    else:
        _dfs(me, op, (d1, d2), 0, BAR, False, [], out, best)
        _dfs(me, op, (d2, d1), 0, BAR, False, [], out, best)
        if best[0] == 1:
            hi = max(d1, d2)
            if any(first == hi for (_, first) in out.values()):
                out = {k: v for k, v in out.items() if v[1] == hi}

    nxt = 1 - turn
    result = []
    for (me_t, op_t), (move, _) in out.items():
        pos = (me_t, op_t) if turn == WHITE else (op_t, me_t)
        result.append((move, State(pos, nxt)))
    return result


def afterstates(state: State, dice: tuple[int, int]) -> list[State]:
    """Convenience wrapper returning only the unique afterstates."""
    return [s for _, s in generate_moves(state, dice)]


def apply_step(state: State, step: Step) -> State:
    """Apply a single checker step for the side to move (turn is NOT passed).

    No legality check is done here; it is used by the GUI to show partial
    moves that were already validated against :func:`generate_moves`.
    """
    src, dst = step
    turn = state.turn
    me = list(state.pos[turn])
    op = list(state.pos[1 - turn])
    me[src] -= 1
    if dst == OFF:
        me[OFF] += 1
    else:
        me[dst] += 1
        if op[25 - dst] == 1:
            op[25 - dst] = 0
            op[BAR] += 1
    pos = (tuple(me), tuple(op)) if turn == WHITE else (tuple(op), tuple(me))
    return State(pos, turn)


def apply_move(state: State, move: Move) -> State:
    """Apply a full move and pass the turn (no legality check)."""
    s = state
    for step in move:
        s = apply_step(s, step)
    return s.swap_turn()


# --------------------------------------------------------------------------
# Notation helpers
# --------------------------------------------------------------------------

def to_absolute(player: int, point: int) -> int:
    """Convert a player's own point number (0..25) to absolute numbering."""
    if point in (OFF, BAR):
        return point
    return point if player == WHITE else 25 - point


def format_move(move: Move, player: int = WHITE, absolute: bool = False) -> str:
    """Human readable notation, e.g. ``"bar/22 13/8*"`` style (hits not marked)."""
    if not move:
        return "(no move)"
    parts = []
    for src, dst in move:
        if absolute:
            s = "bar" if src == BAR else str(to_absolute(player, src))
            t = "off" if dst == OFF else str(to_absolute(player, dst))
        else:
            s = "bar" if src == BAR else str(src)
            t = "off" if dst == OFF else str(dst)
        parts.append(f"{s}/{t}")
    return " ".join(parts)
