"""Plain-text board rendering for the terminal."""

from __future__ import annotations

from .board import BAR, BLACK, OFF, WHITE, State

SYMBOLS = {WHITE: "O", BLACK: "X"}


def _cell(n: int, row: int) -> str:
    """Return the character for checker stack of signed size n at height row."""
    count = abs(n)
    if count == 0:
        return " . " if row == 0 else "   "
    sym = SYMBOLS[WHITE] if n > 0 else SYMBOLS[BLACK]
    if row < 4:
        return f" {sym} " if count > row else "   "
    if row == 4:
        if count == 5:
            return f" {sym} "
        if count > 5:
            return f"{count:>2} "
    return "   "


def render_board(state: State, dice: tuple[int, int] | None = None) -> str:
    """Return a multi-line string with the board from White's point of view.

    White (O) moves 24 -> 1 and bears off at the bottom right.
    Black (X) moves 1 -> 24 and bears off at the top right.
    """
    b = state.absolute_board()
    w_bar, b_bar = state.pos[WHITE][BAR], state.pos[BLACK][BAR]
    w_off, b_off = state.pos[WHITE][OFF], state.pos[BLACK][OFF]

    lines = []
    top = list(range(13, 25))
    bottom = list(range(12, 0, -1))
    lines.append(" " + "".join(f"{p:>3}" for p in top[:6]) + "  |   |" + "".join(f"{p:>3}" for p in top[6:]))
    lines.append("+" + "-" * 18 + "-+---+" + "-" * 18 + "+" + f"  X off: {b_off}")
    for row in range(5):
        left = "".join(_cell(b[p], row) for p in top[:6])
        right = "".join(_cell(b[p], row) for p in top[6:])
        bar = f" {'X' if b_bar > row else ' '} " if row < 4 else (f"{b_bar:>2} " if b_bar > 4 else "   ")
        lines.append(f"|{left} |{bar}|{right}|")
    lines.append("|" + " " * 18 + " |BAR|" + " " * 18 + "|")
    for row in range(4, -1, -1):
        left = "".join(_cell(b[p], row) for p in bottom[:6])
        right = "".join(_cell(b[p], row) for p in bottom[6:])
        bar = f" {'O' if w_bar > row else ' '} " if row < 4 else (f"{w_bar:>2} " if w_bar > 4 else "   ")
        lines.append(f"|{left} |{bar}|{right}|")
    lines.append("+" + "-" * 18 + "-+---+" + "-" * 18 + "+" + f"  O off: {w_off}")
    lines.append(" " + "".join(f"{p:>3}" for p in bottom[:6]) + "  |   |" + "".join(f"{p:>3}" for p in bottom[6:]))
    info = (f"Pips  O: {state.pip_count(WHITE)}   X: {state.pip_count(BLACK)}   "
            f"To move: {SYMBOLS[state.turn]}")
    if dice:
        info += f"   Dice: {dice[0]}-{dice[1]}"
    lines.append(info)
    return "\n".join(lines)
