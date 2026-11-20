"""Step-by-step move input for a human player (independent of pygame).

The human plays one die at a time. A step is offered only if the
position after it can still be completed into one of the legal moves
returned by :func:`generate_moves`, so all rules (bar first, play as many
dice as possible, larger die, bearing off) are enforced automatically.
"""

from __future__ import annotations

from functools import lru_cache

from backgammon.engine.board import (
    Move,
    State,
    Step,
    apply_step,
    generate_moves,
    single_steps,
)


class TurnInput:
    def __init__(self, state: State, dice: tuple[int, int]):
        self.start = state
        self.dice = dice
        self.legal = generate_moves(state, dice)
        self.legal_after = {a.pos: i for i, (_, a) in enumerate(self.legal)}
        d1, d2 = dice
        self.all_dice = (d1,) * 4 if d1 == d2 else (d1, d2)
        self.played: list[tuple[Step, int, State]] = []  # (step, die, state before)
        self.current = state
        self.remaining: tuple[int, ...] = tuple(sorted(self.all_dice))
        self._can_complete = lru_cache(maxsize=None)(self._can_complete_impl)

    # ---------------------------------------------------------------- logic
    def _is_final(self, st: State) -> bool:
        return st.pos in self.legal_after

    def _can_complete_impl(self, pos, turn, remaining: tuple[int, ...]) -> bool:
        st = State(pos, turn)
        if st.pos in self.legal_after:
            return True
        for die in set(remaining):
            rest = list(remaining)
            rest.remove(die)
            rest_t = tuple(rest)
            for step in single_steps(st, die):
                nxt = apply_step(st, step)
                if self._can_complete(nxt.pos, nxt.turn, rest_t):
                    return True
        return False

    def options(self) -> list[tuple[Step, int]]:
        """Legal next single steps as (step, die), smallest die first."""
        if self.is_complete():
            return []
        opts = []
        seen = set()
        for die in sorted(set(self.remaining)):
            rest = list(self.remaining)
            rest.remove(die)
            rest_t = tuple(rest)
            for step in single_steps(self.current, die):
                if step in seen:
                    continue  # same step with a smaller die already offered
                nxt = apply_step(self.current, step)
                if self._can_complete(nxt.pos, nxt.turn, rest_t):
                    opts.append((step, die))
                    seen.add(step)
        return opts

    def sources(self) -> set[int]:
        return {s for (s, _), _ in self.options()}

    def destinations(self, src: int) -> dict[int, int]:
        """Map destination point -> die for steps starting at ``src``."""
        return {d: die for (s, d), die in self.options() if s == src}

    def play(self, step: Step) -> State:
        for (s, d), die in self.options():
            if (s, d) == step:
                self.played.append((step, die, self.current))
                self.current = apply_step(self.current, step)
                rest = list(self.remaining)
                rest.remove(die)
                self.remaining = tuple(rest)
                return self.current
        raise ValueError(f"illegal step {step}")

    def undo(self) -> State | None:
        if not self.played:
            return None
        step, die, before = self.played.pop()
        self.current = before
        self.remaining = tuple(sorted(self.remaining + (die,)))
        return before

    def is_complete(self) -> bool:
        return self._is_final(self.current)

    def no_moves(self) -> bool:
        return len(self.legal) == 1 and not self.legal[0][0]

    def used_dice(self) -> list[int]:
        return [die for _, die, _ in self.played]

    def result(self) -> tuple[int, Move]:
        """Index into generate_moves() result and the steps actually played."""
        return self.legal_after[self.current.pos], tuple(s for s, _, _ in self.played)
