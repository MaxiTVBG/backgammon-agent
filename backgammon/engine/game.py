"""Game flow helpers: dice, opening roll and a simple game loop."""

from __future__ import annotations

import random
from typing import Protocol

from .board import BLACK, WHITE, GameResult, Move, State, generate_moves


class MoveChooser(Protocol):
    def choose(self, state: State, dice: tuple[int, int],
               moves: list[tuple[Move, State]]) -> int: ...


def roll_dice(rng: random.Random) -> tuple[int, int]:
    return rng.randint(1, 6), rng.randint(1, 6)


def opening_roll(rng: random.Random) -> tuple[int, tuple[int, int]]:
    """Each player rolls one die; the higher one starts and plays both dice.

    Returns (starting_player, (white_die, black_die)). Ties are re-rolled.
    """
    while True:
        w, b = rng.randint(1, 6), rng.randint(1, 6)
        if w != b:
            return (WHITE if w > b else BLACK), (w, b)


def play_game(white: MoveChooser, black: MoveChooser, rng: random.Random,
              max_turns: int = 10_000) -> GameResult:
    """Play one full game between two agents and return the result."""
    first, dice = opening_roll(rng)
    state = State.initial(first)
    agents = (white, black)
    turns = 0
    while True:
        moves = generate_moves(state, dice)
        idx = agents[state.turn].choose(state, dice, moves) if len(moves) > 1 else 0
        state = moves[idx][1]
        turns += 1
        if state.is_over():
            return GameResult(state.winner(), state.result_points(), turns)
        if turns >= max_turns:
            raise RuntimeError("game did not finish")
        dice = roll_dice(rng)


class Game:
    """Stateful game object used by the CLI and GUI (keeps a move history)."""

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()
        self.reset()

    def reset(self) -> None:
        self.first_player, self.opening_dice = opening_roll(self.rng)
        self.state = State.initial(self.first_player)
        self.dice: tuple[int, int] | None = self.opening_dice
        self.history: list[tuple[int, tuple[int, int], Move]] = []
        self.moves: list[tuple[Move, State]] = generate_moves(self.state, self.dice)

    def roll(self) -> tuple[int, int]:
        self.dice = roll_dice(self.rng)
        self.moves = generate_moves(self.state, self.dice)
        return self.dice

    def play(self, index: int) -> State:
        assert self.dice is not None
        move, after = self.moves[index]
        self.history.append((self.state.turn, self.dice, move))
        self.state = after
        self.dice = None
        self.moves = []
        return after

    def is_over(self) -> bool:
        return self.state.is_over()
