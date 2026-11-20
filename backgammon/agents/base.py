"""Common agent interface."""

from __future__ import annotations

from backgammon.engine.board import Move, State


class Agent:
    """Base class for all agents.

    An agent receives the current state, the dice and the list of legal
    ``(move, afterstate)`` pairs and returns the index of the chosen move.
    """

    name = "agent"

    def choose(self, state: State, dice: tuple[int, int],
               moves: list[tuple[Move, State]]) -> int:
        ranked = self.rank_moves(state, dice, moves)
        return ranked[0][0]

    def rank_moves(self, state: State, dice: tuple[int, int],
                   moves: list[tuple[Move, State]]) -> list[tuple[int, float]]:
        """Return ``(index, score)`` pairs sorted best first.

        Scores are from the point of view of the player to move; their scale
        depends on the agent (win probability for neural agents).
        """
        raise NotImplementedError

    def win_probability(self, state: State) -> float | None:
        """Estimated probability that the side to move in ``state`` wins.

        ``None`` if the agent has no such estimate.
        """
        return None

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
