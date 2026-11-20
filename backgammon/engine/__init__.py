"""Backgammon rules engine."""

from .board import (
    ALL_ROLLS,
    BAR,
    BLACK,
    NUM_CHECKERS,
    OFF,
    PLAYER_NAMES,
    START_POSITION,
    WHITE,
    GameResult,
    State,
    afterstates,
    apply_move,
    apply_step,
    format_move,
    generate_moves,
    to_absolute,
)
from .game import Game, opening_roll, play_game, roll_dice
from .render import render_board

__all__ = [
    "ALL_ROLLS", "BAR", "BLACK", "NUM_CHECKERS", "OFF", "PLAYER_NAMES",
    "START_POSITION", "WHITE", "GameResult", "State", "afterstates",
    "apply_move", "apply_step", "format_move", "generate_moves",
    "to_absolute", "Game", "opening_roll", "play_game", "roll_dice",
    "render_board",
]
