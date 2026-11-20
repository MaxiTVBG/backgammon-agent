"""Colours and fonts of the GUI."""

from __future__ import annotations

from functools import lru_cache

import pygame

BG_TOP = (28, 34, 40)
BG_BOTTOM = (12, 15, 19)

WOOD = (118, 70, 38)          # frame
WOOD_DARK = (84, 48, 26)      # bar / tray
FELT = (24, 84, 58)           # playing field

POINT_LIGHT = (226, 205, 160)
POINT_DARK = (150, 40, 38)

WHITE_BODY = (238, 230, 214)
WHITE_RIM = (196, 182, 160)
BLACK_BODY = (52, 40, 38)
BLACK_RIM = (24, 18, 17)

DIE_BODY = (250, 247, 240)
DIE_PIP = (30, 30, 34)
DIE_BODY_DARK = (60, 44, 40)
DIE_PIP_DARK = (240, 232, 215)

HIGHLIGHT = (255, 205, 70)
SELECT = (110, 220, 255)
LAST_MOVE = (255, 255, 255)

PANEL_BG = (24, 28, 34)
PANEL_CARD = (34, 40, 48)
PANEL_BORDER = (60, 70, 82)
TEXT = (230, 233, 238)
TEXT_DIM = (150, 158, 170)
ACCENT = (240, 180, 70)
BUTTON = (58, 98, 150)
BUTTON_HOVER = (76, 122, 180)
BUTTON_DISABLED = (52, 58, 66)
GOOD = (90, 190, 120)
BAD = (220, 90, 80)

FONT_NAMES = "segoeui,dejavusans,freesans,arial,liberationsans"


@lru_cache(maxsize=64)
def font(size: int, bold: bool = False) -> pygame.font.Font:
    size = max(8, int(size))
    path = pygame.font.match_font(FONT_NAMES, bold=bold)
    if path:
        return pygame.font.Font(path, size)
    return pygame.font.Font(None, int(size * 1.3))
