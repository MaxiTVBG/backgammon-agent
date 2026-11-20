"""Window geometry: where the board, points, bar, tray and panel are.

Points are addressed in the *viewer's* numbering (1..24): the viewer's home
board (points 1-6) is at the bottom right, next to the bear-off tray.
"""

from __future__ import annotations

from dataclasses import dataclass

import pygame

BAR_UNITS = 1.15
SEP_UNITS = 0.30
TRAY_UNITS = 1.25


@dataclass
class Layout:
    width: int
    height: int

    def __post_init__(self) -> None:
        W, H = self.width, self.height
        self.panel_w = max(300, min(int(W * 0.30), 460))
        margin = max(10, int(min(W, H) * 0.025))
        avail_w = W - self.panel_w - 2 * margin
        avail_h = H - 2 * margin
        aspect = 1.32
        bh = min(avail_h, avail_w / aspect)
        bw = bh * aspect
        bx = margin + (avail_w - bw) / 2
        by = margin + (avail_h - bh) / 2
        self.board = pygame.Rect(int(bx), int(by), int(bw), int(bh))
        self.panel = pygame.Rect(W - self.panel_w, 0, self.panel_w, H)

        f = bh * 0.045
        self.frame = f
        pw = (bw - 2 * f) / (12 + BAR_UNITS + SEP_UNITS + TRAY_UNITS)
        self.pw = pw
        self.field_top = by + f
        self.field_bottom = by + bh - f
        self.field_h = self.field_bottom - self.field_top
        self.mid_y = (self.field_top + self.field_bottom) / 2
        self.left_x = bx + f
        self.bar_x = self.left_x + 6 * pw
        self.bar_w = BAR_UNITS * pw
        self.right_x = self.bar_x + self.bar_w
        self.sep_x = self.right_x + 6 * pw
        self.tray_x = self.sep_x + SEP_UNITS * pw
        self.tray_w = TRAY_UNITS * pw
        self.tri_h = self.field_h * 0.42
        self.r = int(min(pw * 0.47, self.field_h * 0.5 / 5.4))
        self.left_field = pygame.Rect(int(self.left_x), int(self.field_top), int(6 * pw), int(self.field_h))
        self.right_field = pygame.Rect(int(self.right_x), int(self.field_top), int(6 * pw) + 1, int(self.field_h))
        self.bar_rect = pygame.Rect(int(self.bar_x), int(self.field_top), int(self.bar_w) + 1, int(self.field_h))
        self.tray_rect = pygame.Rect(int(self.tray_x), int(self.field_top), int(self.tray_w), int(self.field_h))

    # ------------------------------------------------------------- points
    def point_x(self, vp: int) -> float:
        pw = self.pw
        if 1 <= vp <= 6:
            return self.right_x + (6 - vp + 0.5) * pw
        if 7 <= vp <= 12:
            return self.left_x + (12 - vp + 0.5) * pw
        if 13 <= vp <= 18:
            return self.left_x + (vp - 13 + 0.5) * pw
        return self.right_x + (vp - 19 + 0.5) * pw

    @staticmethod
    def is_top(vp: int) -> bool:
        return vp >= 13

    def point_rect(self, vp: int) -> pygame.Rect:
        x = self.point_x(vp) - self.pw / 2
        if self.is_top(vp):
            return pygame.Rect(int(x), int(self.field_top), int(self.pw), int(self.field_h / 2))
        return pygame.Rect(int(x), int(self.mid_y), int(self.pw), int(self.field_h / 2))

    def _spacing(self, n: int, avail: float) -> float:
        r = self.r
        if n <= 1:
            return 2 * r
        return min(2 * r, (avail - 2 * r) / (n - 1))

    def checker_pos(self, vp: int, k: int, n: int) -> tuple[float, float]:
        """Centre of the k-th checker (0 = at the edge) of a stack of n on point vp."""
        r = self.r
        sp = self._spacing(n, self.field_h / 2 - r * 0.6)
        x = self.point_x(vp)
        if self.is_top(vp):
            return x, self.field_top + r + 1 + k * sp
        return x, self.field_bottom - r - 1 - k * sp

    def bar_pos(self, viewer_side: bool, k: int, n: int) -> tuple[float, float]:
        """Viewer's bar checkers stack upwards from the middle (top half)."""
        r = self.r
        sp = self._spacing(n, self.field_h / 2 - r * 1.2)
        x = self.bar_x + self.bar_w / 2
        if viewer_side:
            return x, self.mid_y - r * 1.25 - k * sp
        return x, self.mid_y + r * 1.25 + k * sp

    def slab_size(self) -> tuple[int, int]:
        h = max(3, int(min(self.r * 0.5, (self.field_h / 2 - self.r) / 15)))
        return int(self.tray_w * 0.78), h

    def off_pos(self, viewer_side: bool, k: int) -> tuple[float, float]:
        """Centre of the k-th borne-off checker in the tray."""
        w, h = self.slab_size()
        x = self.tray_x + self.tray_w / 2
        if viewer_side:
            return x, self.field_bottom - 6 - h / 2 - k * (h + 1)
        return x, self.field_top + 6 + h / 2 + k * (h + 1)

    # ------------------------------------------------------------ hit test
    def hit_test(self, x: float, y: float):
        """Return viewer point 1..24, 'bar', 'off' or None."""
        if not (self.field_top <= y <= self.field_bottom):
            return None
        if self.tray_rect.collidepoint(x, y):
            return "off"
        if self.bar_rect.collidepoint(x, y):
            return "bar"
        top = y < self.mid_y
        if self.left_x <= x < self.left_x + 6 * self.pw:
            col = int((x - self.left_x) // self.pw)
            return 13 + col if top else 12 - col
        if self.right_x <= x < self.right_x + 6 * self.pw:
            col = int((x - self.right_x) // self.pw)
            return 19 + col if top else 6 - col
        return None

    def dice_center(self, viewer_turn: bool) -> tuple[float, float]:
        field = self.right_field if viewer_turn else self.left_field
        return field.centerx, self.mid_y
