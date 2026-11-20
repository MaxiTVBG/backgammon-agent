"""Drawing of the board, checkers, highlights and dice."""

from __future__ import annotations

import math
import random

import pygame

from backgammon.engine.board import BAR, OFF, WHITE, State

from . import textures as tx
from . import theme
from .layout import Layout


def viewer_point(player: int, own_point: int, viewer: int) -> int:
    return own_point if player == viewer else 25 - own_point


def own_point(player: int, vp: int, viewer: int) -> int:
    return vp if player == viewer else 25 - vp


def colors_of(player: int) -> tuple[tuple, tuple]:
    return (theme.WHITE_BODY, theme.WHITE_RIM) if player == WHITE else (theme.BLACK_BODY, theme.BLACK_RIM)


class BoardView:
    def __init__(self, layout: Layout):
        self.layout = layout
        self._static: pygame.Surface | None = None

    def resize(self, layout: Layout) -> None:
        self.layout = layout
        self._static = None

    # ------------------------------------------------------------- static
    def _build_static(self) -> pygame.Surface:
        L = self.layout
        b = L.board
        surf = pygame.Surface(b.size, pygame.SRCALPHA)
        ox, oy = b.x, b.y

        def rel(rect: pygame.Rect) -> pygame.Rect:
            return rect.move(-ox, -oy)

        # drop shadow + wooden frame
        shadow = pygame.Surface(b.size, pygame.SRCALPHA)
        pygame.draw.rect(shadow, (0, 0, 0, 0), shadow.get_rect())
        frame = tx.wood(b.w, b.h, theme.WOOD, 3)
        mask = pygame.Surface(b.size, pygame.SRCALPHA)
        rad = int(L.frame * 0.6)
        pygame.draw.rect(mask, (255, 255, 255, 255), mask.get_rect(), border_radius=rad)
        framed = frame.copy().convert_alpha() if pygame.display.get_surface() else frame.copy()
        framed.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        surf.blit(framed, (0, 0))
        # bevel lines on the frame
        pygame.draw.rect(surf, (255, 225, 180, 50), surf.get_rect().inflate(-3, -3), 2, border_radius=rad)
        pygame.draw.rect(surf, (0, 0, 0, 90), surf.get_rect(), 2, border_radius=rad)

        # fields (felt) with inset shadow
        for field in (L.left_field, L.right_field):
            r = rel(field)
            surf.blit(tx.felt(r.w, r.h), r.topleft)
            pygame.draw.rect(surf, (0, 0, 0, 110), r, 3)
        # bar and tray in darker wood
        for rect, seed in ((L.bar_rect, 7), (L.tray_rect, 9)):
            r = rel(rect)
            surf.blit(tx.wood(r.w, r.h, theme.WOOD_DARK, seed, vertical=True), r.topleft)
            pygame.draw.rect(surf, (0, 0, 0, 120), r, 2)
        # tray divider
        tr = rel(L.tray_rect)
        pygame.draw.line(surf, (0, 0, 0, 120), (tr.x + 4, tr.centery), (tr.right - 4, tr.centery), 3)
        pygame.draw.line(surf, (255, 220, 170, 40), (tr.x + 4, tr.centery + 2), (tr.right - 4, tr.centery + 2), 1)

        # triangles
        tw = int(L.pw * 0.96)
        th = int(L.tri_h)
        for vp in range(1, 25):
            color = theme.POINT_DARK if vp % 2 else theme.POINT_LIGHT
            top = L.is_top(vp)
            tri = tx.triangle(tw, th, color, top)
            x = L.point_x(vp) - tw / 2 - ox
            y = (L.field_top - oy + 2) if top else (L.field_bottom - oy - th - 2)
            surf.blit(tri, (int(x), int(y)))

        # point numbers on the frame
        fnt = theme.font(max(9, int(L.frame * 0.55)), bold=True)
        for vp in range(1, 25):
            txt = fnt.render(str(vp), True, (245, 225, 190))
            x = L.point_x(vp) - ox - txt.get_width() / 2
            if L.is_top(vp):
                y = (L.frame - txt.get_height()) / 2
            else:
                y = b.h - L.frame + (L.frame - txt.get_height()) / 2
            surf.blit(txt, (int(x), int(y)))
        return surf

    def draw_static(self, screen: pygame.Surface) -> None:
        if self._static is None:
            self._static = self._build_static()
        b = self.layout.board
        # soft shadow under the board
        sh = pygame.Surface((b.w + 30, b.h + 30), pygame.SRCALPHA)
        for i in range(6):
            pygame.draw.rect(sh, (0, 0, 0, 22), sh.get_rect().inflate(-i * 5, -i * 5), border_radius=20)
        screen.blit(sh, (b.x - 10, b.y - 4))
        screen.blit(self._static, b.topleft)

    # ------------------------------------------------------------ checkers
    def stack_positions(self, state: State, viewer: int, hide: dict | None = None):
        """Yield (x, y, player, label) for every visible checker."""
        L = self.layout
        hide = hide or {}
        for player in (0, 1):
            arr = state.pos[player]
            for q in range(1, 25):
                n = arr[q] - hide.get((player, q), 0)
                if n <= 0:
                    continue
                vp = viewer_point(player, q, viewer)
                for k in range(n):
                    x, y = L.checker_pos(vp, k, n)
                    yield x, y, player, (n if k == n - 1 and n > 5 else None)
            n = arr[BAR] - hide.get((player, BAR), 0)
            for k in range(max(n, 0)):
                x, y = L.bar_pos(player == viewer, k, n)
                yield x, y, player, (n if k == n - 1 and n > 3 else None)

    def draw_checker(self, screen, x, y, player, label=None, scale: float = 1.0) -> None:
        r = int(self.layout.r * scale)
        body, rim = colors_of(player)
        spr = tx.checker(r, body, rim)
        screen.blit(spr, (int(x - spr.get_width() / 2), int(y - spr.get_height() / 2)))
        if label:
            fnt = theme.font(int(r * 0.9), bold=True)
            col = (40, 30, 28) if player == WHITE else (240, 230, 210)
            t = fnt.render(str(label), True, col)
            screen.blit(t, (int(x - t.get_width() / 2), int(y - t.get_height() / 2)))

    def draw_off(self, screen, state: State, viewer: int, hide: dict | None = None) -> None:
        L = self.layout
        hide = hide or {}
        w, h = L.slab_size()
        for player in (0, 1):
            body, rim = colors_of(player)
            spr = tx.slab(w, h, body, rim)
            n = state.pos[player][OFF] - hide.get((player, OFF), 0)
            for k in range(max(n, 0)):
                x, y = L.off_pos(player == viewer, k)
                screen.blit(spr, (int(x - w / 2), int(y - h / 2)))

    def draw_checkers(self, screen, state: State, viewer: int, hide: dict | None = None) -> None:
        for x, y, player, label in self.stack_positions(state, viewer, hide):
            self.draw_checker(screen, x, y, player, label)
        self.draw_off(screen, state, viewer, hide)

    def top_checker_pos(self, state: State, player: int, point: int, viewer: int,
                        extra: int = 0) -> tuple[float, float]:
        """Position of the topmost checker (or of the next free slot with extra=1)."""
        L = self.layout
        if point == BAR:
            n = state.pos[player][BAR] + extra
            return L.bar_pos(player == viewer, max(n - 1, 0), max(n, 1))
        if point == OFF:
            n = state.pos[player][OFF] + extra
            return L.off_pos(player == viewer, max(n - 1, 0))
        vp = viewer_point(player, point, viewer)
        n = state.pos[player][point] + extra
        return L.checker_pos(vp, max(n - 1, 0), max(n, 1))

    # ---------------------------------------------------------- highlights
    def draw_destinations(self, screen, points, t: float) -> None:
        """Mark destination points (viewer numbering, 'off' for the tray)."""
        L = self.layout
        pulse = 0.5 + 0.5 * math.sin(t * 5)
        for vp in points:
            if vp == "off":
                r = L.tray_rect.inflate(-4, -4)
                half = pygame.Rect(r.x, int(L.mid_y), r.w, r.bottom - int(L.mid_y))
                s = pygame.Surface(half.size, pygame.SRCALPHA)
                s.fill((*theme.HIGHLIGHT, int(60 + 60 * pulse)))
                screen.blit(s, half.topleft)
                pygame.draw.rect(screen, theme.HIGHLIGHT, half, 2, border_radius=4)
                continue
            top = L.is_top(vp)
            tri = tx.triangle(int(L.pw * 0.96), int(L.tri_h), theme.HIGHLIGHT, top, int(90 + 70 * pulse))
            x = L.point_x(vp) - tri.get_width() / 2
            y = (L.field_top + 2) if top else (L.field_bottom - tri.get_height() - 2)
            screen.blit(tri, (int(x), int(y)))
            cx = L.point_x(vp)
            cy = L.field_top + L.tri_h + L.r * 0.6 if top else L.field_bottom - L.tri_h - L.r * 0.6
            pygame.draw.circle(screen, theme.HIGHLIGHT, (int(cx), int(cy)), max(3, int(L.r * 0.22)))

    def draw_ring(self, screen, pos, color, t: float = 0.0, strong: bool = False) -> None:
        r = self.layout.r
        ring = tx.glow_ring(int(r * (1.08 if strong else 1.0)), color)
        if not strong:
            ring = ring.copy()
            ring.set_alpha(int(120 + 80 * (0.5 + 0.5 * math.sin(t * 4))))
        screen.blit(ring, (int(pos[0] - ring.get_width() / 2), int(pos[1] - ring.get_height() / 2)))

    # ---------------------------------------------------------------- dice
    def draw_dice(self, screen, dice: list[int], used: list[bool], player: int, viewer: int,
                  rolling: float | None = None, rng: random.Random | None = None) -> None:
        """Draw dice in the half of the board of the player to move.

        ``rolling`` in [0, 1) is the progress of the roll animation.
        """
        if not dice:
            return
        L = self.layout
        size = int(L.pw * 0.82)
        cx, cy = L.dice_center(player == viewer)
        gap = size * 1.25
        n = len(dice)
        start = cx - gap * (n - 1) / 2
        body, pip = ((theme.DIE_BODY, theme.DIE_PIP) if player == WHITE
                     else (theme.DIE_BODY_DARK, theme.DIE_PIP_DARK))
        for i, (v, u) in enumerate(zip(dice, used)):
            x, y = start + i * gap, cy
            if rolling is not None:
                p = rolling
                face = (rng or random).randint(1, 6) if p < 0.85 else v
                surf = tx.die(size, face, False, body, pip)
                angle = (1 - p) ** 2 * 720 * (1 if i % 2 else -1) + i * 15
                surf = pygame.transform.rotozoom(surf, angle, 1.0 + 0.25 * math.sin(p * math.pi))
                bounce = abs(math.sin(p * math.pi * 3)) * (1 - p) * size * 0.8
                x += (1 - p) * size * 2.2 * (1 if i % 2 else -1)
                y -= bounce
            else:
                surf = tx.die(size, v, u, body, pip)
            screen.blit(surf, (int(x - surf.get_width() / 2), int(y - surf.get_height() / 2)))

    def draw_opening_dice(self, screen, values: tuple[int, int], viewer: int,
                          rolling: float | None, rng=None) -> None:
        """One die for each side: viewer's on the right, opponent's on the left."""
        L = self.layout
        size = int(L.pw * 0.9)
        for i, (v, side_viewer) in enumerate(((values[0], True), (values[1], False))):
            cx, cy = L.dice_center(side_viewer)
            pl = WHITE if side_viewer else 1
            body, pip = ((theme.DIE_BODY, theme.DIE_PIP) if pl == WHITE
                         else (theme.DIE_BODY_DARK, theme.DIE_PIP_DARK))
            if rolling is not None:
                face = (rng or random).randint(1, 6) if rolling < 0.85 else v
                surf = tx.die(size, face, False, body, pip)
                surf = pygame.transform.rotozoom(surf, (1 - rolling) ** 2 * 600 * (1 if i else -1), 1.0)
                cy -= abs(math.sin(rolling * math.pi * 3)) * (1 - rolling) * size
            else:
                surf = tx.die(size, v, False, body, pip)
            screen.blit(surf, (int(cx - surf.get_width() / 2), int(cy - surf.get_height() / 2)))
