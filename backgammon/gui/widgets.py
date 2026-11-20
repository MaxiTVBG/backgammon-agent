"""Minimal pygame widgets: button, option selector and slider."""

from __future__ import annotations

import math

import pygame

from . import theme


class Button:
    def __init__(self, label: str, callback, hotkey: str = ""):
        self.label = label
        self.callback = callback
        self.hotkey = hotkey
        self.rect = pygame.Rect(0, 0, 10, 10)
        self.enabled = True
        self.hover = False
        self.accent = False

    def draw(self, screen: pygame.Surface, font_size: int) -> None:
        if not self.enabled:
            col = theme.BUTTON_DISABLED
        elif self.accent:
            col = (200, 140, 50) if not self.hover else (225, 165, 70)
        else:
            col = theme.BUTTON_HOVER if self.hover else theme.BUTTON
        r = self.rect
        pygame.draw.rect(screen, (0, 0, 0), r.move(0, 3), border_radius=8)
        pygame.draw.rect(screen, col, r, border_radius=8)
        light = tuple(min(255, c + 30) for c in col)
        pygame.draw.rect(screen, light, pygame.Rect(r.x + 2, r.y + 2, r.w - 4, r.h // 2 - 2), border_radius=7)
        pygame.draw.rect(screen, col, pygame.Rect(r.x + 2, r.y + r.h // 3, r.w - 4, r.h // 2), border_radius=7)
        fnt = theme.font(font_size, bold=True)
        txt = fnt.render(self.label, True, theme.TEXT if self.enabled else theme.TEXT_DIM)
        screen.blit(txt, (r.centerx - txt.get_width() // 2, r.centery - txt.get_height() // 2))

    def handle(self, event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.enabled and self.rect.collidepoint(event.pos):
                self.callback()
                return True
        return False


class Selector:
    """``< value >`` cycling selector with a caption."""

    def __init__(self, caption: str, options: list[tuple[str, object]], on_change=None, index: int = 0):
        self.caption = caption
        self.options = options
        self.index = index
        self.on_change = on_change
        self.rect = pygame.Rect(0, 0, 10, 10)
        self.enabled = True

    @property
    def value(self):
        return self.options[self.index][1] if self.options else None

    @property
    def label(self) -> str:
        return self.options[self.index][0] if self.options else "-"

    def set_value(self, value) -> None:
        for i, (_, v) in enumerate(self.options):
            if v == value:
                self.index = i
                return

    def _arrows(self):
        r = self.rect
        s = r.h
        return pygame.Rect(r.x, r.y, s, s), pygame.Rect(r.right - s, r.y, s, s)

    def draw(self, screen, font_size: int) -> None:
        r = self.rect
        cap = theme.font(int(font_size * 0.85)).render(self.caption, True, theme.TEXT_DIM)
        screen.blit(cap, (r.x, r.y - cap.get_height() - 2))
        pygame.draw.rect(screen, theme.PANEL_CARD, r, border_radius=6)
        pygame.draw.rect(screen, theme.PANEL_BORDER, r, 1, border_radius=6)
        left, right = self._arrows()
        col = theme.ACCENT if self.enabled else theme.TEXT_DIM
        for a, d in ((left, -1), (right, 1)):
            cx, cy = a.center
            w = a.w * 0.18
            pts = [(cx - d * w, cy - w * 1.4), (cx + d * w, cy), (cx - d * w, cy + w * 1.4)]
            pygame.draw.polygon(screen, col, pts)
        fnt = theme.font(font_size, bold=True)
        txt = fnt.render(self.label, True, theme.TEXT if self.enabled else theme.TEXT_DIM)
        max_w = r.w - 2 * r.h - 6
        if txt.get_width() > max_w:
            txt = pygame.transform.smoothscale(txt, (max_w, int(txt.get_height() * max_w / txt.get_width())))
        screen.blit(txt, (r.centerx - txt.get_width() // 2, r.centery - txt.get_height() // 2))

    def handle(self, event) -> bool:
        if not (self.enabled and self.options) or event.type != pygame.MOUSEBUTTONDOWN:
            return False
        if event.button not in (1, 4, 5) or not self.rect.collidepoint(event.pos):
            return False
        left, right = self._arrows()
        if event.button == 4 or (event.button == 1 and left.collidepoint(event.pos)):
            step = -1
        else:
            step = 1
        self.index = (self.index + step) % len(self.options)
        if self.on_change:
            self.on_change(self.value)
        return True


class Slider:
    """Horizontal slider with a logarithmic value range."""

    def __init__(self, caption: str, vmin: float, vmax: float, value: float):
        self.caption = caption
        self.vmin, self.vmax = vmin, vmax
        self.value = value
        self.rect = pygame.Rect(0, 0, 10, 10)
        self.dragging = False

    def _frac(self) -> float:
        return (math.log(self.value) - math.log(self.vmin)) / (math.log(self.vmax) - math.log(self.vmin))

    def _set_from_x(self, x: float) -> None:
        f = min(1.0, max(0.0, (x - self.rect.x) / max(self.rect.w, 1)))
        self.value = math.exp(math.log(self.vmin) + f * (math.log(self.vmax) - math.log(self.vmin)))

    def draw(self, screen, font_size: int) -> None:
        r = self.rect
        cap = theme.font(int(font_size * 0.85)).render(f"{self.caption}: x{self.value:.2f}", True, theme.TEXT_DIM)
        screen.blit(cap, (r.x, r.y - cap.get_height() - 2))
        track = pygame.Rect(r.x, r.centery - 3, r.w, 6)
        pygame.draw.rect(screen, theme.PANEL_CARD, track, border_radius=3)
        fill = pygame.Rect(r.x, r.centery - 3, int(r.w * self._frac()), 6)
        pygame.draw.rect(screen, theme.ACCENT, fill, border_radius=3)
        kx = r.x + r.w * self._frac()
        pygame.draw.circle(screen, (0, 0, 0), (int(kx), r.centery + 2), r.h // 2)
        pygame.draw.circle(screen, theme.TEXT, (int(kx), r.centery), r.h // 2)

    def handle(self, event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.rect.inflate(10, 10).collidepoint(event.pos):
            self.dragging = True
            self._set_from_x(event.pos[0])
            return True
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        if event.type == pygame.MOUSEMOTION and self.dragging:
            self._set_from_x(event.pos[0])
            return True
        return False
