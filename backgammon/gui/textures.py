"""Procedurally generated graphics (no external image files).

Everything is rendered with numpy into pygame surfaces and cached by size,
so the board can be re-rendered for any window size.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import pygame

from . import theme


def _surface_from_rgba(arr: np.ndarray) -> pygame.Surface:
    """arr: (H, W, 4) uint8 -> per-pixel-alpha Surface."""
    h, w = arr.shape[:2]
    surf = pygame.image.frombuffer(np.ascontiguousarray(arr).tobytes(), (w, h), "RGBA")
    if pygame.display.get_init() and pygame.display.get_surface() is not None:
        return surf.convert_alpha()
    return surf.copy()


def _surface_from_rgb(arr: np.ndarray) -> pygame.Surface:
    h, w = arr.shape[:2]
    surf = pygame.image.frombuffer(np.ascontiguousarray(arr).tobytes(), (w, h), "RGB")
    return surf.copy()


def _smooth_noise(h: int, w: int, cell_h: int, cell_w: int, rng: np.random.Generator) -> np.ndarray:
    """Value noise in [0, 1] with bilinear interpolation, shape (h, w)."""
    gh, gw = h // cell_h + 2, w // cell_w + 2
    grid = rng.random((gh, gw))
    ys = np.linspace(0, gh - 1.001, h)
    xs = np.linspace(0, gw - 1.001, w)
    y0 = ys.astype(int)
    x0 = xs.astype(int)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]
    fy = fy * fy * (3 - 2 * fy)
    fx = fx * fx * (3 - 2 * fx)
    a = grid[y0][:, x0]
    b = grid[y0][:, x0 + 1]
    c = grid[y0 + 1][:, x0]
    d = grid[y0 + 1][:, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


@lru_cache(maxsize=16)
def wood(w: int, h: int, base: tuple = theme.WOOD, seed: int = 3, vertical: bool = False) -> pygame.Surface:
    """Wood texture with grain lines and knots."""
    if vertical:
        return pygame.transform.rotate(wood(h, w, base, seed, False), 90)
    w, h = max(w, 2), max(h, 2)
    rng = np.random.default_rng(seed)
    n1 = _smooth_noise(h, w, max(h // 6, 4), max(w // 3, 8), rng)
    n2 = _smooth_noise(h, w, 3, 40, rng)
    y = np.arange(h)[:, None] / max(h, 1)
    rings = np.sin((y * 38 + n1 * 5.0 + n2 * 0.6) * math.pi)
    grain = 0.5 + 0.5 * rings
    fine = rng.random((h, w)) * 0.06
    shade = 0.78 + 0.22 * grain ** 3 + fine - 0.04 * n2
    col = np.array(base, dtype=np.float32)[None, None, :] * shade[..., None]
    # subtle darker streaks
    col *= (0.92 + 0.08 * _smooth_noise(h, w, 2, 90, rng))[..., None]
    return _surface_from_rgb(np.clip(col, 0, 255).astype(np.uint8))


@lru_cache(maxsize=8)
def felt(w: int, h: int, base: tuple = theme.FELT, seed: int = 5) -> pygame.Surface:
    """Felt cloth with fine noise and a soft vignette."""
    w, h = max(w, 2), max(h, 2)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2, h / 2
    d = np.sqrt(((xx - cx) / (w / 2)) ** 2 + ((yy - cy) / (h / 2)) ** 2)
    vign = 1.08 - 0.28 * np.clip(d, 0, 1.4) ** 2
    noise = rng.random((h, w)) * 0.08 + _smooth_noise(h, w, 20, 20, rng) * 0.06
    col = np.array(base, dtype=np.float32)[None, None, :] * (vign + noise - 0.07)[..., None]
    return _surface_from_rgb(np.clip(col, 0, 255).astype(np.uint8))


@lru_cache(maxsize=4)
def background(w: int, h: int) -> pygame.Surface:
    w, h = max(w, 2), max(h, 2)
    yy = np.linspace(0, 1, h)[:, None, None]
    top = np.array(theme.BG_TOP, np.float32)
    bot = np.array(theme.BG_BOTTOM, np.float32)
    col = top * (1 - yy) + bot * yy
    col = np.broadcast_to(col, (h, w, 3))
    return _surface_from_rgb(col.astype(np.uint8))


@lru_cache(maxsize=32)
def triangle(w: int, h: int, color: tuple, pointing_down: bool, alpha: int = 255) -> pygame.Surface:
    """Anti-aliased triangle with a gradient from base to tip."""
    ss = 4
    W, H = max(w, 2) * ss, max(h, 2) * ss
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    t = yy / H if pointing_down else 1 - yy / H     # 0 at base, 1 at tip
    half = (1 - t) * W / 2
    inside = np.abs(xx + 0.5 - W / 2) <= half
    base = np.array(color, np.float32)
    shade = 1.06 - 0.22 * t
    stripe = 1 - 0.07 * np.exp(-((np.abs(xx + 0.5 - W / 2) - half * 0.55) ** 2) / (W * 0.5))
    rgb = np.clip(base[None, None, :] * (shade * stripe)[..., None], 0, 255)
    a = inside.astype(np.float32) * alpha
    arr = np.dstack([rgb, a]).astype(np.uint8)
    big = _surface_from_rgba(arr)
    return pygame.transform.smoothscale(big, (max(w, 2), max(h, 2)))


@lru_cache(maxsize=32)
def checker(radius: int, body: tuple, rim: tuple, with_shadow: bool = True) -> pygame.Surface:
    """Checker sprite with radial shading, rim, groove, highlight and shadow.

    The returned surface is (2r + pad) square; draw it centred.
    """
    r = max(radius, 3)
    ss = 4
    pad = int(r * 0.35) + 2
    size = (2 * r + 2 * pad)
    S = size * ss
    R = r * ss
    c = S / 2
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    dx, dy = xx + 0.5 - c, yy + 0.5 - c
    d = np.sqrt(dx * dx + dy * dy) / R

    rgba = np.zeros((S, S, 4), np.float32)
    if with_shadow:
        sx, sy = R * 0.10, R * 0.16
        ds = np.sqrt((dx - sx) ** 2 + (dy - sy) ** 2) / R
        sh = np.clip((1.18 - ds) / 0.3, 0, 1) ** 1.5 * 110
        rgba[..., 3] = sh

    body_c = np.array(body, np.float32)
    rim_c = np.array(rim, np.float32)
    # light from the top-left
    lx, ly = -0.45, -0.55
    shade = 1.0 - 0.22 * ((dx / R - lx * 0.6) ** 2 + (dy / R - ly * 0.6) ** 2) + 0.12
    col = body_c[None, None, :] * shade[..., None]
    # rim
    rim_mask = np.clip((d - 0.86) / 0.04, 0, 1)
    col = col * (1 - rim_mask[..., None]) + (rim_c * shade[..., None]) * rim_mask[..., None]
    # groove ring
    groove = np.exp(-((d - 0.64) ** 2) / 0.0012)
    col *= (1 - 0.22 * groove)[..., None]
    inner_light = np.exp(-((d - 0.60) ** 2) / 0.0010)
    col += 28 * inner_light[..., None] * np.clip(-(dy / R), 0, 1)[..., None]
    # specular highlight
    hx, hy = -0.32, -0.40
    hd = np.sqrt((dx / R - hx) ** 2 + ((dy / R - hy) * 1.4) ** 2)
    spec = np.clip(1 - hd / 0.38, 0, 1) ** 2
    col = col + (255 - col) * (0.55 * spec)[..., None]

    alpha_body = np.clip((1 - d) * R / 1.5, 0, 1)
    out_a = alpha_body * 255 + rgba[..., 3] * (1 - alpha_body)
    rgb = np.zeros((S, S, 3), np.float32)
    rgb[:] = col * alpha_body[..., None]  # shadow is black
    safe_a = np.maximum(out_a, 1e-3)
    rgb = rgb * 255 / safe_a[..., None]
    arr = np.dstack([np.clip(rgb, 0, 255), np.clip(out_a, 0, 255)]).astype(np.uint8)
    big = _surface_from_rgba(arr)
    return pygame.transform.smoothscale(big, (size, size))


@lru_cache(maxsize=16)
def slab(w: int, h: int, body: tuple, rim: tuple) -> pygame.Surface:
    """Side view of a borne-off checker (thin rounded bar)."""
    w, h = max(w, 4), max(h, 3)
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(surf, rim, surf.get_rect(), border_radius=h // 2)
    inner = surf.get_rect().inflate(-2, -2)
    pygame.draw.rect(surf, body, inner, border_radius=max(h // 2 - 1, 1))
    hl = pygame.Rect(inner.x + 3, inner.y + 1, inner.w - 6, max(1, inner.h // 3))
    light = tuple(min(255, c + 45) for c in body)
    pygame.draw.rect(surf, light, hl, border_radius=max(hl.h // 2, 1))
    return surf


PIPS = {
    1: [(0.5, 0.5)],
    2: [(0.27, 0.27), (0.73, 0.73)],
    3: [(0.27, 0.27), (0.5, 0.5), (0.73, 0.73)],
    4: [(0.27, 0.27), (0.73, 0.27), (0.27, 0.73), (0.73, 0.73)],
    5: [(0.27, 0.27), (0.73, 0.27), (0.5, 0.5), (0.27, 0.73), (0.73, 0.73)],
    6: [(0.27, 0.25), (0.73, 0.25), (0.27, 0.5), (0.73, 0.5), (0.27, 0.75), (0.73, 0.75)],
}


@lru_cache(maxsize=64)
def die(size: int, value: int, used: bool = False, body: tuple = theme.DIE_BODY,
        pip: tuple = theme.DIE_PIP) -> pygame.Surface:
    """Die face with rounded corners, bevel shading and pips."""
    ss = 3
    S = max(size, 8) * ss
    surf = pygame.Surface((S, S), pygame.SRCALPHA)
    rad = S // 5
    pygame.draw.rect(surf, (0, 0, 0, 70), pygame.Rect(S * 0.06, S * 0.08, S * 0.94, S * 0.92), border_radius=rad)
    face = pygame.Rect(0, 0, int(S * 0.94), int(S * 0.94))
    pygame.draw.rect(surf, tuple(int(c * 0.82) for c in body), face, border_radius=rad)
    inner = face.inflate(-S * 0.06, -S * 0.06)
    inner.topleft = (int(S * 0.02), int(S * 0.02))
    pygame.draw.rect(surf, body, inner, border_radius=rad)
    gl = pygame.Rect(inner.x + S * 0.08, inner.y + S * 0.05, inner.w - S * 0.16, inner.h * 0.35)
    glow = pygame.Surface(gl.size, pygame.SRCALPHA)
    pygame.draw.rect(glow, (255, 255, 255, 60), glow.get_rect(), border_radius=rad)
    surf.blit(glow, gl.topleft)
    pr = S * 0.075
    for fx, fy in PIPS[value]:
        cx, cy = inner.x + fx * inner.w, inner.y + fy * inner.h
        pygame.draw.circle(surf, tuple(min(255, c + 70) for c in pip), (cx + pr * 0.15, cy + pr * 0.25), pr)
        pygame.draw.circle(surf, pip, (cx, cy), pr)
    out = pygame.transform.smoothscale(surf, (max(size, 8), max(size, 8)))
    if used:
        out.set_alpha(90)
    return out


@lru_cache(maxsize=8)
def glow_ring(radius: int, color: tuple) -> pygame.Surface:
    r = max(radius, 3)
    size = int(r * 2.8)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    d = np.sqrt((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / r
    a = np.exp(-((d - 1.05) ** 2) / 0.012) * 255
    arr = np.zeros((size, size, 4), np.float32)
    arr[..., :3] = color
    arr[..., 3] = a
    return _surface_from_rgba(arr.astype(np.uint8))


def clear_caches() -> None:
    for fn in (wood, felt, background, triangle, checker, slab, die, glow_ring):
        fn.cache_clear()
