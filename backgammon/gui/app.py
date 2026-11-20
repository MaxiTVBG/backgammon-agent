"""Main pygame application: human vs agent and agent vs agent."""

from __future__ import annotations

import argparse
import glob
import math
import os
import random
import threading
import time
from dataclasses import dataclass, field

import pygame

from backgammon.agents.factory import make_agent
from backgammon.engine.board import (
    BAR,
    BLACK,
    OFF,
    WHITE,
    Move,
    State,
    apply_step,
    format_move,
    generate_moves,
)

from . import theme
from .board_view import BoardView, viewer_point
from .layout import Layout
from .move_input import TurnInput
from .widgets import Button, Selector, Slider

COLOR_NAME = {WHITE: "Белите", BLACK: "Черните"}
COLOR_SHORT = {WHITE: "Бели", BLACK: "Черни"}
AGENT_LABELS = [("Random", "random"), ("Heuristic", "heuristic"), ("TD", "td"),
                ("TD + Expectimax", "expectimax")]
WIN_KIND = {1: "обикновена победа", 2: "марс", 3: "сух марс (backgammon)"}

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))


def find_models() -> list[tuple[str, str]]:
    root = os.path.join(PROJECT_ROOT, "checkpoints")
    paths = glob.glob(os.path.join(root, "**", "*.pt"), recursive=True)
    paths += glob.glob(os.path.join(os.getcwd(), "checkpoints", "**", "*.pt"), recursive=True)
    uniq = sorted({os.path.normpath(p) for p in paths},
                  key=lambda p: (os.path.basename(p) != "td_final.pt",
                                 os.path.basename(p) not in ("final.pt", "best.pt"), p))
    return [(os.path.relpath(p, root) if p.startswith(root) else os.path.basename(p), p) for p in uniq]


def ease(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


@dataclass
class Anim:
    duration: float
    base: State
    hide: dict
    player: int | None = None
    src: tuple[float, float] | None = None
    dst: tuple[float, float] | None = None
    on_done: object = None
    start: float = -1.0
    lift: float = 1.0


@dataclass
class Analysis:
    player: int
    agent_name: str
    dice: tuple[int, int]
    top: list[tuple[str, float]] = field(default_factory=list)
    is_prob: bool = True


class App:
    def __init__(self, size=(1440, 880), model: str | None = None, opponent: str = "td",
                 mode: str = "hva"):
        pygame.init()
        pygame.display.set_caption("Табла - самообучаващ се агент TD(λ)")
        self.screen = pygame.display.set_mode(size, pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.layout = Layout(*self.screen.get_size())
        self.view = BoardView(self.layout)
        self.rng = random.Random()
        self.fx_rng = random.Random(1)
        self.t = 0.0

        self.models = find_models()
        if model:
            model = os.path.abspath(model)
            if all(p != model for _, p in self.models):
                self.models.insert(0, (os.path.basename(model), model))
        agent_opts = AGENT_LABELS if self.models else AGENT_LABELS[:2]
        if not self.models and opponent in ("td", "expectimax"):
            opponent = "heuristic"

        self.sel_mode = Selector("Режим", [("Човек срещу агент", "hva"), ("Агент срещу агент", "ava")])
        self.sel_mode.set_value(mode)
        self.sel_a = Selector("Противник", list(agent_opts))
        self.sel_a.set_value(opponent)
        self.sel_b = Selector("Черни (агент)", list(agent_opts))
        self.sel_b.set_value("heuristic")
        self.sel_model = Selector("Модел (checkpoint)", self.models or [("няма обучен модел", None)])
        if model:
            self.sel_model.set_value(model)
        self.sel_mode.on_change = lambda v: self._relayout()
        self.btn_roll = Button("Хвърли", self.on_roll_button, "Space")
        self.btn_undo = Button("Отмени ход", self.on_undo, "U")
        self.btn_new = Button("Нова игра", self.new_game, "N")
        self.btn_new.accent = True
        self.slider = Slider("Скорост", 0.25, 6.0, 1.0)
        self.selectors = [self.sel_mode, self.sel_a, self.sel_b, self.sel_model]
        self.buttons = [self.btn_roll, self.btn_undo, self.btn_new]

        self._agents: dict = {}
        self._evaluator = None
        self._evaluator_model = None
        self.score = [0, 0]
        self.score_mode = None
        self.history_scroll = 0
        self.game_id = 0
        self._relayout()
        self.new_game()

    # ================================================================ setup
    def _relayout(self) -> None:
        self.layout = Layout(*self.screen.get_size())
        self.view.resize(self.layout)
        L = self.layout
        p = L.panel
        pad = max(12, int(p.w * 0.05))
        x, w = p.x + pad, p.w - 2 * pad
        self.fs = max(12, min(20, int(L.height / 50)))
        row = int(self.fs * 2.1)
        cap = int(self.fs * 1.25)
        y = p.y + int(self.fs * 3.2)
        ava = self.sel_mode.value == "ava"
        self.sel_a.caption = "Бели (агент)" if ava else "Противник"
        self.sel_b.enabled = ava
        for sel in (self.sel_mode, self.sel_a) + ((self.sel_b,) if ava else ()) + (self.sel_model,):
            y += cap
            sel.rect = pygame.Rect(x, y, w, row)
            y += row + 6
        if not ava:
            self.sel_b.rect = pygame.Rect(-1000, -1000, 1, 1)
        y += 8
        bw = (w - 16) // 3
        for i, b in enumerate(self.buttons):
            b.rect = pygame.Rect(x + i * (bw + 8), y, bw, int(row * 1.15))
        y += int(row * 1.15) + cap + 8
        self.slider.rect = pygame.Rect(x + 8, y, w - 16, int(self.fs * 1.1))
        y += int(self.fs * 2.2)
        self.info_top = y
        self.panel_x, self.panel_w_inner = x, w

    def agent_for(self, kind: str):
        model = self.sel_model.value
        key = (kind, model)
        if key not in self._agents:
            spec = {"kind": kind, "model": model}
            self._agents[key] = make_agent(spec)
        return self._agents[key]

    def evaluator(self):
        model = self.sel_model.value
        if model is None:
            return None
        if self._evaluator_model != model:
            self._evaluator = make_agent({"kind": "td", "model": model})
            self._evaluator_model = model
        return self._evaluator

    # ================================================================ game flow
    def new_game(self) -> None:
        self.game_id += 1
        self.mode = self.sel_mode.value
        if self.score_mode != (self.mode, self.sel_a.value, self.sel_b.value):
            self.score = [0, 0]
            self.score_mode = (self.mode, self.sel_a.value, self.sel_b.value)
        self.anims: list[Anim] = []
        self.selected = None
        self.turn_input = None
        self.analysis: Analysis | None = None
        self.history: list[str] = []
        self.history_scroll = 0
        self.dice: tuple[int, int] | None = None
        self.dice_used: list[bool] = []
        self.rolling: float | None = None
        self.thinking = False
        self.paused = False
        self.phase = "opening"
        self.message = "Хвърляне за първи ход..."
        self.state = State.initial(WHITE)
        self.viewer = WHITE
        self.human = None
        try:
            if self.mode == "hva":
                self.controllers = {"opp": self.agent_for(self.sel_a.value)}
            else:
                self.controllers = {WHITE: self.agent_for(self.sel_a.value),
                                    BLACK: self.agent_for(self.sel_b.value)}
        except Exception as exc:  # e.g. broken checkpoint
            self.phase = "error"
            self.message = f"Грешка при зареждане на агента: {exc}"
            return
        while True:
            a, b = self.rng.randint(1, 6), self.rng.randint(1, 6)
            if a != b:
                break
        self.opening = (a, b)  # a: right side (viewer / human / white), b: left side
        self.win_white = self._evaluate(self.state)
        self._run_roll_anim(1.1, self._opening_done, opening=True)

    def _opening_done(self) -> None:
        a, b = self.opening
        if self.mode == "hva":
            self.human = WHITE if a > b else BLACK
            self.viewer = self.human
            agent = 1 - self.human
            self.controllers = {agent: self.controllers["opp"]}
            if a > b:
                self.message = f"Вие хвърлихте {a} срещу {b}: играете с белите и започвате."
            else:
                self.message = f"Агентът хвърли {b} срещу вашите {a}: той е с белите и започва."
            first = WHITE
            dice = (max(a, b), min(a, b))
        else:
            first = WHITE if a > b else BLACK
            self.message = f"Бели {a} : Черни {b} - започват {COLOR_NAME[first].lower()}."
            dice = (a, b) if first == WHITE else (b, a)
        self.state = State.initial(first)
        self.win_white = self._evaluate(self.state)
        self.phase = "pause"
        self._wait(1.4 / (self.speed if self.mode == "ava" else 1.0),
                   lambda: self._set_dice_and_play(dice))

    @property
    def speed(self) -> float:
        return self.slider.value

    def is_human_turn(self) -> bool:
        return self.human is not None and self.state.turn == self.human

    def start_turn(self) -> None:
        if self.state.is_over():
            self._game_over()
            return
        self.dice = None
        self.dice_used = []
        if self.is_human_turn():
            self.phase = "await_roll"
            self.message = "Вашият ход - хвърлете заровете (Space)."
        else:
            self.phase = "pause"
            if self.human is not None:
                self.message = f"Агентът ({self._agent_name(self.state.turn)}) хвърля заровете..."
            self._wait(0.35 / self.speed, self._agent_roll)

    def _agent_roll(self) -> None:
        if self.paused:
            self._wait(0.1, self._agent_roll)
            return
        self._roll_dice()

    def _roll_dice(self) -> None:
        dice = (self.rng.randint(1, 6), self.rng.randint(1, 6))
        self.dice = dice
        self.dice_used = [False] * (4 if dice[0] == dice[1] else 2)
        dur = 0.8 if self.is_human_turn() else 0.8 / self.speed
        self._run_roll_anim(dur, lambda: self._set_dice_and_play(dice))

    def _set_dice_and_play(self, dice: tuple[int, int]) -> None:
        self.dice = dice
        self.dice_used = [False] * (4 if dice[0] == dice[1] else 2)
        who = COLOR_SHORT[self.state.turn]
        if self.is_human_turn():
            self.turn_input = TurnInput(self.state, dice)
            if self.turn_input.no_moves():
                self.message = f"Хвърлихте {dice[0]}-{dice[1]}: няма възможен ход."
                self.phase = "pause"
                self._wait(1.6, self._commit_human)
            else:
                self.phase = "human"
                self.message = f"Хвърлихте {dice[0]}-{dice[1]}. Изберете пул и поле."
        else:
            self.phase = "thinking"
            self.message = f"{who} ({self._agent_name(self.state.turn)}) мисли... зарове {dice[0]}-{dice[1]}"
            self._start_thinking()

    def _agent_name(self, player: int) -> str:
        agent = self.controllers.get(player)
        names = dict((v, k) for k, v in AGENT_LABELS)
        return names.get(getattr(agent, "name", ""), "агент")

    # ---------------------------------------------------------------- agent
    def _start_thinking(self) -> None:
        state, dice, gid = self.state, self.dice, self.game_id
        agent = self.controllers[state.turn]
        self.thinking = True
        self._think_result = None
        t0 = time.perf_counter()

        def work():
            moves = generate_moves(state, dice)
            if len(moves) == 1:
                ranked = [(0, float("nan"))]
            else:
                ranked = agent.rank_moves(state, dice, moves)
            self._think_result = (gid, moves, ranked, time.perf_counter() - t0)

        threading.Thread(target=work, daemon=True).start()

    def _poll_thinking(self) -> None:
        if not self.thinking or self._think_result is None:
            return
        gid, moves, ranked, elapsed = self._think_result
        self.thinking = False
        if gid != self.game_id:
            return
        player = self.state.turn
        agent = self.controllers[player]
        is_prob = agent.name in ("td", "expectimax")
        top = []
        for idx, score in ranked[:3]:
            top.append((format_move(moves[idx][0], player, absolute=True), score))
        self.analysis = Analysis(player, self._agent_name(player), self.dice, top, is_prob)
        idx = ranked[0][0]
        move, after = moves[idx]
        min_think = 0.35 / self.speed
        delay = max(0.0, min_think - elapsed)
        if not move:
            self.message = f"{COLOR_SHORT[player]} нямат възможен ход ({self.dice[0]}-{self.dice[1]})."
            self._wait(max(delay, 1.2 / self.speed), lambda: self._commit(move, after))
            return
        self._wait(delay, lambda: self._animate_move(self.state, move, player,
                                                     lambda: self._commit(move, after),
                                                     0.42 / self.speed))

    # ---------------------------------------------------------------- human
    def on_roll_button(self) -> None:
        if self.mode == "ava":
            self.paused = not self.paused
            return
        if self.phase == "await_roll":
            self.phase = "rolling"
            self._roll_dice()
        elif self.phase == "confirm":
            self._commit_human()

    def on_undo(self) -> None:
        if self.phase in ("human", "confirm") and self.turn_input and self.turn_input.played and not self.anims:
            self.turn_input.undo()
            self._sync_human_dice()
            self.selected = None
            self.phase = "human"
            self.message = "Ходът е отменен. Изберете пул."

    def _sync_human_dice(self) -> None:
        used = list(self.turn_input.used_dice())
        values = self._dice_list()
        flags = [False] * len(values)
        for d in used:
            for i, v in enumerate(values):
                if v == d and not flags[i]:
                    flags[i] = True
                    break
        self.dice_used = flags

    def _commit_human(self) -> None:
        ti = self.turn_input
        idx, steps = ti.result()
        move, after = ti.legal[idx]
        self.turn_input = None
        self.selected = None
        self._commit(steps if steps else move, after)

    def _human_click(self, pos) -> None:
        if self.phase != "human" or self.anims:
            return
        target = self.layout.hit_test(*pos)
        ti = self.turn_input
        if target is None:
            self.selected = None
            return
        point = BAR if target == "bar" else OFF if target == "off" else target  # viewer = human
        sources = ti.sources()
        if self.selected is not None:
            dests = ti.destinations(self.selected)
            if point in dests:
                self._play_human_step((self.selected, point))
                return
        if point in sources:
            self.selected = point
        else:
            self.selected = None

    def _play_human_step(self, step) -> None:
        before = self.turn_input.current
        player = self.human
        self.turn_input.play(step)
        self.selected = None
        self._sync_human_dice()

        def done():
            if self.turn_input.is_complete():
                self.phase = "confirm"
                self.message = "Ходът е завършен - натиснете „Готово“ (Space) или „Отмени ход“."
            else:
                # Auto-select when only one checker can move.
                srcs = self.turn_input.sources()
                if len(srcs) == 1:
                    self.selected = next(iter(srcs))
        self._animate_steps(before, [step], player, done, 0.24)

    # --------------------------------------------------------------- commit
    def _commit(self, move: Move, after: State) -> None:
        player = self.state.turn
        dice = self.dice
        n = len(self.history) + 1
        who = COLOR_SHORT[player]
        if self.human is not None:
            who += " (вие)" if player == self.human else " (агент)"
        move_txt = format_move(move, player, absolute=True)
        self.history.append(f"{n:>3}. {who} {dice[0]}-{dice[1]}: {move_txt}")
        if player != self.human:
            self.message = f"{who} изиграха {dice[0]}-{dice[1]}: {move_txt}"
        self.history_scroll = 0
        self.state = after
        self.dice = None
        self.dice_used = []
        self.win_white = self._evaluate(after)
        self.start_turn()

    def _game_over(self) -> None:
        w = self.state.winner()
        pts = self.state.result_points()
        self.phase = "over"
        if self.human is not None:
            idx = 0 if w == self.human else 1
            self.score[idx] += pts
            who = "Вие печелите" if w == self.human else "Агентът печели"
        else:
            self.score[w] += pts
            who = f"{COLOR_NAME[w]} печелят"
        self.message = f"{who}: {WIN_KIND[pts]} (+{pts} т.). Натиснете „Нова игра“."
        self.banner = (who + "!", f"{WIN_KIND[pts].capitalize()} - {pts} точк{'а' if pts == 1 else 'и'}")

    def _evaluate(self, state: State) -> float | None:
        w = state.winner()
        if w is not None:
            return 1.0 if w == WHITE else 0.0
        ev = self.evaluator()
        if ev is None:
            return None
        return float(ev.values([state])[0])

    # ============================================================ animations
    def _wait(self, seconds: float, then) -> None:
        self.anims.append(Anim(max(seconds, 0.0), None, {}, on_done=then))

    def _run_roll_anim(self, seconds: float, then, opening: bool = False) -> None:
        self.rolling = 0.0
        self._rolling_opening = opening

        def done():
            self.rolling = None
            then()
        a = Anim(seconds, None, {}, on_done=done)
        a.lift = -1  # marker: dice roll
        self.anims.append(a)

    def _animate_move(self, state: State, move: Move, player: int, then, step_time: float) -> None:
        self._animate_steps(state, list(move), player, then, step_time, mark_dice=True)

    def _animate_steps(self, state: State, steps, player: int, then, step_time: float,
                       mark_dice: bool = False) -> None:
        view, viewer = self.view, self.viewer
        cur = state
        for step in steps:
            src, dst = step
            nxt = apply_step(cur, step)
            opp = 1 - player
            hit = dst not in (OFF,) and cur.pos[opp][25 - dst] == 1
            p0 = view.top_checker_pos(cur, player, src, viewer)
            p1 = view.top_checker_pos(nxt, player, dst, viewer)
            mark = self._die_marker(step) if mark_dice else None
            self.anims.append(Anim(step_time, cur, {(player, src): 1}, player,
                                   p0, p1, on_done=mark))
            if hit:
                h0 = p1
                h1 = view.top_checker_pos(nxt, opp, BAR, viewer)
                self.anims.append(Anim(step_time * 0.8, nxt, {(opp, BAR): 1}, opp, h0, h1))
            cur = nxt
        self.anims.append(Anim(0.0, None, {}, on_done=then))

    def _die_marker(self, step):
        src, dst = step

        def mark():
            values = self._dice_list()
            need = src - dst if dst != OFF else src
            cands = [i for i, v in enumerate(values) if not self.dice_used[i] and
                     (v == need if dst != OFF else v >= need)]
            if dst == OFF:
                cands.sort(key=lambda i: values[i])
            if cands:
                self.dice_used[cands[0]] = True
        return mark

    def _dice_list(self) -> list[int]:
        if not self.dice:
            return []
        a, b = self.dice
        return [a] * 4 if a == b else [a, b]

    def _update_anims(self) -> None:
        now = self.t
        while self.anims:
            a = self.anims[0]
            if a.start < 0:
                a.start = now
            prog = (now - a.start) / a.duration if a.duration > 0 else 1.0
            if a.lift == -1:
                self.rolling = min(prog, 0.999)
            if prog < 1.0:
                return
            self.anims.pop(0)
            if a.on_done:
                a.on_done()

    # ================================================================ events
    def handle_event(self, event) -> bool:
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.VIDEORESIZE:
            w, h = max(event.w, 900), max(event.h, 600)
            self.screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
            self._relayout()
            # positions of queued animations are stale: finish them instantly
            for a in self.anims:
                a.duration = 0.0
            return True
        for w in self.selectors:
            if w.handle(event):
                if w is not self.sel_mode:
                    self._agents.clear()
                return True
        for b in self.buttons:
            if b.handle(event):
                return True
        if self.slider.handle(event):
            return True
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                self._human_click(event.pos)
            elif event.button == 3:
                self.on_undo()
            elif event.button in (4, 5) and event.pos[0] >= self.layout.panel.x:
                self.history_scroll = max(0, self.history_scroll + (1 if event.button == 4 else -1))
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.on_roll_button()
            elif event.key in (pygame.K_u, pygame.K_BACKSPACE):
                self.on_undo()
            elif event.key == pygame.K_n:
                self.new_game()
            elif event.key == pygame.K_ESCAPE:
                self.selected = None
        return True

    def _update_buttons(self) -> None:
        if self.mode == "ava":
            self.btn_roll.label = "Продължи" if self.paused else "Пауза"
            self.btn_roll.enabled = self.phase not in ("over", "error")
            self.btn_undo.enabled = False
        else:
            self.btn_roll.label = "Готово" if self.phase == "confirm" else "Хвърли"
            self.btn_roll.enabled = self.phase in ("await_roll", "confirm")
            self.btn_undo.enabled = (self.phase in ("human", "confirm") and self.turn_input is not None
                                     and bool(self.turn_input.played))

    # ================================================================ drawing
    def draw(self) -> None:
        from . import textures as tx

        scr = self.screen
        L = self.layout
        scr.blit(tx.background(L.width, L.height), (0, 0))
        self.view.draw_static(scr)

        # what to draw as the base position
        anim = self.anims[0] if self.anims and self.anims[0].base is not None else None
        if anim is not None:
            base, hide = anim.base, anim.hide
        elif self.turn_input is not None and self.phase in ("human", "confirm"):
            base, hide = self.turn_input.current, {}
        else:
            base, hide = self.state, {}
        # sources / destinations for the human
        if self.phase == "human" and not self.anims and self.turn_input is not None:
            if self.selected is not None:
                dests = self.turn_input.destinations(self.selected)
                pts = ["off" if d == OFF else d for d in dests]
                self.view.draw_destinations(scr, pts, self.t)

        self.view.draw_checkers(scr, base, self.viewer, hide)

        if self.phase == "human" and not self.anims and self.turn_input is not None:
            for src in self.turn_input.sources():
                if src != self.selected:
                    pos = self.view.top_checker_pos(base, self.human, src, self.viewer)
                    self.view.draw_ring(scr, pos, theme.HIGHLIGHT, self.t)
            if self.selected is not None:
                pos = self.view.top_checker_pos(base, self.human, self.selected, self.viewer)
                self.view.draw_ring(scr, pos, theme.SELECT, self.t, strong=True)

        # moving checker
        if anim is not None and anim.src is not None:
            p = ease((self.t - anim.start) / anim.duration) if anim.duration > 0 else 1.0
            x = anim.src[0] + (anim.dst[0] - anim.src[0]) * p
            y = anim.src[1] + (anim.dst[1] - anim.src[1]) * p
            lift = math.sin(p * math.pi)
            y -= lift * L.r * 1.6
            self.view.draw_checker(scr, x + lift * 3, y + lift * 6, anim.player, None, 1.0 + 0.12 * lift)

        # dice
        if self.phase == "opening" or (self.rolling is not None and getattr(self, "_rolling_opening", False)):
            self.view.draw_opening_dice(scr, self.opening, self.viewer, self.rolling, self.fx_rng)
        elif self.phase == "pause" and self.dice is None and self.history == [] and hasattr(self, "opening"):
            self.view.draw_opening_dice(scr, self.opening, self.viewer, None)
        elif self.dice:
            values = self._dice_list()
            used = self.dice_used or [False] * len(values)
            self.view.draw_dice(scr, values, used, self.state.turn, self.viewer, self.rolling, self.fx_rng)

        self._draw_side_labels()
        if self.phase == "over":
            self._draw_banner()
        self._draw_panel()

    def _draw_side_labels(self) -> None:
        L = self.layout
        fnt = theme.font(int(self.fs * 0.95), bold=True)
        if self.human is not None:
            me = f"Вие · {COLOR_SHORT[self.viewer]}"
            opp = f"Агент ({self._agent_name(1 - self.human)}) · {COLOR_SHORT[1 - self.viewer]}"
        else:
            me = f"{COLOR_SHORT[WHITE]}: {self._agent_name(WHITE)}"
            opp = f"{COLOR_SHORT[BLACK]}: {self._agent_name(BLACK)}"
        if self.phase == "opening":
            me, opp = ("Вие", "Агент") if self.mode == "hva" else (me, opp)
        b = L.board
        for txt, y, player in ((opp, b.y - self.fs * 1.6, 1 - self.viewer),
                               (me, b.bottom + self.fs * 0.3, self.viewer)):
            if y < 0 or y + self.fs > L.height:
                continue
            col = theme.TEXT
            surf = fnt.render(f"{txt}  ·  пипове {self.state.pip_count(player)}", True, col)
            scr_y = max(2, int(y))
            pygame.draw.circle(self.screen, theme.WHITE_BODY if player == WHITE else theme.BLACK_BODY,
                               (b.x + 10, scr_y + surf.get_height() // 2), int(self.fs * 0.45))
            pygame.draw.circle(self.screen, theme.TEXT_DIM, (b.x + 10, scr_y + surf.get_height() // 2),
                               int(self.fs * 0.45), 1)
            self.screen.blit(surf, (b.x + 22, scr_y))

    def _draw_banner(self) -> None:
        L = self.layout
        title, sub = getattr(self, "banner", ("", ""))
        b = L.board
        w, h = int(b.w * 0.55), int(b.h * 0.22)
        rect = pygame.Rect(0, 0, w, h)
        rect.center = b.center
        s = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(s, (15, 18, 24, 225), s.get_rect(), border_radius=18)
        pygame.draw.rect(s, (*theme.ACCENT, 255), s.get_rect(), 3, border_radius=18)
        self.screen.blit(s, rect.topleft)
        t1 = theme.font(int(h * 0.28), bold=True).render(title, True, theme.ACCENT)
        t2 = theme.font(int(h * 0.16)).render(sub, True, theme.TEXT)
        self.screen.blit(t1, (rect.centerx - t1.get_width() // 2, rect.y + h * 0.16))
        self.screen.blit(t2, (rect.centerx - t2.get_width() // 2, rect.y + h * 0.58))

    def _card(self, y: int, h: int, title: str) -> pygame.Rect:
        r = pygame.Rect(self.panel_x, y, self.panel_w_inner, h)
        pygame.draw.rect(self.screen, theme.PANEL_CARD, r, border_radius=10)
        pygame.draw.rect(self.screen, theme.PANEL_BORDER, r, 1, border_radius=10)
        t = theme.font(int(self.fs * 0.9), bold=True).render(title, True, theme.ACCENT)
        self.screen.blit(t, (r.x + 10, r.y + 6))
        return r

    def _draw_panel(self) -> None:
        L = self.layout
        scr = self.screen
        p = L.panel
        pygame.draw.rect(scr, theme.PANEL_BG, p)
        pygame.draw.line(scr, theme.PANEL_BORDER, p.topleft, p.bottomleft, 2)
        title = theme.font(int(self.fs * 1.35), bold=True).render("Табла · TD(λ) агент", True, theme.TEXT)
        scr.blit(title, (self.panel_x, p.y + int(self.fs * 0.7)))

        self._update_buttons()
        for sel in self.selectors:
            if sel is self.sel_b and self.sel_mode.value != "ava":
                continue
            sel.draw(scr, self.fs)
        for b in self.buttons:
            b.draw(scr, int(self.fs * 0.95))
        self.slider.draw(scr, self.fs)

        fs = self.fs
        line = int(fs * 1.35)
        y = self.info_top

        # status message
        msg_lines = self._wrap(self.message, theme.font(int(fs * 0.9)), self.panel_w_inner - 20)
        r = self._card(y, 10 + line * (len(msg_lines) + 1), "Състояние")
        for i, ln in enumerate(msg_lines):
            scr.blit(theme.font(int(fs * 0.9)).render(ln, True, theme.TEXT), (r.x + 10, r.y + 6 + line * (i + 1)))
        y = r.bottom + 10

        # how the agent thinks
        h = int(line * 5.6 + fs * 1.5 + 18)
        r = self._card(y, h, "Как мисли агентът")
        bar = pygame.Rect(r.x + 10, r.y + line + 8, r.w - 20, int(fs * 1.5))
        if self.win_white is not None:
            pw = self.win_white
            wpx = int(bar.w * pw)
            pygame.draw.rect(scr, theme.BLACK_BODY, bar, border_radius=6)
            pygame.draw.rect(scr, theme.WHITE_BODY, pygame.Rect(bar.x, bar.y, wpx, bar.h),
                             border_top_left_radius=6, border_bottom_left_radius=6)
            pygame.draw.rect(scr, theme.PANEL_BORDER, bar, 1, border_radius=6)
            f = theme.font(int(fs * 0.85), bold=True)
            t1 = f.render(f"Бели {100 * pw:.1f}%", True, (40, 30, 28))
            t2 = f.render(f"Черни {100 * (1 - pw):.1f}%", True, (240, 232, 215))
            if wpx > t1.get_width() + 8:
                scr.blit(t1, (bar.x + 6, bar.centery - t1.get_height() // 2))
            if bar.w - wpx > t2.get_width() + 8:
                scr.blit(t2, (bar.right - t2.get_width() - 6, bar.centery - t2.get_height() // 2))
            cap = theme.font(int(fs * 0.75)).render("вероятност за победа според TD мрежата", True, theme.TEXT_DIM)
            scr.blit(cap, (bar.x, bar.bottom + 2))
        else:
            t = theme.font(int(fs * 0.85)).render("няма модел - обучете агента", True, theme.TEXT_DIM)
            scr.blit(t, (bar.x, bar.y))
        yy = bar.bottom + int(line * 0.95)
        a = self.analysis
        f = theme.font(int(fs * 0.85))
        if a is not None:
            head = f"{COLOR_SHORT[a.player]} ({a.agent_name}), зарове {a.dice[0]}-{a.dice[1]}:"
            scr.blit(f.render(head, True, theme.TEXT_DIM), (r.x + 10, yy))
            yy += line
            for i, (lbl, sc) in enumerate(a.top):
                if a.agent_name == "Random":
                    val = "случаен"
                elif sc != sc:  # NaN
                    val = "единствен"
                elif a.is_prob:
                    val = f"{100 * sc:.1f}%"
                else:
                    val = f"{sc:.2f}"
                col = theme.GOOD if i == 0 else theme.TEXT
                scr.blit(theme.font(int(fs * 0.9), bold=i == 0).render(f"{i + 1}. {lbl}", True, col),
                         (r.x + 10, yy))
                vt = theme.font(int(fs * 0.9), bold=True).render(val, True, col)
                scr.blit(vt, (r.right - vt.get_width() - 10, yy))
                yy += line
        elif self.thinking:
            scr.blit(f.render("мисли...", True, theme.TEXT_DIM), (r.x + 10, yy))
        y = r.bottom + 10

        # score + pips
        r = self._card(y, line * 3 + 8, "Резултат")
        if self.human is not None or self.mode == "hva":
            s = f"Вие {self.score[0]} : {self.score[1]} Агент"
        else:
            s = f"Бели {self.score[0]} : {self.score[1]} Черни"
        scr.blit(theme.font(int(fs * 1.05), bold=True).render(s, True, theme.TEXT), (r.x + 10, r.y + line))
        pips = f"Пипове: бели {self.state.pip_count(WHITE)} · черни {self.state.pip_count(BLACK)}"
        scr.blit(theme.font(int(fs * 0.85)).render(pips, True, theme.TEXT_DIM), (r.x + 10, r.y + 2 * line + 2))
        y = r.bottom + 10

        # history
        h = L.height - y - 12
        if h > line * 2:
            r = self._card(y, h, "История на ходовете")
            rows = max(1, (h - line - 8) // int(fs * 1.2))
            hist = self.history
            self.history_scroll = min(self.history_scroll, max(0, len(hist) - rows))
            end = len(hist) - self.history_scroll
            visible = hist[max(0, end - rows):end]
            hf = theme.font(int(fs * 0.8))
            for i, ln in enumerate(visible):
                col = theme.TEXT if i == len(visible) - 1 and self.history_scroll == 0 else theme.TEXT_DIM
                t = hf.render(ln, True, col)
                if t.get_width() > r.w - 20:
                    t = t.subsurface(pygame.Rect(0, 0, r.w - 20, t.get_height()))
                scr.blit(t, (r.x + 10, r.y + line + 4 + i * int(fs * 1.2)))

    @staticmethod
    def _wrap(text: str, fnt, width: int) -> list[str]:
        words = text.split()
        lines, cur = [], ""
        for w in words:
            cand = (cur + " " + w).strip()
            if fnt.size(cand)[0] <= width:
                cur = cand
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines or [""]

    # ================================================================= loop
    def step(self, dt: float) -> None:
        self.t += dt
        self._poll_thinking()
        self._update_anims()

    def run(self) -> None:
        running = True
        while running:
            dt = self.clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if not self.handle_event(event):
                    running = False
            self.step(dt)
            self.draw()
            pygame.display.flip()
        pygame.quit()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Backgammon GUI (pygame)")
    ap.add_argument("--model", default=None, help="checkpoint to use (default: checkpoints/td_final.pt)")
    ap.add_argument("--opponent", choices=[v for _, v in AGENT_LABELS], default="td")
    ap.add_argument("--mode", choices=["hva", "ava"], default="hva",
                    help="hva = human vs agent, ava = agent vs agent")
    ap.add_argument("--size", default="1440x880", help="window size, e.g. 1280x800")
    args = ap.parse_args(argv)
    w, h = (int(v) for v in args.size.lower().split("x"))
    App((w, h), model=args.model, opponent=args.opponent, mode=args.mode).run()
