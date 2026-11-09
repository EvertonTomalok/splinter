from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from contextvars import ContextVar
from itertools import groupby
from queue import Empty, Full, Queue
from typing import Any

from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.geometry import Region
from textual.strip import Strip
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import Button, RichLog, Static

from splinter.arcade import GAMES, HEIGHT, WIDTH, Arcade
from splinter.tui_naval import NavalLayout, NavalTarget, naval_rows
from splinter.tui_run_palette import AMBER, INK, LIME, MUTED

_job: ContextVar[object | None] = ContextVar("waiting_job", default=None)
_FIELD = "#10150e"
_COLORS = (_FIELD, LIME, AMBER, "#34412a", "#9ca98d")
_PIXEL_STYLES = {
    (top, bottom): Style(color=_COLORS[top], bgcolor=_COLORS[bottom])
    for top in range(len(_COLORS))
    for bottom in range(len(_COLORS))
}


class ActivityHandler(logging.Handler):
    def __init__(self, owner: object, queue: Queue[logging.LogRecord]) -> None:
        super().__init__(logging.INFO)
        self.owner = owner
        self.queue = queue

    def emit(self, record: logging.LogRecord) -> None:
        if _job.get() is not self.owner:
            return
        try:
            self.queue.put_nowait(record)
        except Full:
            pass


class ArcadeBoard(Widget):
    DEFAULT_CSS = """
    ArcadeBoard { width: 64; height: 12; align-horizontal: center; }
    """

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self.scale = 1
        self._rows: list[Strip] = []
        self._row_keys: list[tuple[object, ...]] = []
        self._naval = False

    def render_line(self, y: int) -> Strip:
        if 0 <= y < len(self._rows):
            return self._rows[y]
        return Strip.blank(self.size.width, Style(bgcolor=_FIELD))

    def draw(self, game: Arcade, prompt: str = "") -> None:
        self._naval = game.kind == "naval"
        if self._naval:
            self._rows = naval_rows(game, WIDTH * self.scale, HEIGHT * self.scale // 2)
            self._row_keys = [()] * len(self._rows)
            self.refresh()
            return
        pixels = game.pixels(self.scale)
        width = WIDTH * self.scale
        height = HEIGHT * self.scale // 2
        if len(self._rows) != height:
            self._rows = [Strip.blank(width)] * height
            self._row_keys = [()] * height
        dirty: list[Region] = []
        for y in range(0, HEIGHT * self.scale, 2):
            top_row = pixels[y]
            bottom_row = pixels[y + 1]
            label = f"  {prompt}  " if prompt and y == 6 * self.scale else ""
            key = (tuple(top_row), tuple(bottom_row), self.scale, label)
            if self._row_keys[y // 2] == key:
                continue
            self._row_keys[y // 2] = key
            segments: list[Segment] = []
            pairs = zip(top_row, bottom_row)
            for pair, group in groupby(pairs):
                count = sum(1 for _ in group)
                segments.append(
                    Segment((" " if pair[0] == pair[1] else "▀") * count, _PIXEL_STYLES[pair])
                )
            row = Strip(segments, width)
            if label:
                left = (width - len(label)) // 2
                row = Strip.join(
                    (
                        row.crop(0, left),
                        Strip([Segment(label, Style(color=INK, bgcolor=_FIELD, bold=True))]),
                        row.crop(left + len(label), width),
                    )
                )
            self._rows[y // 2] = row
            dirty.append(Region(0, y // 2, width, 1))
        if dirty:
            self.refresh(*dirty)

    def _target(self, event: events.MouseEvent, *, fire: bool) -> None:
        if not self._naval:
            return
        offset = event.get_content_offset(self)
        if offset is None:
            return
        cell = NavalLayout.for_size(self.size.width, self.size.height).cell_at(offset.x, offset.y)
        if cell is not None:
            event.stop()
            self.post_message(NavalTarget(cell, fire=fire))

    def on_click(self, event: events.Click) -> None:
        if event.button == 1:
            self._target(event, fire=True)

    def on_mouse_move(self, event: events.MouseMove) -> None:
        self._target(event, fire=False)


#: Rows the game column needs around the board: tabs, heading, controls, actions.
_GAME_CHROME_ROWS = 8
#: Narrowest log column worth putting beside the game.
_SIDE_LOG_MIN = 44


def board_layout(width: int, height: int) -> tuple[int, bool]:
    """The board scale and whether the log goes beside the game.

    Beside, the game takes the largest scale whose board fits the height and
    still leaves ``_SIDE_LOG_MIN`` columns of log. Too narrow for that, the log
    goes under the game as before.
    """
    for scale in (3, 2, 1):
        board_width, board_height = WIDTH * scale, HEIGHT * scale // 2
        if (
            width >= board_width + _SIDE_LOG_MIN + 8
            and height >= board_height + _GAME_CHROME_ROWS
        ):
            return scale, True
    return (2 if width >= 134 and height >= 39 else 1), False


class WaitingRoom(Vertical, can_focus=True):
    DEFAULT_CSS = """
    WaitingRoom { height: 1fr; padding: 1 2 0 2; background: #0a0a09;
        align-horizontal: center; }
    WaitingRoom #arcade { width: 128; max-width: 100%; height: auto; }
    WaitingRoom #game-nav { height: 1; }
    WaitingRoom #game-heading { height: 1; }
    WaitingRoom #game-title { width: 1fr; height: 1; color: #ecebe4; text-style: bold; }
    WaitingRoom #game-score { width: auto; height: 1; color: #c6f432; text-style: bold; }
    WaitingRoom #game-board-wrap { height: 12; align-horizontal: center; }
    WaitingRoom #game-controls { height: 1; color: #a6a59c; margin-top: 1; }
    WaitingRoom #game-state { display: none; }
    WaitingRoom #wait-actions { width: 128; max-width: 100%; height: 1; margin: 1 0; }
    WaitingRoom Button { height: 1; min-height: 1; width: auto; min-width: 0;
        border: none; margin: 0 1 0 0; padding: 0 1; background: #1a2116;
        color: #ecebe4; text-style: none; }
    WaitingRoom Button:hover { background: #34412a; color: #ffffff; }
    WaitingRoom Button:focus { color: #c6f432; text-style: bold reverse; }
    WaitingRoom #wait-play { background: #c6f432; color: #0a0a09; text-style: bold; }
    WaitingRoom #wait-body { height: 1fr; align-horizontal: center; }
    WaitingRoom #wait-game { width: auto; height: auto; }
    WaitingRoom #wait-activity { width: 128; max-width: 100%; height: 1fr;
        min-height: 5; border-top: solid #3a3a35; }
    WaitingRoom.-side #wait-body { layout: horizontal; align-horizontal: left; }
    WaitingRoom.-side #wait-game { height: 100%; }
    WaitingRoom.-side #wait-activity { width: 1fr; max-width: 100%; height: 100%;
        border-top: none; border-left: solid #3a3a35; padding: 0 0 0 2; margin-left: 2; }
    WaitingRoom #wait-title { height: auto; color: #ecebe4; text-style: bold; }
    WaitingRoom #wait-meta { height: auto; color: #a6a59c; }
    WaitingRoom #wait-size { height: auto; color: #a6a59c; }
    WaitingRoom #wait-latest { display: none; }
    WaitingRoom #wait-log { height: 1fr; min-height: 1; padding: 0;
        scrollbar-size-vertical: 1; scrollbar-size-horizontal: 0; }
    WaitingRoom.-compact { padding: 0 1; }
    WaitingRoom.-compact #wait-actions { margin: 0; }
    WaitingRoom.-compact #game-nav, WaitingRoom.-compact #game-heading { margin-bottom: 0; }
    WaitingRoom.-compact #game-controls { margin-top: 0; }
    WaitingRoom.-compact #wait-activity { padding-top: 0; min-height: 4; }
    WaitingRoom.-compact #wait-log { margin-top: 0; }
    WaitingRoom.-tiny #arcade { display: none; }
    """

    def __init__(self) -> None:
        super().__init__(id="waiting-room")
        self.display = False
        self.game = Arcade(random.choice(tuple(GAMES)))
        self.active = False
        self.playing = False
        self.logs_only = False
        self.action = ""
        self.model = ""
        self._started = 0.0
        self._last_event = 0.0
        self._last_meta = ""
        self._queue: Queue[logging.LogRecord] = Queue(maxsize=256)
        self._held: list[Text] = []
        self._handler: ActivityHandler | None = None
        self._levels: dict[str, int] = {}
        self._clock: Timer | None = None
        self._frames: Timer | None = None
        self._frame_at = 0.0
        self._hud_key: tuple[object, ...] = ()

    def compose(self) -> ComposeResult:
        with Vertical(id="wait-body"):
            with Vertical(id="wait-game"):
                yield from self._game_widgets()
            with Vertical(id="wait-activity"):
                yield Static(id="wait-title")
                yield Static(id="wait-meta")
                yield Static(id="wait-latest")
                yield RichLog(
                    id="wait-log",
                    wrap=True,
                    markup=False,
                    max_lines=300,
                    auto_scroll=True,
                    min_width=1,
                )

    def _game_widgets(self) -> ComposeResult:
        with Vertical(id="arcade"):
            with Horizontal(id="game-nav"):
                for index, (kind, name) in enumerate(GAMES.items(), 1):
                    yield Button(f"{index} {name}", id=f"game-{kind}", compact=True)
            with Horizontal(id="game-heading"):
                yield Static(id="game-title")
                yield Static(id="game-score")
            with Horizontal(id="game-board-wrap"):
                yield ArcadeBoard(id="game-board")
            yield Static(id="game-controls")
            yield Static(id="game-state")
        with Horizontal(id="wait-actions"):
            yield Button("Space play", id="wait-play", compact=True, variant="primary")
            yield Button("N shuffle", id="wait-next", compact=True)
            yield Button("G logs only", id="wait-logs", compact=True)
            yield Button("Cancel generation", id="wait-cancel", compact=True)
        size_hint = Static("Game needs a larger terminal · showing activity", id="wait-size")
        size_hint.display = False
        yield size_hint

    def on_mount(self) -> None:
        for button in self.query(Button):
            button.active_effect_duration = 0

    def start(self, action: str, model: str) -> None:
        self.action, self.model = action, model
        if not self.active:
            self.active = True
            self.display = True
            self._started = self._last_event = time.monotonic()
            self._queue = Queue(maxsize=256)
            self._held.clear()
            self.query_one("#wait-log", RichLog).clear()
            self._handler = ActivityHandler(self, self._queue)
            for name in ("splinter", "splinter.live"):
                logger = logging.getLogger(name)
                self._levels[name] = logger.level
                logger.setLevel(logging.INFO)
            logging.getLogger("splinter").addHandler(self._handler)
            self._clock = self.set_interval(0.25, self._poll)
            logs_only = self.logs_only
            self.choose_game(random.choice(tuple(GAMES)))
            if logs_only:
                self.toggle_logs()
        self.query_one("#wait-title", Static).update(Text(action[:1].upper() + action[1:]))
        self.note(action)
        self._poll()
        self._focus_content()

    def stop(self) -> None:
        self.active = False
        self.display = False
        self._pause()
        if self._clock:
            self._clock.stop()
            self._clock = None
        if self._handler:
            logging.getLogger("splinter").removeHandler(self._handler)
            self._handler.close()
            self._handler = None
        for name, level in self._levels.items():
            logging.getLogger(name).setLevel(level)
        self._levels.clear()

    def on_unmount(self) -> None:
        self.stop()

    def run_job(self, fn: Callable[..., None], **kw: Any) -> None:
        token = _job.set(self)
        try:
            fn(**kw)
        finally:
            _job.reset(token)

    def note(self, message: str, level: int = logging.INFO) -> None:
        self._last_event = time.monotonic()
        message = " ".join(Text.from_ansi(message).plain.split())[:1200]
        elapsed = int(self._last_event - self._started)
        color = AMBER if level >= logging.WARNING else INK
        row = Text(f"{elapsed // 60:02}:{elapsed % 60:02}  ", style=MUTED)
        row.append(message, style=color)
        self._held.append(row)
        self._flush_rows()
        if self._held:
            self.call_after_refresh(self._flush_rows)
        self.query_one("#wait-latest", Static).update(Text(message, style=INK))

    def _flush_rows(self) -> None:
        """Write the held rows once the log has a width: a row is wrapped when written."""
        log_view = self.query_one("#wait-log", RichLog)
        if log_view.scrollable_content_region.width <= 1:
            return
        for row in self._held:
            log_view.write(row)
        self._held.clear()

    def _poll(self) -> None:
        self._flush_rows()
        for _ in range(64):
            try:
                record = self._queue.get_nowait()
            except Empty:
                break
            self.model = getattr(record, "waiting_model", self.model)
            self.note(record.getMessage(), record.levelno)
        elapsed = int(time.monotonic() - self._started)
        quiet = int(time.monotonic() - self._last_event)
        meta = f"{self.model} · {elapsed // 60:02}:{elapsed % 60:02} elapsed"
        if quiet >= 10:
            meta += f" · No new events for {quiet}s; waiting for provider"
        if meta != self._last_meta:
            self.query_one("#wait-meta", Static).update(Text(meta))
            self._last_meta = meta
        if self.playing and (self.app.screen is not self.screen or not self.has_focus_within):
            self._pause()

    def choose_game(self, kind: str) -> None:
        self._pause()
        self.game = Arcade(kind)
        self.logs_only = False
        self.query_one("#arcade").display = not self.has_class("-tiny")
        self.query_one("#wait-play").display = not self.has_class("-tiny")
        self.query_one("#wait-logs", Button).label = "G logs only"
        self._draw()
        self.focus()

    def _draw(self) -> None:
        state = "" if self.playing else "SPACE TO PLAY"
        if self.game.over:
            state = f"{self.game.result.upper()} · SPACE TO RESTART"
        self.query_one("#game-board", ArcadeBoard).draw(self.game, state)
        hud_key = (self.game.kind, self.game.status, self.game.controls, state, self.playing)
        if hud_key == self._hud_key:
            return
        self._hud_key = hud_key
        self.query_one("#game-title", Static).update(GAMES[self.game.kind].upper())
        self.query_one("#game-score", Static).update(self.game.status)
        self.query_one("#game-controls", Static).update(self.game.controls)
        self.query_one("#game-state", Static).update(state)
        self.query_one("#wait-play", Button).label = "P pause" if self.playing else "Space play"
        for kind, name in GAMES.items():
            button = self.query_one(f"#game-{kind}", Button)
            button.styles.color = LIME if kind == self.game.kind else MUTED

    def _play(self) -> None:
        if self.game.over:
            self.game = Arcade(self.game.kind)
        self.playing = True
        self._frame_at = time.monotonic()
        if self.game.kind != "naval" and self._frames is None:
            self._frames = self.set_interval(1 / 120, self._frame)
        self._draw()

    def on_naval_target(self, event: NavalTarget) -> None:
        event.stop()
        if self.game.kind != "naval" or self.game.over or self.logs_only or not self.active:
            return
        if not event.fire and event.cell == self.game.cursor:
            return
        self.game.cursor = event.cell
        if event.fire:
            self.focus()
            if not self.playing:
                self._play()
            self.game.key("space")
            if self.game.over:
                self._pause()
        self._draw()

    def _pause(self) -> None:
        self.playing = False
        if self._frames:
            self._frames.stop()
            self._frames = None
        if self.is_mounted and self.active:
            self._draw()

    def _frame(self) -> None:
        if self.app.screen is not self.screen or not self.has_focus_within:
            self._pause()
            return
        now = time.monotonic()
        self.game.tick(min(0.1, now - self._frame_at))
        self._frame_at = now
        if self.game.over:
            self._pause()
        self._draw()

    def toggle_logs(self) -> None:
        self._pause()
        self.logs_only = not self.logs_only
        self.query_one("#arcade").display = not self.logs_only and not self.has_class("-tiny")
        self.query_one("#wait-play").display = not self.logs_only and not self.has_class("-tiny")
        self.query_one("#wait-logs", Button).label = (
            "G show game" if self.logs_only else "G logs only"
        )
        self._focus_content()

    def _focus_content(self) -> None:
        if self.logs_only or self.has_class("-tiny"):
            self.query_one("#wait-log", RichLog).focus()
        else:
            self.focus()

    def on_resize(self, event: events.Resize) -> None:
        self.set_class(event.size.width < 90 or event.size.height < 39, "-compact")
        tiny = event.size.width < 66 or event.size.height < 21
        self.set_class(tiny, "-tiny")
        self.query_one("#arcade").display = not tiny and not self.logs_only
        self.query_one("#wait-play").display = not tiny and not self.logs_only
        self.query_one("#wait-size").display = tiny
        scale, side = board_layout(event.size.width, event.size.height)
        self.set_class(side and not tiny, "-side")
        board = self.query_one("#game-board", ArcadeBoard)
        board.scale = scale
        board.styles.width = WIDTH * scale
        board.styles.height = HEIGHT * scale // 2
        self.query_one("#game-board-wrap").styles.height = HEIGHT * scale // 2
        frame_width = WIDTH * scale
        for selector in ("#arcade", "#wait-actions"):
            self.query_one(selector).styles.width = frame_width
        self.query_one("#wait-activity").styles.width = "1fr" if side else frame_width
        self._draw()
        if tiny:
            self._pause()

    def on_key(self, event: events.Key) -> None:
        key = event.key
        if key in {"escape", "ctrl+c", "tab", "shift+tab"}:
            return
        if (self.logs_only or self.has_class("-tiny")) and key in {
            "up",
            "down",
            "pageup",
            "pagedown",
            "home",
            "end",
        }:
            return
        event.stop()
        event.prevent_default()
        if key in {"1", "2", "3", "4"}:
            self.choose_game(tuple(GAMES)[int(key) - 1])
        elif key == "n":
            self.choose_game(random.choice([kind for kind in GAMES if kind != self.game.kind]))
        elif key == "g" or key == "q" and not self.logs_only:
            self.toggle_logs()
        elif key == "p":
            if self.playing:
                self._pause()
            elif not self.logs_only and not self.has_class("-tiny"):
                self._play()
        elif not self.logs_only and not self.has_class("-tiny"):
            if key == "space" and not self.playing:
                self._play()
            elif self.playing:
                self.game.key(key)
                if self.game.over:
                    self._pause()
                self._draw()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        button_id = event.button.id or ""
        if button_id.startswith("game-"):
            self.choose_game(button_id.removeprefix("game-"))
        elif button_id == "wait-play":
            self.focus()
            if self.playing:
                self._pause()
            else:
                self._play()
        elif event.button.id == "wait-next":
            self.choose_game(random.choice([kind for kind in GAMES if kind != self.game.kind]))
        elif event.button.id == "wait-logs":
            self.toggle_logs()
        elif event.button.id == "wait-cancel":
            self._pause()
            await self.app.run_action("abort")
