from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.timer import Timer
from textual.widgets import Button, Static

from splinter.tui_shell import SHELL_CSS, use_splinter_theme
from splinter.tui_waiting import WaitingRoom

_DEMO_EVENTS = (
    "[SIMULATED] Reading project instructions",
    "[SIMULATED] Finding files related to the task",
    "[SIMULATED] Drafting the implementation plan",
    "[SIMULATED] Checking acceptance criteria",
    "[SIMULATED] Plan ready. Demo continues until you exit.",
)


class ArcadeDemoApp(App[None]):
    CSS = SHELL_CSS + """
    Screen { background: #0a0a09; }
    #demo-heading {
        height: 3; padding: 1 2 0 2; color: #c6f432;
        border-bottom: solid #1d1d1a;
    }
    """
    BINDINGS = [
        Binding("escape", "abort", "Exit demo", priority=True),
        Binding("ctrl+c", "abort", "Exit demo", priority=True),
    ]

    def __init__(self, game: str | None = None) -> None:
        super().__init__()
        self.initial_game = game
        self._event_index = 0
        self._demo_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        yield Static("splinter / arcade demo · offline · Esc exits", id="demo-heading")
        yield WaitingRoom()

    def on_mount(self) -> None:
        self.title = "splinter · offline arcade demo"
        use_splinter_theme(self)
        room = self.query_one(WaitingRoom)
        room.start("Demo — simulated planning activity", "local demo · no agents")
        if self.initial_game:
            room.choose_game(self.initial_game)
        room.query_one("#wait-cancel", Button).label = "Exit demo"
        self._next_event()
        self._demo_timer = self.set_interval(4, self._next_event)

    def _next_event(self) -> None:
        self.query_one(WaitingRoom).note(_DEMO_EVENTS[self._event_index % len(_DEMO_EVENTS)])
        self._event_index += 1

    def action_abort(self) -> None:
        if self._demo_timer:
            self._demo_timer.stop()
            self._demo_timer = None
        self.query_one(WaitingRoom).stop()
        self.exit()

    def on_unmount(self) -> None:
        if self._demo_timer:
            self._demo_timer.stop()
