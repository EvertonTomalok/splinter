"""Stop menu for the live run screen.

One modal, three ways out of a running session: pause after the current
iteration, restart from the last checkpoint, or kill the run and exit. The
module owns no pipeline behaviour — it dismisses with a string the host app
maps onto its own actions, so any app can reuse it.

Dismiss values: ``pause`` | ``restart`` | ``kill_exit`` | ``None`` (cancel).
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from splinter.tui_run_palette import Ink

PAUSE = "pause"
RESTART = "restart"
KILL_EXIT = "kill_exit"


class StopMenuModal(ModalScreen[str | None]):
    """Ask what should happen to the running session.

    Dismiss value: ``"pause"``, ``"restart"``, ``"kill_exit"`` or ``None`` when
    the user cancels (esc / the Cancel button).
    """

    DEFAULT_CSS = """
    StopMenuModal {
        align: center middle;
        background: $background 60%;
    }
    StopMenuModal > Vertical#stop-menu {
        width: 64;
        height: auto;
        border: round $warning;
        background: $surface;
        padding: 1 2;
    }
    StopMenuModal #stop-title {
        text-style: bold;
        color: $warning;
        margin-bottom: 1;
    }
    StopMenuModal #stop-hint {
        margin-bottom: 1;
    }
    StopMenuModal Button {
        width: 1fr;
        height: 3;
        margin: 0 0 1 0;
        border: none;
        text-style: bold;
    }
    """

    BINDINGS = [
        Binding("p", "choose_pause", "Pause"),
        Binding("r", "choose_restart", "Restart"),
        Binding("k", "choose_kill", "Kill"),
        Binding("escape", "choose_cancel", "Cancel"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="stop-menu"):
            yield Static("Stop the run", id="stop-title")
            yield Static(
                "[dim]What should happen to this session?[/]",
                id="stop-hint",
            )
            yield Button("Pause after this iteration  (p)", id="stop-pause", variant="warning")
            yield Button("Restart from checkpoint  (r)", id="stop-restart")
            yield Button("Kill & exit  (k)", id="stop-kill", variant="error")
            yield Button("Cancel  (esc)", id="stop-cancel")

    def on_mount(self) -> None:
        self.query_one("#stop-pause", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        choices = {
            "stop-pause": PAUSE,
            "stop-restart": RESTART,
            "stop-kill": KILL_EXIT,
        }
        self.dismiss(choices.get(event.button.id or ""))

    def action_choose_pause(self) -> None:
        self.dismiss(PAUSE)

    def action_choose_restart(self) -> None:
        self.dismiss(RESTART)

    def action_choose_kill(self) -> None:
        self.dismiss(KILL_EXIT)

    def action_choose_cancel(self) -> None:
        self.dismiss(None)


class StopBar(Horizontal):
    can_focus = True

    DEFAULT_CSS = """
    StopBar {
        display: none;
        height: 1;
        margin-top: 1;
        padding: 0 3;
        background: #c6f432;
        color: #0a0a09;
    }
    StopBar Static { width: auto; height: 1; }
    StopBar #stop-fill { width: 1fr; }
    """

    BINDINGS = [
        Binding("p", "choose('pause')", "Pause"),
        Binding("r", "choose('restart')", "Restart"),
        Binding("k", "choose('kill_exit')", "Kill"),
        Binding("escape", "choose_none", "Keep running"),
    ]

    class Chosen(Message):
        def __init__(self, result: str | None) -> None:
            super().__init__()
            self.result = result

    def compose(self) -> ComposeResult:
        dark, kill = "#0a0a09", "#8a1408"

        def key(letter: str, label: str, action: str, tone: str = dark) -> Ink:
            text = f"[@click={action}][b underline {tone}]{letter}[/] [{tone}]{label}[/][/]   "
            return Ink(text)

        yield Ink("[b]Stop how?[/]    ")
        yield key("p", "pause after this iteration", "choose('pause')")
        yield key("r", "restart from checkpoint", "choose('restart')")
        yield key("k", "kill & exit", "choose('kill_exit')", kill)
        yield Static(id="stop-fill")
        yield key("esc", "keep going", "choose_none()")

    def action_choose(self, result: str) -> None:
        self.post_message(self.Chosen(result))

    def action_choose_none(self) -> None:
        self.post_message(self.Chosen(None))
