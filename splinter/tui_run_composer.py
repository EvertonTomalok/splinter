"""Targeted, delivery-aware directives for the live run screen."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime

from rich.markup import escape
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Input

from splinter.memory.session import Session
from splinter.tui_run_palette import AMBER, DIM, INK, LIME, MUTED, Ink
from splinter.tui_run_rail_parts import chip
from splinter.tui_run_text import clip

_PLACEHOLDERS = {
    "story": "e.g. keep the public API unchanged",
    "all": "e.g. never touch migrations/",
    "orch": "e.g. skip US-031, run US-040 next",
}


@dataclass(frozen=True)
class SentMessage:
    target: str
    text: str
    ids: tuple[str, ...]
    sent_at: datetime
    delivered_at: datetime | None = None


def _status(message: SentMessage) -> str:
    """Where a sent message is: waiting for its reader, or read."""
    orchestrator = message.target == "orchestrator only"
    if message.delivered_at is None:
        waiting = "sending to the orchestrator…"
        return f"[{AMBER}]{waiting}[/]"
    done = (
        "✓ with the orchestrator · it decides where it applies"
        if orchestrator
        else "✓ with the orchestrator · applied to the story's next step"
    )
    return f"[{LIME}]{done}[/]"


class ComposerInput(Input):
    BINDINGS = [Binding("tab", "cycle_target", priority=True)]

    def action_cycle_target(self) -> None:
        composer = self.parent
        if isinstance(composer, Composer):
            composer.cycle_target()


class Composer(Vertical):
    """Compose guidance for one story, every active story, or the orchestrator."""

    DEFAULT_CSS = """
    Composer { height: auto; }
    Composer #composer-msgs { height: auto; }
    Composer #composer-chips { height: 1; }
    Composer #composer-row { height: 1; margin-top: 1; }
    Composer #composer-prompt { width: 2; color: $lime; }
    Composer #user-cmd-input {
        width: 1fr;
        height: 1;
        border: none;
        padding: 0;
        background: transparent;
        color: $ink;
    }
    Composer #user-cmd-input:focus { border: none; background: transparent; }
    Composer #user-cmd-input > .input--placeholder { color: $dim; }
    Composer #composer-send { width: auto; height: 1; }
    """

    TARGETS = ("story", "all", "orch")

    def __init__(self, *, story: int = 1) -> None:
        super().__init__(id="composer")
        self.target = "story"
        self.story = story
        self.story_label = f"T{story}"
        self.live = True
        self.messages: list[SentMessage] = []

    def compose(self) -> ComposeResult:
        yield Ink(id="composer-msgs")
        yield Ink(id="composer-chips")
        with Horizontal(id="composer-row"):
            yield Ink("›", id="composer-prompt")
            yield ComposerInput(id="user-cmd-input", placeholder=_PLACEHOLDERS["story"])
            yield Ink(
                chip(f"[{INK}]send[/] [{LIME}]↵[/]", False, "[@click=app.composer_send]"),
                id="composer-send",
            )

    def on_mount(self) -> None:
        self._render_all()

    def on_resize(self) -> None:
        self._render_chips()

    def set_story(self, story: int | None, label: str = "", live: bool = True) -> None:
        if story is None:
            return
        shown = label or f"T{story}"
        if (story, shown, live) == (self.story, self.story_label, self.live):
            return
        self.story, self.story_label, self.live = story, shown, live
        self._render_chips()
        if self.target == "story":
            self.select_target("story")

    def select_target(self, target: str) -> None:
        if target not in self.TARGETS:
            return
        self.target = target
        self._render_chips()
        placeholder = _PLACEHOLDERS[target]
        if target == "story" and not self.live:
            placeholder = f"note for {self.story_label} when it runs"
        self.query_one("#user-cmd-input", ComposerInput).placeholder = placeholder

    def cycle_target(self) -> None:
        index = self.TARGETS.index(self.target)
        self.select_target(self.TARGETS[(index + 1) % len(self.TARGETS)])

    def prefill(self, target: str, text: str) -> None:
        self.select_target(target)
        input_widget = self.query_one("#user-cmd-input", ComposerInput)
        input_widget.value = text
        input_widget.focus()

    def record(
        self, target: str, text: str, ids: tuple[str, ...], sent_at: datetime
    ) -> SentMessage:
        message = SentMessage(target, text, ids, sent_at)
        self.messages.append(message)
        self._render_messages()
        return message

    def poll(self, session: Session) -> tuple[SentMessage, ...]:
        delivered: list[SentMessage] = []
        updated: list[SentMessage] = []
        now = datetime.now().astimezone()
        for message in self.messages:
            is_delivered = all(session.receipt_exists(message_id) for message_id in message.ids)
            if message.delivered_at is None and is_delivered:
                message = replace(message, delivered_at=now)
                delivered.append(message)
            updated.append(message)
        if delivered:
            self.messages = updated
            self._render_messages()
        return tuple(delivered)

    def _render_all(self) -> None:
        self._render_messages()
        self._render_chips()

    def _render_messages(self) -> None:
        visible = self.messages[-2:]
        older = len(self.messages) - len(visible)
        lines = [f"[{DIM}]+{older} earlier[/]"] if older else []
        for message in visible:
            target = self._shown_target(message.target)
            status = _status(message)
            lines.append(
                f"[{LIME}]you[/] [{DIM}]→ {target}[/]  {clip(message.text, 40)}  {status}"
            )
        self.query_one("#composer-msgs", Ink).update("\n".join(lines))

    def _shown_target(self, target: str) -> str:
        return self.story_label if target == f"T{self.story}" else target

    def _render_chips(self) -> None:
        story = self.story_label + (" · next iteration" if self.live else "")
        labels = {"story": story, "all": "all agents", "orch": "orchestrator only"}
        room = sum(len(label) + 5 for label in labels.values()) + 21
        caption = "to orchestrator, for" if room <= (self.size.width or 200) else "for"
        chips = [f"[{MUTED}]{caption}[/]"]
        for target in self.TARGETS:
            go = f"[@click=app.composer_target('{target}')]"
            chips.append(chip(escape(labels[target]), target == self.target, go))
        self.query_one("#composer-chips", Ink).update(" ".join(chips))
