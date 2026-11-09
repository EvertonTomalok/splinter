from __future__ import annotations

from typing import Any

from rich.markup import escape
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Input, Static

from splinter.run_view import StoryRow
from splinter.tui_questions import Option, parse_questions
from splinter.tui_run_palette import RED
from splinter.tui_run_text import story_label


def _options(reason: str) -> tuple[Option, ...]:
    questions = parse_questions(reason)
    if not questions:
        questions = parse_questions(f"1. {reason}")
    return tuple(option for question in questions for option in question.options)


def _title(markup: str) -> VerticalScroll:
    """The banner's text; a long question scrolls instead of being cut."""
    return VerticalScroll(Static(markup, classes="nb-title"), classes="nb-scroll")


class NeedsYouBanner(Vertical):
    DEFAULT_CSS = """
    NeedsYouBanner {
        display: none;
        height: auto;
        max-height: 18;
        padding: 0 1;
        margin-bottom: 1;
        border: solid #5a1f1a;
        background: #170b09;
        color: #ecebe4;
    }
    NeedsYouBanner .nb-scroll { height: auto; max-height: 8; scrollbar-size-vertical: 1; }
    NeedsYouBanner .nb-title { height: auto; }
    NeedsYouBanner .nb-row { height: 1; margin-top: 1; }
    NeedsYouBanner .nb-row Button {
        width: auto;
        height: 1;
        min-width: 0;
        border: none;
        padding: 0 1;
        margin: 0 1 0 0;
        background: #26201c;
        color: #ecebe4;
        text-style: none;
    }
    NeedsYouBanner .nb-row Button:hover { color: #c6f432; background: #26201c; }
    NeedsYouBanner .nb-row Button:focus { text-style: bold; background-tint: transparent; }
    NeedsYouBanner .nb-row Button.nb-go:hover { color: #0a0a09; background: #d4fa5a; }
    NeedsYouBanner .nb-row Button.nb-go {
        background: #c6f432;
        color: #0a0a09;
        text-style: bold;
    }
    NeedsYouBanner Input {
        height: 1;
        margin-top: 1;
        border: none;
        padding: 0 1;
        background: #0f0807;
        color: #ecebe4;
    }
    NeedsYouBanner Input:focus { border: none; }
    NeedsYouBanner Input > .input--placeholder { color: #7a7972; }
    """

    class Chosen(Message):
        def __init__(self, kind: str, story: int, result: tuple[str, str] | None) -> None:
            super().__init__()
            self.kind = kind
            self.story = story
            self.result = result

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self._kind = ""
        self._story = 0
        self._key: tuple[object, ...] | None = None
        self._option_labels: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        yield from ()

    def show(self, kind: str, story: StoryRow, status: dict[str, Any]) -> None:
        key = (
            kind,
            story.number,
            str(status.get("ask_reason", "")),
            str(status.get("final_eval_summary", "")),
        )
        if key == self._key:
            return
        self._key = key
        self._kind = kind
        self._story = story.number
        self.remove_children()
        if kind == "asked":
            widgets = self._asked(story, status)
        elif kind == "paused":
            widgets = self._paused(story, status)
        else:
            widgets = self._failed(story)
        self.mount_all(widgets)
        self.display = True

    def hide(self) -> None:
        self.display = False

    def prefill(self, text: str) -> None:
        input_id = {"asked": "nb-answer", "paused": "nb-text"}.get(self._kind, "nb-guidance")
        field = self.query_one(f"#{input_id}", Input)
        field.value = text
        field.focus()

    def _asked(self, story: StoryRow, status: dict[str, Any]) -> list[Widget]:
        reason = str(status.get("ask_reason", "")) or "The runner needs your input."
        options = _options(reason)
        self._option_labels = {option.letter: option.text for option in options}
        buttons = [
            Button(f"{option.letter.lower()}  {option.text}", id=f"nb-opt-{option.letter}")
            for option in options
        ]
        ident = story_label(story)[0]
        head = (
            f"[b {RED}]PAUSED[/] the runner asked a question in iteration "
            f"{max(1, story.iterations)}:\n"
        )
        widgets: list[Widget] = [_title(head + escape(reason))]
        if buttons:
            widgets.append(Horizontal(*buttons, classes="nb-row"))
        widgets.append(Input(id="nb-answer", placeholder=f"or type here, target {ident}"))
        widgets.append(
            Horizontal(
                Button("answer ↵", id="nb-submit", classes="nb-go"),
                Button("accept as done", id="nb-accept"),
                Button("jump to premium", id="nb-jump"),
                Button("edit config", id="nb-edit"),
                classes="nb-row",
            )
        )
        return widgets

    def _paused(self, story: StoryRow, status: dict[str, Any]) -> list[Widget]:
        summary = str(status.get("final_eval_summary", "")) or "Final eval complete."
        buttons = [Button("approve", id="nb-approve", classes="nb-go")]
        if bool(status.get("phased", False)):
            buttons.append(Button("next phase", id="nb-next"))
        buttons.extend(
            [
                Button("plan changes", id="nb-changes"),
                Button("reject", id="nb-reject"),
                Button("edit config", id="nb-edit"),
            ]
        )
        head = f"[b {RED}]WAIT[/] final eval is done and waits for your validation:\n"
        return [
            _title(head + escape(summary)),
            Input(id="nb-text", placeholder="your response (↵ approves)"),
            Horizontal(*buttons, classes="nb-row"),
        ]

    def _failed(self, story: StoryRow) -> list[Widget]:
        last = " ".join(part for part in (story.tier, story.model) if part)
        where = f", last on {escape(last)}" if last else ""
        title = f"[b {RED}]FAIL[/] after {story.iterations} iterations{where}."
        return [
            _title(title),
            Input(id="nb-guidance", placeholder="guidance for the retry"),
            Horizontal(
                Button("retry with guidance", id="nb-retry", classes="nb-go"),
                Button("send to orchestrator", id="nb-orchestrator"),
                Button("route next model", id="nb-reroute"),
                Button("raise tier", id="nb-tier_up"),
                Button("copy worktree path", id="nb-open"),
                classes="nb-row",
            ),
        ]

    def _text(self, input_id: str) -> str:
        return self.query_one(f"#{input_id}", Input).value.strip()

    def _choose(self, result: tuple[str, str] | None) -> None:
        self.post_message(self.Chosen(self._kind, self._story, result))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id.startswith("nb-opt-"):
            letter = button_id.removeprefix("nb-opt-")
            self._choose(("answer", f"{letter}. {self._option_labels[letter]}"))
            return
        if self._kind == "asked":
            self._asked_choice(button_id)
        elif self._kind == "paused":
            self._paused_choice(button_id)
        elif button_id == "nb-retry":
            self._choose(("retry", self._text("nb-guidance")))
        elif button_id == "nb-open":
            self._choose(("open_worktree", ""))
        elif button_id in ("nb-orchestrator", "nb-reroute", "nb-tier_up"):
            self._choose((button_id.removeprefix("nb-"), self._text("nb-guidance")))

    def _asked_choice(self, button_id: str) -> None:
        text = self._text("nb-answer")
        if button_id == "nb-submit" and not text:
            self._ask_for_text("nb-answer")
            return
        results = {
            "nb-submit": ("answer", text),
            "nb-accept": ("accept", ""),
            "nb-jump": ("jump_premium", text),
            "nb-edit": ("edit_config", text),
        }
        if button_id in results:
            self._choose(results[button_id])

    def _paused_choice(self, button_id: str) -> None:
        text = self._text("nb-text")
        if button_id == "nb-reject":
            self._choose(None)
            return
        results = {
            "nb-approve": ("approve", text),
            "nb-next": ("next_phase", text),
            "nb-changes": ("changes", text),
            "nb-edit": ("edit_config", text),
        }
        if button_id in results:
            self._choose(results[button_id])

    def _ask_for_text(self, input_id: str) -> None:
        field = self.query_one(f"#{input_id}", Input)
        field.focus()
        self.app.notify("Type the answer first, or pick one of the options.", severity="warning")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "nb-answer":
            if not event.value.strip():
                self._ask_for_text("nb-answer")
                return
            self._choose(("answer", event.value.strip()))
        elif event.input.id == "nb-guidance":
            self._choose(("retry", event.value.strip()))
        elif event.input.id == "nb-text":
            self._choose(("approve", event.value.strip()))
