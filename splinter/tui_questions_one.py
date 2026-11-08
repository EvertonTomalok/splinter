"""Widgets for the one-question-at-a-time display mode.

:class:`QModeHeader` is the clickable ``one at a time | all at once`` toggle,
:class:`OneQuestionPage` renders a single question with large answer rows,
auto-advancing on pick, and :class:`QuestionIndex` is the right-pane list of
open and answered questions. The question data and the answer bookkeeping
live in :class:`splinter.tui_questions.QuestionForm`; these widgets render
and forward, they hold no state of their own.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

from rich.markup import escape
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Input, Static

from splinter.tui_run_palette import BG, DIM, HOVER, INK, LIME, MUTED, Ink

if TYPE_CHECKING:
    from splinter.tui_questions import Question, QuestionForm

_INDEX_TEXT_MAX = 48


def mode_header_markup(mode: str, meta: str = "") -> str:
    """``PRD writer · sonnet · 3 questions   show [one at a time][all at once]``."""

    def chip(name: str, label: str) -> str:
        tone = f"{BG} on {INK}" if mode == name else f"{MUTED} on {HOVER}"
        return f"[@click=set_mode('{name}')][{tone}] {label} [/][/]"

    head = f"[{MUTED}]{escape(meta)}[/]    " if meta else ""
    return f"{head}[{DIM}]show[/] {chip('one', 'one at a time')}{chip('all', 'all at once')}"


def answer_label(letter: str, text: str, picked: bool) -> str:
    """An answer row: the key, the answer, and the reason under it in grey."""
    main, _, why = text.partition(" — ")
    key = f"{BG} on #a9d12a" if picked else f"{LIME} on #1d1d1a"
    note = BG if picked else MUTED
    label = f"[b {key}] {letter} [/]  {escape(main.strip())}"
    return label + (f"\n     [{note}]{escape(why.strip())}[/]" if why.strip() else "")


class QModeHeader(Ink):
    """The form header carrying the mode toggle; ``m`` does the same."""

    def action_toggle_mode(self) -> None:
        """Markup ``@click`` actions land here; forward them to the form."""
        if self.parent is not None:
            cast("QuestionForm", self.parent).action_toggle_mode()

    def action_set_mode(self, mode: str) -> None:
        """A segment click: switch only when it is not the active one."""
        form = cast("QuestionForm | None", self.parent)
        if form is not None and form.mode != mode:
            form.action_toggle_mode()


class OneQuestionPage(Vertical):
    """One question per page: counter, large answer rows, free text, back."""

    DEFAULT_CSS = """
    OneQuestionPage {
        height: auto;
        padding: 0 1;
    }
    OneQuestionPage #q-counter {
        height: 1;
        margin: 1 0 1 0;
    }
    OneQuestionPage #q-text {
        height: auto;
        text-style: bold;
        color: #ecebe4;
        margin: 0 0 1 0;
    }
    OneQuestionPage .ans {
        width: 100%;
        height: auto;
        min-height: 3;
        margin: 0;
        padding: 0 1;
        border: none;
        border-top: solid #1d1d1a;
        background: #0a0a09;
        color: #ecebe4;
        content-align: left middle;
        text-align: left;
        text-style: none;
    }
    OneQuestionPage .ans:hover {
        background: #121210;
        border-top: solid #1d1d1a;
    }
    OneQuestionPage .ans.-picked {
        background: #c6f432;
        color: #0a0a09;
    }
    OneQuestionPage #ans-other {
        height: 3;
        margin: 0;
        border: none;
        border-top: solid #1d1d1a;
        border-bottom: solid #1d1d1a;
        background: #0a0a09;
    }
    OneQuestionPage #q-back {
        margin: 1 0 0 0;
        color: #ecebe4;
    }
    """

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self._question: Question | None = None
        self._position = 0
        self._total = 0
        self._answer = ""

    def compose(self) -> ComposeResult:
        question = self._question
        if question is None:
            return
        yield Static(
            f"[b {LIME}]{self._position}[/] [{DIM}]/ {self._total}[/]", id="q-counter"
        )
        yield Static(escape(question.text), id="q-text")
        letters = {option.letter for option in question.options}
        for option in question.options:
            picked = option.letter == self._answer
            yield Button(
                answer_label(option.letter, option.text, picked),
                id=f"ans-{option.letter}",
                classes="ans -picked" if picked else "ans",
            )
        free_text = "" if self._answer in letters else self._answer
        yield Input(placeholder="other answer…", id="ans-other", value=free_text)
        yield Button("← back", id="q-back")

    def show(self, question: Question, position: int, total: int, answer: str) -> None:
        """Point the page at a question; the picked answer keeps its class."""
        self._question = question
        self._position = position
        self._total = total
        self._answer = answer
        self.refresh(recompose=True)


class QuestionIndex(Ink):
    """Open and answered questions; a row click jumps to that question."""

    DEFAULT_CSS = """
    QuestionIndex {
        height: auto;
        max-height: 8;
        padding: 0;
        display: none;
    }
    """

    def refresh_from(self, questions: Sequence[Question], answers: Mapping[int, str]) -> None:
        """Re-render the rows; hidden while there are no questions."""
        if not questions:
            self.display = False
            return
        rows = [_index_row(question, answers.get(question.number)) for question in questions]
        self.update("\n".join(rows))
        self.display = True


def _index_row(question: Question, answer: str | None) -> str:
    if answer is None:
        label = f"[{LIME}]○ {question.number} · {escape(_clip(question.text))}[/]"
    elif any(option.letter == answer for option in question.options):
        label = f"[{MUTED}]● {question.number} · {escape(answer)}[/]"
    else:
        label = f"[{MUTED}]● {question.number} · {escape(_clip(answer))}[/]"
    return f"[@click=app.jump_question({question.number})]{label}[/]"


def _clip(text: str, limit: int = _INDEX_TEXT_MAX) -> str:
    plain = " ".join(text.split())
    return plain if len(plain) <= limit else plain[: limit - 1] + "…"
