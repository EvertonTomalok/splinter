"""Widgets for the one-question-at-a-time display mode.

:class:`QModeHeader` is the clickable ``one at a time | all at once`` toggle,
:class:`OneQuestionPage` renders a single question with large answer rows
(:class:`AnswerRow`: a boxed key, the answer and the reason under it),
auto-advancing on pick, :class:`QuestionIndex` is the right-pane list of open
questions and :class:`AnsweredList` the answers picked so far. The question
data and the answer bookkeeping live in
:class:`splinter.tui_questions.QuestionForm`; these widgets render and
forward, they hold no state of their own.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

from rich.markup import escape
from textual import events
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Static

from splinter.tui_frame import KEY_EDGE, frame_markup, segments_markup
from splinter.tui_run_palette import BG, INK, LIME, MUTED, Ink

if TYPE_CHECKING:
    from splinter.tui_questions import Question, QuestionForm

#: The design's faintest text: separators, unanswered marks, the spend line.
FAINT = "#55554e"
_INDEX_TEXT_MAX = 38


def mode_header_markup(mode: str) -> str:
    """``show [one at a time|all at once]`` on three rows, the switch framed."""
    chips = segments_markup(
        [
            ("one at a time", "set_mode('one')", mode == "one"),
            ("all at once", "set_mode('all')", mode == "all"),
        ]
    )
    return frame_markup(chips, lead=f"[{MUTED}]show[/]  ")


def key_markup(letter: str, *, picked: bool = False) -> str:
    """An answer key: the letter in a tight frame drawn in the widget's colour."""
    tone = BG if picked else LIME
    return frame_markup(f"[{tone}] {escape(letter)} [/]", "")


def split_answer(text: str) -> tuple[str, str]:
    """``Stripe — current live traffic`` → (``Stripe``, ``current live traffic``)."""
    main, _, why = text.partition(" — ")
    return main.strip(), why.strip()


class QModeHeader(Ink):
    """The form header carrying the mode toggle; ``m`` does the same."""

    def action_toggle_mode(self) -> None:
        """Markup ``@click`` actions land here; forward them to the form."""
        form = self._form()
        if form is not None:
            form.action_toggle_mode()

    def action_set_mode(self, mode: str) -> None:
        """A segment click: switch only when it is not the active one."""
        form = self._form()
        if form is not None and form.mode != mode:
            form.action_toggle_mode()

    def _form(self) -> QuestionForm | None:
        from splinter.tui_questions import QuestionForm

        node = self.parent
        while node is not None and not isinstance(node, QuestionForm):
            node = node.parent
        return cast("QuestionForm | None", node)


class AnswerRow(Horizontal, can_focus=True):
    """One answer: the boxed key, the answer and the reason under it.

    A click, ``enter`` or ``space`` posts :class:`AnswerRow.Picked`.
    """

    DEFAULT_CSS = f"""
    AnswerRow {{
        width: 100%;
        height: auto;
        padding: 0 1;
        border-bottom: solid #1d1d1a;
        background: {BG};
    }}
    AnswerRow:hover {{
        background: #121210;
    }}
    AnswerRow .kk {{
        width: 5;
        height: 3;
        color: {KEY_EDGE};
    }}
    AnswerRow:focus .kk {{
        color: {LIME};
    }}
    AnswerRow.-picked .kk {{
        color: {BG};
    }}
    AnswerRow .ans-body {{
        width: 1fr;
        height: auto;
        padding: 1 0 0 2;
    }}
    AnswerRow .ans-main {{
        color: {INK};
        height: auto;
    }}
    AnswerRow .ans-why {{
        color: {MUTED};
        height: auto;
    }}
    AnswerRow.-picked {{
        background: {LIME};
    }}
    AnswerRow.-picked .ans-main {{
        color: {BG};
    }}
    AnswerRow.-picked .ans-why {{
        color: #3d4a10;
    }}
    """

    class Picked(Message):
        """The user chose this answer."""

        def __init__(self, row: AnswerRow, letter: str) -> None:
            super().__init__()
            self.row = row
            self.letter = letter

        @property
        def control(self) -> AnswerRow:
            return self.row

    def __init__(self, letter: str, text: str, *, picked: bool = False, id: str | None = None):
        super().__init__(id=id, classes="ans -picked" if picked else "ans")
        self.letter = letter
        self._picked = picked
        self._main, self._why = split_answer(text)

    def compose(self) -> ComposeResult:
        yield Static(key_markup(self.letter, picked=self._picked), classes="kk")
        with Vertical(classes="ans-body"):
            yield Static(self._main, classes="ans-main", markup=False)
            if self._why:
                yield Static(self._why, classes="ans-why", markup=False)

    def on_click(self, event: events.Click) -> None:
        event.stop()
        self.post_message(self.Picked(self, self.letter))

    def on_key(self, event: events.Key) -> None:
        if event.key in ("enter", "space"):
            event.stop()
            event.prevent_default()
            self.post_message(self.Picked(self, self.letter))


class OneQuestionPage(Vertical):
    """One question per page: counter, the question, its answer rows, back."""

    DEFAULT_CSS = f"""
    OneQuestionPage {{
        height: auto;
        padding: 0;
    }}
    OneQuestionPage #q-counter {{
        height: 1;
        margin: 1 0 0 0;
    }}
    OneQuestionPage #q-text {{
        height: auto;
        max-width: 60;
        text-style: bold;
        color: {INK};
        margin: 1 0 1 0;
    }}
    OneQuestionPage #q-why {{
        height: auto;
        max-width: 72;
        color: {MUTED};
        margin: 0 0 1 0;
    }}
    OneQuestionPage #q-text.-has-why {{
        margin: 1 0 0 0;
    }}
    OneQuestionPage #ans-list {{
        height: auto;
        border-top: solid #1d1d1a;
    }}
    OneQuestionPage #q-back {{
        margin: 1 0 0 0;
        color: {INK};
    }}
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
            f"[b {LIME}]{self._position}[/][{FAINT}] / {self._total}[/]", id="q-counter"
        )
        yield Static(
            escape(question.text), id="q-text", classes="-has-why" if question.context else ""
        )
        if question.context:
            yield Static(escape(question.context), id="q-why")
        with Vertical(id="ans-list"):
            for option in question.options:
                yield AnswerRow(
                    option.letter,
                    option.text,
                    picked=option.letter == self._answer,
                    id=f"ans-{option.letter}",
                )
        yield Button("← back", id="q-back")

    def show(self, question: Question, position: int, total: int, answer: str) -> None:
        """Point the page at a question; the picked answer keeps its class."""
        self._question = question
        self._position = position
        self._total = total
        self._answer = answer
        self.refresh(recompose=True)


class QuestionIndex(Ink):
    """The open questions by their short label; a row click jumps to one."""

    DEFAULT_CSS = """
    QuestionIndex {
        height: auto;
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


def question_label(question: Question) -> str:
    """The short name of a question: its ``[label]``, else its clipped text."""
    return question.label or _clip(question.text)


def answer_text(question: Question, answer: str) -> str:
    """The picked option's answer (reason dropped), or the typed answer."""
    for option in question.options:
        if option.letter == answer:
            return split_answer(option.text)[0]
    return answer


def _index_row(question: Question, answer: str | None) -> str:
    label = f"[{LIME}]{question.number} {escape(question_label(question))}[/]"
    return f"[@click=app.jump_question({question.number})]{label}[/]"


class AnsweredList(Ink):
    """``answered`` and one row per question: the pick, or a dash."""

    DEFAULT_CSS = """
    AnsweredList {
        height: auto;
        border-top: solid #1d1d1a;
        padding: 1 0 0 0;
        display: none;
    }
    """

    def refresh_from(self, questions: Sequence[Question], answers: Mapping[int, str]) -> None:
        """Re-render the rows; hidden while there are no questions."""
        if not questions:
            self.display = False
            return
        rows = [f"[{MUTED}]answered[/]"]
        for question in questions:
            answer = answers.get(question.number)
            if answer:
                picked = escape(_clip(answer_text(question, answer)))
                text = f"[{LIME}]{question.number} {picked}[/]"
            else:
                text = f"[{FAINT}]{question.number} —[/]"
            rows.append(f"[@click=app.jump_question({question.number})]{text}[/]")
        self.update("\n".join(rows))
        self.display = True


def _clip(text: str, limit: int = _INDEX_TEXT_MAX) -> str:
    plain = " ".join(text.split())
    return plain if len(plain) <= limit else plain[: limit - 1] + "…"
