"""Parse model questions into data and render them as an answer form.

The PRD and plan models answer with numbered questions and lettered options.
:func:`parse_questions` turns that text into structured data; :class:`QuestionForm`
renders it so the user answers with the mouse instead of typing ``1A,2C`` by
hand. The form has two displays of the same questions: ``all`` (compact radio
rows, every question on one page) and ``one`` (one question per page, large
answer rows, auto-advance on pick). Answers live in one map under both
displays, so switching modes keeps them, and :meth:`QuestionForm.compile_answer`
reads that map — its output is the same string in either mode.

The module is deliberately free of app state so the same widgets can be reused by
the PRD app, the plan app, and any future conversation screen.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from rich.markup import escape
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button, RadioButton, RadioSet, Static

from splinter.tui_questions_one import (
    AnswerRow,
    OneQuestionPage,
    QModeHeader,
    mode_header_markup,
    split_answer,
)
from splinter.ui_prefs import questions_mode, write_ui_pref

#: ``1. text``, ``1) text`` and ``**1.** text`` all open a question.
_QUESTION_RE = re.compile(r"^\s*(?:\*\*|__)?\s*(\d{1,2})\s*[.)]\s*(?:\*\*|__)?\s*(.*)$")
#: ``(A)``, ``A)``, ``A.`` and ``A:`` all open an option.
_OPTION_RE = re.compile(r"(?:\(\s*([A-Za-z])\s*\)|(?<![A-Za-z0-9])([A-Za-z])[.):])")
#: A leading bullet does not turn a question line into an option line.
_BULLET_RE = re.compile(r"^[-*•]\s*")
#: ``[short label] The question?``: the label names the decision in the index.
_LABEL_RE = re.compile(r"^\[([^\]]{1,40})\]\s*")
#: ``Why: …``: the one line saying what in the PRD forces the decision.
_WHY_RE = re.compile(r"^\s*(?:\*\*|__)?(?:why|context)\s*:\s*(?:\*\*|__)?\s*(.+)$", re.I)
#: ``Label: …``: the short name of the decision, shown in the index.
_LABEL_LINE_RE = re.compile(r"^\s*(?:\*\*|__)?label\s*:\s*(?:\*\*|__)?\s*(.+)$", re.I)
#: More options than this (or longer labels) stop fitting on one row.
_MAX_INLINE_OPTIONS = 4
_MAX_INLINE_LABEL = 32


@dataclass(frozen=True)
class Option:
    """One lettered answer to a question."""

    letter: str
    text: str


@dataclass(frozen=True)
class Question:
    """A numbered question with its lettered options."""

    number: int
    text: str
    options: tuple[Option, ...] = ()
    label: str = ""
    context: str = ""


def _clean(text: str) -> str:
    """Strip whitespace and the markdown emphasis the models like to add."""
    text = text.strip()
    text = re.sub(r"^(\*\*|__)", "", text)
    text = re.sub(r"(\*\*|__)$", "", text)
    return text.strip()


def _split_options(content: str, hits: list[re.Match[str]]) -> list[Option]:
    """Split a line into options at every marker found in it."""
    options: list[Option] = []
    for index, hit in enumerate(hits):
        letter = (hit.group(1) or hit.group(2) or "").upper()
        start = hit.end()
        end = hits[index + 1].start() if index + 1 < len(hits) else len(content)
        label = content[start:end].strip()
        if not letter or not label:
            continue
        if any(option.letter == letter for option in options):
            continue
        options.append(Option(letter=letter, text=label))
    return options


def _question_from_block(number: int, lines: list[str]) -> Question | None:
    """Build a question from its lines; ``None`` when it carries no options."""
    question_lines: list[str] = []
    options: list[Option] = []
    context = ""
    label = ""
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        content = _BULLET_RE.sub("", stripped)
        why = _WHY_RE.match(content)
        if why and not options:
            context = _clean(why.group(1))
            continue
        named = _LABEL_LINE_RE.match(content)
        if named and not options:
            label = _clean(named.group(1))
            continue
        hits = list(_OPTION_RE.finditer(content))
        if not hits:
            question_lines.append(stripped)
        elif hits[0].start() == 0:
            options.extend(_split_options(content, hits))
        elif len(hits) >= 2:
            # Question text and options share the line: "Pick one? A) x  B) y".
            question_lines.append(content[: hits[0].start()])
            options.extend(_split_options(content, hits))
        else:
            question_lines.append(stripped)
    if not options:
        return None
    text = _clean(" ".join(question_lines))
    match = _LABEL_RE.match(text)
    if match:
        label = label or match.group(1).strip()
        text = text[match.end():].strip()
    return Question(
        number=number, text=text, options=tuple(options), label=label, context=context
    )


def parse_questions(text: str) -> list[Question]:
    """Parse numbered questions with lettered options out of model output.

    Handles ``1.``/``1)``/``**1.**`` question headers, question text spanning
    several lines, and options written on one line (``A) x  B) y``) or one per
    line (``A) x``, ``- A: x``, ``(A) x``). Questions without options are
    dropped — they cannot be answered with the radio form.
    """
    blocks: list[tuple[int, list[str]]] = []
    current: tuple[int, list[str]] | None = None
    for line in text.splitlines():
        match = _QUESTION_RE.match(line)
        if match:
            current = (int(match.group(1)), [match.group(2)])
            blocks.append(current)
        elif current is not None:
            current[1].append(line)
    questions: list[Question] = []
    for number, lines in blocks:
        question = _question_from_block(number, lines)
        if question is not None:
            questions.append(question)
    return questions


def extract_open_questions(text: str) -> list[Question]:
    """Parse the questions out of a refinement turn.

    Refinement replies carry a ``## Open Questions`` section; only that part is
    parsed so the working draft in the same reply cannot be mistaken for
    questions. Text without the section is parsed as-is.
    """
    match = re.search(r"##\s*open questions\b(.*)", text, re.IGNORECASE | re.DOTALL)
    return parse_questions(match.group(1) if match else text)


def _radio_label(option: Option) -> Text:
    """``A  Stripe  current live traffic``: the key in lime, the reason muted."""
    main, why = split_answer(option.text)
    label = Text()
    label.append(option.letter, style="bold #c6f432")
    label.append(f"  {main}")
    if why:
        label.append(f"  {why}", style="#8f8e85")
    return label


def _fits_on_one_row(question: Question) -> bool:
    """Whether the options of a question fit side by side."""
    if len(question.options) > _MAX_INLINE_OPTIONS:
        return False
    return all(len(option.text) <= _MAX_INLINE_LABEL for option in question.options)


class QuestionForm(VerticalScroll):
    """Answer form with two displays of the same questions.

    ``all`` shows every question with its radio set on one page; ``one``
    shows a single question with large answer rows, auto-advancing on pick.
    Answers live in ``_answers`` (question number → letter or free text), so
    switching modes keeps them and :meth:`compile_answer` reads that map.
    The form carries the id ``question-form`` unless overridden: every host
    app and test targets it by that id.
    """

    DEFAULT_CSS = """
    QuestionForm {
        height: auto;
        max-height: 20;
        width: 1fr;
        padding: 0;
        margin: 0 0 1 0;
    }
    QuestionForm #q-head {
        height: 3;
        width: 100%;
    }
    QuestionForm #q-meta {
        width: 1fr;
        height: 1;
        margin: 1 0 0 0;
        color: #8f8e85;
    }
    QuestionForm #q-mode {
        height: 3;
        width: auto;
    }
    QuestionForm RadioButton {
        background: #0a0a09;
        color: #ecebe4;
        padding: 0 1;
        margin: 0 1 0 0;
        border: none;
    }
    QuestionForm RadioButton.-selected {
        background: #0a0a09;
    }
    QuestionForm RadioSet:focus > RadioButton.-selected {
        background: #121210;
    }
    QuestionForm RadioButton.-on {
        background: #1c2409;
        color: #c6f432;
    }
    QuestionForm RadioSet:blur > RadioButton.-selected > .toggle--label {
        background: transparent;
    }
    QuestionForm RadioSet:focus > RadioButton.-selected > .toggle--label {
        background: #1a1a17;
        color: #ecebe4;
        text-style: bold;
    }
    QuestionForm RadioSet:focus {
        background-tint: #ecebe4 0%;
    }
    QuestionForm #q-list {
        height: auto;
    }
    QuestionForm .question-text {
        text-style: bold;
        width: 100%;
        height: auto;
        margin: 1 0 0 0;
    }
    QuestionForm RadioSet {
        height: auto;
        width: 1fr;
        border: none;
        padding: 0;
        background: transparent;
        overflow: hidden hidden;
    }
    """

    BINDINGS = [
        Binding("m", "toggle_mode", "Mode"),
        Binding("left", "prev_question", "Back", show=False),
        Binding("right", "next_question", "Next", show=False),
        *(Binding(key, f"pick('{key.upper()}')", show=False) for key in "abcd"),
    ]

    mode: reactive[str] = reactive("one")

    class AnswersChanged(Message):
        """Posted whenever an answer is picked, typed or restored."""

        def __init__(self, form: "QuestionForm", answers: dict[int, str]) -> None:
            super().__init__()
            self.form = form
            self.answers = answers

        @property
        def control(self) -> "QuestionForm":
            return self.form

    class PageChanged(Message):
        """Posted when the page or the display mode changes (the nav label follows)."""

        def __init__(self, form: "QuestionForm") -> None:
            super().__init__()
            self.form = form

        @property
        def control(self) -> "QuestionForm":
            return self.form

    class FinishRequested(Message):
        """``→`` past the last question, or the host's finish key: finalize now."""

        def __init__(self, form: "QuestionForm") -> None:
            super().__init__()
            self.form = form

        @property
        def control(self) -> "QuestionForm":
            return self.form

    class Completed(Message):
        """Posted after the last question is answered; the form never submits."""

        def __init__(self, form: "QuestionForm") -> None:
            super().__init__()
            self.form = form

        @property
        def control(self) -> "QuestionForm":
            return self.form

    def __init__(
        self,
        questions: Sequence[Question] = (),
        *,
        id: str | None = None,
        prefs_root: Path | None = None,
    ) -> None:
        super().__init__(id=id or "question-form")
        self._questions: list[Question] = list(questions)
        self._answers: dict[int, str] = {}
        self._index = 0
        self._prefs_root = prefs_root
        self.meta = ""
        self.mode = questions_mode(prefs_root)
        self.display = bool(self._questions)

    @property
    def questions(self) -> list[Question]:
        """The questions currently shown by the form."""
        return list(self._questions)

    @property
    def answers(self) -> dict[int, str]:
        """Question number → picked letter or typed answer."""
        return dict(self._answers)

    @property
    def position(self) -> int:
        """1-based number of the page shown in ``one`` mode (0 without questions)."""
        return self._index + 1 if self._questions else 0

    @property
    def current_question(self) -> Question | None:
        """The question on screen: the page in ``one`` mode, the focused set in ``all``."""
        if self.mode == "one":
            return self._current_question()
        return self._focused_question() or (self._questions[0] if self._questions else None)

    @property
    def on_last(self) -> bool:
        """Whether the ``one`` page shows the last question."""
        return bool(self._questions) and self._index >= len(self._questions) - 1

    def compose(self) -> ComposeResult:
        with Horizontal(id="q-head"):
            yield Static(escape(self.meta), id="q-meta", markup=True)
            yield QModeHeader(id="q-mode")
        yield OneQuestionPage(id="q-page")
        with Vertical(id="q-list"):
            for question in self._questions:
                yield from self._question_widgets(question)

    def on_mount(self) -> None:
        self._apply_answers_to_radios()
        self._refresh_page()
        self._sync_mode_ui()

    def _question_widgets(self, question: Question) -> list[Widget]:
        return [
            Static(
                f"[b #c6f432]{question.number}[/]  [b]{escape(question.text)}[/]",
                classes="question-text",
            ),
            self._build_radio_set(question),
        ]

    def _build_radio_set(self, question: Question) -> RadioSet:
        buttons = [RadioButton(_radio_label(option)) for option in question.options]
        radio = RadioSet(*buttons, id=f"q-{question.number}", compact=True)
        if _fits_on_one_row(question):
            radio.styles.layout = "horizontal"
            for button in buttons:
                button.styles.width = "auto"
        return radio

    async def set_questions(self, questions: Sequence[Question]) -> None:
        """Replace the questions shown by the form (answers start over)."""
        self._questions = list(questions)
        self._answers = {}
        self._index = 0
        self.display = bool(self._questions)
        if self.is_attached:
            holder = self.query_one("#q-list", Vertical)
            await holder.remove_children()
            widgets = [w for q in self._questions for w in self._question_widgets(q)]
            if widgets:
                await holder.mount_all(widgets)
        self._sync_mode_ui()
        self._refresh_page()

    # --- mode ---
    def action_toggle_mode(self) -> None:
        """Key ``m`` / header click: flip the display and remember the choice."""
        self.mode = "all" if self.mode == "one" else "one"
        write_ui_pref("questions_mode", self.mode, root=self._prefs_root)
        if self.mode == "one":
            self.focus_page()
            return
        radios = list(self.query(RadioSet))
        if radios:
            self.call_after_refresh(radios[0].focus)

    def watch_mode(self, mode: str) -> None:
        self._sync_mode_ui()

    def _sync_mode_ui(self) -> None:
        try:
            self.query_one("#q-meta", Static).update(escape(self.meta))
            self.query_one("#q-mode", QModeHeader).update(mode_header_markup(self.mode))
            self.query_one("#q-list", Vertical).display = self.mode == "all"
            self.query_one("#q-page", OneQuestionPage).display = self.mode == "one"
        except Exception:
            return
        self._apply_answers_to_radios()
        self.post_message(self.PageChanged(self))

    # --- one-mode navigation ---
    def action_prev_question(self) -> None:
        """Key ``←`` / back button: show the previous question."""
        if self._index > 0:
            self._index -= 1
            self._refresh_page()
            self.focus_page()

    def action_next_question(self) -> None:
        """Key ``→`` / next: the next question, or finish past the last one."""
        if not self._questions:
            return
        if self.on_last:
            self.post_message(self.FinishRequested(self))
            return
        self._index += 1
        self._refresh_page()
        self.focus_page()

    def action_pick(self, letter: str) -> None:
        """Keys ``a``-``d``: pick that option of the shown (or focused) question."""
        question = self._current_question() if self.mode == "one" else self._focused_question()
        if question is None or not any(o.letter == letter for o in question.options):
            return
        self._set_answer(question.number, letter)
        if self.mode == "one":
            self._advance()
        else:
            self._apply_answers_to_radios()

    def _focused_question(self) -> Question | None:
        node = self.app.focused if self.is_attached else None
        while node is not None and not isinstance(node, RadioSet):
            node = node.parent if isinstance(node.parent, Widget) else None
        if node is None or node.id is None or not node.id.startswith("q-"):
            return None
        try:
            return self._question_by_number(int(node.id[2:]))
        except ValueError:
            return None

    def _advance(self) -> None:
        if self._index < len(self._questions) - 1:
            self._index += 1
            self._refresh_page()
            self.focus_page()
            return
        self.post_message(self.Completed(self))

    def focus_page(self) -> None:
        """Focus the shown page's picked answer, else its first one.

        The picked button is gone after the page swap, so the page is refocused.

        The recompose is queued ahead of this callback on the same pump, so
        the new first button exists by the time the callback runs.
        """
        try:
            page = self.query_one("#q-page", OneQuestionPage)
        except Exception:
            return
        page.call_next(self._focus_first_button, page)

    def _focus_first_button(self, page: OneQuestionPage) -> None:
        picked: list[Widget] = [row for row in page.query(AnswerRow) if row.has_class("-picked")]
        (picked[0] if picked else self).focus()

    def _refresh_page(self) -> None:
        if not self._questions:
            return
        self._index = max(0, min(self._index, len(self._questions) - 1))
        question = self._questions[self._index]
        try:
            page = self.query_one("#q-page", OneQuestionPage)
        except Exception:
            return
        page.show(
            question,
            self._index + 1,
            len(self._questions),
            self._answers.get(question.number, ""),
        )
        self.post_message(self.PageChanged(self))

    def _current_question(self) -> Question | None:
        if 0 <= self._index < len(self._questions):
            return self._questions[self._index]
        return None

    def _question_by_number(self, number: int) -> Question | None:
        for question in self._questions:
            if question.number == number:
                return question
        return None

    def jump_to(self, number: int) -> None:
        """Jump to a question by its number (right-pane list click)."""
        for index, question in enumerate(self._questions):
            if question.number != number:
                continue
            if self.mode == "one":
                self._index = index
                self._refresh_page()
                self.focus_page()
            else:
                self._scroll_to_question(number)
            return

    def _scroll_to_question(self, number: int) -> None:
        try:
            radio = self.query_one(f"#q-{number}", RadioSet)
        except Exception:
            return
        self.scroll_to_widget(radio)
        radio.focus()

    # --- answers ---
    def restore_answers(self, answers: Mapping[int, str]) -> None:
        """Re-fill the form with saved answers (a back-to-questions jump)."""
        numbers = {question.number for question in self._questions}
        self._answers = {n: a for n, a in answers.items() if n in numbers}
        self._apply_answers_to_radios()
        self._refresh_page()
        self.post_message(self.AnswersChanged(self, dict(self._answers)))

    def _apply_answers_to_radios(self) -> None:
        """Mirror ``_answers`` onto the radio sets; echoes dedupe in _set_answer."""
        for question in self._questions:
            answer = self._answers.get(question.number, "")
            try:
                radio = self.query_one(f"#q-{question.number}", RadioSet)
            except Exception:
                continue
            buttons = list(radio.query(RadioButton))
            for index, option in enumerate(question.options):
                if index < len(buttons):
                    buttons[index].value = option.letter == answer

    def _set_answer(self, number: int, answer: str) -> None:
        if self._answers.get(number) == answer:
            return
        self._answers[number] = answer
        self.post_message(self.AnswersChanged(self, dict(self._answers)))

    # --- events ---
    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        radio = event.radio_set
        if radio.id is None or not radio.id.startswith("q-"):
            return
        try:
            number = int(radio.id[2:])
        except ValueError:
            return
        question = self._question_by_number(number)
        if question is None or event.index >= len(question.options):
            return
        self._set_answer(number, question.options[event.index].letter)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "q-back":
            event.stop()
            self.action_prev_question()

    def on_answer_row_picked(self, event: AnswerRow.Picked) -> None:
        event.stop()
        question = self._current_question()
        if question is None:
            return
        if not any(option.letter == event.letter for option in question.options):
            return
        self._set_answer(question.number, event.letter)
        self._advance()

    def compile_answer(self, notes: str = "") -> str:
        """Build the answer text from the picked options plus free-text notes.

        Unanswered questions are left out. Notes are appended after a ``+`` so
        the model can tell the picked options from the user's own words.
        """
        parts: list[str] = []
        for question in self._questions:
            answer = self._answers.get(question.number)
            if answer is None:
                continue
            picked = answer.strip()
            if any(option.letter == picked for option in question.options):
                parts.append(f"{question.number}{picked}")
            elif picked:
                parts.append(picked)
        answer = ", ".join(parts)
        note = notes.strip()
        if not note:
            return answer
        return f"{answer} + {note}" if answer else note
