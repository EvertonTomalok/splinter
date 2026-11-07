"""Parse model questions into data and render them as an answer form.

The PRD and plan models answer with numbered questions and lettered options.
:func:`parse_questions` turns that text into structured data; :class:`QuestionForm`
renders it as one radio set per question so the user answers with the mouse
instead of typing ``1A,2C`` by hand.

The module is deliberately free of app state so the same widgets can be reused by
the PRD app, the plan app, and any future conversation screen.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import RadioButton, RadioSet, Static

#: ``1. text``, ``1) text`` and ``**1.** text`` all open a question.
_QUESTION_RE = re.compile(r"^\s*(?:\*\*|__)?\s*(\d{1,2})\s*[.)]\s*(?:\*\*|__)?\s*(.*)$")
#: ``(A)``, ``A)``, ``A.`` and ``A:`` all open an option.
_OPTION_RE = re.compile(r"(?:\(\s*([A-Za-z])\s*\)|(?<![A-Za-z0-9])([A-Za-z])[.):])")
#: A leading bullet does not turn a question line into an option line.
_BULLET_RE = re.compile(r"^[-*•]\s*")
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
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        content = _BULLET_RE.sub("", stripped)
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
    return Question(number=number, text=text, options=tuple(options))


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


def _fits_on_one_row(question: Question) -> bool:
    """Whether the options of a question fit side by side."""
    if len(question.options) > _MAX_INLINE_OPTIONS:
        return False
    return all(len(option.text) <= _MAX_INLINE_LABEL for option in question.options)


class QuestionForm(VerticalScroll):
    """Compact form: one bold question line plus a radio set of its options."""

    DEFAULT_CSS = """
    QuestionForm {
        height: auto;
        max-height: 16;
        width: 1fr;
        border: round $panel;
        padding: 0 1;
        margin: 0 0 1 0;
    }
    QuestionForm > .question-text {
        text-style: bold;
        width: 100%;
        height: auto;
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

    def __init__(self, questions: Sequence[Question] = (), *, id: str | None = None) -> None:
        super().__init__(id=id)
        self._questions: list[Question] = list(questions)
        self.display = bool(self._questions)

    @property
    def questions(self) -> list[Question]:
        """The questions currently shown by the form."""
        return list(self._questions)

    def compose(self) -> ComposeResult:
        yield from self._build_widgets()

    def _build_widgets(self) -> list[Widget]:
        widgets: list[Widget] = []
        for question in self._questions:
            widgets.append(Static(f"{question.number}. {question.text}", classes="question-text"))
            widgets.append(self._build_radio_set(question))
        return widgets

    def _build_radio_set(self, question: Question) -> RadioSet:
        buttons = [RadioButton(f"{option.letter}) {option.text}") for option in question.options]
        radio = RadioSet(*buttons, id=f"q-{question.number}", compact=True)
        if _fits_on_one_row(question):
            radio.styles.layout = "horizontal"
            for button in buttons:
                button.styles.width = "auto"
        return radio

    async def set_questions(self, questions: Sequence[Question]) -> None:
        """Replace the questions shown by the form."""
        self._questions = list(questions)
        await self.remove_children()
        widgets = self._build_widgets()
        if widgets:
            await self.mount_all(widgets)
        self.display = bool(self._questions)

    def compile_answer(self, notes: str = "") -> str:
        """Build the answer text from the pressed options plus free-text notes.

        Unanswered questions are left out. Notes are appended after a ``+`` so
        the model can tell the picked options from the user's own words.
        """
        parts: list[str] = []
        for question in self._questions:
            try:
                radio = self.query_one(f"#q-{question.number}", RadioSet)
            except Exception:
                continue
            buttons = list(radio.query(RadioButton))
            for index, button in enumerate(buttons):
                if not button.value or index >= len(question.options):
                    continue
                parts.append(f"{question.number}{question.options[index].letter}")
                break
        answer = ", ".join(parts)
        note = notes.strip()
        if not note:
            return answer
        return f"{answer} + {note}" if answer else note
