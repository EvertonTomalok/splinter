"""Right pane and nav row of the PRD questions screen, as Rich markup.

While the writer asks, the right pane shows a short outline of the draft
instead of the raw markdown: the file and its story count, the title, a few
user stories, then the open questions (:class:`QuestionIndex`), the answers
(:class:`AnsweredList`) and the pre-run spend. ``e`` swaps the outline for the
editor. Every function here is pure: text in, markup out.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from rich.markup import escape

from splinter.question_context import Explanation, Section
from splinter.tui_questions import Question
from splinter.tui_questions_one import FAINT, question_label
from splinter.tui_run_palette import INK, LIME, MUTED, RED

_STORY_RE = re.compile(r"^#{2,4}\s+(US-\d+)\s*[:—-]\s*(.+?)\s*$", re.MULTILINE)
_TITLE_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
#: Stories shown in full when there are this many or fewer.
_ALL_STORIES = 7
_HEAD_STORIES = 2
_TAIL_STORIES = 3
_STORY_TITLE_MAX = 34


def cap_markup(editing: bool) -> str:
    """``PRD draft · e to edit · click a question to jump``."""
    hint = "editing · esc to close" if editing else "e to edit · click a question to jump"
    return f"[{MUTED}]PRD draft[/] [{FAINT}]· {hint}[/]"


def _body(prd: str) -> str:
    """The PRD without its leading frontmatter block."""
    text = prd.lstrip()
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[end + 4 :]
    return text


def _clip(text: str, limit: int) -> str:
    plain = " ".join(text.split())
    return plain if len(plain) <= limit else plain[: limit - 1] + "…"


def outline_markup(prd: str, name: str) -> str:
    """File · story count, the title, then the first and the last user stories."""
    stories = _STORY_RE.findall(prd)
    count = len(stories)
    counted = f"{count} {'story' if count == 1 else 'stories'}" if count else "no stories yet"
    lines = [f"[{FAINT}]{escape(name)} · {counted}[/]"]
    title = _TITLE_RE.search(_body(prd))
    if title:
        lines.append(f"[b {INK}]{escape(title.group(1))}[/]")
    shown: list[tuple[str, str] | None] = list(stories)
    if count > _ALL_STORIES:
        shown = [*stories[:_HEAD_STORIES], None, *stories[-_TAIL_STORIES:]]
    for story in shown:
        if story is None:
            lines.append(f"[{FAINT}]…[/]")
            continue
        story_id, story_title = story
        lines.append(f"[{MUTED}]{story_id} {escape(_clip(story_title, _STORY_TITLE_MAX))}[/]")
    return "\n".join(lines)


def nav_markup(*, mode: str, on_last: bool, answered: int, total: int, letters: str) -> str:
    """``← back  → next`` in one mode, ``↵ finalize  2/3 answered`` in all mode."""
    context = f"   [@click=app.question_context()][b {LIME}]?[/] [{INK}]context[/][/]"
    if mode == "all":
        picks = f"a-{letters[-1].lower()} picks, " if len(letters) > 1 else ""
        return (
            f"[@click=app.finish_questions()][b {LIME}]↵[/] [b {INK}]finalize[/][/]{context}   "
            f"[{FAINT}]{answered}/{total} answered · {picks}tab moves[/]"
        )
    forward = "finalize" if on_last else "next"
    return (
        f"[@click=app.question_back()][b {LIME}]←[/] [{INK}]back[/][/]   "
        f"[@click=app.question_next()][b {LIME}]→[/] [{INK}]{forward}[/][/]{context}"
    )


def cowabunga_hint() -> str:
    return f"[@click=app.cowabunga()][{FAINT}]^k cowabunga answers the rest[/][/]"


def spend_markup(usage: Mapping[str, Any], jev: Mapping[str, Any]) -> str:
    """``pre-run $0.0412 · jev $0.0018 ×2``; the jev part only once it ran."""
    total = float(usage.get("cost", 0.0) or 0.0)
    line = f"pre-run ${total:.4f}"
    calls = int(jev.get("calls", 0) or 0)
    if calls:
        line += f" · jev ${float(jev.get('cost', 0.0) or 0.0):.4f} ×{calls}"
    return f"[{FAINT}]{line}[/]"


def context_cap_markup(number: int) -> str:
    """``question 2 · context · ? or esc to close``."""
    return f"[{MUTED}]question {number}[/] [{FAINT}]· context · ? or esc to close[/]"


def context_markup(
    question: Question,
    sections: list[Section],
    *,
    explanation: Explanation | None = None,
    error: str = "",
    model: str = "",
    waiting: str = "",
    idle: bool = False,
) -> str:
    """Where the PRD talks about the question, then the plain-words explanation."""
    lines = [f"[b {INK}]{question.number} {escape(question_label(question))}[/]"]
    if question.context:
        lines.append(f"[{MUTED}]{escape(question.context)}[/]")
    lines += ["", f"[{MUTED}]in the PRD[/]"]
    if not sections:
        lines.append(f"[{FAINT}]nothing in the draft names this directly[/]")
    for section in sections:
        lines.append(f"[{FAINT}]§ {escape(section.heading)}[/]")
        lines += [f"[{INK}]{escape(line)}[/]" for line in section.lines]
    lines.append("")
    if explanation is not None:
        cost = f" · ${explanation.cost:.4f}" if explanation.cost else ""
        lines.append(f"[{MUTED}]in plain words[/]")
        lines.append(f"[{FAINT}]{escape(explanation.model)}{cost} · x asks again[/]")
        lines += [_explanation_line(line) for line in explanation.text.splitlines()]
    elif error:
        lines.append(f"[{MUTED}]in plain words[/]")
        lines.append(f"[{RED}]couldn't explain: {escape(error)}[/]")
        lines.append(f"[@click=app.explain_again()][{FAINT}]x to try again[/][/]")
    elif idle:
        lines.append(f"[{MUTED}]in plain words[/]")
        lines.append(
            f"[@click=app.question_context()][b {LIME}]?[/] [{INK}]explains this one[/][/]"
            f" [{FAINT}]· {escape(model)}[/]"
        )
    else:
        lines.append(f"[{MUTED}]in plain words[/]")
        spinner = f"[b {LIME}]{escape(waiting)}[/] " if waiting else ""
        lines.append(f"{spinner}[{FAINT}]{escape(model)} is reading the question…[/]")
    return "\n".join(lines)


_OPTION_LINE = re.compile(r"^([A-H])([).:]?\s)")


def _explanation_line(line: str) -> str:
    """One line of the explanation; an option line gets its letter in lime."""
    match = _OPTION_LINE.match(line)
    if not match:
        return f"[{INK}]{escape(line)}[/]"
    rest = escape(line[match.end(1):])
    return f"[b {LIME}]{match.group(1)}[/][{INK}]{rest}[/]"
