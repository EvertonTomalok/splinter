"""The live iteration as it happens: every streamed 🔧 tool call and 💬 thought.

Providers stream each action of the runner to the log. The run screen keeps the
recent ones, tagged with the story and the iteration they belong to, and the
activity view draws them under the live iteration: ``│ 14:55:09  think  round the
wait up``. The newest row lights up when it lands and a caret blinks after it, so
the view moves while the runner works. The sentence above the stage strip says the
same thing in words: ``sonnet@high is editing src/api/middleware.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rich.text import Text

from splinter.tui_run_iteration import _short_path
from splinter.tui_run_log import KIND_COLOR, classify
from splinter.tui_run_palette import DIM, INK, LIME, LINE, MUTED

#: Live rows drawn under the live iteration; older ones fold into one row.
LIVE_ROWS = 12
#: Seconds a new row stays lit.
FRESH_S = 1.2
_KIND_W = 6
_VERBS = {"read": "reading", "edit": "editing", "run": "running"}


@dataclass(frozen=True)
class LiveAction:
    """One streamed action of a runner."""

    stamp: str
    kind: str
    text: str
    story: int | None
    iteration: int
    at: float


def live_action(
    message: str, stamp: str, story: int | None, iteration: int, at: float
) -> LiveAction | None:
    """The action a streamed log message carries, or ``None`` for other lines."""
    body = message.strip()
    if not body.startswith(("💬", "🔧")):
        return None
    kind, text = classify(body)
    text = _short_path(" ".join(text.split()))
    if not text:
        return None
    return LiveAction(stamp, kind or "note", text, story, iteration, at)


def actions_for(
    actions: Sequence[LiveAction], story: int | None, iteration: int
) -> list[LiveAction]:
    return [a for a in actions if a.story == story and a.iteration == iteration]


def live_rows(
    actions: Sequence[LiveAction], width: int, phase: bool, now: float
) -> Text:
    """The live feed: guide, stamp, kind, text; the newest row lit, a caret after it."""
    room = max(12, width - _KIND_W - 16)
    rows: list[Text] = []
    hidden = len(actions) - LIVE_ROWS
    if hidden > 0:
        rows.append(_guide().append(f"… {hidden} earlier actions", style=DIM))
    shown = list(actions[-LIVE_ROWS:])
    for index, action in enumerate(shown):
        newest = index == len(shown) - 1
        fresh = newest and now - action.at < FRESH_S
        line = _guide()
        line.append(f"{action.stamp}  ", style=MUTED if fresh else DIM)
        kind_tone = LIME if fresh else KIND_COLOR.get(action.kind, MUTED)
        line.append(f"{action.kind:<{_KIND_W}}", style=kind_tone)
        text = action.text if len(action.text) <= room else action.text[: room - 1] + "…"
        tone = INK if newest or action.kind == "edit" else MUTED
        line.append(text, style=f"bold {tone}" if fresh else tone)
        if newest:
            line.append(" ▌" if not phase else "  ", style=LIME)
        rows.append(line)
    if not shown:
        line = _guide().append("waiting for the runner's first action", style=DIM)
        line.append(" ▌" if not phase else "  ", style=LIME)
        rows.append(line)
    return Text("\n").join(rows)


def _guide() -> Text:
    return Text("  │ ", style=LINE)


def live_phrase(runner: str, action: LiveAction, room: int = 70) -> str:
    """``sonnet@high is editing src/api/middleware.py`` from the newest action."""
    text = action.text
    if action.kind == "think":
        phrase = f"{runner} · {text[:1].lower()}{text[1:]}"
    else:
        verb = _VERBS.get(action.kind, f"calling {action.kind}")
        phrase = f"{runner} is {verb} {text}"
    return phrase if len(phrase) <= room else phrase[: room - 1] + "…"
