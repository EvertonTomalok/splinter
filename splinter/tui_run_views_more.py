"""The `spend`, `tier` and `diff` views of the run screen.

The views read the session files through the run view model — no view calls a
model (README D4). The diff view reaches git through the injectable seam of
``splinter.vcs.diff``, and the breaker states are read, never tripped.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import zip_longest

from rich.console import Group, RenderableType
from rich.markup import escape
from rich.text import Text

from splinter.engines import runtime
from splinter.engines.health import BreakerState
from splinter.run_view import LadderRung, RunView, SpendRow, StoryRow
from splinter.tui_run_palette import (
    ADD_BG,
    ADD_FG,
    DEL_BG,
    DEL_FG,
    DIM,
    GREY_BAR,
    INK,
    LIME,
    LINE,
    MUTED,
    RED,
)
from splinter.tui_run_text import clip, pad, story_label
from splinter.tui_run_views import VIEWS, ViewContext, ViewFactory
from splinter.vcs.diff import StoryDiff

_TOP_STORIES = 8
_NEEDS = ("fail", "paused", "asked")


@dataclass(frozen=True)
class MoreState:
    """The screen state the `spend`, `tier` and `diff` views read."""

    spend_by: str = "cost"
    tier: str = ""
    diff: StoryDiff | None = None
    diff_cwd: str = ""
    budget: float | None = None


_state = MoreState()


def set_more_state(state: MoreState) -> None:
    """Publish the screen state the extra views read."""
    global _state
    _state = state


def more_state() -> MoreState:
    return _state


def _span(seconds: float) -> str:
    """Human span: ``45s`` · ``2m14s`` · ``1h05m`` · ``3d04h``."""
    if seconds <= 0:
        return "0s"
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h{minutes:02d}m"
    days, hours = divmod(hours, 24)
    return f"{days}d{hours:02d}h"


def _bar(value: float, peak: float, width: int, tone: str = GREY_BAR) -> str:
    """A thin bar on its track, scaled to the biggest row."""
    filled = max(1, round(width * value / peak)) if peak > 0 and value > 0 else 0
    filled = min(width, filled)
    return f"[{tone}]{'━' * filled}[/][{LINE}]{'━' * (width - filled)}[/]"


def _chip(label: str, value: str, by: str) -> str:
    action = f"[@click=app.set_spend_by('{value}')]"
    if by == value:
        return f"{action}[{LIME} on #1c2409] [bold]{label}[/bold] [/][/]"
    return f"{action}[{MUTED} on #141412] {label} [/][/]"


def _money(value: float) -> str:
    return f"${value:.4f}"


def _measure(row: SpendRow, by: str) -> float:
    return row.cost if by == "cost" else row.seconds


def _current_runner(view: RunView) -> str:
    tier = next((rung.tier for rung in view.ladder if rung.current), "")
    return f"runner {tier}" if tier else ""


def _step_lines(view: RunView, by: str, width: int, budget: float | None) -> list[str]:
    rows = view.spend
    lines = [f"[{MUTED}]by step · whole run[/]"]
    if not rows:
        return [*lines, f"[{DIM}]no spend yet[/]"]
    peak = max(_measure(row, by) for row in rows)
    label_w, value_w = min(20, width // 2 - 2), 9
    hot = _current_runner(view)
    for row in rows:
        shown = _money(row.cost) if by == "cost" else _span(row.seconds)
        tone = LIME if row.label == hot else GREY_BAR
        label = f"[{LIME if row.label == hot else INK}]{clip(row.label, label_w)}[/]"
        bar = _bar(_measure(row, by), peak, max(4, width - label_w - value_w - 2), tone)
        lines.append(f"{pad(label, label_w)} {bar} {shown:>{value_w}}")
    total = _money(view.total_cost) if by == "cost" else _span(view.elapsed_s)
    lines.append(f"[{LINE}]{'─' * width}[/]")
    lines.append(pad(f"[{MUTED}]total[/]", width - len(total)) + f"[b {LIME}]{total}[/]")
    if budget:
        used = (view.total_cost / budget) * 100
        lines.append(f"[{MUTED}]budget {_money(budget)} · used {used:.0f}%[/]")
    return lines


def _story_measure(story: StoryRow, by: str) -> float:
    return story.cost if by == "cost" else story.elapsed_s


def _story_lines(view: RunView, by: str, width: int) -> list[str]:
    head = f"[{MUTED}]top stories[/] {_chip('by cost', 'cost', by)} {_chip('by time', 'time', by)}"
    stories = sorted(view.stories, key=lambda s: _story_measure(s, by), reverse=True)
    top = [story for story in stories[:_TOP_STORIES] if _story_measure(story, by) > 0]
    if not top:
        return [head, f"[{DIM}]no stories yet[/]"]
    peak = max(_story_measure(story, by) for story in top)
    lines = [head]
    title_w = max(4, width - 7 - 9 - 9)
    for story in top:
        value = _story_measure(story, by)
        shown = _money(story.cost) if by == "cost" else _span(story.elapsed_s)
        ident, title = story_label(story)
        tone = RED if story.state in _NEEDS else LIME
        text = (
            f"[b {INK}]{escape(ident):<6}[/] {pad(f'[{MUTED}]{clip(title, title_w)}[/]', title_w)}"
            f" {_bar(value, peak, 7, tone)} {shown:>8}"
        )
        lines.append(f"[@click=app.spend_story({story.number})]{text}[/]")
    lines.append(f"[{DIM}]click a story to open it[/]")
    return lines


def spend_view(ctx: ViewContext) -> RenderableType:
    """Where the money and the time went, by step and by story."""
    state = more_state()
    by = state.spend_by if state.spend_by in ("cost", "time") else ctx.spend_by
    budget = state.budget if state.budget is not None else ctx.budget
    width = ctx.width or 76
    if width < 64:
        left = _step_lines(ctx.view, by, width, budget)
        return "\n".join([*left, "", *_story_lines(ctx.view, by, width)])
    column = (width - 4) // 2
    left = _step_lines(ctx.view, by, column, budget)
    right = _story_lines(ctx.view, by, column)
    rows = zip_longest(left, right, fillvalue="")
    return "\n".join(pad(a, column + 4) + b for a, b in rows)


def _pick_rung(view: RunView, tier: str) -> LadderRung | None:
    rungs = view.ladder
    if not rungs:
        return None
    if tier:
        for rung in rungs:
            if rung.tier == tier:
                return rung
    for rung in rungs:
        if rung.current:
            return rung
    return rungs[0]


def _tier_headline(rung: LadderRung) -> str:
    """``T1 sonnet@high · claude``: tier, primary model and its provider."""
    primary = rung.models[0] if rung.models else "—"
    provider, _, model = primary.rpartition("/")
    if rung.effort and "@" not in model:
        model = f"{model}@{rung.effort}"
    where = f" [{MUTED}]· {escape(provider)}[/]" if provider else ""
    return f"[b {LIME}]{escape(rung.tier)}[/] [{INK}]{escape(model)}[/]{where}"


def _breaker_note(state: BreakerState, index: int) -> str:
    if state is BreakerState.CLOSED:
        role = "primary · healthy" if index == 0 else "healthy"
        return f"{role} · breaker {state.value}"
    if state is BreakerState.OPEN:
        return f"breaker {state.value} · rerouted"
    return f"breaker {state.value} · one trial left"


def _tier_stats(view: RunView, rung: LadderRung, width: int) -> list[str]:
    stories = iterations = passed = 0
    spent = 0.0
    for rows in view.iterations_by_story.values():
        mine = [row for row in rows if row.tier == rung.tier]
        stories += 1 if mine else 0
        iterations += len(mine)
        spent += sum(row.cost for row in mine)
        passed += sum(1 for row in mine if row.verdict == "PASS")
    cell = max(12, width // 4)
    caps = ("stories reached", "iterations", "passed here", "spent")
    values = (str(stories), str(iterations), str(passed), _money(spent))
    top = "".join(pad(f"[{MUTED}]{cap}[/]", cell) for cap in caps)
    tones = (INK, INK, INK, LIME)
    bottom = "".join(pad(f"[b {tone}]{v}[/]", cell) for v, tone in zip(values, tones, strict=True))
    return [f"[{LINE}]{'─' * min(width, cell * 4)}[/]", top, bottom]


def _escalated_lines(view: RunView, rung: LadderRung) -> list[str]:
    numbers = sorted(
        number
        for number, rows in view.iterations_by_story.items()
        if any(row.escalated_to == rung.tier for row in rows)
    )
    if not numbers:
        return [f"[{DIM}]no story escalated into this rung[/]"]
    lines = [f"[{MUTED}]escalated into this rung[/]"]
    for number in numbers:
        story = _story_of(view, number)
        ident, title = story_label(story) if story else (f"T{number}", "")
        lines.append(
            f"[@click=app.select_story({number})][b {INK}]{escape(ident):<7}[/]"
            f"[{MUTED}]{escape(title)}[/][/]"
        )
    return lines


def _story_of(view: RunView, number: int) -> StoryRow | None:
    for story in view.stories:
        if story.number == number:
            return story
    return None


def tier_view(ctx: ViewContext) -> RenderableType:
    """The pool of one ladder rung, what ran on it and who escalated into it."""
    state = more_state()
    tier = state.tier or ctx.tier
    rung = _pick_rung(ctx.view, tier)
    if rung is None:
        return f"[{DIM}]no ladder in this run[/]"
    width = ctx.width or 76
    lines = [_tier_headline(rung), f"[{MUTED}]pool · first healthy model wins[/]"]
    model_w = max(16, min(38, width - 34))
    for index, model in enumerate(rung.models):
        breaker = runtime.model_state(model)
        closed = breaker is BreakerState.CLOSED
        mark = f"[{LIME}]●[/]" if closed else f"[{RED}]○[/]"
        note = _breaker_note(breaker, index)
        shown = pad(f"[{INK}]{clip(model, model_w)}[/]", model_w)
        lines.append(f"{mark} {shown} [{MUTED}]{note}[/]")
    lines.append("")
    lines.extend(_tier_stats(ctx.view, rung, width))
    lines.append("")
    lines.extend(_escalated_lines(ctx.view, rung))
    return "\n".join(lines)


def _diff_of(ctx: ViewContext) -> StoryDiff | None:
    state = more_state()
    if state.diff is not None:
        return state.diff
    candidate = ctx.diff_text
    if isinstance(candidate, StoryDiff):
        return candidate
    return None


def _patch_lines(patch: str) -> Text:
    """The patch with line numbers, additions and deletions in the design colours."""
    out = Text()
    number = 0
    for raw in patch.rstrip().splitlines():
        if raw.startswith("@@"):
            number = _hunk_start(raw)
            out.append(raw + "\n", style=DIM)
            continue
        if raw.startswith(("diff ", "index ", "--- ")):
            continue
        if raw.startswith("+++ "):
            path = raw[4:].removeprefix("b/")
            out.append(("\n" if out.plain else "") + f"{path}\n", style=f"bold {INK}")
            continue
        if raw.startswith(("new file", "deleted file", "similarity", "rename ")):
            out.append(raw + "\n", style=DIM)
            continue
        sign, code = (raw[:1], raw[1:]) if raw[:1] in "+- " else (" ", raw)
        shown = "" if sign == "-" else str(number)
        out.append(f"{shown:>4}  ", style=DIM)
        if sign == "+":
            out.append(f"+ {code}\n", style=f"{ADD_FG} on {ADD_BG}")
        elif sign == "-":
            out.append(f"- {code}\n", style=f"{DEL_FG} on {DEL_BG}")
        else:
            out.append(f"  {code}\n", style=MUTED)
        number += 0 if sign == "-" else 1
    out.rstrip()
    return out


def _stat_lines(stat: str) -> Text:
    """``path | 12 ++++----``: path light, the pluses lime, the minuses red."""
    out = Text()
    for raw in stat.rstrip().splitlines():
        path, bar, rest = raw.partition("|")
        if not bar:
            out.append(raw.strip() + "\n", style=MUTED)
            continue
        count, _, marks = rest.strip().partition(" ")
        out.append(path.rstrip().ljust(len(path)), style=INK)
        out.append(f"| {count:>4} ", style=DIM)
        out.append(marks.count("+") * "+", style=LIME)
        out.append(marks.count("-") * "-", style=RED)
        out.append("\n")
    out.rstrip()
    return out


def _hunk_start(header: str) -> int:
    try:
        new = header.split("+", 1)[1].split(" ", 1)[0]
        return int(new.split(",", 1)[0])
    except (IndexError, ValueError):
        return 0


def diff_view(ctx: ViewContext) -> RenderableType:
    """The changes of the selected story: the stat first, then the patch."""
    state = more_state()
    story = ctx.story if ctx.story is not None else ctx.view.live_story
    row = _story_of(ctx.view, story) if story is not None else None
    where = state.diff_cwd or "main tree"
    ident = story_label(row)[0] if row is not None else (f"T{story}" if story else "—")
    header = f"story {ident} · {where}"
    diff = _diff_of(ctx)
    if diff is None:
        return f"[{MUTED}]{escape(header)}[/]\n[{DIM}]loading diff… (r to refresh)[/]"
    if diff.error:
        return f"[{MUTED}]{escape(header)}[/]\n[{RED}]{escape(diff.error)}[/]"
    parts: list[RenderableType] = [Text(header, style=MUTED)]
    if diff.stat.strip():
        parts.append(_stat_lines(diff.stat))
    if diff.patch.strip():
        parts.append(_patch_lines(diff.patch))
    if len(parts) == 1:
        parts.append(Text("nothing yet", style=DIM))
    return Group(*parts)


MORE_VIEWS: dict[str, ViewFactory] = {
    "spend": spend_view,
    "tier": tier_view,
    "diff": diff_view,
}

VIEWS.update(MORE_VIEWS)
