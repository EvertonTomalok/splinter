from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from rich.console import Group, RenderableType
from rich.markup import escape
from rich.text import Text

from splinter.memory.session import Session
from splinter.run_snapshot import fmt_span
from splinter.run_view import IterationRow, RunView
from splinter.run_view_files import (
    FileCache,
    gate_lines,
    iteration_lines,
    plan_path,
    review_note_name,
)
from splinter.tui_run_iteration import iteration_detail
from splinter.tui_run_live import LiveAction, actions_for, live_rows
from splinter.tui_run_palette import (
    AMBER,
    DIM,
    INK,
    LIME,
    MUTED,
    RED,
    VERDICT_COLOR,
    verdict_label,
)
from splinter.tui_run_text import clip


@dataclass(frozen=True)
class ViewContext:
    view: RunView
    story: int | None
    opened: frozenset[tuple[int, int]]
    files: FileCache
    session: Session
    multi: bool
    spend_by: str = "cost"
    tier: str = ""
    budget: float | None = None
    diff_text: object = None
    width: int = 0
    live: tuple[LiveAction, ...] = ()
    phase: bool = False
    now: float = 0.0


ViewFactory = Callable[[ViewContext], RenderableType]


def _empty(message: str) -> Text:
    return Text(message, style="dim")


def _session_path(session: Session, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else session.dir / path


def _iteration_paths(ctx: ViewContext) -> tuple[IterationRow, ...]:
    if ctx.story is None:
        return ()
    return ctx.view.iterations_by_story.get(ctx.story, ())


def _run_path(ctx: ViewContext, row: IterationRow) -> Path:
    return _session_path(ctx.session, row.run_file)


def _review_path(ctx: ViewContext, row: IterationRow) -> Path:
    note = row.review_note or review_note_name(ctx.story or 0, row.n)
    filename = note if note.endswith(".md") else f"{note}.md"
    return ctx.session.dir / "knowledge" / filename


def _retry_count(rows: Sequence[IterationRow], index: int) -> int:
    tier = rows[index].tier
    count = 0
    for row in reversed(rows[: index + 1]):
        if row.tier != tier or row.verdict not in {"RETRY", "ESCALATE"}:
            break
        count += 1
    return count


def _model(row: IterationRow) -> str:
    model = " ".join(part for part in (row.tier, row.model) if part)
    return f"{model}@{row.effort}" if row.effort else model


def review_fields(text: str) -> dict[str, str]:
    """``decision``, ``reason`` and ``corrections`` of a review note."""
    fields: dict[str, str] = {}
    lines = text.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip().lstrip("-* ").strip()
        key, _, value = stripped.partition(":")
        if key.lower() in ("decision", "reason") and value.strip():
            fields.setdefault(key.lower(), value.strip().strip("*").strip())
        if stripped.lower().startswith("## corrections"):
            fields["corrections"] = "\n".join(lines[index + 1 :]).strip()
            break
    return fields


def _why(ctx: ViewContext, row: IterationRow) -> str:
    reason = ""
    if row.review_note:
        reason = review_fields(ctx.files.read(_review_path(ctx, row))).get("reason", "")
    if reason:
        room = (ctx.width or 100) - len(_model(row)) - 30
        return clip(reason, max(20, room))
    return f"${row.cost:.2f} · {fmt_span(row.duration_s)}"


def _iteration_header(
    ctx: ViewContext, row: IterationRow, live: bool, opened: bool = False
) -> Text:
    story = ctx.story or 0
    go = f"[@click=app.toggle_iteration({story},{row.n})]"
    model = escape(_model(row))
    if live:
        return Text.from_markup(
            f"{go}[{LIME}]▾[/] [{INK}]iteration {row.n}[/] [{DIM}]·[/] [{INK}]{model}[/]"
            f" [{DIM}]·[/] [{LIME}]live[/][/]"
        )
    arrow = "▾" if opened else "▸"
    verdict = verdict_label(row.verdict) or "live"
    tone = VERDICT_COLOR.get(row.verdict, MUTED)
    return Text.from_markup(
        f"{go}[{DIM}]{arrow}[/] [{MUTED}]iteration {row.n}[/] [{DIM}]·[/] [{MUTED}]{model}[/]"
        f" [{DIM}]·[/] [b {tone}]{verdict}[/] [{DIM}]{_why(ctx, row)}[/][/]"
    )


def act_view(ctx: ViewContext) -> RenderableType:
    rows = _iteration_paths(ctx)
    live_story = ctx.story is not None and ctx.story == ctx.view.live_story
    if not rows:
        if live_story and actions_for(ctx.live, ctx.story, 1):
            return Group(_starting_header(1), _live_feed(ctx, 1))
        return _queued_note(ctx)
    rendered: list[RenderableType] = []
    for index, row in enumerate(rows):
        live = live_story and index == len(rows) - 1 and not row.verdict
        failed_last = index == len(rows) - 1 and row.verdict == "FAIL"
        opened = failed_last != ((ctx.story or 0, row.n) in ctx.opened)
        rendered.append(_iteration_header(ctx, row, live, opened))
        if live:
            rendered.append(_live_feed(ctx, row.n))
        elif opened:
            rendered.append(_detail(ctx, row))
        if row.escalated_to:
            retries = _retry_count(rows, index)
            text = f"↑ escalated {row.tier} → {row.escalated_to} after {retries} RETRY"
            rendered.append(Text(text, style=AMBER))
    upcoming = rows[-1].n + 1
    if live_story and rows[-1].verdict and actions_for(ctx.live, ctx.story, upcoming):
        rendered.append(_starting_header(upcoming))
        rendered.append(_live_feed(ctx, upcoming))
    return Group(*rendered)


def _live_feed(ctx: ViewContext, n: int) -> Text:
    actions = actions_for(ctx.live, ctx.story, n)
    return live_rows(actions, ctx.width or 100, ctx.phase, ctx.now)


def _starting_header(n: int) -> Text:
    return Text.from_markup(
        f"[{LIME}]▾[/] [{INK}]iteration {n}[/] [{DIM}]·[/] [{LIME}]live[/]"
    )


def _detail(ctx: ViewContext, row: IterationRow) -> Text:
    loop = ctx.files.read(ctx.session.dir / "loop.md")
    story = ctx.story or 0
    verdict = next(iter(iteration_lines(loop, story, row.n, "- verdict:")), "")
    review = review_fields(ctx.files.read(_review_path(ctx, row))) if row.review_note else {}
    run_text = ctx.files.read(_run_path(ctx, row))
    gate = gate_lines(loop, story, row.n)
    return iteration_detail(run_text, review, gate, verdict, ctx.width or 100)


def _queued_note(ctx: ViewContext) -> Text:
    live = ctx.view.live_story
    if ctx.story is None or live is None or ctx.story == live:
        return _empty("no iterations yet")
    return Text(
        "Not started. Runs after the live story merges, starting on T0. "
        "Notes you send to this story wait for it.",
        style=MUTED,
    )


def _plan_line(line: str) -> str:
    stripped = line.strip()
    if stripped.startswith("# "):
        return f"[b {INK}]{escape(stripped[2:])}[/]"
    if stripped.startswith("#"):
        return f"[{MUTED}]{escape(stripped.lstrip('#').strip())}[/]"
    number, dot, rest = stripped.partition(". ")
    if dot and number.isdigit():
        return f"[{LIME}]{number}[/]  [{INK}]{escape(rest)}[/]"
    if stripped[:2] in ("- ", "* "):
        return f"   [{DIM}]·[/] [{MUTED}]{escape(stripped[2:])}[/]"
    return f"[{MUTED}]{escape(stripped)}[/]"


def plan_view(ctx: ViewContext) -> RenderableType:
    if ctx.story is None:
        return _empty("no plan yet")
    path = plan_path(ctx.session, ctx.story)
    content = ctx.files.read(path)
    if not content.strip():
        return _empty("not planned yet · the planner plans it when it starts")
    head = f"[{DIM}]knowledge/{escape(path.name)}[/]"
    body = [_plan_line(line) for line in content.splitlines() if line.strip()]
    return Text.from_markup("\n".join([head, *body]))


def _gate_marks(line: str) -> list[str]:
    body = line.removeprefix("- gate:").strip()
    verdict, _, detail = body.partition(" ")
    if verdict.upper() == "PASS":
        return [f"[{LIME}]✓[/]  [b {INK}]{'gate':<8}[/] [{MUTED}]all checks passed[/]"]
    failed = [name.strip() for name in detail.removeprefix("failed:").split(",") if name.strip()]
    if not failed:
        return [f"[{RED}]✗[/]  [b {INK}]{'gate':<8}[/] [{MUTED}]{escape(detail) or 'failed'}[/]"]
    return [f"[{RED}]✗[/]  [b {INK}]{escape(name):<8}[/] [{MUTED}]failed[/]" for name in failed]


def gate_view(ctx: ViewContext) -> RenderableType:
    rows = _iteration_paths(ctx)
    if not rows:
        return _empty("not run yet")
    loop = ctx.files.read(ctx.session.dir / "loop.md")
    lines: list[str] = []
    for row in reversed(rows):
        if lines:
            lines.append("")
        lines.append(f"[{MUTED}]iteration {row.n} · {escape(_model(row))}[/]")
        gated = gate_lines(loop, ctx.story or 0, row.n)
        marks = [mark for line in gated for mark in _gate_marks(line)]
        lines.extend(marks or [f"[{DIM}]no gate output yet[/]"])
    return Text.from_markup("\n".join(lines))


def eval_view(ctx: ViewContext) -> RenderableType:
    rows = _iteration_paths(ctx)
    if not rows:
        return _empty("Not evaluated yet.")
    lines: list[str] = []
    for row in reversed(rows):
        if lines:
            lines.append("")
        lines.append(f"[{MUTED}]eval · iteration {row.n} · {escape(_model(row))}[/]")
        content = ctx.files.read(_review_path(ctx, row)) if row.review_note else ""
        if not content.strip():
            lines.append(f"[{DIM}]no review yet[/]")
            continue
        fields = review_fields(content)
        decision = fields.get("decision", row.verdict)
        tone = VERDICT_COLOR.get(decision.upper(), MUTED)
        reason = fields.get("reason", "") or content.strip()
        word = escape(verdict_label(decision.upper()))
        lines.append(f"[b {tone}]{word}[/] [{INK}]{escape(reason)}[/]")
        if fields.get("corrections"):
            lines.append(f"[{MUTED}]{escape(fields['corrections'])}[/]")
    return Text.from_markup("\n".join(lines))


def _localization_path(ctx: ViewContext) -> Path:
    numbered = ctx.session.dir / "knowledge" / f"localization-{ctx.story}.md"
    return numbered if numbered.exists() else ctx.session.dir / "knowledge" / "localization.md"


def files_view(ctx: ViewContext) -> RenderableType:
    """The code the localizer pointed the plan at: path, lines, symbol and why."""
    if ctx.story is None:
        return _empty("no files yet")
    content = ctx.files.read(_localization_path(ctx))
    lines: list[str] = []
    for raw in content.splitlines():
        line = raw.strip()
        if line.startswith("### "):
            where, _, symbol = line[4:].partition(" — ")
            if lines:
                lines.append("")
            lines.append(f"[b {INK}]{escape(where)}[/]  [{MUTED}]{escape(symbol)}[/]")
        elif line.startswith("reason:"):
            lines.append(f"[{DIM}]{escape(line.removeprefix('reason:').strip())}[/]")
    if not lines:
        return _empty("not localized yet · the localizer picks files before the plan")
    head = f"[{DIM}]knowledge/{escape(_localization_path(ctx).name)}[/]"
    return Text.from_markup("\n".join([head, "", *lines]))


def logs_view(ctx: ViewContext) -> RenderableType:
    """The live log has its own widget; the view body stays empty behind it."""
    return Text("")


VIEWS: dict[str, ViewFactory] = {
    "logs": logs_view,
    "act": act_view,
    "plan": plan_view,
    "files": files_view,
    "gate": gate_view,
    "eval": eval_view,
}


def view_registry() -> dict[str, ViewFactory]:
    from splinter.tui_run_views_plan import PLAN_VIEWS

    registry = dict(VIEWS)
    registry.update(PLAN_VIEWS)
    try:
        from splinter.tui_run_views_more import MORE_VIEWS
    except ImportError:
        return registry
    registry.update(MORE_VIEWS)
    return registry


def relevant_paths(ctx: ViewContext, name: str) -> tuple[Path, ...]:
    if name in ("plan", "files", "filter") and ctx.story is None:
        from splinter.tui_run_views_plan import overview_paths

        return overview_paths(ctx)
    if name == "filter":
        return (ctx.session.dir / "knowledge" / f"filter-{ctx.story}.md",)
    if ctx.story is None:
        return ()
    if name == "act":
        rows = _iteration_paths(ctx)
        expanded = tuple(
            row for index, row in enumerate(rows)
            if not (ctx.story == ctx.view.live_story and index == len(rows) - 1 and not row.verdict)
            and (
                (index == len(rows) - 1 and row.verdict == "FAIL")
                != ((ctx.story, row.n) in ctx.opened)
            )
        )
        runs = tuple(_run_path(ctx, row) for row in expanded)
        reviews = tuple(_review_path(ctx, row) for row in rows if row.review_note)
        loop = (ctx.session.dir / "loop.md",) if expanded else ()
        return (*runs, *reviews, *loop)
    if name == "plan":
        numbered = ctx.session.dir / "knowledge" / f"plan-{ctx.story}.md"
        fallback = plan_path(ctx.session, ctx.story)
        return (numbered,) if numbered == fallback else (numbered, fallback)
    if name == "gate":
        return (ctx.session.dir / "loop.md",)
    if name == "files":
        return (_localization_path(ctx),)
    if name == "eval":
        return tuple(_review_path(ctx, row) for row in _iteration_paths(ctx))
    return ()
