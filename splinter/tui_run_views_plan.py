"""PLAN stage views: the run plan, the files each story got and what the filter read.

Before the first story runs there is no story to look at, yet the pipeline has
already written most of the plan: ``knowledge/localization-N.md`` (the files
the localizer picked per story), ``knowledge/filter-N.md`` (what the filter
read in them) and, for the agents strategy, ``agents-run/briefs.json`` (the
waves of briefs). With no story chosen these views show every story at once;
with a story chosen they show that story. They read files only.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from rich.console import RenderableType
from rich.markup import escape
from rich.text import Text

from splinter.tui_run_palette import DIM, INK, LIME, MUTED
from splinter.tui_run_views import ViewContext

_STORY_RE = re.compile(r"^#{2,4}\s+(US-\d+)\s*:\s*(.+?)\s*$", re.MULTILINE)
_PATH_RE = re.compile(r"\*\*`?([^*`]+?)`?\*\*")
_FIRST_SENTENCE = re.compile(r"(.+?[.!?])(\s|$)")
_LINE_MAX = 150


def _empty(message: str) -> Text:
    return Text(message, style="dim")


def _knowledge(ctx: ViewContext, name: str) -> Path:
    return ctx.session.dir / "knowledge" / name


def _title(ctx: ViewContext, number: int) -> str:
    """The story's title: the run's own, else its ``US-NNN: title`` in the PRD."""
    for row in ctx.view.stories:
        if row.number == number and row.title and row.title != f"T{number}":
            return row.title
    titles = _STORY_RE.findall(ctx.files.read(ctx.session.dir / "prd.md"))
    if 0 < number <= len(titles):
        story_id, title = titles[number - 1]
        return f"{story_id} {title.strip()}"
    return ""


def _numbers(ctx: ViewContext, prefix: str) -> list[int]:
    folder = ctx.session.dir / "knowledge"
    found: list[int] = []
    try:
        for path in folder.glob(f"{prefix}-*.md"):
            tail = path.stem.removeprefix(f"{prefix}-")
            if tail.isdigit():
                found.append(int(tail))
    except OSError:
        return []
    return sorted(found)


def _clip(text: str, limit: int = _LINE_MAX) -> str:
    plain = " ".join(text.split())
    return plain if len(plain) <= limit else plain[: limit - 1] + "…"


def localized_files(content: str) -> list[tuple[str, str, str]]:
    """``(file, symbol, reason)`` for every anchor of a localization note."""
    found: list[tuple[str, str, str]] = []
    where = symbol = ""
    for raw in content.splitlines():
        line = raw.strip()
        if line.startswith("### "):
            if where:
                found.append((where, symbol, ""))
            where, _, symbol = line[4:].rstrip(" —").partition(" — ")
        elif line.startswith("reason:") and where:
            found.append((where, symbol, line.removeprefix("reason:").strip()))
            where = symbol = ""
    if where:
        found.append((where, symbol, ""))
    return found


def _story_rule(ctx: ViewContext, number: int, extra: str = "") -> str:
    head = f"[@click=app.select_story({number})][b {LIME}]T{number}[/][/]"
    title = escape(_clip(_title(ctx, number), 70))
    return f"{head} [{INK}]{title}[/]" + (f"  [{DIM}]{escape(extra)}[/]" if extra else "")


def files_overview(ctx: ViewContext) -> RenderableType:
    """Every story with the files the localizer gave it."""
    numbers = _numbers(ctx, "localization")
    if not numbers:
        return _empty("not localized yet · the localizer picks files before the plan")
    lines = [f"[{DIM}]files per story · click a story to open it[/]", ""]
    for number in numbers:
        anchors = localized_files(ctx.files.read(_knowledge(ctx, f"localization-{number}.md")))
        count = f"{len(anchors)} file{'s' if len(anchors) != 1 else ''}"
        lines.append(_story_rule(ctx, number, count))
        lines += [f"   [{MUTED}]{escape(where)}[/]" for where, _, _ in anchors]
        lines.append("")
    return Text.from_markup("\n".join(lines).rstrip())


def _filter_lines(content: str) -> list[str]:
    lines: list[str] = []
    for raw in content.splitlines():
        line = raw.strip()
        if not line:
            continue
        match = _PATH_RE.match(line)
        if match:
            rest = line[match.end():].lstrip(":* ").strip()
            lines += ["", f"[b {INK}]{escape(match.group(1))}[/]"]
            if rest:
                lines.append(f"[{MUTED}]{escape(rest)}[/]")
        elif line.lower().startswith("**recommendation"):
            body = line.split(":", 1)[-1].strip(" *")
            lines += ["", f"[b {LIME}]recommendation[/] [{INK}]{escape(body)}[/]"]
        else:
            lines.append(f"[{INK}]{escape(line.replace('**', '').replace('`', ''))}[/]")
    return lines


def filter_view(ctx: ViewContext) -> RenderableType:
    """What the filter read in a story's files; every story when none is chosen."""
    if ctx.story is not None:
        path = _knowledge(ctx, f"filter-{ctx.story}.md")
        content = ctx.files.read(path)
        if not content.strip():
            return _empty("not filtered yet · the filter reads the files before the plan")
        head = f"[{DIM}]knowledge/{escape(path.name)}[/]"
        return Text.from_markup("\n".join([head, *_filter_lines(content)]))
    numbers = _numbers(ctx, "filter")
    if not numbers:
        return _empty("not filtered yet · the filter reads the files before the plan")
    lines = [f"[{DIM}]what the filter read, per story · click a story to open it[/]", ""]
    for number in numbers:
        content = ctx.files.read(_knowledge(ctx, f"filter-{number}.md")).strip()
        first = _FIRST_SENTENCE.match(" ".join(content.split()))
        summary = first.group(1) if first else content
        lines.append(_story_rule(ctx, number))
        lines.append(f"   [{MUTED}]{escape(_clip(summary.replace('**', '').replace('`', '')))}[/]")
        lines.append("")
    return Text.from_markup("\n".join(lines).rstrip())


def _briefs(ctx: ViewContext) -> list[dict[str, object]]:
    raw = ctx.files.read(ctx.session.dir / "agents-run" / "briefs.json")
    if not raw.strip():
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []


def _waves(briefs: list[dict[str, object]]) -> list[list[dict[str, object]]]:
    """Dependency levels of the briefs, input order kept (unknown deps go last)."""
    ids = {str(b.get("id")) for b in briefs}
    placed: dict[str, int] = {}
    waves: list[list[dict[str, object]]] = []
    pending = list(briefs)
    while pending:
        level = len(waves)
        ready = [
            b
            for b in pending
            if all(str(d) in placed or str(d) not in ids for d in _list(b.get("deps")))
        ] or pending
        waves.append(ready)
        for brief in ready:
            placed[str(brief.get("id"))] = level
        pending = [b for b in pending if b not in ready]
    return waves


def _list(value: object) -> list[str]:
    return [str(v) for v in value] if isinstance(value, list) else []


def _refining(ctx: ViewContext) -> RenderableType:
    """The plan while it forms: stories refined so far and their briefs."""
    raw = ctx.files.read(ctx.session.dir / "agents-run" / "briefs-partial.json")
    try:
        data = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        data = {}
    done = data.get("done", []) if isinstance(data, dict) else []
    total = data.get("total", 0) if isinstance(data, dict) else 0
    localized = len(_numbers(ctx, "localization"))
    filtered = len(_numbers(ctx, "filter"))
    lines = [
        f"[b {INK}]writing the plan[/] [{MUTED}]· {len(done)}/{total or localized} stories "
        f"split into briefs[/]",
        f"[{DIM}]{localized} stories localized · {filtered} filtered · "
        f"files and filters show what the planner reads[/]",
    ]
    for story in done if isinstance(done, list) else []:
        if not isinstance(story, dict):
            continue
        lines += ["", f"[b {LIME}]{escape(str(story.get('task', '')))}[/]"]
        for brief in story.get("briefs", []):
            if isinstance(brief, dict):
                lines.append(
                    f"   [b {INK}]{escape(str(brief.get('id', '')))}[/]  "
                    f"[{MUTED}]{escape(_clip(str(brief.get('goal', '')), 110))}[/]"
                )
    return Text.from_markup("\n".join(lines))


def _runs_parallel(ctx: ViewContext) -> bool:
    try:
        return bool(ctx.session.read_status().get("parallel"))
    except (OSError, ValueError):
        return False


def run_plan_view(ctx: ViewContext) -> RenderableType:
    """The run's plan: waves of briefs (agents), else the planner's notes so far."""
    briefs = _briefs(ctx)
    if not briefs:
        return _refining(ctx)
    parallel = _runs_parallel(ctx)
    pace = "waves overlap, briefs of a wave in parallel" if parallel else "one brief at a time"
    lines = [f"[{DIM}]agents-run/briefs.json · {len(briefs)} briefs · {pace}[/]"]
    for index, wave in enumerate(_waves(briefs), 1):
        size = f"{len(wave)} in parallel" if parallel else f"{len(wave)} brief" + (
            "s" if len(wave) != 1 else ""
        )
        lines += ["", f"[b {LIME}]wave {index}[/] [{DIM}]· {size}[/]"]
        for brief in wave:
            task = str(brief.get("task_id") or brief.get("task") or "")
            effort = str(brief.get("effort") or "")
            meta = " · ".join(part for part in (task, effort) if part)
            lines.append(
                f"[b {INK}]{escape(str(brief.get('id', '')))}[/]  "
                f"[{INK}]{escape(_clip(str(brief.get('goal', '')), 110))}[/]"
                + (f"  [{DIM}]{escape(meta)}[/]" if meta else "")
            )
            files = _list(brief.get("files"))
            if files:
                lines.append(f"   [{MUTED}]{escape(_clip(', '.join(files), 140))}[/]")
            deps = _list(brief.get("deps"))
            if deps:
                lines.append(f"   [{DIM}]after {escape(', '.join(deps))}[/]")
    return Text.from_markup("\n".join(lines))


def overview_paths(ctx: ViewContext) -> tuple[Path, ...]:
    """Files the no-story PLAN views read, so a change repaints them."""
    folder = ctx.session.dir / "knowledge"
    run = ctx.session.dir / "agents-run"
    notes = tuple(
        folder / f"{prefix}-{n}.md"
        for prefix in ("localization", "filter")
        for n in _numbers(ctx, prefix)
    )
    return (run / "briefs.json", run / "briefs-partial.json", folder, *notes)


def _plan_or_run(ctx: ViewContext) -> RenderableType:
    from splinter.tui_run_views import plan_view

    return run_plan_view(ctx) if ctx.story is None else plan_view(ctx)


def _files_or_all(ctx: ViewContext) -> RenderableType:
    from splinter.tui_run_views import files_view

    return files_overview(ctx) if ctx.story is None else files_view(ctx)


PLAN_VIEWS = {"plan": _plan_or_run, "files": _files_or_all, "filter": filter_view}
