"""Textual TUIs for splinter.

* :class:`AnalyzeApp` — ``splinter analyze`` inspector: a tree of steps + the
  escalation trajectory on the left, a markdown detail pane on the right.
* :class:`RunApp` — ``splinter run`` dashboard: a live overview on the left and a
  real-time log pane on the right streaming what the pipeline is doing, while the
  pipeline executes on a worker thread.

``q`` or ``Ctrl-C`` quits either app (and, for a run, aborts it).
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.markup import escape
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult, SystemCommand
from textual.binding import Binding
from textual.command import DiscoveryHit, Hit, Hits
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.screen import ModalScreen, Screen
from textual.system_commands import SystemCommandsProvider
from textual.timer import Timer
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Markdown,
    OptionList,
    RichLog,
    Rule,
    Select,
    Static,
    TextArea,
    Tree,
)
from textual.widgets.option_list import Option
from textual.widgets.tree import TreeNode
from textual.worker import Worker, WorkerState

from splinter.analyze import (
    _VERDICT_GLYPH,
    _collapse_phases,
    _escalations,
    _eval_segments,
    _has_final_eval_artifacts,
    _is_parallel_mode,
    _iterations,
    _knowledge_notes,
    _loop_block,
    _parse_parallel_tasks,
    _plan_files,
    _plans_from_agentic,
    _prd_feature_name,
    _prd_phases,
    _prd_story_titles,
    _run_state,
    _running_tasks,
    _task_iters,
    _tasks,
    _trace_metrics,
    _verdict_glyph,
    format_run_completion,
)
from splinter.enums import DEFAULT_RUNNER_MODE, RunnerMode
from splinter.memory.session import Session, delete_session, list_sessions
from splinter.obs.events import events_path
from splinter.question_context import Explanation, explain, explain_model, prd_excerpt
from splinter.run_trace import TraceNode, build_trace
from splinter.run_view import RunView, RunViewBuilder
from splinter.run_view_files import FileCache
from splinter.tui_blockword import BlockWord
from splinter.tui_prd_mode import MODES, task_source
from splinter.tui_prd_outline import (
    cap_markup,
    context_cap_markup,
    context_markup,
    cowabunga_hint,
    nav_markup,
    outline_markup,
    spend_markup,
)
from splinter.tui_questions import (
    Question,
    QuestionForm,
    extract_open_questions,
    parse_questions,
)
from splinter.tui_questions_one import AnsweredList, QuestionIndex
from splinter.tui_run_banner import NeedsYouBanner
from splinter.tui_run_chrome import FooterKeys, RunHeader, cowabunga_markup
from splinter.tui_run_composer import Composer
from splinter.tui_run_focus import FocusColumn
from splinter.tui_run_live import LiveAction, live_action
from splinter.tui_run_log import format_line, stamp_prefix
from splinter.tui_run_motion import MotionClock, ToastTracker
from splinter.tui_run_palette import DIM as _DIM
from splinter.tui_run_palette import INK as _INK
from splinter.tui_run_palette import LIME as _LIME
from splinter.tui_run_palette import MUTED as _MUTED
from splinter.tui_run_palette import RED as _RED
from splinter.tui_run_palette import Ink, palette_css, palette_variables
from splinter.tui_run_rail import RunRail
from splinter.tui_run_stop import StopBar, StopMenuModal
from splinter.tui_run_text import story_label
from splinter.tui_run_trace import TraceScreen
from splinter.tui_run_view_key import view_content_key
from splinter.tui_run_views import ViewContext, view_registry
from splinter.tui_run_views_more import MoreState, set_more_state
from splinter.tui_shell import SHELL_CSS, ShellBar, crumb_markup, use_splinter_theme
from splinter.tui_status import BusyStatus
from splinter.tui_stepper import PRD_STEPS, PhaseStepper
from splinter.tui_task_pick import TaskPicker, looks_like_path, system_pick
from splinter.tui_waiting import WaitingRoom
from splinter.vcs.diff import (
    Runner,
    StoryDiff,
    git_runner,
    shown_tree,
    story_cwd,
    story_diff,
    story_worktree,
)

REFRESH_SECONDS = 2.0
AUTO_REFRESH_SECONDS = 2.0
TRACE_TAIL_MAX_BYTES = 256 * 1024
# Default trace view shows only the last 200 events (tail). Older events stay
# reachable via `[` / `]` paging — this caps render work + memory on huge traces.
TRACE_TAIL_MAX_LINES = 200
TRACE_SUMMARY_MIN_LINES = 60
TRACE_SUMMARY_MAX_LINES = 200
TRACE_PAGE_BYTES = 128 * 1024
TRACE_MSG_PREVIEW = 140

_STATE_EMOJI = {
    "RUNNING": "🟡",
    "COMPLETED": "🟢",
    "FAILED": "🔴",
    "INTERRUPTED": "🟠",
    "AWAITING_USER": "🟣",
    "PAUSED": "🟠",
    "DONE": "🟢",
    "UNKNOWN": "⚪",
}

#: Matches the task number inside a log line ("planning task 3 with opus").
_TASK_NO_RE = re.compile(r"\btask (\d+)\b")


def _cap_payload(text: str, limit: int = 20_000) -> str:
    """Return full payload unchanged."""
    _ = limit
    return text


_SPLINTER = """\
```
        🐀   ~ Master Splinter ~

      🐢 🐢 🐢 🐢   cowabunga!
```\
"""


def _overview_md(session: Session, state: str) -> str:
    status = session.read_status()
    metrics = _trace_metrics(session)
    loop = session.read("loop.md")
    iters = _iterations(loop)
    from splinter.agents.localizer import _count_anchors, _parse_anchors

    localization_text = session.read("knowledge/localization.md")
    anchors_count = _count_anchors(localization_text)
    parsed = _parse_anchors(localization_text) if localization_text else []
    hot = [a for a in parsed if a.relevance == "hot"]

    lines = [
        _SPLINTER,
        f"# {session.id}",
        f"{_STATE_EMOJI.get(state, '⚪')} **{state}** · "
        f"strategy `{status.get('strategy', '?')}` · tasks {status.get('tasks', '?')}",
        "",
    ]
    if metrics:
        lines.append(
            f"💰 **${metrics.get('cost', '0')}** · "
            f"runs {metrics.get('runs', '0')} · tokens `{metrics.get('tokens', '{}')}`"
        )
        lines.append("")

    lines.append("## Steps")
    lines.append(f"- localize — {anchors_count} anchors")
    if hot:
        for a in hot[:5]:
            reason_snip = a.reason
            if len(reason_snip) > 100:
                reason_snip = reason_snip[:97] + "..."
            loc = a.file + (f":L{a.line_start}" if a.line_start else "")
            lines.append(f"  - `{loc}` — {reason_snip}")
        if len(hot) > 5:
            lines.append(f"  - + {len(hot) - 5} more hot anchors")
    all_plans = _plan_files(session)
    if len(all_plans) > 1:
        lines.append(f"- plan — {len(all_plans)} plans")
        for filename, label in all_plans:
            content = session.read(filename)
            has_plan = "✓" if content.strip() else "pending"
            lines.append(f"  - {label} — {has_plan}")
    else:
        lines.append(f"- plan — {'✓' if session.read('knowledge/plan.md') else 'pending'}")
    if iters:
        n, tier, verdict = iters[-1]
        lines.append(
            f"- run/eval — iter {n}/{status.get('max_iterations', '?')} "
            f"· {tier} · last **{verdict}**"
        )
    else:
        lines.append("- run/eval — pending")

    phases = _prd_phases(session.read("prd_phases.md"))
    if phases or iters:
        lines.append("")
        lines.append("## Trajectory")
        if phases:
            collapsed = _collapse_phases(phases)
            prd_parts = [f"{name} x{count}" if count > 1 else name for name, count in collapsed]
            lines.append("**PRD** " + " -> ".join(prd_parts))
        if iters:
            tally: dict[str, int] = {}
            for _, _, v in iters:
                tally[v] = tally.get(v, 0) + 1
            order = list(_VERDICT_GLYPH)
            ranked = [v for v in order if v in tally] + [v for v in tally if v not in order]
            tally_parts = []
            for v in ranked:
                glyph, _ = _verdict_glyph(v)
                tally_parts.append(f"{glyph} {tally[v]}")
            lines.append(f"**Run** · {len(iters)} iters · " + " · ".join(tally_parts))
        for task_no, _title, task_iters in _task_iters(loop, session=session):
            if not task_iters:
                continue
            esc = _escalations(task_iters)
            cells = []
            for idx, tier, verdict in task_iters:
                glyph, _ = _verdict_glyph(verdict)
                prefix = "⤴ " if (idx - 1) in esc else ""
                cells.append(f"{prefix}{idx} `{tier}` {glyph}")
            lines.append(f"- **Task {task_no}** " + "   ".join(cells))
    return "\n".join(lines)


def _iteration_md(session: Session, task_no: int, n: int) -> str:
    """Render iteration detail for a specific task.

    Checks ``runs/phase-{n}.md`` for phase runs before falling back to
    ``runs/iter-{n}.md`` for main-loop iterations.
    """
    tasks = (
        _parse_parallel_tasks(session, session.read("loop.md"))
        if _is_parallel_mode(session.read("loop.md"))
        else _tasks(session.read("loop.md"))
    )
    if task_no < 1 or task_no > len(tasks):
        return f"_task {task_no} not found_"

    _, _, task_body = tasks[task_no - 1]
    summary = _loop_block(task_body, n)
    task_run_file = f"runs/task-{task_no}-iter-{n}.md" if task_no > 1 else f"runs/iter-{n}.md"
    run_out = (
        session.read(f"runs/phase-{n}.md").strip()
        or session.read(task_run_file).strip()
        or session.read(f"runs/iter-{n}.md").strip()
    )

    eval_segments = _eval_segments(session.read("eval.md"), len(tasks))
    eval_block = ""
    if task_no <= len(eval_segments):
        parts = re.split(r"^### Iter (\d+):", eval_segments[task_no - 1], flags=re.MULTILINE)
        for i in range(1, len(parts), 2):
            if int(parts[i]) == n:
                eval_block = parts[i + 1].strip()
                break

    md = [f"# Iteration {n}"]
    if summary:
        md.append("## Summary")
        md.append(summary)
    if run_out:
        md.append("## Runner output")
        md.append(f"```\n{_cap_payload(run_out)}\n```")
    if eval_block:
        md.append("## Eval verdict")
        md.append(eval_block)
    if len(md) == 1:
        md.append("_no data for this iteration yet._")
    return "\n\n".join(md)


def _file_md(session: Session, label: str, filename: str) -> str:
    content = session.read(filename).strip()
    if not content:
        return f"# {label}\n\n_empty_"
    return f"# {label}\n\n{_cap_payload(content)}"


def _build_plan_label(idx: int, total: int) -> str:
    """Tree label for the single paginated plan node. ``idx`` 0 = overview,
    1..N = plan-N; ◂/▸ hints appear only when there is somewhere to page to."""
    if total <= 1:
        return "plan"
    name = "plans" if idx == 0 else f"plan-{idx}"
    return f"◂ {name} ▸ ({idx}/{total})"


def _plan_overview_md(session: Session) -> str:
    """Summary screen shown when the plan node is first selected (before
    paginating). Renders a table of every plan; ◂/▸ switch between them."""
    _plans_from_agentic(session)
    plans = _plan_files(session)
    if not plans:
        content = session.read("knowledge/plan.md").strip()
        if content:
            return _file_md(session, "Plan", "knowledge/plan.md")
        # Recover plan files from agentic trace (legacy sessions where they were
        # deleted by an older eval-fix round) then re-scan so normal flow applies.
        if _plans_from_agentic(session) > 0:
            plans = _plan_files(session)
        if not plans:
            return "# Plans\n\n_no plan yet._"
    lines = [
        "# Plans",
        "",
        f"{len(plans)} plan(s) · use ◂ / ▸ to page through them",
        "",
        "| # | Plan | Steps | Title |",
        "|---|------|-------|-------|",
    ]
    for i, (filename, label) in enumerate(plans, 1):
        content = session.read(filename).strip()
        steps_n = len(re.findall(r"^\s*\d+\.", content, re.MULTILINE))
        steps = str(steps_n) if content else "—"
        title_m = re.search(
            r"^##\s+(?:Implementation Plan\s+[—-]+\s+)?(.+)$", content, re.MULTILINE
        )
        title = title_m.group(1).strip() if title_m else ("pending" if not content else label)
        lines.append(f"| {i} | {label} | {steps} | {title} |")
    return "\n".join(lines)


@dataclass(frozen=True)
class _TraceRow:
    ts: str
    stage: str
    message: str
    count: int


@dataclass(frozen=True)
class _TraceRender:
    markdown: str
    row_count: int
    has_older: bool
    has_newer: bool
    max_offset: int
    normalized_offset: int


def _trace_tail_lines(visible_rows: int | None) -> int:
    if visible_rows is None or visible_rows <= 0:
        return TRACE_TAIL_MAX_LINES
    target = max(visible_rows * 3, TRACE_SUMMARY_MIN_LINES)
    return min(target, TRACE_SUMMARY_MAX_LINES)


def _trace_source(session: Session) -> Path:
    """The single physical file backing the Trace pane's byte-offset paging."""
    return events_path(session)


def _trace_rows_from_text(text: str) -> list[_TraceRow]:
    """Parse ``log``-typed records out of a raw ``events.jsonl`` slice. ``run``-typed
    records (model cost entries) are skipped — this pane shows the chronological
    narrative, not the cost ledger."""
    rows: list[_TraceRow] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            rows.append(_TraceRow(ts="", stage="event", message=line, count=1))
            continue
        if not isinstance(data, dict) or data.get("type") != "log":
            continue
        payload = data.get("payload") or {}
        stage = str(payload.get("stage", "event")) or "event"
        msg = str(payload.get("message", "")).strip() or line
        ts = str(data.get("ts", ""))
        rows.append(_TraceRow(ts=ts, stage=stage, message=msg, count=1))
    return rows


def _format_trace_rows(
    rows: list[_TraceRow],
    *,
    start_index: int = 1,
    selected_index: int | None = None,
    expanded_index: int | None = None,
) -> list[str]:
    """Render trace rows to markdown blocks (one per row). ``start_index`` lets
    appended rows continue the numbering of rows already on screen."""
    out: list[str] = []
    for offset, row in enumerate(rows):
        idx = start_index + offset
        prefix = ">" if idx == selected_index else " "
        count = f" x{row.count}" if row.count > 1 else ""
        msg = row.message
        if len(msg) > TRACE_MSG_PREVIEW:
            msg = msg[: TRACE_MSG_PREVIEW - 3].rstrip() + "..."
        ts = row.ts[11:19] if row.ts and "T" in row.ts else ""
        ts_field = f"{ts} " if ts else ""
        out.append(f"{prefix} {idx:>3}. {ts_field}{row.stage:<14} {msg}{count}")
        if expanded_index == idx:
            detail = row.message
            if row.count > 1:
                detail = f"{row.message}\n(repeated {row.count} times)"
            out.append(f"```\n{detail}\n```")
    return out


def _read_appended_text(path: Path, start: int) -> tuple[str, int]:
    """Read bytes appended to ``path`` since byte offset ``start``. Returns
    ``(text, new_end)``. Event logs append whole lines, so ``start`` is always a
    line boundary and the decoded chunk needs no partial-line trimming."""
    if not path.exists():
        return ("", start)
    size = path.stat().st_size
    if size <= start:
        return ("", size)
    with path.open("rb") as f:
        f.seek(start)
        chunk = f.read(size - start)
    return (chunk.decode("utf-8", errors="replace"), size)


def _collapse_trace_rows(rows: list[_TraceRow]) -> list[_TraceRow]:
    if not rows:
        return []
    collapsed: list[_TraceRow] = []
    for row in rows:
        if collapsed and collapsed[-1].stage == row.stage and collapsed[-1].message == row.message:
            prev = collapsed[-1]
            collapsed[-1] = _TraceRow(
                ts=prev.ts or row.ts,
                stage=prev.stage,
                message=prev.message,
                count=prev.count + 1,
            )
            continue
        collapsed.append(row)
    return collapsed


def _trace_render(
    session: Session,
    *,
    offset_from_end: int = 0,
    visible_rows: int | None = None,
    selected_row: int | None = None,
    expanded_row: int | None = None,
) -> _TraceRender:
    metrics = _trace_metrics(session)
    path = _trace_source(session)
    target_rows = _trace_tail_lines(visible_rows)
    text, truncated_before, has_newer, source_size, normalized_offset = _tail_file_text(
        path,
        max_bytes=TRACE_TAIL_MAX_BYTES,
        max_lines=max(target_rows * 3, target_rows),
        offset_from_end=offset_from_end,
    )
    rows = _collapse_trace_rows(_trace_rows_from_text(text))
    if len(rows) > target_rows:
        rows = rows[-target_rows:]
        truncated_before = True
    row_count = len(rows)
    selected = row_count if selected_row is None else selected_row
    if row_count > 0:
        selected = max(1, min(selected, row_count))
    else:
        selected = 0
    expanded = expanded_row if expanded_row is not None and expanded_row == selected else None

    parts = ["# Trace"]
    if metrics:
        parts.append(
            f"💰 **${metrics.get('cost', '0')}** · runs {metrics.get('runs', '0')} · "
            f"tokens `{metrics.get('tokens', '{}')}`"
        )

    has_older = truncated_before
    parts.append(
        f"_offset {normalized_offset} bytes · "
        f"{'older yes' if has_older else 'older no'} · "
        f"{'newer yes' if has_newer else 'newer no'}_"
    )
    parts.append(
        "_keys: `[` older · `]` newer · `0` newest · `j/k` select row · `enter` expand selected_"
    )

    if not rows:
        parts.append("## Events\n\nrun in progress — no events yet")
        return _TraceRender(
            markdown="\n\n".join(parts),
            row_count=0,
            has_older=has_older,
            has_newer=has_newer,
            max_offset=max(0, source_size - TRACE_PAGE_BYTES),
            normalized_offset=normalized_offset,
        )

    parts.append("## Events")
    parts.extend(_format_trace_rows(rows, selected_index=selected, expanded_index=expanded))

    return _TraceRender(
        markdown="\n\n".join(parts),
        row_count=row_count,
        has_older=has_older,
        has_newer=has_newer,
        max_offset=max(0, source_size - TRACE_PAGE_BYTES),
        normalized_offset=normalized_offset,
    )


def _trace_md(
    session: Session,
    *,
    offset_from_end: int = 0,
    visible_rows: int | None = None,
    selected_row: int | None = None,
    expanded_row: int | None = None,
) -> str:
    return _trace_render(
        session,
        offset_from_end=offset_from_end,
        visible_rows=visible_rows,
        selected_row=selected_row,
        expanded_row=expanded_row,
    ).markdown


def _final_eval_rounds_md(session: Session) -> str:
    """Render all final eval rounds — LLM review + user decision, split by PRD round."""
    kdir = session.dir / "knowledge"
    round_files = sorted(
        kdir.glob("final-eval-*.md"),
        key=lambda p: int(
            re.match(r"final-eval-(\d+)", p.stem).group(1)  # type: ignore[union-attr]
            if re.match(r"final-eval-(\d+)", p.stem)
            else 0
        ),
    )

    if round_files:
        parts = ["# Final Eval"]
        for rf in round_files:
            content = rf.read_text().strip()
            parts.append(f"\n\n---\n\n{content}")
        return "\n".join(parts)

    # Fallback: single final_eval.md with decision inferred from status.
    content = session.read("final_eval.md").strip()
    if not content:
        if (session.dir / "final_eval.yaml").exists():
            return "# Final Eval\n\n_pending — eval has not run yet._"
        return "# Final Eval\n\n_no final eval configured._"
    status = session.read_status()
    state = str(status.get("state", ""))
    fe_passed = status.get("final_eval_passed")
    if fe_passed:
        decision = "\n\n---\n\n**User Decision: ✅ Approved**"
    elif state == "awaiting_validation":
        decision = "\n\n---\n\n*⏳ Awaiting review…*"
    elif state == "failed" and str(status.get("stage", "")) == "final_eval":
        decision = "\n\n---\n\n**User Decision: ❌ Rejected**"
    else:
        decision = ""
    return f"# Final Eval\n\n{content}{decision}"


def _task_md(session: Session, task_no: int, title: str) -> str:
    """Detail for an expanded task node — title, iteration tally, last verdict."""
    tasks = (
        _parse_parallel_tasks(session, session.read("loop.md"))
        if _is_parallel_mode(session.read("loop.md"))
        else _tasks(session.read("loop.md"))
    )
    if task_no < 1 or task_no > len(tasks):
        return f"_task {task_no} not found_"

    _, _, task_body = tasks[task_no - 1]
    iters = _iterations(task_body)

    lines = [f"# Task {task_no}" + (f" · {title}" if title else "")]
    if iters:
        tally: dict[str, int] = {}
        for _, _, verdict in iters:
            tally[verdict] = tally.get(verdict, 0) + 1

        parts = []
        for verdict in tally:
            glyph, color = _verdict_glyph(verdict)
            parts.append(f"[{color}]{glyph}[/] {tally[verdict]} {verdict.lower()}")
        if parts:
            lines.append("")
            lines.append("  " + "   ".join(parts))

        lines.append("")
        lines.append("## Iterations")
        for n, tier, verdict in iters:
            glyph, color = _verdict_glyph(verdict)
            lines.append(f"- #{n} · {tier} · [{color}]{glyph} {verdict}[/]")
    else:
        lines.append("")
        lines.append("_no iterations yet_")
    return "\n".join(lines)


def _story_md(session: Session, us_id: str, title: str) -> str:
    """Detail for a single-shot story leaf — that story's ``### US-NNN`` PRD block,
    with its acceptance checkboxes showing done/pending state."""
    prd = session.read("prd.md")
    if not prd.strip():
        return f"_{title or us_id}_\n\n_no PRD on disk_"
    block = re.search(rf"(###\s+{re.escape(us_id)}\b.*?)(?=###\s+US-\d+|\Z)", prd, re.DOTALL)
    if not block:
        return f"# {title or us_id}\n\n_story block not found_"
    return block.group(1).strip()


def _prd_phase_md(session: Session, phase: str, detail: str, occurrence: int = 0) -> str:
    """Detail for a trajectory phase — routed to the artifact that phase produced."""
    from splinter import prd_session

    phase_l = phase.lower()
    head = f"# {phase}" + (f" — {detail}" if detail else "")

    if phase_l == "run":
        metrics = _trace_metrics(session)
        iters = _iterations(session.read("loop.md"))
        parts = [head]
        if metrics:
            parts.append(
                f"💰 **${metrics.get('cost', '0')}** · runs {metrics.get('runs', '0')} · "
                f"tokens `{metrics.get('tokens', '{}')}`"
            )
        if iters:
            parts.append(
                "## Iterations\n" + "\n".join(f"- #{n} · {tier} · {v}" for n, tier, v in iters)
            )
        # Live tail of the event log so an in-flight run shows the model working
        # (it logs tool calls / text before any iteration is finalized in loop.md).
        tail_text = session.render_events_tail(64 * 1024)
        tail = tail_text.splitlines()[-25:] if tail_text.strip() else []
        if tail:
            parts.append("## Live\n\n```\n" + "\n".join(tail) + "\n```")
        elif not iters:
            parts.append("_starting run…_")
        return "\n\n".join(parts)

    if phase_l == "strategy":
        titles = prd_session.user_story_titles(session.read("prd.md"))
        chosen = detail or str(session.read_status().get("strategy", "?"))
        parts = [head, f"**Strategy:** `{chosen}`"]
        if titles:
            parts.append("## Tasks\n" + "\n".join(f"- {escape(t)}" for t in titles))
        return "\n\n".join(parts)

    versions = prd_session.list_prd_versions(session)
    matched = prd_session.version_for_phase(versions, phase, occurrence)
    if matched is not None:
        return prd_session.render_prd_version_compare(session, matched)

    prd = session.read("prd.md").strip()
    body = _cap_payload(prd) if prd else "_PRD draft not captured yet._"
    return f"# PRD · {phase}{f' — {detail}' if detail else ''}\n\n{body}"


_PALETTE_CSS = """
    CommandPalette { align-horizontal: right; }
    CommandPalette > Vertical { width: 55%; }
"""

_SCROLL_LEFT_PANE_CSS = """
    #nav {
        height: 1fr;
        overflow-y: auto;
        overflow-x: hidden;
        scrollbar-size-vertical: 1;
    }
    #overview-scroll {
        height: 1fr;
    }
"""

_MAXIMIZE_CSS = """
    App.--maximized {
        #nav { display: none; }
        #overview { display: none; }
        #draftpane { display: none; }
        #rail { display: none; }
        #cmd-bar { display: none; }
    }
"""

#: Thin scrollbars for the run screen: no horizontal bar on the log or the
#: overview, a one-cell vertical bar.
_RUN_PANE_CSS = """
    #overview-scroll {
        overflow-x: hidden;
        scrollbar-size-horizontal: 0;
        scrollbar-size-vertical: 1;
    }
    RichLog {
        overflow-x: hidden;
        scrollbar-size-horizontal: 0;
        scrollbar-size-vertical: 1;
    }
"""


def _path_sig(path: Path) -> tuple[int, int]:
    if not path.exists():
        return (0, 0)
    stat = path.stat()
    return (stat.st_mtime_ns, stat.st_size)


def _tail_file_text(
    path: Path,
    *,
    max_bytes: int,
    max_lines: int,
    offset_from_end: int = 0,
) -> tuple[str, bool, bool, int, int]:
    if not path.exists():
        return ("", False, False, 0, 0)
    size = path.stat().st_size
    max_offset = max(0, size - max_bytes)
    normalized_offset = max(0, min(offset_from_end, max_offset))
    end = size - normalized_offset
    start = max(0, end - max_bytes)
    with path.open("rb") as f:
        f.seek(start)
        chunk = f.read(end - start)
    text = chunk.decode("utf-8", errors="replace")
    truncated_before = start > 0
    if truncated_before:
        nl = text.find("\n")
        text = text[nl + 1 :] if nl >= 0 else ""
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
        truncated_before = True
    has_newer = end < size
    return ("\n".join(lines).strip(), truncated_before, has_newer, size, normalized_offset)


def _find_shortcuts_cmd(screen: Any, app: Any) -> SystemCommand:
    if screen.query("HelpPanel"):
        return SystemCommand(
            "Find Shortcuts",
            "Hide the keys and widget help panel",
            app.action_hide_help_panel,
        )
    return SystemCommand(
        "Find Shortcuts",
        "Show keys and shortcuts for the focused widget",
        app.action_show_help_panel,
    )


def _filter_models(models: list[str], text: str) -> list[str]:
    q = text.strip().lower()
    if not q:
        return models
    return [m for m in models if q in m.lower()]


class _OrderedCommandsProvider(SystemCommandsProvider):
    """Preserves get_system_commands insertion order (no alphabetical sort)."""

    async def discover(self) -> Hits:
        for cmd in self.app.get_system_commands(self.screen):
            if cmd.discover:
                yield DiscoveryHit(cmd.title, cmd.callback, help=cmd.help)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for cmd in self.app.get_system_commands(self.screen):
            if (match := matcher.match(cmd.title)) > 0:
                yield Hit(match, matcher.highlight(cmd.title), cmd.callback, help=cmd.help)


class AnalyzeApp(App[None]):
    """Live session inspector."""

    CSS = (
        """
    Tree { width: 38%; border-right: solid $primary; }
    #detail { padding: 0 1; }
    """
        + _PALETTE_CSS
        + _SCROLL_LEFT_PANE_CSS
        + _MAXIMIZE_CSS
    )

    COMMANDS = {_OrderedCommandsProvider}

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("ctrl+c", "quit", "Quit"),
        ("r", "reload", "Refresh"),
        ("R", "toggle_auto", "Auto-refresh"),
    ]

    _maximized: reactive[bool] = reactive(False)
    _auto: reactive[bool] = reactive(False)

    def __init__(self, session: Session) -> None:
        super().__init__()
        self.session = session
        self._traj_node: TreeNode[Any] | None = None
        self._kn_node: TreeNode[Any] | None = None
        self._kn_labels: set[str] = set()
        self._timer: Any = None
        self._expanded_tasks: set[int] = set()
        self._steps_node: TreeNode[Any] | None = None
        self._final_eval_node: TreeNode[Any] | None = None
        self._plan_node: TreeNode[Any] | None = None
        self._plan_idx: int = 0  # 0 = overview, 1..N = plan-N.md
        self._traj_sig: tuple[Any, ...] | None = None
        self._trace_offset: int = 0
        self._trace_selected_row: int = 1
        self._trace_expanded_row: int | None = None
        self._trace_row_count: int = 0
        self._trace_sig: tuple[Any, ...] | None = None
        # Incremental-append state: track the source file + byte offset already
        # rendered so a refresh appends only new events instead of re-rendering.
        self._trace_append_path: Path | None = None
        self._trace_append_bytes: int = 0
        self._trace_last_index: int = 0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            yield Tree("session", id="nav")
            with VerticalScroll():
                yield Markdown(id="detail")
        yield Footer()

    def on_mount(self) -> None:
        self._build_tree()
        self._do_reload()
        # Auto-refresh stays OFF by default (even on a live run) — polling a big
        # trace every tick is slow + memory-hungry. Shift+R toggles it on; `r`
        # does a one-shot manual refresh whenever auto is off.
        self._auto = False

    # --- tree ---
    def _final_eval_step_label(self) -> str:
        status = self.session.read_status()
        final_eval_md = self.session.read("final_eval.md").strip()
        fe_passed = status.get("final_eval_passed")
        state = str(status.get("state", ""))
        fe_summary = str(status.get("final_eval_summary", "")).strip()
        if fe_passed:
            fe_icon = "✅"
        elif state == "awaiting_validation":
            fe_icon = "🔍"
        elif state == "failed" and status.get("stage") == "final_eval":
            fe_icon = "❌"
        elif final_eval_md or fe_summary:
            fe_icon = "📋"
        else:
            fe_icon = "⏳"
        return f"{fe_icon} final_eval"

    def _refresh_steps(self) -> None:
        if self._steps_node is None:
            return
        has_fe = _has_final_eval_artifacts(self.session)
        if not has_fe:
            if self._final_eval_node is not None:
                self._final_eval_node.remove()
                self._final_eval_node = None
            return
        label = self._final_eval_step_label()
        if self._final_eval_node is None:
            self._final_eval_node = self._steps_node.add_leaf(
                label,
                data={"kind": "file", "label": "Final Eval", "file": "final_eval.md"},
            )
            return
        self._final_eval_node.label = label

    def _build_tree(self) -> None:
        tree = self.query_one("#nav", Tree)
        tree.root.expand()

        overview = tree.root.add_leaf("📊 Overview", data={"kind": "overview"})
        overview.allow_expand = False

        trace = tree.root.add_leaf("🔍 trace", data={"kind": "trace"})
        trace.allow_expand = False

        steps = tree.root.add("🧩 Steps", expand=True)
        self._steps_node = steps
        if self.session.read("prd.md"):
            from splinter import prd_session

            prd_versions = prd_session.prd_version_files(self.session)
            if prd_versions:
                prd_node = steps.add("📄 PRD", expand=True)
                prd_node.add_leaf(
                    "current",
                    data={"kind": "file", "label": "PRD (current)", "file": "prd.md"},
                )
                for vfile, vlabel in prd_versions:
                    prd_node.add_leaf(
                        vlabel,
                        data={"kind": "prd_version", "label": vlabel, "file": vfile},
                    )
            else:
                steps.add_leaf("prd", data={"kind": "file", "label": "PRD", "file": "prd.md"})
        steps.add_leaf(
            "localize",
            data={"kind": "file", "label": "Localization", "file": "knowledge/localization.md"},
        )
        plans = _plan_files(self.session)
        self._plan_node = steps.add_leaf(
            _build_plan_label(self._plan_idx, len(plans)),
            data={"kind": "file", "label": "Plan", "file": "knowledge/plan.md"},
        )
        steps.add_leaf("eval", data={"kind": "file", "label": "Eval", "file": "eval.md"})
        self._refresh_steps()

        notes = _knowledge_notes(self.session)
        extra = [
            (fn, lbl)
            for fn, lbl in notes
            if lbl not in ("plan", "localization")
            and not lbl.startswith("plan-")
            and not lbl.startswith("final-eval-")
        ]
        if extra:
            self._kn_node = tree.root.add("📝 Knowledge", expand=True)
            for filename, label in extra:
                self._kn_labels.add(label)
                self._kn_node.add_leaf(
                    label, data={"kind": "file", "label": label, "file": filename}
                )

        self._traj_node = tree.root.add("📈 Trajectory", expand=True)
        self._refresh_trajectory()

    def _refresh_trajectory(self) -> None:
        if self._traj_node is None:
            return

        sig = (
            _path_sig(self.session.dir / "prd_phases.md"),
            _path_sig(self.session.dir / "loop.md"),
            _path_sig(self.session.dir / "prd.md"),
            _path_sig(self.session.dir / "prd_versions.md"),
            _path_sig(self.session.dir / "phases.md"),
        )
        if sig == self._traj_sig:
            return
        self._traj_sig = sig

        currently_expanded: set[int] = set()
        if self._traj_node.children:
            for child in self._traj_node.children:
                try:
                    is_expanded = bool(getattr(child, "_expanded", False))
                except Exception:
                    is_expanded = False
                if is_expanded and child.data and child.data.get("kind") == "task":
                    task_no = child.data.get("n")
                    if task_no:
                        currently_expanded.add(task_no)
        self._expanded_tasks.update(currently_expanded)

        self._traj_node.remove_children()
        phase_counts: dict[str, int] = {}
        for phase, detail in _prd_phases(self.session.read("prd_phases.md")):
            occ = phase_counts.get(phase, 0)
            phase_counts[phase] = occ + 1
            label = f"📝 {phase}" + (f" · {detail}" if detail else "")
            self._traj_node.add_leaf(
                label,
                data={
                    "kind": "prd_phase",
                    "phase": phase,
                    "detail": detail,
                    "occurrence": occ,
                },
            )

        loop_md = self.session.read("loop.md")
        tasks = (
            _parse_parallel_tasks(self.session, loop_md)
            if _is_parallel_mode(loop_md)
            else _tasks(loop_md)
        )
        if len(tasks) > 1:
            for task_no, title, task_body in tasks:
                task_iters = _iterations(task_body)
                task_node = self._traj_node.add(
                    f"🗂 Task {task_no} · {title}" if title else f"🗂 Task {task_no}",
                    data={"kind": "task", "n": task_no, "title": title},
                )
                for n, tier, verdict in task_iters:
                    task_node.add_leaf(
                        f"#{n} · {tier} · {verdict}",
                        data={"kind": "iter", "task": task_no, "n": n},
                    )
                if task_no in self._expanded_tasks:
                    task_node.expand()
        else:
            iters = _iterations(loop_md)
            stories = _prd_story_titles(self.session)
            if len(stories) > 1:
                # Single-shot (raphael): one task node holding the PRD stories + iters.
                feature = _prd_feature_name(self.session)
                task_node = self._traj_node.add(
                    f"🗂 Task · {feature} ({len(stories)} stories)",
                    data={"kind": "task", "n": 1, "title": feature},
                    expand=True,
                )
                for st in stories:
                    sid = st.split(":", 1)[0].strip()
                    task_node.add_leaf(f"📋 {st}", data={"kind": "story", "id": sid, "title": st})
                for n, tier, verdict in iters:
                    task_node.add_leaf(
                        f"#{n} · {tier} · {verdict}",
                        data={"kind": "iter", "task": 1, "n": n},
                    )
            else:
                for n, tier, verdict in iters:
                    self._traj_node.add_leaf(
                        f"#{n} · {tier} · {verdict}",
                        data={"kind": "iter", "task": 1, "n": n},
                    )

        from splinter.analyze import _phase_entries

        phase_md = self.session.read("phases.md")
        if phase_md.strip():
            for pnum, pstatus, pmodel, pcost in _phase_entries(phase_md):
                glyph = "✓" if pstatus == "PASS" else "✗"
                self._traj_node.add_leaf(
                    f"Phase {pnum} · {pstatus} · {glyph} · {pmodel} · ${pcost}",
                    data={"kind": "phase", "n": pnum},
                )

    # --- detail ---
    def _detail(self) -> Markdown:
        return self.query_one("#detail", Markdown)

    def _show_overview(self) -> None:
        self._detail().update(_overview_md(self.session, _run_state(self.session)))

    def _render_data(self, data: dict[str, Any] | None) -> None:
        if not data:
            self._show_overview()
            return
        kind = data.get("kind")
        if kind == "iter":
            task_no = data.get("task", 1)
            self._detail().update(_iteration_md(self.session, task_no, data["n"]))
        elif kind == "task":
            self._detail().update(_task_md(self.session, data["n"], data.get("title", "")))
        elif kind == "story":
            self._detail().update(_story_md(self.session, data["id"], data.get("title", "")))
        elif kind == "prd_phase":
            self._detail().update(
                _prd_phase_md(
                    self.session,
                    data["phase"],
                    data["detail"],
                    int(data.get("occurrence", 0) or 0),
                )
            )
        elif kind == "phase":
            self._detail().update(_phase_detail_md(self.session, data["n"]))
        elif kind == "trace":
            self._render_trace(force=False)
        elif kind == "prd_version":
            from splinter import prd_session

            versions = prd_session.list_prd_versions(self.session)
            num_s = re.search(r"v(\d+)", str(data.get("label", "")))
            if num_s:
                ver = next((v for v in versions if v.num == int(num_s.group(1))), None)
                if ver is not None:
                    self._detail().update(prd_session.render_prd_version_compare(self.session, ver))
                    return
            self._detail().update(_file_md(self.session, data["label"], data["file"]))
        elif kind == "file":
            if data.get("file") == "knowledge/plan.md":
                self._render_plan()
            elif data.get("file") == "final_eval.md":
                self._detail().update(_final_eval_rounds_md(self.session))
            else:
                self._detail().update(_file_md(self.session, data["label"], data["file"]))
        else:
            self._show_overview()

    def _render_plan(self) -> None:
        """Render the plan pane for the current ``_plan_idx`` (0 = overview,
        1..N = plan-N.md). Shared by ◂/▸ pagination and auto-reload so a refresh
        never snaps the visible plan back to the first one."""
        _plans_from_agentic(self.session)
        plans = _plan_files(self.session)
        total = len(plans)
        if self._plan_idx > total:  # plan count shrank between reloads
            self._plan_idx = total
        if self._plan_node is not None:
            self._plan_node.label = _build_plan_label(self._plan_idx, total)
        if self._plan_idx == 0:
            self._detail().update(_plan_overview_md(self.session))
        else:
            file = plans[self._plan_idx - 1][0]
            self._detail().update(_file_md(self.session, f"Plan {self._plan_idx}", file))

    def _is_plan_node_focused(self) -> bool:
        try:
            node = self.query_one("#nav", Tree).cursor_node
            return node is not None and node is self._plan_node
        except Exception:
            return False

    def _is_trace_node_focused(self) -> bool:
        try:
            node = self.query_one("#nav", Tree).cursor_node
            if node is None:
                return False
            data = node.data if isinstance(node.data, dict) else {}
            return data.get("kind") == "trace"
        except Exception:
            return False

    def _trace_visible_rows(self) -> int | None:
        try:
            height = int(self._detail().size.height)
            return height if height > 0 else None
        except Exception:
            return None

    def _compute_trace_sig(self, visible_rows: int | None) -> tuple[Any, ...]:
        return (
            _path_sig(self.session.dir / "events.jsonl"),
            self._trace_offset,
            self._trace_selected_row,
            self._trace_expanded_row,
            visible_rows or 0,
        )

    def _try_append_trace(self, visible_rows: int | None) -> bool:
        """Append only newly-written events to the Markdown widget. Returns True
        when it handled the refresh; False to fall back to a full re-render
        (source switched/rotated, file shrank, or the tail cap was hit)."""
        if self._trace_append_path is None:
            return False
        path = _trace_source(self.session)
        if path != self._trace_append_path:
            return False
        size = path.stat().st_size if path.exists() else 0
        if size <= self._trace_append_bytes:
            return False  # no growth, shrank, or rotated — let the caller decide
        text, new_end = _read_appended_text(path, self._trace_append_bytes)
        new_rows = _collapse_trace_rows(_trace_rows_from_text(text))
        if not new_rows:
            self._trace_append_bytes = new_end
            return False
        # Cap the live widget: once appends pile past 2x the tail target, do a
        # full re-render so the trimmed-tail view reclaims widget memory.
        target_rows = _trace_tail_lines(visible_rows)
        if self._trace_last_index + len(new_rows) > target_rows * 2:
            return False
        body = _format_trace_rows(
            new_rows,
            start_index=self._trace_last_index + 1,
            selected_index=self._trace_selected_row,
            expanded_index=None,
        )
        self._detail().append("\n\n".join(body))
        self._trace_append_bytes = new_end
        self._trace_last_index += len(new_rows)
        self._trace_row_count = self._trace_last_index
        self._trace_sig = self._compute_trace_sig(visible_rows)
        return True

    def _reset_trace_append(self, row_count: int) -> None:
        """After a full render, re-anchor append tracking to the newest tail.
        Disabled while paged into history (offset != 0) — appends only apply at
        the live tail."""
        self._trace_last_index = row_count
        if self._trace_offset != 0:
            self._trace_append_path = None
            return
        path = _trace_source(self.session)
        self._trace_append_path = path
        self._trace_append_bytes = path.stat().st_size if path.exists() else 0

    def _render_trace(self, force: bool) -> None:
        visible_rows = self._trace_visible_rows()
        sig = self._compute_trace_sig(visible_rows)
        if not force and sig == self._trace_sig:
            return
        # Live tail grew with no paging/expand interaction — append just the new
        # events instead of rebuilding the whole trace markdown.
        if (
            not force
            and self._trace_offset == 0
            and self._trace_expanded_row is None
            and self._try_append_trace(visible_rows)
        ):
            return
        rendered = _trace_render(
            self.session,
            offset_from_end=self._trace_offset,
            visible_rows=visible_rows,
            selected_row=self._trace_selected_row,
            expanded_row=self._trace_expanded_row,
        )
        self._trace_row_count = rendered.row_count
        self._trace_offset = rendered.normalized_offset
        if self._trace_row_count > 0:
            self._trace_selected_row = max(
                1,
                min(self._trace_selected_row, self._trace_row_count),
            )
            if (
                self._trace_expanded_row is not None
                and self._trace_expanded_row > self._trace_row_count
            ):
                self._trace_expanded_row = None
        else:
            self._trace_selected_row = 1
            self._trace_expanded_row = None
        self._trace_sig = self._compute_trace_sig(visible_rows)
        self._detail().update(rendered.markdown)
        self._reset_trace_append(rendered.row_count)

    def on_key(self, event: events.Key) -> None:
        if self._is_plan_node_focused():
            total = len(_plan_files(self.session))
            if total == 0:
                return
            if event.key == "right":
                event.prevent_default()
                self._plan_idx = 0 if self._plan_idx >= total else self._plan_idx + 1
            elif event.key == "left":
                event.prevent_default()
                self._plan_idx = total if self._plan_idx <= 0 else self._plan_idx - 1
            else:
                return
            self._render_plan()
            return

        if not self._is_trace_node_focused():
            return

        if event.key == "[":
            event.prevent_default()
            path = _trace_source(self.session)
            size = path.stat().st_size if path.exists() else 0
            max_offset = max(0, size - TRACE_PAGE_BYTES)
            self._trace_offset = min(self._trace_offset + TRACE_PAGE_BYTES, max_offset)
            self._render_trace(force=True)
            return
        if event.key == "]":
            event.prevent_default()
            self._trace_offset = max(0, self._trace_offset - TRACE_PAGE_BYTES)
            self._render_trace(force=True)
            return
        if event.key == "0":
            event.prevent_default()
            self._trace_offset = 0
            self._render_trace(force=True)
            return
        if event.key == "j":
            event.prevent_default()
            if self._trace_row_count > 0:
                self._trace_selected_row = min(self._trace_selected_row + 1, self._trace_row_count)
                self._trace_expanded_row = None
                self._render_trace(force=True)
            return
        if event.key == "k":
            event.prevent_default()
            if self._trace_row_count > 0:
                self._trace_selected_row = max(self._trace_selected_row - 1, 1)
                self._trace_expanded_row = None
                self._render_trace(force=True)
            return
        if event.key == "enter":
            event.prevent_default()
            if self._trace_row_count > 0:
                if self._trace_expanded_row == self._trace_selected_row:
                    self._trace_expanded_row = None
                else:
                    self._trace_expanded_row = self._trace_selected_row
                self._render_trace(force=True)

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted[Any]) -> None:
        self._render_data(event.node.data)

    # --- actions ---
    def action_reload(self) -> None:
        # `r` — one-shot manual refresh, only while auto-refresh is off
        # (when auto is on, the interval timer already does the polling).
        if self._auto:
            return
        self._do_reload()

    def action_toggle_auto(self) -> None:
        """Shift+R — flip the 1s auto-refresh on/off."""
        self._auto = not self._auto

    def _refresh_knowledge(self) -> None:
        notes = _knowledge_notes(self.session)
        extra = [
            (fn, lbl)
            for fn, lbl in notes
            if lbl not in ("plan", "localization")
            and not lbl.startswith("plan-")
            and not lbl.startswith("final-eval-")
        ]
        new = [(fn, lbl) for fn, lbl in extra if lbl not in self._kn_labels]
        if not new:
            return
        tree = self.query_one("#nav", Tree)
        if self._kn_node is None:
            self._kn_node = tree.root.add("📝 Knowledge", expand=True)
        for filename, label in new:
            self._kn_labels.add(label)
            self._kn_node.add_leaf(label, data={"kind": "file", "label": label, "file": filename})

    def _do_reload(self) -> None:
        state = _run_state(self.session)
        self.title = f"splinter analyze · {self.session.id}"
        self._refresh_steps()
        self._refresh_trajectory()
        self._refresh_knowledge()

        node = self.query_one("#nav", Tree).cursor_node
        self._render_data(node.data if node is not None else None)

        # Run finished — stop auto-polling (watcher tears the timer down).
        if state != "RUNNING" and self._auto:
            self._auto = False
        else:
            self._update_subtitle(state)

    def _update_subtitle(self, state: str) -> None:
        emoji = _STATE_EMOJI.get(state, "⚪")
        self.sub_title = f"{emoji} {state} · auto-refresh {'on' if self._auto else 'off'}"

    def watch__auto(self, val: bool) -> None:
        if val and self._timer is None:
            self._timer = self.set_interval(AUTO_REFRESH_SECONDS, self._do_reload)
        elif not val and self._timer is not None:
            self._timer.stop()
            self._timer = None
        self._update_subtitle(_run_state(self.session))

    def get_system_commands(self, screen: Any) -> Iterable[SystemCommand]:
        yield _find_shortcuts_cmd(screen, self)
        yield SystemCommand("Theme", "Change the current theme", self.action_change_theme)
        if self._maximized:
            yield SystemCommand("Minimize", "Restore default layout", self.action_toggle_maximize)
        else:
            yield SystemCommand("Maximize", "Maximize right panel", self.action_toggle_maximize)
        yield SystemCommand(
            "Screenshot",
            "Save an SVG screenshot of the current screen",
            lambda: self.set_timer(0.1, self.deliver_screenshot),
        )
        yield SystemCommand("Quit", "Quit the application", self.action_quit)

    def action_toggle_maximize(self) -> None:
        self._maximized = not self._maximized

    def watch__maximized(self, val: bool) -> None:
        self.set_class(val, "--maximized")


def run_tui(session: Session) -> None:
    AnalyzeApp(session).run()


# --- session browser -------------------------------------------------------


class SessionPicker(App[str | None]):
    """Pick a session: ↑/↓ navigate, Enter open, d delete, q quit."""

    CSS = "DataTable { height: 1fr; }"

    BINDINGS = [
        ("enter", "open", "Open"),
        ("d", "delete", "Delete"),
        ("q", "quit", "Quit"),
        ("escape", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(cursor_type="row")
        yield Footer()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        # Enter on a row — DataTable emits this; open the session.
        self.exit(event.row_key.value)

    def on_mount(self) -> None:
        self.title = "splinter · sessions"
        table = self.query_one(DataTable)
        table.add_columns("session", "state", "cost")
        self._reload()
        table.focus()

    def _reload(self) -> None:
        from splinter.prd_session import prune_dead_prd_sessions

        prune_dead_prd_sessions()  # drop abandoned empty refinements before listing
        table = self.query_one(DataTable)
        table.clear()
        sessions = list_sessions()
        if not sessions:
            self.sub_title = "no sessions"
            return
        self.sub_title = f"{len(sessions)} session(s)"
        for sid in sessions:
            session = Session(sid)
            metrics = _trace_metrics(session)
            table.add_row(sid, _run_state(session), f"${metrics.get('cost', '0')}", key=sid)

    def _current_id(self) -> str | None:
        table = self.query_one(DataTable)
        if table.row_count == 0:
            return None
        row = table.coordinate_to_cell_key(table.cursor_coordinate).row_key
        return str(row.value) if row.value is not None else None

    def action_open(self) -> None:
        self.exit(self._current_id())

    def action_delete(self) -> None:
        sid = self._current_id()
        if sid:
            delete_session(sid)
            self._reload()


def run_session_browser() -> int:
    """Loop: pick a session in the TUI, view it, return to the picker on quit."""
    while True:
        sid = SessionPicker().run()
        if not sid:
            return 0
        AnalyzeApp(Session(sid)).run()


# --- run dashboard ---------------------------------------------------------


class _TextualLogHandler(logging.Handler):
    """Forwards ``splinter`` log records to a RichLog on the app thread."""

    def __init__(self, app: Any) -> None:
        super().__init__()
        self.app = app

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        try:
            self.app.call_from_thread(self.app.write_log, msg, record.levelno)
        except Exception:
            pass  # app shutting down


class _GapModal(ModalScreen[str]):
    """Shown when the pipeline pauses due to a provider gap (rc=2).

    Countdown sleeps then retries; or the user can switch to Claude or exit.
    """

    DEFAULT_CSS = """
    _GapModal {
        align: center middle;
        background: $background 60%;
    }
    _GapModal > Vertical#gap-dialog {
        width: 70;
        height: auto;
        border: round $warning;
        background: $surface;
        padding: 1 3;
    }
    _GapModal #gap-header {
        height: auto;
        margin-bottom: 1;
    }
    _GapModal #gap-title {
        text-style: bold;
        color: $warning;
        width: 1fr;
    }
    _GapModal #gap-provider-label {
        color: $text-muted;
        text-align: right;
        width: auto;
    }
    _GapModal #gap-body {
        color: $text;
        margin-bottom: 1;
    }
    _GapModal #gap-countdown {
        text-align: center;
        text-style: bold;
        color: $warning;
        background: $warning 15%;
        height: 1;
        margin-bottom: 1;
        display: none;
    }
    _GapModal.counting #gap-countdown {
        display: block;
    }
    _GapModal #gap-picker {
        height: 3;
        align: center middle;
        margin-bottom: 1;
        display: none;
    }
    _GapModal.picking #gap-picker {
        display: block;
    }
    _GapModal.picking #gap-actions {
        display: none;
    }
    _GapModal #gap-picker-label {
        height: 3;
        content-align: center middle;
        margin-right: 1;
        color: $text-muted;
    }
    _GapModal #gap-picker-unit {
        height: 3;
        content-align: center middle;
        margin: 0 1;
        color: $text-muted;
    }
    _GapModal #gap-duration {
        width: 12;
    }
    _GapModal #start {
        width: auto;
        min-width: 12;
        margin-left: 1;
    }
    _GapModal #gap-actions {
        height: 3;
        align-horizontal: center;
    }
    _GapModal Button {
        width: 1fr;
        height: 3;
        margin: 0 1;
        border: none;
        text-style: bold;
    }
    _GapModal Button:focus {
        text-style: bold underline;
    }
    _GapModal Button.-active {
        border: none;
    }
    """

    BINDINGS = [
        ("s", "press('sleep')", "Sleep & retry"),
        ("c", "press('claude')", "Use Claude"),
        ("e", "press('exit')", "Exit"),
        ("escape", "exit_modal", "Cancel"),
    ]

    countdown: reactive[int] = reactive(0)

    def __init__(
        self,
        kind: str = "",
        provider: str = "",
        retry_after: int | None = None,
    ) -> None:
        super().__init__()
        self._kind = kind
        self._provider = provider
        self._sleep_secs = retry_after or 60
        self._tick_timer: Any = None

    def compose(self) -> ComposeResult:
        kind_label = self._kind.replace("_", " ").title() if self._kind else "Provider Gap"
        is_billing = self._kind == "insufficient_balance"
        body = (
            "Provider is out of balance. Top up, then retry — or switch this run to "
            "Claude to keep going."
            if is_billing
            else "Provider is unavailable after repeated retries. Wait and retry, "
            "switch to Claude, or stop the run."
        )
        with Vertical(id="gap-dialog"):
            with Horizontal(id="gap-header"):
                yield Static(f"⏸  Run Paused · {kind_label}", id="gap-title")
                if self._provider:
                    yield Static(f"via {self._provider}", id="gap-provider-label")
            yield Rule()
            yield Static(body, id="gap-body")
            yield Label("", id="gap-countdown")
            with Horizontal(id="gap-picker"):
                yield Label("Sleep for", id="gap-picker-label")
                yield Input(
                    value=str(self._sleep_secs),
                    type="integer",
                    id="gap-duration",
                )
                yield Label("seconds", id="gap-picker-unit")
                yield Button("Start", id="start", variant="success")
            with Horizontal(id="gap-actions"):
                yield Button("  Sleep & Retry  (s)", id="sleep", variant="warning")
                yield Button("  Use Claude  (c)", id="claude", variant="primary")
                yield Button("  Exit  (e)", id="exit", variant="error")

    def on_mount(self) -> None:
        self.query_one("#claude", Button).focus()

    def watch_countdown(self, value: int) -> None:
        label = self.query_one("#gap-countdown", Label)
        label.update(f"⏳  Retrying in {value}s — press Esc to cancel" if value > 0 else "")

    def action_press(self, button_id: str) -> None:
        try:
            self.query_one(f"#{button_id}", Button).press()
        except Exception:
            if button_id == "exit":
                self.action_exit_modal()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "sleep":
            self._open_picker()
        elif bid == "start":
            self._start_countdown()
        else:
            self._cancel_timer()
            self.dismiss(bid or "exit")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "gap-duration":
            self._start_countdown()

    def _open_picker(self) -> None:
        self.add_class("picking")
        inp = self.query_one("#gap-duration", Input)
        inp.value = str(self._sleep_secs)
        inp.focus()

    def _start_countdown(self) -> None:
        raw = self.query_one("#gap-duration", Input).value.strip()
        try:
            secs = int(raw)
        except ValueError:
            secs = self._sleep_secs
        secs = max(1, secs)
        self._sleep_secs = secs
        self.remove_class("picking")
        self.add_class("counting")
        self.countdown = secs
        self._tick_timer = self.set_interval(1.0, self._tick)

    def _tick(self) -> None:
        self.countdown -= 1
        if self.countdown <= 0:
            self._cancel_timer()
            self.dismiss("retry")

    def _cancel_timer(self) -> None:
        if self._tick_timer is not None:
            self._tick_timer.stop()
            self._tick_timer = None

    def action_exit_modal(self) -> None:
        # Esc cancels the active countdown or duration picker before exiting.
        if self.has_class("counting"):
            self._cancel_timer()
            self.remove_class("counting")
            self.countdown = 0
            self.query_one("#sleep", Button).focus()
            return
        if self.has_class("picking"):
            self.remove_class("picking")
            self.query_one("#sleep", Button).focus()
            return
        self._cancel_timer()
        self.dismiss("exit")


class _GateLangModal(ModalScreen[str | None]):
    """Picker for language presets to append to gate checks."""

    DEFAULT_CSS = """
    _GateLangModal {
        align: center middle;
        background: $background 60%;
    }
    _GateLangModal > Vertical#lang-dialog {
        width: 50;
        height: auto;
        border: round $primary;
        background: $surface;
        padding: 1 2;
    }
    _GateLangModal #lang-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }
    _GateLangModal #lang-list {
        height: 10;
        margin-bottom: 1;
    }
    _GateLangModal #lang-actions {
        height: 3;
        align-horizontal: center;
    }
    _GateLangModal Button {
        width: 1fr;
        height: 3;
        margin: 0 1;
        border: none;
        text-style: bold;
    }
    """

    BINDINGS = [
        ("enter", "confirm", "Confirm"),
        ("escape", "exit_modal", "Cancel"),
        ("q", "exit_modal", "Cancel"),
    ]

    def __init__(self) -> None:
        super().__init__()
        from splinter.configure import gate_default_languages

        self._languages = gate_default_languages()

    def compose(self) -> ComposeResult:
        with Vertical(id="lang-dialog"):
            yield Static("Choose language preset to append", id="lang-title")
            yield Rule()
            lang_list = OptionList(*self._languages, id="lang-list")
            yield lang_list
            with Horizontal(id="lang-actions"):
                yield Button("  Select  (enter)", id="confirm", variant="success")
                yield Button("  Cancel  (esc)", id="cancel", variant="error")

    def on_mount(self) -> None:
        self.query_one("#lang-list", OptionList).focus()

    def action_confirm(self) -> None:
        self.query_one("#confirm", Button).press()

    def action_exit_modal(self) -> None:
        self.query_one("#cancel", Button).press()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        idx = event.option_index
        if idx is not None and 0 <= idx < len(self._languages):
            self.dismiss(self._languages[idx])

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or "cancel"
        if bid == "confirm":
            opt_list = self.query_one("#lang-list", OptionList)
            highlighted = opt_list.highlighted
            if highlighted is not None and 0 <= highlighted < len(self._languages):
                self.dismiss(self._languages[highlighted])
            else:
                self.dismiss(None)
        else:
            self.dismiss(None)


class _AskUserModal(ModalScreen[tuple[str, str] | None]):
    """Shown when the eval loop needs human judgment (ASK_USER / max-tier escalate)."""

    DEFAULT_CSS = """
    _AskUserModal {
        align: center middle;
        background: $background 60%;
    }
    _AskUserModal > Vertical#ask-dialog {
        width: 95%;
        height: 92%;
        border: round $primary;
        background: $surface;
        padding: 0;
    }
    _AskUserModal #ask-header {
        padding: 1 2 0 2;
        height: auto;
    }
    _AskUserModal #ask-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }
    _AskUserModal #ask-skill-scroll {
        height: 1fr;
        border: round $panel;
        margin: 0 2 1 2;
        padding: 0 1;
    }
    _AskUserModal #ask-input-label {
        margin: 0 2 0 2;
        height: auto;
        color: $text-muted;
    }
    _AskUserModal #ask-response {
        height: 8;
        margin: 0 2 1 2;
    }
    _AskUserModal #ask-actions {
        height: 3;
        align-horizontal: center;
        margin: 0 2 1 2;
    }
    _AskUserModal Button {
        width: 1fr;
        height: 3;
        margin: 0 1;
        border: none;
        text-style: bold;
    }
    """

    BINDINGS = [
        ("a", "submit_answer", "Answer"),
        ("f", "accept_done", "Accept / Done"),
        ("p", "jump_premium", "Jump Premium"),
        ("c", "action_cowabunga", "Cowabunga"),
        ("d", "edit_config", "Edit Config"),
        ("e", "exit_modal", "Exit"),
        ("escape", "exit_modal", "Cancel"),
    ]

    def __init__(self, reason: str = "", corrections: str = "", prefill: str = "") -> None:
        super().__init__()
        self._reason = reason
        self._corrections = corrections
        self._prefill = prefill

    def compose(self) -> ComposeResult:
        display = self._corrections or self._reason or "The evaluator needs guidance to continue."
        with Vertical(id="ask-dialog"):
            with Vertical(id="ask-header"):
                yield Static("❓  Run Paused · Your input needed", id="ask-title")
                yield Rule()
            with VerticalScroll(id="ask-skill-scroll"):
                yield Markdown(display)
            yield Label("Your feedback (optional — sent to the runner):", id="ask-input-label")
            yield TextArea(self._prefill, id="ask-response")
            with Horizontal(id="ask-actions"):
                yield Button("  Answer  (a)", id="answer", variant="success")
                yield Button("  ✓ Accept / Done  (f)", id="accept_done", variant="success")
                yield Button("  Jump Premium  (p)", id="jump_premium", variant="primary")
                yield Button("  Cowabunga  (c)", id="cowabunga", variant="warning")
                yield Button("  Edit Config  (d)", id="edit_config", variant="default")
                yield Button("  Exit  (e)", id="exit", variant="error")

    def on_mount(self) -> None:
        self.query_one("#ask-response", TextArea).focus()

    def action_submit_answer(self) -> None:
        self.query_one("#answer", Button).press()

    def action_accept_done(self) -> None:
        self.query_one("#accept_done", Button).press()

    def action_jump_premium(self) -> None:
        self.query_one("#jump_premium", Button).press()

    def action_cowabunga(self) -> None:
        self.query_one("#cowabunga", Button).press()

    def action_edit_config(self) -> None:
        self.query_one("#edit_config", Button).press()

    def action_exit_modal(self) -> None:
        self.query_one("#exit", Button).press()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or "exit"
        if bid == "answer":
            text = self.query_one("#ask-response", TextArea).text.strip()
            self.dismiss(("answer", text))
        elif bid == "jump_premium":
            text = self.query_one("#ask-response", TextArea).text.strip()
            self.dismiss(("jump_premium", text))
        elif bid == "accept_done":
            self.dismiss(("accept", ""))
        elif bid == "cowabunga":
            self.dismiss(("cowabunga", ""))
        elif bid == "edit_config":
            text = self.query_one("#ask-response", TextArea).text.strip()
            self.dismiss(("edit_config", text))
        else:
            self.dismiss(None)


class _FinalEvalModal(ModalScreen[dict[str, str | None] | None]):
    """Configure the session-scoped final eval gate.

    Dismiss value: dict with keys {kind, name, cmd, skill, provider, model, effort}
    or None to cancel.
    """

    DEFAULT_CSS = """
    _FinalEvalModal {
        align: center middle;
        background: $background 60%;
    }
    _FinalEvalModal > Vertical#fe-dialog {
        width: 68;
        height: 90%;
        border: round $primary;
        background: $surface;
        padding: 0;
    }
    _FinalEvalModal #fe-header {
        padding: 1 2 0 2;
        height: auto;
    }
    _FinalEvalModal #fe-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }
    _FinalEvalModal #fe-scroll {
        height: 1fr;
        padding: 0 2;
    }
    _FinalEvalModal #fe-kind-list {
        height: 5;
        margin-bottom: 1;
    }
    _FinalEvalModal #fe-detail-area {
        height: auto;
        margin-bottom: 1;
    }
    _FinalEvalModal .fe-label {
        margin-top: 1;
    }
    _FinalEvalModal #fe-provider-list {
        height: 6;
    }
    _FinalEvalModal .fe-model-filter {
        width: 1fr;
        height: 3;
    }
    _FinalEvalModal #fe-model-list {
        height: 8;
    }
    _FinalEvalModal #fe-effort-list {
        height: 7;
    }
    _FinalEvalModal #fe-actions {
        height: 3;
        align-horizontal: center;
        padding: 0 2;
        margin-top: 1;
        margin-bottom: 1;
    }
    _FinalEvalModal Button {
        width: 1fr;
        height: 3;
        margin: 0 1;
        border: none;
        text-style: bold;
    }
    """

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("enter", "confirm", "Confirm"),
    ]

    _KINDS = ["User Review (ask_user)", "Run Skill", "Run Command"]
    _PROVIDERS = ["(default)", "claude", "opencode", "codex", "cursor"]
    _EFFORTS = ["(default)", "low", "medium", "high", "max"]

    def __init__(self, current: dict[str, str | None] | None = None) -> None:
        super().__init__()
        self._current = current or {}
        self._all_models: dict[str, list[str]] = {}
        self._current_model_opts: list[str] = ["(default)"]
        self._fe_filter: str = ""
        self._fe_full_models: list[str] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="fe-dialog"):
            with Vertical(id="fe-header"):
                yield Static("⚙️  Set Final Eval", id="fe-title")
                yield Rule()
            with VerticalScroll(id="fe-scroll"):
                yield Label("Kind:", classes="fe-label")
                yield OptionList(*self._KINDS, id="fe-kind-list")
                with Vertical(id="fe-detail-area"):
                    yield Label("Skill name / Shell command:", classes="fe-label")
                    yield Input(placeholder="skill name or shell command", id="fe-input")
                    yield Label("Provider (optional):", classes="fe-label")
                    yield OptionList(*self._PROVIDERS, id="fe-provider-list")
                    yield Label("Model (optional):", classes="fe-label")
                    yield Input(
                        placeholder="filter models…",
                        id="fe-model-filter",
                        classes="fe-model-filter",
                    )
                    yield OptionList("(default)", id="fe-model-list")
                    yield Label("Effort (optional):", classes="fe-label")
                    yield OptionList(*self._EFFORTS, id="fe-effort-list")
            with Horizontal(id="fe-actions"):
                yield Button("  Confirm  (enter)", id="fe-confirm", variant="success")
                yield Button("  Cancel  (esc)", id="fe-cancel", variant="error")

    def on_mount(self) -> None:
        self._all_models = self._load_models()
        full_list = self._all_models.get("(default)", [])
        self._fe_full_models = full_list
        self._rebuild_model_list(["(default)"] + full_list)
        self.query_one("#fe-kind-list", OptionList).focus()
        self._update_detail_visibility()

    def _load_models(self) -> dict[str, list[str]]:
        try:
            from splinter.configure import available_models_by_provider

            by_provider = available_models_by_provider()
        except Exception:
            by_provider = {
                "claude": ["sonnet", "opus"],
                "opencode": [],
                "codex": ["codex/gpt-5-codex"],
            }
        all_m = sorted({m for models in by_provider.values() for m in models})
        return {
            "(default)": all_m,
            "claude": by_provider.get("claude", []),
            "opencode": by_provider.get("opencode", []),
            "codex": by_provider.get("codex", []),
            "cursor": by_provider.get("cursor", []),
        }

    def _rebuild_model_list(self, options: list[str]) -> None:
        model_list = self.query_one("#fe-model-list", OptionList)
        current_idx: Any = model_list.highlighted
        current: str = ""
        if current_idx is not None and current_idx < len(self._current_model_opts):
            current = self._current_model_opts[current_idx]
        self._current_model_opts = options
        model_list.clear_options()
        for opt in options:
            model_list.add_option(opt)
        if current and current in options:
            for i, opt in enumerate(options):
                if opt == current:
                    model_list.highlighted = i
                    break

    def _update_detail_visibility(self) -> None:
        kind_idx = self.query_one("#fe-kind-list", OptionList).highlighted or 0
        self.query_one("#fe-detail-area", Vertical).display = kind_idx > 0

    def _update_model_list(self) -> None:
        provider_idx = self.query_one("#fe-provider-list", OptionList).highlighted
        raw_provider = self._PROVIDERS[provider_idx] if provider_idx is not None else "(default)"
        models = self._all_models.get(raw_provider, self._all_models.get("(default)", []))
        self._fe_full_models = models
        opts = _filter_models(models, self._fe_filter)
        self._rebuild_model_list(["(default)"] + opts)

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option_list.id == "fe-kind-list":
            self._update_detail_visibility()
        elif event.option_list.id == "fe-provider-list":
            self._update_model_list()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "fe-model-filter":
            self._fe_filter = event.value
            opts = _filter_models(self._fe_full_models, self._fe_filter)
            self._rebuild_model_list(["(default)"] + opts)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_confirm(self) -> None:
        self.query_one("#fe-confirm", Button).press()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "fe-cancel":
            self.dismiss(None)
            return
        if bid != "fe-confirm":
            return
        kind_idx = self.query_one("#fe-kind-list", OptionList).highlighted or 0
        detail = self.query_one("#fe-input", Input).value.strip()
        provider_idx = self.query_one("#fe-provider-list", OptionList).highlighted
        raw_provider = self._PROVIDERS[provider_idx] if provider_idx is not None else "(default)"
        provider = None if raw_provider == "(default)" else raw_provider
        model_idx = self.query_one("#fe-model-list", OptionList).highlighted
        raw_model = self._current_model_opts[model_idx] if model_idx is not None else "(default)"
        model = None if raw_model == "(default)" else raw_model
        effort_idx = self.query_one("#fe-effort-list", OptionList).highlighted
        raw_effort = self._EFFORTS[effort_idx] if effort_idx is not None else "(default)"
        effort = None if raw_effort == "(default)" else raw_effort
        if kind_idx == 0:
            self.dismiss(
                {
                    "kind": "ask_user",
                    "name": "review",
                    "cmd": None,
                    "skill": None,
                    "provider": None,
                    "model": None,
                    "effort": None,
                }
            )
        elif kind_idx == 1:
            self.dismiss(
                {
                    "kind": "skill",
                    "name": detail or "skill-eval",
                    "cmd": None,
                    "skill": detail or None,
                    "provider": provider,
                    "model": model,
                    "effort": effort,
                }
            )
        else:
            self.dismiss(
                {
                    "kind": "command",
                    "name": detail.split()[0] if detail else "cmd",
                    "cmd": detail or None,
                    "skill": None,
                    "provider": provider,
                    "model": model,
                    "effort": effort,
                }
            )


class _ConfirmStopModal(ModalScreen[str | None]):
    """Confirm before pausing or killing the run.

    Dismiss values:
      "pause"  → graceful stop (finish current iteration, then pause)
      "kill"   → terminate subprocesses immediately and restart from checkpoint
      "kill_exit" → terminate subprocesses immediately and exit run UI
      None     → cancel
    """

    DEFAULT_CSS = """
    _ConfirmStopModal {
        align: center middle;
        background: $background 60%;
    }
    _ConfirmStopModal > Vertical#stop-dialog {
        width: 60;
        height: auto;
        border: round $warning;
        background: $surface;
        padding: 1 2;
    }
    _ConfirmStopModal #stop-title {
        text-style: bold;
        color: $warning;
        margin-bottom: 1;
    }
    _ConfirmStopModal #stop-actions {
        height: 3;
        align-horizontal: center;
        margin-top: 1;
    }
    _ConfirmStopModal Button {
        width: 1fr;
        height: 3;
        margin: 0 1;
        border: none;
        text-style: bold;
    }
    """

    BINDINGS = [
        ("p", "do_pause", "Pause/Chat"),
        ("escape", "do_cancel", "Cancel"),
    ]

    def __init__(self, action: str) -> None:
        super().__init__()
        self._action = action  # "pause" | "kill" | "kill_exit"

    def compose(self) -> ComposeResult:
        if self._action == "pause":
            title = "⏸  Pause/Chat after current iteration?"
            detail = "Current step finishes, then run pauses. Resume with 'splinter resume'."
        elif self._action == "kill_exit":
            title = "🛑  Kill process now and exit?"
            detail = "Subprocesses terminate immediately. Resume later with 'splinter resume'."
        else:
            title = "🛑  Kill process now?"
            detail = "Subprocesses terminate immediately, then run restarts from checkpoint."
        with Vertical(id="stop-dialog"):
            yield Static(title, id="stop-title")
            yield Static(f"[dim]{detail}[/]")
            with Horizontal(id="stop-actions"):
                yield Button("  Confirm  (Enter)", id="confirm", variant="warning")
                yield Button("  Cancel  (Esc)", id="cancel", variant="primary")

    def on_mount(self) -> None:
        self.query_one("#confirm", Button).focus()

    def action_do_pause(self) -> None:
        self.dismiss("pause")

    def action_do_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm":
            self.dismiss(self._action)
        else:
            self.dismiss(None)


def _load_all_models_flat() -> list[str]:
    try:
        from splinter.configure import available_models_by_provider

        by_provider = available_models_by_provider()
    except Exception:
        by_provider = {
            "claude": ["sonnet", "opus"],
            "opencode": [],
            "codex": ["codex/gpt-5-codex"],
            "cursor": ["cursor/composer-2.5", "cursor/opus-4.8", "cursor/sonnet-4.6"],
        }
    return sorted({m for models in by_provider.values() for m in models})


class _EditConfigModal(ModalScreen["dict[str, str | None] | None"]):
    """Edit planner / runner / eval model + effort for the next round only.

    Dismiss value: dict with keys {planner_model, planner_effort, runner_model,
    runner_effort, eval_model, eval_effort} — each str or None when (default).
    Cancel / esc → None.
    """

    DEFAULT_CSS = """
    _EditConfigModal {
        align: center middle;
        background: $background 60%;
    }
    _EditConfigModal > Vertical#ec-dialog {
        width: 72;
        height: 92%;
        border: round $primary;
        background: $surface;
        padding: 0;
    }
    _EditConfigModal #ec-header {
        padding: 1 2 0 2;
        height: auto;
    }
    _EditConfigModal #ec-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }
    _EditConfigModal #ec-scroll {
        height: 1fr;
        padding: 0 2;
    }
    _EditConfigModal .ec-section {
        text-style: bold;
        color: $accent;
        margin-top: 1;
        margin-bottom: 0;
    }
    _EditConfigModal .ec-label {
        margin-top: 1;
        margin-bottom: 0;
        color: $text-muted;
    }
    _EditConfigModal .ec-model-list {
        height: 8;
        margin-bottom: 0;
    }
    _EditConfigModal .ec-effort-list {
        height: 7;
        margin-bottom: 1;
    }
    _EditConfigModal #ec-actions {
        height: 3;
        align-horizontal: center;
        padding: 0 2;
        margin-top: 1;
        margin-bottom: 1;
    }
    _EditConfigModal Button {
        width: 1fr;
        height: 3;
        margin: 0 1;
        border: none;
        text-style: bold;
    }
    _EditConfigModal .ec-skip-section {
        height: auto;
        margin: 0 0 1 0;
        padding: 0 0 0 0;
    }
    _EditConfigModal Checkbox {
        height: 3;
        margin: 0;
        padding: 0 0;
    }
    """

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("enter", "confirm", "Confirm"),
    ]

    _EFFORTS = ["(default)", "low", "medium", "high", "max"]
    _current: dict[str, str | None]

    def __init__(self, current: dict[str, str | None] | None = None) -> None:
        super().__init__()
        self._current = current or {}

    def compose(self) -> ComposeResult:
        with Vertical(id="ec-dialog"):
            with Vertical(id="ec-header"):
                yield Static("⚙️  Edit Config — Next Round Only", id="ec-title")
                yield Rule()
            with VerticalScroll(id="ec-scroll"):
                yield Label("── Skip Steps ──", classes="ec-section")
                with Vertical(classes="ec-skip-section"):
                    yield Checkbox("Skip Planner  (run without replanning)", id="ec-skip-planner")
                    yield Checkbox("Skip Eval  (auto-pass, no LLM judge)", id="ec-skip-eval")
                    yield Checkbox(
                        "Skip Final Eval  (bypass final eval gate)", id="ec-skip-final-eval"
                    )

                yield Label("── Planner ──", classes="ec-section")
                yield Label("Model:", classes="ec-label")
                yield OptionList("(default)", id="ec-plan-model", classes="ec-model-list")
                yield Label("Effort:", classes="ec-label")
                yield OptionList(*self._EFFORTS, id="ec-plan-effort", classes="ec-effort-list")

                yield Label("── Runner ──", classes="ec-section")
                yield Label("Model:", classes="ec-label")
                yield OptionList("(default)", id="ec-run-model", classes="ec-model-list")
                yield Label("Effort:", classes="ec-label")
                yield OptionList(*self._EFFORTS, id="ec-run-effort", classes="ec-effort-list")

                yield Label("── Eval ──", classes="ec-section")
                yield Label("Model:", classes="ec-label")
                yield OptionList("(default)", id="ec-eval-model", classes="ec-model-list")
                yield Label("Effort:", classes="ec-label")
                yield OptionList(*self._EFFORTS, id="ec-eval-effort", classes="ec-effort-list")
            with Horizontal(id="ec-actions"):
                yield Button("  Confirm  (enter)", id="ec-confirm", variant="success")
                yield Button("  Cancel  (esc)", id="ec-cancel", variant="error")

    def on_mount(self) -> None:
        all_models = _load_all_models_flat()
        opts = ["(default)"] + all_models
        model_keys = {
            "ec-plan-model": self._current.get("planner_model"),
            "ec-run-model": self._current.get("runner_model"),
            "ec-eval-model": self._current.get("eval_model"),
        }
        for list_id, cur in model_keys.items():
            ml = self.query_one(f"#{list_id}", OptionList)
            ml.clear_options()
            for opt in opts:
                ml.add_option(opt)
            if cur and cur in opts:
                ml.highlighted = opts.index(cur)
        effort_keys = {
            "ec-plan-effort": self._current.get("planner_effort"),
            "ec-run-effort": self._current.get("runner_effort"),
            "ec-eval-effort": self._current.get("eval_effort"),
        }
        for list_id, cur in effort_keys.items():
            if cur and cur in self._EFFORTS:
                self.query_one(f"#{list_id}", OptionList).highlighted = self._EFFORTS.index(cur)
        if self._current.get("skip_planner") == "true":
            self.query_one("#ec-skip-planner", Checkbox).value = True
        if self._current.get("skip_eval") == "true":
            self.query_one("#ec-skip-eval", Checkbox).value = True
        if self._current.get("skip_final_eval") == "true":
            self.query_one("#ec-skip-final-eval", Checkbox).value = True
        self.query_one("#ec-plan-model", OptionList).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_confirm(self) -> None:
        self.query_one("#ec-confirm", Button).press()

    def _pick(self, list_id: str, options: list[str]) -> str | None:
        idx = self.query_one(f"#{list_id}", OptionList).highlighted
        raw = options[idx] if idx is not None and idx < len(options) else "(default)"
        return None if raw == "(default)" else raw

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "ec-cancel":
            self.dismiss(None)
            return
        if bid != "ec-confirm":
            return
        all_models = ["(default)"] + _load_all_models_flat()
        self.dismiss(
            {
                "skip_planner": str(self.query_one("#ec-skip-planner", Checkbox).value).lower(),
                "skip_eval": str(self.query_one("#ec-skip-eval", Checkbox).value).lower(),
                "skip_final_eval": str(
                    self.query_one("#ec-skip-final-eval", Checkbox).value
                ).lower(),
                "planner_model": self._pick("ec-plan-model", all_models),
                "planner_effort": self._pick("ec-plan-effort", self._EFFORTS),
                "runner_model": self._pick("ec-run-model", all_models),
                "runner_effort": self._pick("ec-run-effort", self._EFFORTS),
                "eval_model": self._pick("ec-eval-model", all_models),
                "eval_effort": self._pick("ec-eval-effort", self._EFFORTS),
            }
        )


class _RunErrorModal(ModalScreen[bool]):
    """Shown when the pipeline worker fails unexpectedly.

    Returns True → retry/resume from last checkpoint.
    Returns False → abort.
    """

    DEFAULT_CSS = """
    _RunErrorModal {
        align: center middle;
        background: $background 60%;
    }
    _RunErrorModal > Vertical#err-dialog {
        width: 72;
        height: auto;
        border: round $error;
        background: $surface;
        padding: 1 2;
    }
    _RunErrorModal #err-title {
        text-style: bold;
        color: $error;
        margin-bottom: 1;
    }
    _RunErrorModal #err-body {
        color: $text;
        margin-bottom: 1;
    }
    _RunErrorModal #err-actions {
        height: 3;
        align-horizontal: center;
        margin-top: 1;
    }
    _RunErrorModal Button {
        width: 1fr;
        height: 3;
        margin: 0 1;
        border: none;
        text-style: bold;
    }
    """

    BINDINGS = [
        ("c", "do_continue", "Continue"),
        ("escape", "do_abort", "Abort"),
    ]

    def __init__(self, error: str) -> None:
        super().__init__()
        self._error = error

    def compose(self) -> ComposeResult:
        with Vertical(id="err-dialog"):
            yield Static("⚠  Run failed", id="err-title")
            yield Static(self._error[:300], id="err-body")
            yield Static("[dim]Continue resumes from the last checkpoint.[/]")
            with Horizontal(id="err-actions"):
                yield Button("  Continue  (c)", id="continue", variant="warning")
                yield Button("  Abort  (Esc)", id="abort", variant="error")

    def on_mount(self) -> None:
        self.query_one("#continue", Button).focus()

    def action_do_continue(self) -> None:
        self.dismiss(True)

    def action_do_abort(self) -> None:
        self.dismiss(False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "continue")


class _ManualValidationModal(ModalScreen[tuple[str, str] | None]):
    """Shown after final eval — user approves, rejects, or requests corrections.

    Dismiss values:
      ("approve", "")          → run accepted
      ("changes", "<text>")    → plan changes and resume loop with corrections
      None                     → rejected / exit
    """

    DEFAULT_CSS = """
    _ManualValidationModal {
        align: center middle;
        background: $background 60%;
    }
    _ManualValidationModal > Vertical#val-dialog {
        width: 95%;
        height: 95%;
        border: round $primary;
        background: $surface;
        padding: 1 2;
    }
    _ManualValidationModal #val-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
        height: auto;
    }
    _ManualValidationModal #val-summary-scroll {
        height: 1fr;
        border: round $panel;
        margin-bottom: 1;
        padding: 0 1;
    }
    _ManualValidationModal #val-response-label {
        height: auto;
        margin-bottom: 0;
    }
    _ManualValidationModal #val-response {
        height: 8;
        margin-bottom: 1;
    }
    _ManualValidationModal #val-actions {
        height: 3;
        align-horizontal: center;
    }
    _ManualValidationModal Button {
        width: 1fr;
        height: 3;
        margin: 0 1;
        border: none;
        text-style: bold;
    }
    """

    BINDINGS = [
        ("a", "approve", "Approve"),
        ("n", "next_phase", "Next Phase"),
        ("f", "plan_changes", "Final Eval"),
        ("c", "edit_config", "Edit Config"),
        ("r", "reject", "Reject"),
        ("escape", "reject", "Reject"),
    ]

    def __init__(
        self,
        summary: str = "",
        all_passed: bool = True,
        show_phase: bool = False,
        prefill: str = "",
    ) -> None:
        super().__init__()
        self._summary = summary
        self._all_passed = all_passed
        self._show_phase = show_phase
        self._prefill = prefill

    def compose(self) -> ComposeResult:
        status = "✅ checks passed" if self._all_passed else "⚠️  some checks failed"
        with Vertical(id="val-dialog"):
            yield Static(f"🔍  Final Eval · {status}", id="val-title")
            yield Rule()
            with VerticalScroll(id="val-summary-scroll"):
                yield Markdown(self._summary or "_Final eval complete._")
            yield Rule()
            yield Label("Your response (leave blank to approve as-is):", id="val-response-label")
            yield TextArea(self._prefill, id="val-response")
            with Horizontal(id="val-actions"):
                yield Button("  Approve  (a)", id="approve", variant="success")
                if self._show_phase:
                    yield Button("  Next Phase  (n)", id="next_phase", variant="primary")
                yield Button("  Final Eval  (f)", id="plan_changes", variant="primary")
                yield Button("  Edit Config  (c)", id="edit_config", variant="default")
                yield Button("  Reject  (r)", id="reject", variant="error")

    def on_mount(self) -> None:
        self.query_one("#val-response", TextArea).focus()

    def action_approve(self) -> None:
        self.query_one("#approve", Button).press()

    def action_next_phase(self) -> None:
        self.query_one("#next_phase", Button).press()

    def action_plan_changes(self) -> None:
        self.query_one("#plan_changes", Button).press()

    def action_edit_config(self) -> None:
        self.query_one("#edit_config", Button).press()

    def action_reject(self) -> None:
        self.query_one("#reject", Button).press()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or "reject"
        text = self.query_one("#val-response", TextArea).text.strip()
        if bid == "approve":
            self.dismiss(("approve", text))
        elif bid == "next_phase":
            self.dismiss(("next_phase", text))
        elif bid == "plan_changes":
            self.dismiss(("changes", text))
        elif bid == "edit_config":
            self.dismiss(("edit_config", text))
        else:
            self.dismiss(None)


def _phase_detail_md(session: Session, n: int) -> str:
    """Render phase detail: plan, runner output, and gate status."""
    plan = session.read(f"knowledge/phase-plan-{n}.md").strip()
    run_out = session.read(f"runs/phase-{n}.md").strip()
    phase_loop = session.read("phase_loop.md")
    from splinter.analyze import _phase_entries

    phases_data = _phase_entries(session.read("phases.md"))
    phase_info = next((p for p in phases_data if p[0] == n), None)

    lines = [f"# Phase {n}"]
    if phase_info:
        _, pstatus, pmodel, pcost = phase_info
        lines.append(f"- status: **{pstatus}** · model: {pmodel} · cost: ${pcost}")

    if plan:
        lines.append("## Plan")
        lines.append(plan)
    if run_out:
        lines.append("## Runner output")
        lines.append(f"```\n{_cap_payload(run_out)}\n```")

    phase_loop_lines: list[str] = []
    for line in phase_loop.splitlines():
        if line.startswith(f"## Phase {n}"):
            phase_loop_lines.append(line)
        elif phase_loop_lines and line.startswith("## Phase "):
            break
        elif phase_loop_lines:
            phase_loop_lines.append(line)
    if phase_loop_lines:
        lines.append("## Loop")
        lines.extend(phase_loop_lines[1:])

    return "\n\n".join(lines)


class _PhaseConfigModal(ModalScreen[dict[str, str] | None]):
    """Configure and launch the next development phase.

    Dismiss value: dict with keys {description, plan_model, plan_effort, run_model, run_effort}
    or None to finish.
    """

    DEFAULT_CSS = """
    _PhaseConfigModal {
        align: center middle;
        background: $background 60%;
    }
    _PhaseConfigModal > Vertical#phase-dialog {
        width: 72;
        height: auto;
        max-height: 94%;
        border: round $primary;
        background: $surface;
        padding: 1 2;
    }
    _PhaseConfigModal #phase-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }
    _PhaseConfigModal #phase-desc {
        height: 4;
        margin-bottom: 1;
    }
    _PhaseConfigModal .phase-label {
        margin-top: 1;
        margin-bottom: 0;
    }
    _PhaseConfigModal .phase-row {
        height: auto;
        margin-bottom: 0;
    }
    _PhaseConfigModal .phase-row Select {
        width: 1fr;
        margin-right: 1;
    }
    _PhaseConfigModal #phase-actions {
        height: 3;
        align-horizontal: center;
        margin-top: 1;
    }
    _PhaseConfigModal Button {
        width: 1fr;
        height: 3;
        margin: 0 1;
        border: none;
        text-style: bold;
    }
    """

    BINDINGS = [
        ("escape", "done", "Done"),
    ]

    _PLAN_MODELS: list[str] = []
    _RUN_MODELS: list[str] = []
    _EFFORTS = ["(default)", "auto", "low", "medium", "high", "max"]

    def __init__(
        self,
        phase_num: int,
        default_plan_model: str = "",
        default_plan_effort: str = "",
        default_run_model: str = "",
        default_run_effort: str = "",
    ) -> None:
        super().__init__()
        self._phase_num = phase_num
        self._default_plan_model = default_plan_model
        self._default_plan_effort = default_plan_effort
        self._default_run_model = default_run_model
        self._default_run_effort = default_run_effort

    def compose(self) -> ComposeResult:
        with Vertical(id="phase-dialog"):
            yield Static(
                f"Phase {self._phase_num} · What would you like to implement?",
                id="phase-title",
            )
            yield Label("Description:")
            yield TextArea("", id="phase-desc")

            yield Label("Plan model:", classes="phase-label")
            with Horizontal(classes="phase-row"):
                plan_opts = [(m, m) for m in self._PLAN_MODELS] or [("opus", "opus")]
                yield Select(
                    plan_opts,
                    id="phase-plan-model",
                    value=self._default_plan_model or plan_opts[0][0],
                )
                yield Select(
                    [(e, e) for e in self._EFFORTS],
                    id="phase-plan-effort",
                    value=self._default_plan_effort or "(default)",
                )

            yield Label("Run model:", classes="phase-label")
            with Horizontal(classes="phase-row"):
                run_opts = [(m, m) for m in self._RUN_MODELS] or [("haiku", "haiku")]
                run_defaults = [
                    m
                    for m in (self._PLAN_MODELS or ["opus"])
                    if "flash" in m.lower() or "haiku" in m.lower()
                ]
                run_def = self._default_run_model or (
                    run_defaults[0] if run_defaults else run_opts[0][0]
                )
                run_val = run_def if run_def in {m for m, _ in run_opts} else run_opts[0][0]
                yield Select(
                    run_opts,
                    id="phase-run-model",
                    value=run_val,
                )
                yield Select(
                    [(e, e) for e in self._EFFORTS],
                    id="phase-run-effort",
                    value=self._default_run_effort or "auto",
                )

            with Horizontal(id="phase-actions"):
                yield Button("  Go  (Enter)", id="phase-go", variant="success")
                yield Button("  Done  (Esc)", id="phase-done", variant="primary")

    def on_mount(self) -> None:
        self.query_one("#phase-desc", TextArea).focus()

    def action_done(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or "phase-done"
        if bid == "phase-go":
            desc = self.query_one("#phase-desc", TextArea).text.strip()
            if not desc:
                self.dismiss(None)
                return
            try:
                plan_model = self.query_one("#phase-plan-model", Select).value
            except Exception:
                plan_model = self._default_plan_model
            try:
                plan_effort = self.query_one("#phase-plan-effort", Select).value
            except Exception:
                plan_effort = self._default_plan_effort
            try:
                run_model = self.query_one("#phase-run-model", Select).value
            except Exception:
                run_model = self._default_run_model
            try:
                run_effort = self.query_one("#phase-run-effort", Select).value
            except Exception:
                run_effort = self._default_run_effort

            result: dict[str, str] = {
                "description": desc,
                "plan_model": str(plan_model),
                "plan_effort": str(plan_effort) if plan_effort != "(default)" else "",
                "run_model": str(run_model),
                "run_effort": str(run_effort) if run_effort != "(default)" else "auto",
            }
            self.dismiss(result)
        else:
            self.dismiss(None)


class RunApp(App[int]):
    """Live dashboard for ``splinter run``: story views + streaming activity log."""

    CSS = (
        """
    Screen { background: $focus-bg; color: $ink; }
    RunApp * {
        scrollbar-background: $focus-bg;
        scrollbar-background-hover: $focus-bg;
        scrollbar-background-active: $focus-bg;
        scrollbar-color: $line2;
        scrollbar-color-hover: $edge;
        scrollbar-color-active: $dim;
        scrollbar-corner-color: $focus-bg;
    }
    #run-body { height: 1fr; min-height: 0; }
    #left { width: 1fr; padding: 1 3 0 3; }
    #focus { height: 1fr; min-height: 0; }
    #rail { width: 36%; min-width: 44; max-width: 62; border-left: solid $line; padding: 1 2 0 2; }
    #overview { height: auto; }
    #view-scroll { display: none; }
    #log {
        height: 1fr;
        padding: 0 0 0 2;
        border-left: solid $live-line;
        background: $focus-bg;
    }
    #groups { height: 1fr; min-height: 0; }
    #cmd-bar {
        height: auto;
        max-height: 8;
        border-top: solid $line;
        padding: 0;
    }
    #composer { width: 1fr; }
    #footer-bar { height: 2; }
    """
        + palette_css()
        + _PALETTE_CSS
        + _SCROLL_LEFT_PANE_CSS
        + _MAXIMIZE_CSS
        + _RUN_PANE_CSS
    )

    COMMANDS = {_OrderedCommandsProvider}

    BINDINGS = [
        Binding("s", "stop", "Stop"),
        Binding("ctrl+k", "toggle_kowabunga", "Cowabunga"),
        Binding("i", "instruct", "Instruct"),
        Binding("o", "toggle_latest_iteration", "Iteration", show=False),
        Binding("slash", "find", "Find", show=False),
        Binding("g", "follow_live", "Live"),
        Binding("left", "prev_story", "Prev", show=False),
        Binding("right", "next_story", "Next", show=False),
        Binding("1", "set_view('act')", "Activity", show=False),
        Binding("3", "set_view('plan')", "Plan", show=False),
        Binding("4", "set_view('gate')", "Gate", show=False),
        Binding("5", "set_view('eval')", "Eval", show=False),
        Binding("q", "quit_or_stop", "Quit"),
        Binding("ctrl+c", "quit", "Quit", show=False),
        Binding("dollar_sign", "open_spend('cost')", "Spend"),
        Binding("d", "open_diff", "Diff"),
        Binding("t", "open_tier('')", "Tier"),
        Binding("b", "toggle_spend_by", "By", show=False),
        Binding("r", "refresh_diff", "Refresh", show=False),
        Binding("2", "open_diff", "Diff", show=False),
        Binding("6", "open_spend('cost')", "Spend", show=False),
        Binding("7", "open_tier('')", "Tier", show=False),
        Binding("f", "toggle_needs_filter", "Needs you", show=False),
        Binding("L", "trace", "Trace"),
        Binding("l", "set_view('logs')", "Logs", show=False),
    ]

    #: The spend measure (cost or time), the picked ladder rung and the loaded diff.
    spend_by: str = "cost"
    tier_pick: str = ""
    _diff: StoryDiff | None = None
    _diff_cwd: str = ""
    _diff_gen: int = 0

    _maximized: reactive[bool] = reactive(False)
    _kowabunga: reactive[RunnerMode] = reactive(DEFAULT_RUNNER_MODE)
    selected_story: reactive[int | None] = reactive(None)
    view: reactive[str] = reactive("act")
    opened_iterations: reactive[frozenset[tuple[int, int]]] = reactive(frozenset())

    def __init__(self, session: Session, run_kwargs: dict[str, Any]) -> None:
        super().__init__()
        self.session = session
        self.run_kwargs = run_kwargs
        self.rc = 0
        self.error = ""
        self._timer: Any = None
        self._handler: logging.Handler | None = None
        self._prev_propagate: bool = True
        self._run_config: dict[str, str | None] = {}
        self._restart_after_kill = False
        self._parallel = False
        self.follow_live = True
        self._view_builder = RunViewBuilder()
        self._view_files = FileCache()
        self._last_view: RunView | None = None
        self._view_key: tuple[object, ...] | None = None
        self._live_actions: list[LiveAction] = []
        self._log_history: list[str] = []
        self._motion_phase = False
        self._diff_runner: Runner = git_runner
        budget = self.run_kwargs.get("budget")
        self._budget: float | None = float(budget) if budget else None
        self._needs_notified: set[int] = set()
        self._motion = MotionClock(self)
        self._toasts = ToastTracker()

    def _story_number(self, view: RunView) -> int | None:
        numbers = {story.number for story in view.stories}
        if self.selected_story in numbers:
            return self.selected_story
        return view.live_story

    def get_css_variables(self) -> dict[str, str]:
        return palette_variables(super().get_css_variables())

    def _apply_run_view(self, view: RunView) -> None:
        if not self.is_running:
            return
        story = self._story_number(view)
        focus = self.query_one("#focus", FocusColumn)
        focus.available_views = tuple(view_registry())
        focus.current_view = self.view
        focus.tier_pick = getattr(self, "tier_pick", "")
        focus.apply(view, story)
        rail = self.query_one("#rail", RunRail)
        rail.tier_pick = focus.tier_pick if self.view == "tier" else ""
        rail.max_iterations = self._max_iterations()
        rail.apply(view, story)
        self._apply_chrome(view, story)
        self._render_view()

    def _max_iterations(self) -> int:
        try:
            return int(self.session.read_status().get("max_iterations", 5) or 5)
        except (TypeError, ValueError, OSError):
            return 5

    def _apply_chrome(self, view: RunView, story: int | None) -> None:
        status = self.session.read_status()
        strategy = str(self.run_kwargs.get("strategy") or status.get("strategy") or "direct")
        source = self._run_source() if self.run_kwargs.get("prd_path") or self.run_kwargs.get(
            "task_path"
        ) else ""
        header = self.query_one(RunHeader)
        header.show(view, story, strategy, source, self._max_iterations())
        row = next((item for item in view.stories if item.number == story), None)
        if row is not None:
            composer = self.query_one(Composer)
            composer.set_story(row.number, story_label(row)[0], row.number == view.live_story)

    def action_toggle_needs_filter(self) -> None:
        self.query_one("#rail", RunRail).toggle_needs()
        if self._last_view is not None:
            self._apply_run_view(self._last_view)

    def action_composer_send(self) -> None:
        self._send_live_command()

    def _reapply_view(self) -> None:
        view = getattr(self, "_last_view", None)
        if view is not None:
            self._apply_run_view(view)
            self._refresh_needs_you()

    def watch_selected_story(self) -> None:
        self._reapply_view()
        if self.view == "diff":
            self._start_diff()

    def watch_view(self, old: str, new: str) -> None:
        self._reapply_view()
        if new == "diff" and old != "diff":
            self._start_diff()

    def watch_opened_iterations(self) -> None:
        self._render_view()

    def select_story(self, n: int) -> None:
        self.follow_live = False
        self.selected_story = n

    def action_select_story(self, n: int) -> None:
        self.select_story(n)

    def action_follow_live(self) -> None:
        self.follow_live = True
        self.selected_story = None

    def _step_story(self, offset: int) -> None:
        if isinstance(self.focused, Input) or self._last_view is None:
            return
        numbers = [story.number for story in self._last_view.stories]
        if not numbers:
            return
        current = self._story_number(self._last_view)
        try:
            index = numbers.index(current) if current is not None else 0
        except ValueError:
            index = 0
        self.select_story(numbers[min(max(index + offset, 0), len(numbers) - 1)])

    def action_prev_story(self) -> None:
        self._step_story(-1)

    def action_next_story(self) -> None:
        self._step_story(1)

    def action_set_view(self, name: str) -> None:
        if name in view_registry():
            self.view = name

    def action_stage_view(self, stage: str) -> None:
        """A stage of the strip opens its first sub-topic (or keeps the open one)."""
        from splinter.tui_run_focus import TOPICS, VIEW_STAGE

        topics = TOPICS.get(stage, ())
        if not topics or VIEW_STAGE.get(self.view) == stage:
            return
        self.action_set_view(topics[0][0])

    def action_open_iteration(self, story: int, n: int) -> None:
        self.select_story(story)
        self.view = "act"
        self.opened_iterations = self.opened_iterations | {(story, n)}
        self.call_after_refresh(self._scroll_view_end)

    def action_toggle_iteration(self, story: int, n: int) -> None:
        key = (story, n)
        if key in self.opened_iterations:
            self.opened_iterations = self.opened_iterations - {key}
        else:
            self.opened_iterations = self.opened_iterations | {key}

    def action_instruct(self) -> None:
        """i — write to the orchestrator: focus the composer."""
        self.query_one("#user-cmd-input", Input).focus()

    def action_find(self) -> None:
        """/ — find a story by id or title in the right rail."""
        self.query_one("#rail", RunRail).open_find()

    def action_toggle_latest_iteration(self) -> None:
        if self._last_view is None:
            return
        story = self._story_number(self._last_view)
        rows = self._last_view.iterations_by_story.get(story or 0, ())
        if rows:
            self.view = "act"
            self.action_toggle_iteration(story or 0, rows[-1].n)

    def _scroll_view_end(self) -> None:
        self.query_one("#view-scroll", VerticalScroll).scroll_end(animate=False)

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            event.stop()
            self.action_follow_live()

    def compose(self) -> ComposeResult:
        yield RunHeader(id="run-header")
        with Horizontal(id="run-body"):
            with Vertical(id="left"):
                yield FocusColumn(id="focus")
                with Horizontal(id="cmd-bar"):
                    yield Composer()
            yield RunRail(id="rail")
        yield FooterKeys(id="footer-bar")
        yield StopBar(id="stop-bar")

    def on_mount(self) -> None:
        use_splinter_theme(self)
        self._kowabunga = self.session.read_kowabunga()
        self._render_kowabunga_ind(self._kowabunga)
        self._refresh()
        self.query_one("#overview-scroll", VerticalScroll).focus()
        self._timer = self.set_interval(0.5, self._refresh)

        self._handler = _TextualLogHandler(self)
        self._handler.setFormatter(logging.Formatter("%(message)s"))
        splog = logging.getLogger("splinter")
        splog.setLevel(logging.INFO)
        self._prev_propagate = splog.propagate
        splog.propagate = False
        splog.addHandler(self._handler)
        logging.getLogger("splinter.live").setLevel(logging.INFO)

        _status = self.session.read_status()
        for key, status_key in (
            ("planner_model", "next_planner_model"),
            ("planner_effort", "next_planner_effort"),
            ("runner_model", "next_runner_model"),
            ("runner_effort", "next_runner_effort"),
            ("eval_model", "next_eval_model"),
            ("eval_effort", "next_eval_effort"),
            ("skip_planner", "next_skip_planner"),
            ("skip_eval", "next_skip_eval"),
            ("skip_final_eval", "next_skip_final_eval"),
        ):
            value = str(_status.get(status_key, "")).strip()
            if value:
                self._run_config[key] = value
        _state = _status.get("state")
        if _state == "awaiting_user" and _status.get("stage") == "final_eval":
            self.call_after_refresh(self._open_validation)
        elif _state == "awaiting_user":
            self.call_after_refresh(self._open_ask)
        elif _state == "awaiting_validation":
            self.call_after_refresh(self._open_validation)
        else:
            self.run_worker(self._work, thread=True, name="pipeline", exclusive=True)

    def _open_ask(self, prefill: str = "") -> None:
        if not self._show_selected_banner("asked"):
            self._show_ask_user_modal(prefill)

    def _open_validation(self, prefill: str = "") -> None:
        if not self._show_selected_banner("paused"):
            self._show_manual_validation_modal(prefill)

    def _show_ask_user_modal(self, prefill: str = "") -> None:
        st = self.session.read_status()
        reason = str(st.get("ask_reason", ""))
        corrections = str(st.get("ask_corrections", ""))
        cp = self.session.read("run_checkpoint.json").strip()
        if cp:
            try:
                data = json.loads(cp)
                gate = str(data.get("gate_output", "")).strip()
                corr = str(data.get("corrections", "")).strip()
                parts = [p for p in (corr, gate) if p]
                if parts:
                    corrections = "\n\n".join(parts)
            except (json.JSONDecodeError, TypeError):
                pass

        def reopen(text: str) -> None:
            self.call_after_refresh(lambda: self._show_ask_user_modal(prefill=text))

        self.push_screen(
            _AskUserModal(reason, corrections, prefill=prefill),
            callback=lambda result: self._handle_ask_choice(result, reopen),
        )

    def _record_user_guidance(self, text: str) -> None:
        if not text:
            return
        self._log_line(
            self._log_prefix("") + f"[bold cyan]👤 User guidance:[/] [cyan]{escape(text)}[/]"
        )
        self.session.append("events.md", f"[USER GUIDANCE] {text}\n")

    def _handle_ask_choice(
        self, result: tuple[str, str] | None, reopen: Callable[[str], None]
    ) -> None:
        if self._timer is None:
            self._timer = self.set_interval(0.5, self._refresh)
        if result is None:
            self.exit(3)
            return
        action, text = result
        if action in {"answer", "jump_premium"}:
            self._record_user_guidance(text)
            message = "continuing with your answer"
            if action == "jump_premium":
                message = "jumping to premium tier"
            self.write_log(f"— {message} —", logging.WARNING)
            self._run_pipeline_worker(
                resume=True,
                user_guidance=text or None,
                jump_premium=action == "jump_premium",
                cowabunga=False,
            )
        elif action in {"accept", "cowabunga"}:
            message = (
                "✓ accepted by user — marking done and continuing"
                if action == "accept"
                else "cowabunga — proceeding autonomously"
            )
            level = logging.INFO if action == "accept" else logging.WARNING
            self.write_log(f"— {message} —", level)
            self.run_kwargs["cowabunga"] = True
            if action == "cowabunga":
                state = str(self.session.read_status().get("state", "awaiting_user"))
                self.session.set_status(state, cowabunga=True)
            self._run_pipeline_worker(resume=True, jump_premium=False, cowabunga=True)
        elif action == "edit_config":
            self._edit_run_config(text, reopen)
        else:
            self.exit(3)

    def _edit_run_config(self, text: str, reopen: Callable[[str], None]) -> None:
        def _on_cfg(cfg: dict[str, str | None] | None) -> None:
            if cfg is not None:
                self._store_config_overrides(cfg)
            reopen(text)

        self.push_screen(_EditConfigModal(self._run_config), callback=_on_cfg)

    def _show_manual_validation_modal(self, prefill: str = "") -> None:
        st = self.session.read_status()
        summary = str(st.get("final_eval_summary", ""))
        all_passed = bool(st.get("final_eval_passed", True))

        def reopen(text: str) -> None:
            self.call_after_refresh(lambda: self._show_manual_validation_modal(prefill=text))

        self.push_screen(
            _ManualValidationModal(summary, all_passed, show_phase=self._phased, prefill=prefill),
            callback=lambda result: self._handle_validation_choice(result, reopen),
        )

    def _append_validation_decision(self, label: str, text: str = "") -> None:
        round_index = int(self.session.read_status().get("round_index", 1))
        path = f"knowledge/final-eval-{max(0, round_index - 1)}.md"
        if self.session.has(path):
            body = f"\n## User Decision: {label}\n"
            if text:
                body += f"\n{text}\n"
            self.session.append(path, body)

    def _handle_validation_choice(
        self, result: tuple[str, str] | None, reopen: Callable[[str], None]
    ) -> None:
        if result is None:
            self.write_log("— rejected ❌ — run marked failed —", logging.WARNING)
            self._append_validation_decision("Rejected")
            self.session.set_status("failed", stage="final_eval")
            self.exit(1)
            return
        action, text = result
        if action == "approve":
            self.write_log("— validated ✅ — run accepted —", logging.INFO)
            self._append_validation_decision("Approved")
            self.session.set_status("completed", stage="done")
            if self._phased:
                self.call_after_refresh(self._show_phase_modal)
            else:
                self._write_run_complete()
        elif action == "changes":
            summary = str(self.session.read_status().get("final_eval_summary", ""))
            guidance = text or summary
            self._append_validation_decision("Requested Changes", guidance)
            self.write_log(f"— planning corrections: {guidance[:80]}… —", logging.INFO)
            if self._timer is None:
                self._timer = self.set_interval(0.5, self._refresh)
            self._run_pipeline_worker(resume=True, user_guidance=guidance)
        elif action == "next_phase":
            self._append_validation_decision("Approved (next phase)", text)
            self.session.set_status("completed", stage="done")
            if self._timer is None:
                self._timer = self.set_interval(0.5, self._refresh)
            self.call_after_refresh(self._show_phase_modal)
        elif action == "edit_config":
            self._edit_run_config(text, reopen)

    def _show_selected_banner(self, kind: str, prefill: str = "") -> bool:
        if not self._refresh_needs_you(expected=kind):
            return False
        if prefill:
            self.query_one("#needs-you", NeedsYouBanner).prefill(prefill)
        return True

    def _refresh_needs_you(self, expected: str | None = None) -> bool:
        slots = self.query("#needs-you-slot")
        view = getattr(self, "_last_view", None)
        if not slots or view is None:
            return False
        banner = next(iter(slots.first().query(NeedsYouBanner)), None)
        status = self.session.read_status()
        story = self.selected_story
        if story is None:
            try:
                story = int(status.get("task_index", -1)) + 1
            except (TypeError, ValueError):
                story = 0
            if story <= 0:
                story = view.live_story
            if story is None and len(view.needs_you) == 1:
                story = view.needs_you[0]
        row = next((item for item in view.stories if item.number == story), None)
        if row is None or row.state not in {"asked", "paused", "fail"}:
            if banner is not None:
                banner.hide()
            return False
        if row.state in ("paused", "fail") and row.number not in self._needs_notified:
            self.notify(f"{row.id} needs you")
            self._needs_notified.add(row.number)
        if self.size.height < 30 or (expected is not None and row.state != expected):
            if banner is not None:
                banner.hide()
            return False
        if banner is None:
            banner = NeedsYouBanner(id="needs-you")
            slots.first().mount(banner)
        banner.show(row.state, row, {**status, "phased": self._phased})
        return True

    def on_needs_you_banner_chosen(self, event: NeedsYouBanner.Chosen) -> None:
        banner = self.query_one("#needs-you", NeedsYouBanner)
        if event.kind == "asked":
            self._handle_ask_choice(event.result, banner.prefill)
            return
        if event.kind == "paused":
            self._handle_validation_choice(event.result, banner.prefill)
            return
        if event.result is None:
            return
        action, text = event.result
        if action == "retry":
            self.session.queue_live_command(text, task_no=event.story)
            self._do_pause_kill()
        elif action == "open_worktree":
            path = story_worktree(self.session, event.story, self._story_task_id(event.story))
            if path is None:
                self.notify(
                    f"T{event.story} ran in the main tree: {Path.cwd()}", severity="warning"
                )
                return
            self.copy_to_clipboard(str(path))
            self.notify(f"Copied {path}")

    @property
    def _phased(self) -> bool:
        return bool(self.run_kwargs.get("phased", False))

    def _run_pipeline_worker(
        self,
        *,
        resume: bool = False,
        user_guidance: str | None = None,
        jump_premium: bool = False,
        cowabunga: bool = False,
        claude_runner_fallback: bool = False,
    ) -> None:
        def _run() -> None:
            from splinter.pipeline import run_pipeline

            _pipeline_keys = {
                "strategy",
                "prd_path",
                "task_path",
                "effort",
                "budget",
                "max_iterations",
                "eval_skill",
                "eval_model",
                "eval_effort",
                "cowabunga",
                "resume",
                "session",
                "claude_runner_fallback",
                "user_guidance",
                "jump_premium",
                "no_ground",
                "phased",
                "phase_plan_model",
                "phase_plan_effort",
                "phase_run_model",
                "phase_run_effort",
                "parallel",
                "max_concurrency",
            }
            kwargs = {
                **{k: v for k, v in self.run_kwargs.items() if k in _pipeline_keys},
                "resume": resume,
                "user_guidance": user_guidance,
                "jump_premium": jump_premium,
                "cowabunga": cowabunga or bool(self.run_kwargs.get("cowabunga")),
                "claude_runner_fallback": claude_runner_fallback,
            }
            try:
                self.rc = run_pipeline(**kwargs)
            except BaseException as exc:  # noqa: BLE001
                self.rc = 1
                self.error = str(exc)
                try:
                    self.call_from_thread(self.write_log, f"ERROR: {exc}", logging.ERROR)
                except Exception:
                    pass

        self.run_worker(_run, thread=True, name="pipeline", exclusive=True)

    def _store_config_overrides(self, cfg: dict[str, str | None]) -> None:
        # Accumulate into run-scoped config so re-opening EditConfig shows current values.
        self._run_config.update({k: v for k, v in cfg.items() if v})
        data = self.session.read_status()
        state = str(data.get("state", "running"))
        self.session.set_status(
            state,
            next_planner_model=cfg.get("planner_model") or "",
            next_planner_effort=cfg.get("planner_effort") or "",
            next_runner_model=cfg.get("runner_model") or "",
            next_runner_effort=cfg.get("runner_effort") or "",
            next_eval_model=cfg.get("eval_model") or "",
            next_eval_effort=cfg.get("eval_effort") or "",
            next_skip_planner=cfg.get("skip_planner") or "",
            next_skip_eval=cfg.get("skip_eval") or "",
            next_skip_final_eval=cfg.get("skip_final_eval") or "",
        )
        parts = []
        if cfg.get("runner_model"):
            eff = cfg.get("runner_effort") or "default"
            parts.append(f"runner → {cfg['runner_model']} @ {eff}")
        if cfg.get("planner_model"):
            eff = cfg.get("planner_effort") or "default"
            parts.append(f"planner → {cfg['planner_model']} @ {eff}")
        if cfg.get("eval_model"):
            eff = cfg.get("eval_effort") or "default"
            parts.append(f"eval → {cfg['eval_model']} @ {eff}")
        if parts:
            self.write_log(f"— config override: {' · '.join(parts)} —", logging.WARNING)

    def on_unmount(self) -> None:
        self._motion.stop()
        if self._handler is not None:
            splog = logging.getLogger("splinter")
            splog.removeHandler(self._handler)
            splog.propagate = self._prev_propagate
        logging.getLogger("splinter.live").setLevel(logging.NOTSET)

    async def action_quit(self) -> None:
        # Kill any running provider subprocess so the worker thread can unblock.
        from splinter import procreg

        procreg.terminate_all()
        self.exit(self.rc)

    async def action_quit_or_stop(self) -> None:
        """q — quit an idle run, open the stop menu while the run is live."""
        if _run_state(self.session) == "RUNNING":
            self.action_stop()
            return
        await self.action_quit()

    def action_stop(self) -> None:
        bars = self.query("#stop-bar")
        if self.size.height >= 30 and bars:
            self.query_one("#footer-bar", Horizontal).display = False
            bar = bars.first()
            bar.display = True
            bar.focus()
            return
        self.push_screen(StopMenuModal(), callback=self._on_stop_choice)

    def on_stop_bar_chosen(self, event: StopBar.Chosen) -> None:
        self.query_one("#stop-bar", StopBar).display = False
        self.query_one("#footer-bar", Horizontal).display = True
        self._on_stop_choice(event.result)

    def _on_stop_choice(self, result: str | None) -> None:
        if result == "pause":
            self._do_pause_graceful()
        elif result == "restart":
            self._do_pause_kill()
        elif result == "kill_exit":
            self._do_kill_exit()

    def _sync_more_state(self) -> None:
        """Publish the state the spend / tier / diff views read."""
        set_more_state(
            MoreState(
                spend_by=self.spend_by,
                tier=self.tier_pick,
                diff=self._diff,
                diff_cwd=self._diff_cwd,
                budget=self._budget,
            )
        )

    def action_open_spend(self, by: str = "cost") -> None:
        """Rail elapsed / spent and `$` open the spend view."""
        self.spend_by = by if by in ("cost", "time") else "cost"
        self._sync_more_state()
        self.view = "spend"

    def action_set_spend_by(self, by: str) -> None:
        """The `by cost | by time` switch of the spend view."""
        self.spend_by = by if by in ("cost", "time") else "cost"
        self._sync_more_state()
        self._render_view()

    def action_toggle_spend_by(self) -> None:
        """b — flip the spend measure while the spend view is open."""
        if self.view != "spend":
            return
        self.spend_by = "time" if self.spend_by == "cost" else "cost"
        self._sync_more_state()
        self._render_view()

    def action_spend_story(self, n: int) -> None:
        """A top story of the spend view — the view stays spend, filtered by it."""
        self.select_story(n)
        self._sync_more_state()
        self.view = "spend"

    def action_open_tier(self, tier: str = "") -> None:
        """A ladder rung opens the tier view."""
        self.tier_pick = tier
        self._sync_more_state()
        self.view = "tier"

    def action_open_diff(self) -> None:
        """d — the diff of the selected story (a second d reloads it)."""
        if self.view == "diff":
            self._start_diff()
        self.view = "diff"

    def action_refresh_diff(self) -> None:
        """r — refresh the diff on demand, never on the refresh tick."""
        if self.view != "diff":
            return
        self._start_diff()

    def _start_diff(self) -> None:
        self._diff = None
        self._diff_gen += 1
        self._sync_more_state()
        gen = self._diff_gen
        self.run_worker(
            lambda: self._load_diff(gen), thread=True, name="diff", exclusive=True, group="diff"
        )

    def _story_task_id(self, story: int) -> str:
        view = getattr(self, "_last_view", None)
        rows = view.stories if view is not None else ()
        return next((row.id for row in rows if row.number == story), "")

    def _load_diff(self, gen: int) -> None:
        story = self.selected_story
        if story is None and self._last_view is not None:
            story = self._last_view.live_story
        cwd = Path.cwd()
        try:
            if story is not None:
                cwd = story_cwd(self.session, story, self._story_task_id(story))
            result = story_diff(cwd, self._diff_runner)
        except Exception as exc:
            result = StoryDiff(stat="", patch="", error=str(exc))
        self.call_from_thread(self._set_diff, result, shown_tree(cwd), gen)

    def _set_diff(self, result: StoryDiff, where: str = "", gen: int = 0) -> None:
        if gen and gen != self._diff_gen:
            return
        self._diff = result
        self._diff_cwd = where
        self._sync_more_state()
        self._render_view()

    async def action_pause_graceful(self) -> None:
        """Finish the current iteration, then pause."""

        def _on_choice(result: str | None) -> None:
            if result == "pause":
                self._do_pause_graceful()

        self.push_screen(_ConfirmStopModal("pause"), callback=_on_choice)

    async def action_pause_kill(self) -> None:
        """Kill subprocesses immediately and restart from the same checkpoint."""

        def _on_choice(result: str | None) -> None:
            if result == "kill":
                self._do_pause_kill()

        self.push_screen(_ConfirmStopModal("kill"), callback=_on_choice)

    async def action_kill_exit(self) -> None:
        """Kill subprocesses immediately and exit the run screen."""
        if _run_state(self.session) != "RUNNING":
            await self.action_quit()
            return
        self._do_kill_exit()

    def _do_pause_graceful(self) -> None:
        from splinter import procreg

        procreg.request_stop()
        self.write_log(
            "— graceful pause requested — will stop after current iteration —",
            logging.WARNING,
        )

    def _do_pause_kill(self) -> None:
        from splinter import procreg

        procreg.terminate_all()
        cur = self.session.read_status()
        self.session.set_status(
            "paused",
            reason="user_kill_restart",
            stage=str(cur.get("stage", "run")),
        )
        self._restart_after_kill = True
        self.write_log(
            "— killed by user — restarting from last checkpoint… —",
            logging.WARNING,
        )

    def _do_kill_exit(self) -> None:
        from splinter import procreg

        procreg.terminate_all()
        cur = self.session.read_status()
        self.session.set_status(
            "paused",
            reason="user_kill",
            stage=str(cur.get("stage", "run")),
        )
        self.write_log("— killed by user — resume with: splinter resume —", logging.WARNING)
        self.rc = 2
        self.exit(2)

    def _work(self) -> None:
        self._run_pipeline_worker(resume=bool(self.run_kwargs.get("resume", False)))

    def write_log(self, msg: str, level: int = logging.INFO) -> None:
        # Streamed model text/tool args are arbitrary — format_line escapes them so
        # stray `[` markup (e.g. "fix [bug]") never raises MarkupError.
        self._log_line(format_line(msg, level, self._log_prefix(msg)))
        self._record_action(msg)

    def _record_action(self, msg: str) -> None:
        """Keep a streamed tool call or thought for the live feed of the activity view."""
        view = getattr(self, "_last_view", None)
        task = _TASK_NO_RE.search(msg)
        story = int(task.group(1)) if self._parallel and task else None
        if story is None and view is not None:
            story = view.live_story
        rows = view.iterations_by_story.get(story, ()) if view and story is not None else ()
        iteration = 1 if not rows else rows[-1].n + (1 if rows[-1].verdict else 0)
        stamp = datetime.now().strftime("%H:%M:%S")
        action = live_action(msg, stamp, story, iteration, time.monotonic())
        if action is None:
            return
        actions: list[LiveAction] = self.__dict__.setdefault("_live_actions", [])
        actions.append(action)
        if len(actions) > 600:
            del actions[:200]
        if not self.is_running:
            return
        with contextlib.suppress(Exception):
            focus = self.query_one("#focus", FocusColumn)
            focus.live_action = action
            if view is not None:
                focus.apply(view, self._story_number(view))
            self._render_view()

    def on_motion(self, phase: bool) -> None:
        """Blink the caret of the live feed and fade its newest row."""
        self._motion_phase = phase
        if self.view == "act":
            self._render_view()

    def _log_line(self, markup: str) -> None:
        """Write one formatted line to the log, keep it for the trace screen."""
        history = self._log_history
        history.append(markup)
        if len(history) > 5000:
            del history[:1000]
        self.query_one("#log", RichLog).write(markup)
        screen = self.screen
        if isinstance(screen, TraceScreen):
            screen.write(markup)

    def build_trace_root(self) -> TraceNode:
        """The run's trace tree from the files on disk and the latest view."""
        view = getattr(self, "_last_view", None)
        if view is None:
            view = self._view_builder.build(
                self.session, str(self.session.read_status().get("state", "")),
                datetime.now(timezone.utc),
            )
        return build_trace(self.session, view)

    def action_trace(self) -> None:
        """L — the full-screen trace: tree, selected text and the live log."""
        if isinstance(self.screen, TraceScreen):
            self.pop_screen()
            return
        history = self._log_history
        crumbs, stats = self._trace_head()
        self.push_screen(TraceScreen(self.build_trace_root(), history, crumbs, stats))

    def _trace_head(self) -> tuple[str, str]:
        """``splinter / direct / prd.md / trace`` and ``23/58 done · 2 need you · $0.39``."""
        status = self.session.read_status()
        strategy = str(self.run_kwargs.get("strategy") or status.get("strategy") or "direct")
        has_source = self.run_kwargs.get("prd_path") or self.run_kwargs.get("task_path")
        parts = [f"[b {_LIME}]splinter[/]", f"[{_MUTED}]{escape(strategy)}[/]"]
        if has_source:
            parts.append(f"[{_MUTED}]{escape(self._run_source())}[/]")
        parts.append(f"[b {_INK}]trace[/]")
        crumbs = f" [{_DIM}]/[/] ".join(parts)
        view = getattr(self, "_last_view", None)
        if view is None:
            return crumbs, ""
        done = int(view.counts.get("pass", 0))
        stats = f"[b {_INK}]{done}/{len(view.stories)}[/] [{_MUTED}]done[/]"
        if view.needs_you:
            stats += f" [{_DIM}]·[/] [b {_RED}]{len(view.needs_you)}[/] [{_MUTED}]need you[/]"
        stats += f" [{_DIM}]·[/] [b {_LIME}]${view.total_cost:.4f}[/]"
        return crumbs, stats

    def _log_prefix(self, msg: str) -> str:
        """Dim ``HH:MM:SS`` stamp, plus ``[T<n>]`` when the record names a task."""
        stamp = datetime.now().strftime("%H:%M:%S")
        task = _TASK_NO_RE.search(msg)
        return stamp_prefix(stamp, task.group(1) if self._parallel and task else "")

    def _running_task_choices(self) -> list[tuple[int, str]]:
        """Currently-running parallel tasks as (task_no, title). Empty in
        single-task mode, where there is nothing to disambiguate."""
        loop_md = self.session.read("loop.md")
        if not _is_parallel_mode(loop_md):
            return []
        titles = {no: title for no, title, _ in _parse_parallel_tasks(self.session, loop_md)}
        return [(no, titles.get(no, "")) for no in _running_tasks(self.session)]

    def _dispatch_directive(self, text: str, *, task_no: int | None) -> None:
        """Queue a directive for the runner without touching any live process.

        The in-flight provider is never killed. The directive is merged into
        corrections at the top of the target task's next iteration, so the model
        decides when to act on it.
        """
        target = f"Task {task_no}" if task_no is not None else "runner"
        self._log_line(
            self._log_prefix("")
            + f"[bold cyan]👤 User → {escape(target)}:[/] [cyan]{escape(text)}[/]"
        )
        self.session.queue_live_command(text, task_no=task_no)
        self._log_line(
            self._log_prefix("") + "[dim]queued — applies on the target session's next iteration[/]"
        )

    def _send_live_command(self) -> None:
        composer = self.query_one(Composer)
        inp = composer.query_one("#user-cmd-input", Input)
        text = inp.value.strip()
        if not text:
            return
        view = getattr(self, "_last_view", None)
        story = (
            getattr(self, "selected_story", None) or getattr(view, "live_story", None) or 1
        )
        now = tuple(getattr(view, "now", ()) or ())
        if composer.target == "story":
            label = f"T{story}"
            ids: tuple[str, ...] = (self.session.queue_live_command(text, task_no=story),)
            self.session.append("events.md", f"[USER GUIDANCE] ({label}) {text}")
        elif composer.target == "all":
            label = "all agents"
            ids = tuple(self.session.broadcast_live_command(text, now or (story,)))
            self.session.append("events.md", f"[USER GUIDANCE] ({label}) {text}")
        else:
            label = "orchestrator only"
            ids = (self.session.queue_orchestrator_note(text),)
            self.session.append("events.md", f"[ORCHESTRATOR NOTE] {text}")
        composer.record(label, text, ids, datetime.now().astimezone())
        self.write_log(f"👤 User → {label}: {text}")
        self.write_log("queued — applies on the target's next iteration")
        inp.value = ""
        inp.focus()

    def action_composer_target(self, target: str) -> None:
        self.query_one(Composer).select_target(target)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "user-cmd-input":
            self._send_live_command()

    def _write_run_complete(self) -> None:
        summary = format_run_completion(self.session)
        self._log_line(format_line(f"RUN COMPLETE — {summary}", logging.INFO, self._log_prefix("")))
        self._log_line(
            self._log_prefix("")
            + " " * 7
            + "[#7a7972]press q to quit · uv run splinter analyze to inspect[/]"
        )

    def _render_view(self) -> None:
        view = getattr(self, "_last_view", None)
        if view is None or not self.is_running:
            return
        story = self._story_number(view)
        ctx = ViewContext(
            view=view,
            story=story,
            opened=self.opened_iterations,
            files=self._view_files,
            session=self.session,
            multi=self._parallel,
            spend_by=getattr(self, "spend_by", "cost"),
            tier=getattr(self, "tier_pick", ""),
            budget=self.run_kwargs.get("budget"),
            diff_text=getattr(self, "_diff", None),
            width=max(0, self.query_one("#view-body").content_size.width - 1),
            live=tuple(self.__dict__.get("_live_actions", ())),
            phase=self.__dict__.get("_motion_phase", False),
            now=time.monotonic(),
        )
        streaming = self.view == "act" and story is not None and story == view.live_story
        base = (
            self.view,
            *view_content_key(ctx, self.view),
        )
        key = (*base, ctx.phase if streaming else None)
        logs = self.view == "logs"
        self.query_one("#view-body").display = not logs
        self.query_one("#log-pane").display = logs
        scroll = self.query_one("#view-scroll", VerticalScroll)
        scroll.display = True
        scroll.set_class(self.view == "act", "-bottom")
        self.query_one("#needs-you-slot").display = self.view == "act"
        if key != self._view_key:
            factory = view_registry()[self.view]
            self.query_one("#view-static", Static).update(factory(ctx))
            moved = self._view_key is None or self._view_key[:-1] != base
            self._view_key = key
            if self.view == "act":
                if moved:
                    self.call_after_refresh(self._scroll_view_end)
            else:
                scroll.scroll_home(animate=False)

    def _refresh_view_model(self, state: str) -> None:
        view: RunView | None
        try:
            view = self._view_builder.build(
                self.session, state, datetime.now(timezone.utc)
            )
        except Exception:
            view = self._last_view
        if view is None:
            return
        previous = self._last_view
        self._last_view = view
        self._apply_run_view(view)
        self._sync_motion(view)
        self._notify_toasts(previous, view)
        source_path = self.run_kwargs.get("prd_path") or self.run_kwargs.get("task_path")
        source = Path(str(source_path)).name if source_path else self.session.id
        if view.live_story is None:
            progress = f"{view.counts.get('pass', 0)}/{len(view.stories)} done"
        else:
            progress = f"story {view.live_story}/{len(view.stories)}"
        self.sub_title = f"{source} · {progress} · {state}"

    def _sync_motion(self, view: RunView) -> None:
        """Pulse while the run lives; stop for good when it is done."""
        if view.stage == "done":
            self._motion.stop()
        else:
            self._motion.start()

    def _notify_toasts(self, previous: RunView | None, view: RunView) -> None:
        """Show the run's own events — each of them once."""
        for message in self._toasts.events(previous, view):
            self.notify(message)

    def _refresh(self) -> None:
        state = _run_state(self.session)
        self.title = f"splinter run · {self.session.id}"
        with contextlib.suppress(Exception):
            clock = datetime.now().strftime("%H:%M:%S")
            self.query_one("#footer-clock", Static).update(clock)
        self._parallel = _is_parallel_mode(self.session.read("loop.md"))
        self._refresh_view_model(state)
        try:
            composer = self.query_one(Composer)
            view = getattr(self, "_last_view", None)
            live_story = getattr(view, "live_story", None)
            for message in composer.poll(self.session):
                is_live_target = message.target in {f"T{live_story}", "all agents"}
                if live_story is not None and is_live_target:
                    self.write_log(f"orch → T{live_story}: forwarded your note")
        except Exception:
            pass
        self._refresh_needs_you()
        # Re-sync the mode + read-error flag from the session each tick. The
        # pipeline can seed the mode ON from --cowabunga on the worker thread
        # *after* on_mount snapshots it, so the reactive would otherwise show a
        # stale OFF while scheduling runs as ON. Both reads touch status.json and
        # may raise a transient OSError (read_status only swallows JSONDecodeError);
        # keep the last-known mode / no error on failure so one bad read never
        # throws inside the interval callback.
        try:
            self._kowabunga = self.session.read_kowabunga()
            read_error = bool(self.session.read_status().get("kowabunga_read_error"))
        except Exception:
            read_error = False
        self._render_kowabunga_ind(self._kowabunga, error=read_error)

    def _run_source(self) -> str:
        for key in ("prd_path", "task_path"):
            value = self.run_kwargs.get(key)
            if value:
                return Path(str(value)).name
        return self.session.id

    def _show_gap_modal(self, kind: str, provider: str = "", retry_after: object = None) -> None:
        try:
            ra: int | None = (
                int(retry_after) if isinstance(retry_after, (int, str, float)) else None
            )
        except (TypeError, ValueError):
            ra = None

        def _on_choice(choice: str | None) -> None:
            if self._timer is None:
                self._timer = self.set_interval(0.5, self._refresh)
            if choice == "claude":
                self.write_log("— switching to Claude (sonnet @ high) —", logging.WARNING)
                self._run_pipeline_worker(resume=True, claude_runner_fallback=True)
            elif choice == "retry":
                self.write_log("— sleep done, retrying… —", logging.WARNING)
                self._run_pipeline_worker(resume=True)
            else:
                self.exit(2)

        self.push_screen(_GapModal(kind, provider, ra), callback=_on_choice)

    def _show_phase_modal(self) -> None:
        from splinter.models.roster import load_ladder
        from splinter.phases import phase_count

        ladder = load_ladder()
        plan_models = sorted(set([ladder.planner_model] + [t.models[0] for t in ladder.tiers]))
        run_models = sorted(set(t.models[0] for t in ladder.tiers))
        next_num = phase_count(self.session) + 1

        _PhaseConfigModal._PLAN_MODELS = plan_models
        _PhaseConfigModal._RUN_MODELS = run_models

        def _on_choice(result: dict[str, str] | None) -> None:
            if result is None:
                self.write_log("— phase mode ended —", logging.INFO)
                self._write_run_complete()
                return

            if self._timer is None:
                self._timer = self.set_interval(0.5, self._refresh)

            self.write_log(
                f"— phase {next_num}: {result['description'][:80]}… —",
                logging.INFO,
            )
            self._run_phase_worker(result)

        self.push_screen(
            _PhaseConfigModal(
                next_num,
                default_plan_model=ladder.planner_model,
                default_plan_effort=ladder.planner_effort,
                default_run_model=ladder.tiers[0].models[0] if ladder.tiers else "",
                default_run_effort="auto",
            ),
            callback=_on_choice,
        )

    def _run_phase_worker(self, cfg: dict[str, str]) -> None:
        from splinter.models.roster import load_ladder
        from splinter.phases import PhaseConfig, run_phase

        def _run() -> None:
            ladder = load_ladder()
            phase_cfg = PhaseConfig(
                description=cfg["description"],
                plan_model=cfg["plan_model"] or ladder.planner_model,
                plan_effort=cfg["plan_effort"] or ladder.planner_effort,
                run_model=cfg["run_model"]
                or (ladder.tiers[0].models[0] if ladder.tiers else "haiku"),
                run_effort=cfg["run_effort"] or "auto",
            )
            try:
                result = run_phase(phase_cfg, self.session, ladder)
                self.rc = 0
                try:
                    self.call_from_thread(
                        self.write_log,
                        f"— phase {result.phase_number} · "
                        f"{'PASS' if result.gate_passed else 'FAIL'} · "
                        f"{result.run_result.model} · "
                        f"${result.run_result.cost:.4f} —",
                        logging.INFO,
                    )
                except Exception:
                    pass
            except BaseException as exc:  # noqa: BLE001
                self.rc = 1
                self.error = str(exc)
                try:
                    self.call_from_thread(self.write_log, f"ERROR: {exc}", logging.ERROR)
                except Exception:
                    pass

        def _on_phase_done() -> None:
            self._refresh()
            if self._timer is not None:
                self._timer.stop()
                self._timer = None
            if self.rc == 0:
                self.call_after_refresh(self._show_phase_modal)
            else:
                self.write_log(
                    f"— phase failed: {self.error[:120]} —",
                    logging.ERROR,
                )
                self.call_after_refresh(self._show_phase_modal)

        self.run_worker(_run, thread=True, name="phase", exclusive=True)
        self._phase_callback = _on_phase_done

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.name == "phase":
            if event.state in (WorkerState.SUCCESS, WorkerState.ERROR):
                if hasattr(self, "_phase_callback"):
                    self._phase_callback()
                    del self._phase_callback
            return
        if event.worker.name != "pipeline":
            return
        if event.state not in (WorkerState.SUCCESS, WorkerState.ERROR):
            return

        self._refresh()
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

        if self._restart_after_kill:
            self._restart_after_kill = False
            self.write_log("— restarting run from checkpoint… —", logging.WARNING)
            if self._timer is None and self.is_running:
                self._timer = self.set_interval(0.5, self._refresh)
            self._run_pipeline_worker(resume=True)
            return

        # --- original dispatch (moved here from previous on_worker_state_changed) ---
        if self.rc == 0:
            if self._phased:
                self.write_log("— run complete — entering phase mode —", logging.INFO)
                self.call_after_refresh(self._show_phase_modal)
            else:
                self._write_run_complete()
        elif self.rc == 2:
            self.write_log("— run PAUSED (provider gap) —", logging.WARNING)
            st = self.session.read_status()
            kind = str(st.get("kind", ""))
            provider = str(st.get("provider", ""))
            retry_after = st.get("retry_after")
            self.call_after_refresh(self._show_gap_modal, kind, provider, retry_after)
        elif self.rc == 3:
            # rc=3 covers both ASK_USER pauses and the final-eval manual
            # validation pause — route by stage, same as on_mount.
            st = self.session.read_status()
            if str(st.get("stage", "")) == "final_eval" or str(st.get("state", "")) == (
                "awaiting_validation"
            ):
                self.write_log("— final eval complete — awaiting manual validation —", logging.INFO)
                self.call_after_refresh(self._open_validation)
            else:
                self.write_log("— run PAUSED (needs your input) —", logging.WARNING)
                self.call_after_refresh(self._open_ask)
        else:
            from splinter import procreg

            procreg.terminate_all()
            st = self.session.read_status()
            err_msg = str(st.get("error", "") or f"rc={self.rc}")
            self.write_log(f"— run failed: {err_msg} —", logging.ERROR)
            self.call_after_refresh(self._show_error_modal, err_msg)

    def _show_error_modal(self, error: str) -> None:
        def _on_choice(resume: bool | None) -> None:
            if resume:
                if self._timer is None:
                    self._timer = self.set_interval(0.5, self._refresh)
                self.write_log("— resuming from last checkpoint… —", logging.WARNING)
                self._run_pipeline_worker(resume=True)
            else:
                self.exit(self.rc)

        self.push_screen(_RunErrorModal(error), callback=_on_choice)

    def get_system_commands(self, screen: Any) -> Iterable[SystemCommand]:
        yield _find_shortcuts_cmd(screen, self)
        yield SystemCommand("Theme", "Change the current theme", self.action_change_theme)
        if self._maximized:
            yield SystemCommand("Minimize", "Restore default layout", self.action_toggle_maximize)
        else:
            yield SystemCommand("Maximize", "Maximize right panel", self.action_toggle_maximize)
        yield SystemCommand(
            "Screenshot",
            "Save an SVG screenshot of the current screen",
            lambda: self.set_timer(0.1, self.deliver_screenshot),
        )
        yield SystemCommand("Quit", "Quit the application", self.action_quit)

    def action_toggle_maximize(self) -> None:
        self._maximized = not self._maximized

    def watch__maximized(self, val: bool) -> None:
        self.set_class(val, "--maximized")

    def action_toggle_kowabunga(self) -> None:
        new_mode = (
            RunnerMode.KOWABUNGA_OFF
            if self._kowabunga == RunnerMode.KOWABUNGA_ON
            else RunnerMode.KOWABUNGA_ON
        )
        self.session.set_kowabunga(new_mode)
        self._kowabunga = new_mode

    def watch__kowabunga(self, val: RunnerMode) -> None:
        self._render_kowabunga_ind(val)

    def _render_kowabunga_ind(self, val: RunnerMode, *, error: bool = False) -> None:
        ind = self.query_one("#kowabunga-ind", Static)
        ind.update(cowabunga_markup(val == RunnerMode.KOWABUNGA_ON, error))


def run_with_tui(run_kwargs: dict[str, Any], session: Session | None = None) -> int:
    """Run the pipeline on a worker thread under RunApp.

    A ``session`` may be supplied (e.g. the one the PRD conversation just wrote
    into) so the PRD draft and its execution live in the same session dir;
    otherwise a fresh session is created.
    """
    from splinter.memory.session import new_session_id

    if session is None:
        session = Session(new_session_id())
    app = RunApp(session, {**run_kwargs, "session": session})
    app.run()
    if app.rc != 0:
        print(f"run failed (session {session.id}){': ' + app.error if app.error else ''}")
    else:
        print(f"run complete. session: {session.id}")
    return app.rc


# --- configure -------------------------------------------------------------


class _TemplateEditor(TextArea):
    """Markdown editor with the keymap users expect.

    - ``ctrl+a`` selects all (TextArea's default ctrl+a is line-start).
    - ``ctrl+c`` copies the selection, or the whole buffer when nothing is
      selected, to the system clipboard (OSC-52) with a confirmation toast.
    - click-and-drag selects text inside the widget; ``ctrl+x`` cuts.
    """

    BINDINGS = [
        Binding("ctrl+a", "select_all", "Select all", show=False),
        Binding("ctrl+shift+a", "select_all", "Select all", show=False),
        Binding("ctrl+c", "copy", "Copy", show=False),
    ]

    def action_copy(self) -> None:
        text = self.selected_text or self.text
        if not text:
            return
        self.app.copy_to_clipboard(text)
        scope = "selection" if self.selected_text else "whole template"
        self.notify(f"Copied ({scope}) ✓", timeout=2)


# Slot names the harness actually fills per template (see splinter/phases.py,
# agents/runner.py, agents/evaluator.py). Templates not listed here (PRD, AGENTS.md)
# are not ``str.format``-rendered, so they have no insertable slots — any ``{...}``
# in their text is literal and must not be offered.
_TEMPLATE_SLOTS: dict[str, list[str]] = {
    "plan": [
        "task_section",
        "acceptance_section",
        "code_context_section",
        "standards_section",
    ],
    "eval": [
        "task_section",
        "acceptance_section",
        "plan_section",
        "output_section",
        "gate_section",
        "previous_evals_section",
        "skill_section",
        "standards_section",
    ],
}


def _template_placeholder_keys(key: str) -> list[str]:
    """Return the insertable ``{placeholder}`` slot names for template ``key``."""
    return list(_TEMPLATE_SLOTS.get(key, []))


class TemplatesScreen(Screen[None]):
    """Edit project-level prompt overrides for PRD / Planner / Eval / AGENTS.md."""

    DEFAULT_CSS = """
    TemplatesScreen { layout: vertical; background: $background; }

    #tmpl-top {
        height: 3;
        align: left middle;
        padding: 0 1;
        background: $panel;
        border-bottom: tall $primary;
    }
    #tmpl-title {
        width: 1fr;
        content-align: left middle;
        margin: 0 1;
    }
    #tmpl-top Button {
        height: 3;
        min-width: 12;
        border: tall $boost;
        background: $boost;
        color: $text;
        text-style: bold;
        text-align: center;
        content-align: center middle;
        margin-right: 1;
    }
    #tmpl-top Button:hover { border: tall $accent; background: $panel-lighten-1; }
    #tmpl-back { color: $text; }
    #tmpl-back:hover { border: tall $primary; }
    #tmpl-copy { color: $accent-lighten-2; border: tall $accent-darken-1; }
    #tmpl-copy:hover { background: $accent; color: $text; border: tall $accent; }
    #tmpl-default { color: $warning-lighten-2; border: tall $warning-darken-2; }
    #tmpl-default:hover { background: $warning; color: $text; border: tall $warning; }
    #tmpl-save { color: $success-lighten-2; border: tall $success-darken-1; }
    #tmpl-save:hover { background: $success; color: $text; border: tall $success; }

    #tmpl-body { height: 1fr; }

    #tmpl-sidebar {
        width: 26;
        background: $panel;
        border-right: tall $primary;
    }
    #tmpl-sidebar-head, #tmpl-keys-head {
        height: 1;
        padding: 0 1;
        color: $text-muted;
        text-style: bold;
        background: $boost;
    }
    #tmpl-keys-head { margin-top: 1; }
    #tmpl-list {
        height: auto;
        max-height: 12;
        background: $panel;
        padding: 1 0;
        scrollbar-size-vertical: 1;
    }
    #tmpl-keys {
        height: 1fr;
        padding: 0;
        background: $panel;
        color: $accent;
        scrollbar-size-vertical: 1;
    }
    #tmpl-keys > .option-list--option { padding: 0 1; color: $accent; }
    #tmpl-keys > .option-list--option-highlighted {
        background: $accent;
        color: $text;
        text-style: bold;
    }
    #tmpl-keys > .option-list--option-disabled { color: $text-muted; }
    #tmpl-list > .option-list--option {
        padding: 0 1;
    }
    #tmpl-list > .option-list--option-highlighted {
        background: $primary;
        color: $text;
        text-style: bold;
    }

    #tmpl-pane { width: 1fr; height: 1fr; }
    #tmpl-filebar {
        height: 1;
        padding: 0 1;
        color: $text-muted;
        background: $surface;
        border-bottom: solid $boost;
    }
    #tmpl-editor {
        width: 1fr;
        height: 1fr;
        border: none;
    }
    """

    BINDINGS = [
        ("escape", "go_models", "◂ Models"),
        ("ctrl+s", "save_template", "Save"),
    ]

    def __init__(self) -> None:
        super().__init__()
        from splinter.configure import EDITABLE_TEMPLATES

        self._keys = [k for k, _ in EDITABLE_TEMPLATES]
        self._labels = [lbl for _, lbl in EDITABLE_TEMPLATES]
        self._current_idx = 0
        self._cur_keys: list[str] = []

    def _item_label(self, idx: int) -> "Text":
        from rich.text import Text

        from splinter.configure import template_is_overridden

        overridden = template_is_overridden(self._keys[idx])
        label = Text()
        if overridden:
            label.append("● ", style="bold yellow")
        else:
            label.append("○ ", style="dim")
        label.append(self._labels[idx])
        return label

    def compose(self) -> ComposeResult:
        yield Horizontal(
            Button("◂ Models", id="tmpl-back"),
            Static("[dim]configure ·[/] [b]templates[/]", id="tmpl-title"),
            Button("⧉ Copy", id="tmpl-copy"),
            Button("↺ Default", id="tmpl-default"),
            Button("✓ Save", id="tmpl-save"),
            id="tmpl-top",
        )
        with Horizontal(id="tmpl-body"):
            with Vertical(id="tmpl-sidebar"):
                yield Static("TEMPLATES", id="tmpl-sidebar-head")
                yield OptionList(
                    *[self._item_label(i) for i in range(len(self._keys))],
                    id="tmpl-list",
                )
                yield Static("INSERT KEY  [dim]· click to insert[/]", id="tmpl-keys-head")
                yield OptionList(id="tmpl-keys")
            with Vertical(id="tmpl-pane"):
                yield Static("", id="tmpl-filebar")
                yield self._make_editor()
        yield Footer()

    def _refresh_keys(self) -> None:
        """Rebuild the clickable placeholder palette for the current template."""
        box = self.query_one("#tmpl-keys", OptionList)
        box.clear_options()
        self._cur_keys = _template_placeholder_keys(self._keys[self._current_idx])
        if not self._cur_keys:
            box.add_option(Option("no insertable keys", disabled=True))
            return
        for name in self._cur_keys:
            box.add_option(f"{{{name}}}")

    def _insert_key(self, name: str) -> None:
        editor = self.query_one("#tmpl-editor", TextArea)
        editor.replace(f"{{{name}}}", editor.selection.start, editor.selection.end)
        editor.focus()

    def _make_editor(self) -> TextArea:
        # "focus" tab behavior (not "indent") so Escape bubbles to the screen
        # ("◂ Models") instead of being captured by the editor.
        try:
            return _TemplateEditor(
                "",
                id="tmpl-editor",
                language="markdown",
                soft_wrap=True,
                show_line_numbers=True,
                tab_behavior="focus",
                theme="vscode_dark",
            )
        except Exception:
            # syntax extras / theme unavailable — fall back to a plain editor
            return _TemplateEditor("", id="tmpl-editor", soft_wrap=True)

    def on_mount(self) -> None:
        self._load_current()
        # Focus the editor so typing, selection and copy work immediately.
        self.query_one("#tmpl-editor", TextArea).focus()

    def _do_copy(self) -> None:
        editor = self.query_one("#tmpl-editor", TextArea)
        text = editor.selected_text or editor.text
        if not text:
            return
        self.app.copy_to_clipboard(text)
        scope = "selection" if editor.selected_text else "whole template"
        self.notify(f"Copied ({scope}) ✓", timeout=2)

    def _load_current(self) -> None:
        from splinter.configure import (
            template_current_text,
            template_is_overridden,
            template_override_path,
        )

        key = self._keys[self._current_idx]
        self.query_one("#tmpl-editor", TextArea).load_text(template_current_text(key))
        overridden = template_is_overridden(key)
        path = template_override_path(key)
        origin = "project override" if overridden else "packaged default"
        self.query_one("#tmpl-filebar", Static).update(
            f"[b]{self._labels[self._current_idx]}[/]  [dim]·[/]  {path}  [dim]·[/]  {origin}"
        )
        self._refresh_list()
        self._refresh_keys()

    def _refresh_list(self) -> None:
        ol = self.query_one("#tmpl-list", OptionList)
        ol.clear_options()
        for i in range(len(self._keys)):
            ol.add_option(self._item_label(i))
        if ol.highlighted != self._current_idx:
            ol.highlighted = self._current_idx

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option_list.id != "tmpl-list":
            return
        # Re-highlighting the same template (e.g. when _refresh_list resets the
        # OptionList) must NOT reload the editor — that would wipe unsaved edits.
        if event.option_index == self._current_idx:
            return
        self._current_idx = event.option_index
        self._load_current()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "tmpl-list":
            # Enter / click on a template → jump straight into editing.
            self._current_idx = event.option_index
            self._load_current()
            self.query_one("#tmpl-editor", TextArea).focus()
            event.stop()
        elif event.option_list.id == "tmpl-keys":
            if 0 <= event.option_index < len(self._cur_keys):
                self._insert_key(self._cur_keys[event.option_index])
            event.stop()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "tmpl-back":
            self.dismiss()
        elif bid == "tmpl-copy":
            self._do_copy()
        elif bid == "tmpl-save":
            self._do_save()
        elif bid == "tmpl-default":
            self._do_default()

    def _do_save(self) -> None:
        from splinter.configure import write_template_override

        key = self._keys[self._current_idx]
        write_template_override(key, self.query_one("#tmpl-editor", TextArea).text)
        self.notify(f"Saved {self._labels[self._current_idx]} ✓", timeout=2)
        self._refresh_list()

    def _do_default(self) -> None:
        from splinter.configure import reset_template_override

        key = self._keys[self._current_idx]
        reset_template_override(key)
        self.notify(f"Reset {self._labels[self._current_idx]} to default", timeout=2)
        self._load_current()

    def action_go_models(self) -> None:
        self.dismiss()

    def action_save_template(self) -> None:
        self._do_save()


# --- interactive PRD session -----------------------------------------------


def _fm_block(prd_text: str) -> tuple[dict[str, Any], str]:
    """Split a PRD into (frontmatter dict, body)."""
    import yaml

    if prd_text.startswith("---"):
        parts = prd_text.split("---", 2)
        if len(parts) >= 3:
            try:
                fm = yaml.safe_load(parts[1]) or {}
            except yaml.YAMLError:
                fm = {}
            return (fm if isinstance(fm, dict) else {}), parts[2]
    return {}, prd_text


def _set_fm_strategy(prd_text: str, strategy: str) -> str:
    """Force the ``strategy:`` field in the PRD frontmatter to ``strategy``."""
    fm, body = _fm_block(prd_text)
    if not fm:
        return prd_text
    fm["strategy"] = strategy
    import yaml

    return f"---\n{yaml.safe_dump(fm, sort_keys=False).strip()}\n---{body}"


class ConfirmQuit(ModalScreen[bool]):
    """Are-you-sure dialog before abandoning a PRD session."""

    CSS = """
    ConfirmQuit { align: center middle; }
    #box {
        width: 80; max-width: 90%; height: auto; padding: 1 2;
        border: thick $warning; background: $surface;
    }
    #box Static { width: 100%; height: auto; }
    #cmd { color: $text; background: $boost; padding: 0 1; margin: 1 0; }
    #qbuttons { height: 3; align-horizontal: center; margin-top: 1; }
    #qbuttons Button { margin: 0 1; }
    """

    BINDINGS = [("escape", "stay", "Stay")]

    def __init__(self, session_id: str, what: str = "PRD") -> None:
        super().__init__()
        self.session_id = session_id
        self.what = what

    def compose(self) -> ComposeResult:
        with Vertical(id="box"):
            yield Static(f"Leave this {self.what} session?")
            yield Static("The draft is saved — resume later with:")
            yield Static(f"uv run splinter resume {self.session_id}", id="cmd")
            with Horizontal(id="qbuttons"):
                yield Button("Leave", id="leave", variant="error")
                yield Button("Stay", id="stay", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "leave")

    def action_stay(self) -> None:
        self.dismiss(False)


class ComposerTextArea(TextArea):
    """PRD composer box: Enter inserts a newline; Ctrl+S submits."""

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        if event.key == "ctrl+s":
            event.stop()
            event.prevent_default()
            send = getattr(self.app, "action_send", None)
            if callable(send):
                send()
            return
        await super()._on_key(event)


class EditorPane(Static):
    """Shared editor pane: TextArea (editable) + Markdown (preview)."""

    CSS = """
    EditorPane {
        width: 100%;
        height: 100%;
    }
    EditorPane > Horizontal {
        width: 100%;
        height: 100%;
    }
    EditorPane #editor {
        width: 50%;
        border-right: solid $primary;
    }
    EditorPane #preview {
        width: 50%;
        padding: 0 1;
    }
    """

    def __init__(self, initial_content: str = "", id: str | None = None) -> None:
        super().__init__(id=id)
        self._initial_content = initial_content
        self._update_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield TextArea(id="editor")
            yield Markdown(id="preview")

    def on_mount(self) -> None:
        editor = self.query_one("#editor", TextArea)
        editor.text = self._initial_content
        self._refresh_preview()
        self._update_timer = self.set_interval(0.5, self._refresh_preview)

    def on_unmount(self) -> None:
        if self._update_timer:
            self._update_timer.stop()

    def _refresh_preview(self) -> None:
        try:
            editor = self.query_one("#editor", TextArea)
            preview = self.query_one("#preview", Markdown)
            content = editor.text or "_(empty)_"
            preview.update(content)
        except Exception:
            pass

    def get_content(self) -> str:
        try:
            return str(self.query_one("#editor", TextArea).text)
        except Exception:
            return ""

    def set_content(self, content: str) -> None:
        try:
            editor = self.query_one("#editor", TextArea)
            editor.text = content
            self._refresh_preview()
        except Exception:
            pass


#: Which step of the PRD stepper each conversation phase belongs to.
_PHASE_STEPS: dict[str, int] = {
    "init": 0,
    "generate": 0,
    "chat": 1,
    "finalize": 2,
    "trust": 2,
    "strategy": 3,
    "review": 3,
    "run": 4,
}

#: The stepper step the questions live on (clickable to go back to them).
_QUESTIONS_STEP = 1

#: What the Cowabunga button does in each phase.
_COWABUNGA_TOOLTIPS: dict[str, str] = {
    "chat": "Answer the open questions yourself and finalize",
    "strategy": "Pick the recommended strategy and run",
    "review": "Pick the recommended strategy and run",
    "trust": "Accept the PRD as it stands and run it now",
}


class PrdSessionApp(App[int | None]):
    """Refine a PRD with the user, pick a strategy, then hand off to the runner.

    Left pane: empty editable instructions. Right pane: PRD preview + conversation.
    Phases: ``generate`` (from instructions) → ``chat`` (Q&A until "fulfilled") →
    ``strategy`` (pick a turtle) → ``review`` (eyeball the user stories) → exit with the run kwargs.
    """

    CSS = (
        """
    PrdSessionApp.-waiting #session-body { display: none; }
    PrdSessionApp.-waiting Footer { display: none; }
    #draftpane {
        width: 34%;
        padding: 1 2 0 2;
        border-left: solid #1d1d1a;
    }
    #draft-body {
        height: 1fr;
    }
    #draft-cap, #outline-scroll, #context-scroll, #answered, #spend, #prompt-glyph, #q-nav {
        display: none;
    }
    #context-scroll { height: 1fr; scrollbar-size-vertical: 0; }
    #context-body { height: auto; }
    #draft-cap { height: auto; margin-bottom: 1; }
    #outline-scroll {
        height: 1fr;
        scrollbar-size-vertical: 0;
    }
    #outline { height: auto; }
    #q-index-head {
        height: 1;
        margin-top: 1;
        text-style: bold;
        color: #ecebe4;
    }
    #spend { height: 1; margin-top: 1; }
    #prompt-row { height: auto; }
    #prompt-glyph { width: 2; height: 1; color: #c6f432; }
    #q-nav { height: 1; margin: 1 0 0 0; }
    #q-keys { width: 1fr; height: 1; }
    #q-hint { width: auto; height: 1; }
    PrdSessionApp.-asking Footer { display: none; }
    PrdSessionApp.-asking #convo { display: none; }
    PrdSessionApp.-asking #chatpane { padding: 0 7 1 7; }
    PrdSessionApp.-asking #draftpane {
        width: 30%;
        min-width: 40;
        max-width: 52;
        padding: 1 3 1 3;
    }
    PrdSessionApp.-asking #draft-cap { display: block; }
    PrdSessionApp.-asking #draft-body { display: none; }
    PrdSessionApp.-asking #outline-scroll { display: block; }
    PrdSessionApp.-asking #answered { display: block; }
    PrdSessionApp.-asking #spend { display: block; }
    PrdSessionApp.-asking.-context #outline-scroll { display: none; }
    PrdSessionApp.-asking.-context #context-scroll { display: block; }
    PrdSessionApp.-asking.-editing #draft-body { display: block; }
    PrdSessionApp.-asking.-editing #outline-scroll { display: none; }
    PrdSessionApp.-asking.-editing #context-scroll { display: none; }
    PrdSessionApp.-asking #draft-edit { border: none; }
    PrdSessionApp.-asking #question-form {
        height: 1fr;
        max-height: 100%;
        margin: 0;
    }
    PrdSessionApp.-asking #q-back { display: none; }
    PrdSessionApp.-asking #actions { display: none; }
    PrdSessionApp.-asking #q-nav { display: block; }
    PrdSessionApp.-asking #prompt-glyph { display: block; }
    PrdSessionApp.-asking #composer { padding: 0; }
    PrdSessionApp.-asking #entry, PrdSessionApp.-asking #entry:focus {
        height: auto;
        min-height: 1;
        max-height: 4;
        border: none;
        padding: 0;
    }
    PrdSessionApp.-asking #entry .text-area--cursor-line { background: #0a0a09; }
    #draft-stage {
        display: none;
        height: 1fr;
    }
    #draft-word {
        margin: 1 0 1 0;
    }
    #draft-sentence {
        height: auto;
        margin-bottom: 1;
    }
    #instructions {
        height: 1fr;
        min-height: 5;
        border: none;
        border-left: wide #7a7972;
        padding: 0 1;
    }
    #instructions:focus {
        border: none;
        border-left: wide #c6f432;
    }
    #draft-keys {
        height: 1;
        margin: 1 0;
    }
    #task-picker { display: none; }
    PrdSessionApp.-task #instructions { display: none; }
    PrdSessionApp.-task #task-picker { display: block; }
    PrdSessionApp.-drafting #draft-stage { display: block; }
    PrdSessionApp.-drafting #draftpane { display: none; }
    PrdSessionApp.-drafting #actions { display: none; }
    PrdSessionApp.-drafting #question-index { display: none; }
    PrdSessionApp.-drafting #convo { height: auto; max-height: 4; }
    PrdSessionApp.-drafting #composer { border-top: none; padding: 0; }
    PrdSessionApp.-drafting Footer { display: none; }
    #chatpane {
        width: 1fr;
        padding: 1 4 0 4;
    }
    #convo {
        height: 1fr;
        padding: 0 1;
        overflow-x: hidden;
        scrollbar-size-horizontal: 0;
        scrollbar-size-vertical: 1;
    }
    #questions-host {
        height: auto;
    }
    #composer {
        dock: bottom;
        height: auto;
        border-top: solid #1d1d1a;
        padding: 1 0 0 0;
    }
    #busy {
        height: 1;
    }
    #entry {
        height: 3;
        border: none;
        border-left: wide #7a7972;
        padding: 0 1;
    }
    #entry:focus {
        border: none;
        border-left: wide #c6f432;
    }
    #draft-edit {
        border: none;
        border-top: solid #1d1d1a;
        border-title-color: #8f8e85;
        border-title-align: left;
        padding: 0;
    }
    #actions {
        height: 1;
        margin: 1 0;
        padding: 0;
        align: left middle;
    }
    """
        + SHELL_CSS
        + _PALETTE_CSS
        + _MAXIMIZE_CSS
    )

    COMMANDS = {_OrderedCommandsProvider}

    BINDINGS = [
        ("ctrl+c", "abort", "Abort"),
        ("escape", "escape", "Abort"),
        Binding("ctrl+s", "send", "Send"),
        Binding("ctrl+k", "cowabunga", "Cowabunga", show=False, priority=True),
        Binding("enter", "finish_questions", "Finalize", show=False, priority=True),
        Binding("e", "edit_draft", "Edit draft", show=False),
        Binding("question_mark", "question_context", "Context", show=False),
        Binding("x", "explain_again", "Explain again", show=False),
        Binding("ctrl+t", "plan_task", "Plan first", show=False),
        Binding("f2", "configure", "Configure", show=False),
        Binding("ctrl+o", "pick('file')", "File", show=False),
        Binding("ctrl+l", "pick('folder')", "Folder", show=False),
    ]

    _maximized: reactive[bool] = reactive(False)

    def __init__(self, session: Session, run_kwargs: dict[str, Any]) -> None:
        super().__init__()
        self.session = session
        self.run_kwargs = run_kwargs
        self.task_plan = False
        self.cowabunga = bool(run_kwargs.get("cowabunga"))
        self.resuming = bool(run_kwargs.get("resume"))
        self.no_ground = bool(run_kwargs.get("no_ground"))
        self.trusted = False
        self.phase = "init"
        self.claude_session = ""
        self.final_prd = ""
        self.strategy: str | None = run_kwargs.get("strategy")
        self._busy = False
        self._initial_prd = ""
        self._desc = run_kwargs.get("description", "")
        self._convo_lines: list[str] = []
        self._generating = False
        self._started_at: str | None = None
        self._source_prd_path = str(run_kwargs.get("prd_path") or "")
        self._failed = False
        self._last_job: tuple[Any, dict[str, Any]] | None = None
        self._prd_name = Path(self._source_prd_path).name if self._source_prd_path else "draft"
        self._prd_base = ""
        self._questions: list[Question] = []
        self._last_questions: list[Question] = []
        self._last_answers: dict[int, str] = {}
        self._model_name = ""
        self.mode = "prd"
        self.task_handoff: str | None = None
        self._explanations: dict[str, Explanation] = {}
        self._explain_errors: dict[str, str] = {}
        self._explaining: set[str] = set()

    def _save_state(self) -> None:
        """Persist enough to resume this refinement: conversation id, phase, strategy."""
        from datetime import datetime, timezone

        if self._started_at is None:
            self._started_at = datetime.now(timezone.utc).isoformat()
        self.session.set_status(
            "refining",
            source="prd",
            phase=self.phase,
            claude_session=self.claude_session,
            strategy=self.strategy or "?",
            started_at=self._started_at,
        )

    def compose(self) -> ComposeResult:
        label = self._prd_name if self._source_prd_path else f"new {self.mode}"
        yield ShellBar(crumb_markup("prd", label), PhaseStepper(PRD_STEPS, id="stepper"))
        with Horizontal(id="session-body"):
            with Vertical(id="chatpane"):
                with Vertical(id="draft-stage"):
                    mode = MODES[self.mode]
                    yield BlockWord(mode.word, id="draft-word")
                    yield Static(mode.sentence(), id="draft-sentence")
                    entry = ComposerTextArea(id="instructions", soft_wrap=True)
                    entry.placeholder = mode.placeholder
                    yield entry
                    yield Ink(mode.keys(), id="draft-keys")
                yield RichLog(id="convo", markup=True, wrap=True)
                with Vertical(id="questions-host"):
                    yield QuestionForm(id="question-form")
                with Vertical(id="composer"):
                    yield BusyStatus(id="busy")
                    with Horizontal(id="prompt-row"):
                        yield Static("›", id="prompt-glyph")
                        entry_reply = ComposerTextArea(id="entry", soft_wrap=True)
                        entry_reply.placeholder = "anything else the writer should know"
                        yield entry_reply
                    with Horizontal(id="actions"):
                        yield Button("Generate PRD", id="generate", variant="primary")
                    with Horizontal(id="q-nav"):
                        yield Ink("", id="q-keys")
                        yield Ink(cowabunga_hint(), id="q-hint")
            with Vertical(id="draftpane"):
                yield Ink(cap_markup(False), id="draft-cap")
                yield Vertical(id="draft-body")
                with VerticalScroll(id="outline-scroll"):
                    yield Ink("", id="outline")
                    yield Static("Open questions", id="q-index-head")
                    yield QuestionIndex(id="question-index")
                with VerticalScroll(id="context-scroll"):
                    yield Ink("", id="context-body")
                yield AnsweredList(id="answered")
                yield Ink("", id="spend")
        yield WaitingRoom()
        yield Footer()

    def on_mount(self) -> None:
        from pathlib import Path

        from splinter import prd_session

        self.title = "splinter · PRD"
        use_splinter_theme(self)
        self._update_stepper()

        if not self.resuming:
            from datetime import datetime, timezone

            self._started_at = datetime.now(timezone.utc).isoformat()

        path = self.run_kwargs.get("prd_path")
        try:
            self._initial_prd = Path(path).read_text() if path else ""
        except OSError as exc:
            self._say(f"[red]ERROR: {escape(f'cannot read PRD: {exc}')}[/]")
            self.set_timer(2.0, lambda: self.exit(None))
            return
        if not self._initial_prd.strip():
            self._initial_prd = self.session.read("prd.md")
        fm, _ = _fm_block(self._initial_prd)
        if not self._desc:
            self._desc = str(fm.get("feature", "")) or self._first_line(self._initial_prd)
        self._set_preview(self._initial_prd, label="initial")

        if self.resuming:
            self._resume()
            return

        route = prd_session.route_prd(self._initial_prd)
        if route == "generate":
            self._enter_generate_phase()
        elif self.cowabunga:
            self._set_phase("finalize")
            self._set_busy(True, "cowabunga — the model is deciding everything…")
            self._say("[magenta]🤙 cowabunga — no questions, finalizing the PRD myself.[/]")
            self._spawn(self._finalize_worker, autodecide=True)
        else:
            if self._initial_prd.strip():
                self._mount_draft_editor(self._initial_prd)
            self._set_phase("chat")
            self._set_busy(True, "reading the PRD, drafting questions…")
            self._say("[dim]reading the PRD, drafting clarifying questions…[/]")
            self._spawn(self._questions_worker)

    def _enter_generate_phase(self) -> None:
        """Draft phase: one column, the stage word, one input and its keys."""
        self._generating = True
        self._set_phase("generate")
        self._set_composer_visible(False)
        self._render_actions("generate")
        self._set_busy(False)
        self._focus_instructions()

    def action_mode(self, name: str) -> None:
        """Header chip: switch between ``--prd`` and ``--task`` while drafting."""
        if name not in MODES or name == self.mode:
            return
        if not self._generating or self._busy:
            self.notify("The mode is fixed once the PRD exists.", severity="warning")
            return
        self.mode = name
        self.set_class(name == "task", "-task")
        if name == "task" and not self.query(TaskPicker):
            self.query_one("#draft-stage").mount(
                TaskPicker(id="task-picker"), before="#draft-keys"
            )
        mode = MODES[name]
        label = self._prd_name if self._source_prd_path else f"new {name}"
        self.query_one(ShellBar).set_crumb(crumb_markup(name, label))
        self.query_one("#stepper", PhaseStepper).set_steps(mode.steps)
        self.query_one("#draft-word", BlockWord).word = mode.word
        self.query_one("#draft-sentence", Static).update(mode.sentence())
        self.query_one("#draft-keys", Ink).update(mode.keys())
        self.query_one("#instructions", TextArea).placeholder = mode.placeholder
        self._update_stepper()
        self._focus_instructions()

    def _run_task(self, plan: bool = False) -> None:
        """``^s`` hands the task to the run screen; ``^t`` plans it with you first."""
        picker = self.query_one(TaskPicker)
        text = picker.value
        if not text:
            self.notify("Type a path, pick a file, or write the task in a line.",
                        severity="warning")
            return
        if looks_like_path(text) and not Path(text).expanduser().exists():
            self.notify(f"No such file or folder: {text}", severity="warning")
            picker.focus_path()
            return
        self.task_plan = plan
        self.task_handoff = str(task_source(self.session, text))
        self.exit(0)

    def on_task_picker_chosen(self, event: TaskPicker.Chosen) -> None:
        """A file was taken: back to the path, where ↵ runs it."""
        self.query_one(TaskPicker).focus_path()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "task-path" and self._generating and self.mode == "task":
            event.stop()
            self._run_task()

    def action_plan_task(self) -> None:
        if self._generating and self.mode == "task" and not self._busy:
            self._run_task(plan=True)

    def action_pick(self, kind: str) -> None:
        """``^o`` / ``^l``: the system's own file or folder chooser, off the UI thread."""
        if not (self._generating and self.mode == "task"):
            return
        if sys.platform != "darwin":
            self.notify("No system chooser here: type the path or pick it in the tree.")
            return

        def work() -> None:
            picked = system_pick(kind == "folder")
            self.call_from_thread(self.query_one(TaskPicker).take_system_pick, picked)

        self.run_worker(work, thread=True, name="system-pick")

    def _set_composer_visible(self, visible: bool) -> None:
        """Show or hide the right-hand composer (hidden while generating)."""
        try:
            self.query_one("#entry", TextArea).display = visible
        except Exception:
            pass

    def _set_phase(self, phase: str) -> None:
        """Move to a phase and keep the stepper in sync."""
        self.phase = phase
        self.set_class(phase == "generate", "-drafting")
        self._update_stepper()
        if phase != "chat":
            # The question form belongs to the questions phase only.
            self._set_questions([])

    def _update_stepper(self) -> None:
        """Point the stepper (and the subtitle) at the current step."""
        try:
            stepper = self.query_one("#stepper", PhaseStepper)
        except Exception:
            return
        stepper.set_current(_PHASE_STEPS.get(self.phase, 0))
        label = stepper.current_label
        if isinstance(label, str):
            self.sub_title = label

    @staticmethod
    def _first_line(text: str) -> str:
        for ln in text.splitlines():
            if ln.strip() and not ln.strip().startswith("---"):
                return ln.strip()
        return "feature"

    def _resume(self) -> None:
        """Re-enter a saved refinement at its last phase, reusing the conversation."""
        status = self.session.read_status()
        self.claude_session = str(status.get("claude_session", "") or "")
        saved = status.get("strategy")
        if saved and saved != "?":
            self.strategy = str(saved)
        self.final_prd = self.session.read("prd.md")
        phase = str(status.get("phase") or "chat")

        prd_for_left = self.final_prd or self._initial_prd
        if prd_for_left.strip() and phase != "trust":
            self._mount_draft_editor(prd_for_left)

        self._replay_convo()
        self._say(f"[magenta]⟳ resumed {self.session.id} at phase '{phase}'.[/]")
        if not self.claude_session:
            self._say(
                "[yellow]No saved conversation id — prior context may be lost; "
                "answers still apply to the current draft.[/]"
            )

        if phase == "trust":
            self._set_phase("trust")
            self.trusted = True
            self._enter_trusted()
        elif phase in ("review", "strategy"):
            self._set_phase("review")
            self._show_stories()
            self._show_final_eval_hint()
            self._render_actions("review")
            self._set_busy(False, "accept / gate: <cmds> / changes / cowabunga")
        else:  # clarify / refine both live in the chat phase
            self._set_phase("chat")
            self._render_actions("chat")
            self._set_busy(False, "your answers / fulfilled / cowabunga")
        self._save_state()

    def _enter_trusted(self) -> None:
        """Load non-empty PRD as-is into editable left pane; no generation step."""
        from splinter import prd_session

        self._set_phase("trust")
        self.final_prd = self._initial_prd

        def _complete_mount() -> None:
            prd_session.log_phase(self.session, "trust")
            self._say("[green]PRD loaded — edit it on the right if needed.[/]")
            self._render_actions("trust")
            self._set_busy(False, "edit / send PRD / cowabunga")
            self._save_state()

        async def _mount_edit() -> None:
            draftpane = self.query_one("#draft-body", Vertical)
            await draftpane.remove_children()
            self._prd_base = self._initial_prd
            edit = TextArea(
                id="draft-edit", soft_wrap=True, text=self._initial_prd, language="markdown"
            )
            edit.border_title = self._draft_title()
            edit.border_subtitle = ""
            await draftpane.mount(edit)
            self.call_after_refresh(_complete_mount)

        self.run_worker(_mount_edit(), name="mount-edit")

    def _read_draft(self) -> str:
        """Read current PRD from #draft-edit if mounted, else fall back to final_prd."""
        try:
            return str(self.query_one("#draft-edit", TextArea).text)
        except Exception:
            return self.final_prd or self._initial_prd

    def _read_trusted_draft(self) -> str:
        return self._read_draft()

    def _to_strategy_phase(self) -> None:
        """Transition to review phase and open RunSetupModal."""
        self._say("[green]✅ PRD finalized.[/]")
        self._set_phase("review")
        self._save_state()
        self._show_stories()
        self._show_final_eval_hint()
        self._render_actions("review")
        self._set_busy(False, "accept / gate: <cmds> / changes / cowabunga")
        self._open_run_setup_modal()

    def _accept_trusted(self) -> None:
        """Accept edited PRD and proceed to strategy selection."""
        from splinter import prd_session

        text = self._read_trusted_draft()
        self.final_prd = prd_session.ensure_frontmatter(
            text, description=self._desc, strategy=self.strategy
        )
        self._set_preview(self.final_prd, label="trust-accept")
        prd_session.log_phase(self.session, "trust-accept")
        n_stories = len(prd_session.user_story_titles(self.final_prd))
        if n_stories:
            self._show_stories()
        self._to_strategy_phase()

    def _on_trust(self, text: str) -> None:
        """Handle composer input in trust phase: cowabunga triggers run."""
        from splinter import prd_session

        if prd_session.is_cowabunga(text):
            self.final_prd = self._read_trusted_draft()
            self.final_prd = prd_session.ensure_frontmatter(
                self.final_prd, description=self._desc, strategy=self.strategy
            )
            self._begin_run(autopick=True)

    # --- ui helpers ---
    def _say(self, msg: str, *, persist: bool = True) -> None:
        self.query_one("#convo", RichLog).write(msg)
        if persist:
            self._convo_lines.append(msg)
            # NUL-delimit records: a message may contain newlines, and splitting on
            # them would break a markup tag across lines (MarkupError on replay).
            self.session.write("convo.md", "\x00".join(self._convo_lines))

    def _replay_convo(self) -> None:
        """Reprint the saved conversation so a resumed session shows what was asked."""
        prior = self.session.read("convo.md")
        if not prior.strip():
            return
        # New sessions are NUL-delimited; tolerate the old newline format too.
        records = prior.split("\x00") if "\x00" in prior else prior.splitlines()
        records = [r for r in records if r]
        convo = self.query_one("#convo", RichLog)
        for rec in records:
            # RichLog defers markup parsing to render time, so a try/except around
            # write() won't catch bad markup — validate now and escape if it fails.
            try:
                Text.from_markup(rec)
            except Exception:  # noqa: BLE001 — bad markup in an old log shouldn't abort resume
                rec = escape(rec)
            convo.write(rec)
        self._convo_lines = records
        convo.write("[dim]─── resumed ───[/]")

    def _set_busy(self, busy: bool, placeholder: str = "") -> None:
        self._busy = busy
        self.set_class(busy, "-waiting")
        waiting = self.query_one(WaitingRoom)
        if busy:
            waiting.start(self._busy_action(placeholder), self._busy_model())
        else:
            waiting.stop()
        entry = self.query_one("#entry", TextArea)
        entry.disabled = busy
        for btn in self.query("#actions Button"):
            btn.disabled = busy
        for selector in (
            "#question-form RadioSet",
            "#question-form Button",
            "#question-form Input",
        ):
            for widget in self.query(selector):
                widget.disabled = busy
        try:
            status = self.query_one("#busy", BusyStatus)
        except Exception:
            status = None
        if status is not None:
            if busy:
                status.start(self._busy_action(placeholder), self._busy_model())
            else:
                status.stop()
        if placeholder and not busy:
            # The border title carries the phase hint; the busy line carries the action.
            if self._generating:
                self.query_one("#instructions", TextArea).border_title = placeholder
            else:
                entry.border_title = placeholder
        if not busy:
            if self._generating:
                self._focus_instructions()
            else:
                entry.focus()

    def _busy_action(self, placeholder: str) -> str:
        """The action shown by the busy line (placeholder without the ellipsis)."""
        return placeholder.rstrip("…").strip()

    async def action_configure(self) -> None:
        from splinter.tui_session_config import edit_session_config

        previous_model = self._busy_model()
        if await edit_session_config(self, busy=self._busy):
            self._model_name = ""
            if self._busy_model() != previous_model:
                self.claude_session = ""
            self._save_state()

    def _busy_model(self) -> str:
        """The model doing the work, for the busy line."""
        if self._model_name:
            return self._model_name
        try:
            from splinter.models.roster import load_ladder

            self._model_name = load_ladder().prd_model
        except Exception:
            self._model_name = ""
        return self._model_name

    def _draft_title(self) -> str:
        """Title of the left pane: the PRD file name, or 'draft'."""
        return f"PRD draft · {self._prd_name} · edit freely"

    def _set_questions(self, questions: list[Question]) -> None:
        """Show the question form for these questions (empty list hides it)."""
        self._questions = list(questions)
        if questions:
            self._last_questions = list(questions)
        try:
            form = self.query_one("#question-form", QuestionForm)
        except Exception:
            return
        if not self.is_running:
            return

        async def _swap() -> None:
            await form.set_questions(self._questions)
            form.display = bool(self._questions)
            if self._questions and form.mode == "one":
                form.focus_page()

        self.set_class(bool(self._questions), "-asking")
        if not self._questions:
            self.remove_class("-editing", "-context")
        count = len(self._questions)
        plural = "s" if count != 1 else ""
        form.meta = f"PRD writer · {self._writer_model()} · {count} question{plural}"
        self.run_worker(_swap, name="questions")  # type: ignore[arg-type]
        self._refresh_question_index()
        self._refresh_outline()
        self._refresh_spend()

    def _writer_model(self) -> str:
        """``sonnet@high``: the PRD model and its effort, as the header names it."""
        model = self._model_name or self._busy_model() or "sonnet"
        try:
            from splinter.models.roster import load_ladder

            effort = load_ladder().prd_effort
        except Exception:
            effort = ""
        return f"{model}@{effort}" if effort and "@" not in model else model

    def _refresh_question_index(self) -> None:
        """Point the right-pane question list and answers at the current questions."""
        try:
            index = self.query_one("#question-index", QuestionIndex)
            answered = self.query_one("#answered", AnsweredList)
        except Exception:
            return
        index.refresh_from(self._questions, self._last_answers)
        answered.refresh_from(self._questions, self._last_answers)
        self._refresh_nav()

    def _refresh_outline(self) -> None:
        """The right-pane outline of the draft (file, title, a few stories)."""
        try:
            outline = self.query_one("#outline", Ink)
        except Exception:
            return
        draft = self._read_draft() or self.session.read("prd.md")
        outline.update(outline_markup(draft, self._prd_name))

    def _refresh_spend(self) -> None:
        try:
            spend = self.query_one("#spend", Ink)
        except Exception:
            return
        spend.update(spend_markup(self.session.read_pre_run_usage(), self.session.read_jev_usage()))

    def _refresh_nav(self) -> None:
        """``← back  → next`` (``finalize`` on the last question) under the composer."""
        try:
            form = self.query_one("#question-form", QuestionForm)
            keys = self.query_one("#q-keys", Ink)
        except Exception:
            return
        letters = "".join(
            sorted({option.letter for q in self._questions for option in q.options})
        )
        answered = sum(1 for q in self._questions if self._last_answers.get(q.number))
        keys.update(
            nav_markup(
                mode=form.mode,
                on_last=form.on_last,
                answered=answered,
                total=len(self._questions),
                letters=letters or "A",
            )
        )

    def on_question_form_answers_changed(self, event: QuestionForm.AnswersChanged) -> None:
        self._last_answers = dict(event.answers)
        self._refresh_question_index()

    def on_question_form_page_changed(self, event: QuestionForm.PageChanged) -> None:
        self._refresh_nav()
        if self.has_class("-context"):
            self._refresh_context()

    def action_question_context(self) -> None:
        """``?``: the PRD sections behind the question and a plain-words explanation."""
        if not self.has_class("-asking"):
            return
        opening = not self.has_class("-context")
        self.remove_class("-editing")
        self.set_class(opening, "-context")
        if opening:
            self._refresh_context()
            return
        self.query_one("#draft-cap", Ink).update(cap_markup(False))

    def action_explain_again(self) -> None:
        """``x``: ask the cheap model again (after a failure, or for a fresh take)."""
        question = self._context_question()
        if question is None or not self.has_class("-context"):
            return
        self._explanations.pop(question.text, None)
        self._explain_errors.pop(question.text, None)
        self._refresh_context()

    def _context_question(self) -> Question | None:
        try:
            return self.query_one("#question-form", QuestionForm).current_question
        except Exception:
            return None

    def _refresh_context(self) -> None:
        """Render the context pane for the question on screen; explain it once."""
        question = self._context_question()
        if question is None:
            return
        draft = self._read_draft() or self.session.read("prd.md")
        sections = prd_excerpt(draft, question)
        key = question.text
        try:
            model = explain_model()[0]
        except Exception:
            model = ""
        self.query_one("#draft-cap", Ink).update(context_cap_markup(question.number))
        self.query_one("#context-body", Ink).update(
            context_markup(
                question,
                sections,
                explanation=self._explanations.get(key),
                error=self._explain_errors.get(key, ""),
                model=model,
            )
        )
        if key in self._explanations or key in self._explain_errors or key in self._explaining:
            return
        self._explaining.add(key)
        title = self._outline_title(draft)
        self.run_worker(
            lambda: self._explain_worker(question, sections, title),
            thread=True,
            name="explain",
            group="explain",
        )

    @staticmethod
    def _outline_title(draft: str) -> str:
        for line in draft.splitlines():
            if line.startswith("# "):
                return line[2:].strip()
        return ""

    def _explain_worker(self, question: Question, sections: list[Any], title: str) -> None:
        try:
            result = explain(question, sections, title)
        except Exception as exc:  # noqa: BLE001 — shown in the pane, x retries
            self.call_from_thread(self._explained, question.text, None, str(exc) or "failed")
            return
        self.session.log_llm_usage(result.model, result.tokens, result.cost)
        self.call_from_thread(self._explained, question.text, result, "")

    def _explained(self, key: str, result: Explanation | None, error: str) -> None:
        self._explaining.discard(key)
        if result is not None:
            self._explanations[key] = result
        else:
            self._explain_errors[key] = error[:200]
        self._refresh_spend()
        if self.has_class("-context"):
            self._refresh_context()

    def on_question_form_finish_requested(self, event: QuestionForm.FinishRequested) -> None:
        self.action_finish_questions()

    def on_question_form_completed(self, event: QuestionForm.Completed) -> None:
        """Auto-advance ended on the last question: ``→`` now finalizes."""
        self._refresh_nav()

    def action_question_back(self) -> None:
        try:
            self.query_one("#question-form", QuestionForm).action_prev_question()
        except Exception:
            return

    def action_question_next(self) -> None:
        try:
            self.query_one("#question-form", QuestionForm).action_next_question()
        except Exception:
            return

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """``enter`` finalizes only in ``all at once`` mode, away from the text boxes;
        ``^k`` is cowabunga only where cowabunga means something."""
        if action == "finish_questions":
            if not self.has_class("-asking") or isinstance(self.focused, TextArea):
                return False
            try:
                return self.query_one("#question-form", QuestionForm).mode == "all"
            except Exception:
                return False
        if action == "cowabunga":
            return self.phase in _COWABUNGA_TOOLTIPS and not self._busy
        return True

    def action_finish_questions(self) -> None:
        """``→ finalize`` / ``↵ finalize``: finalize the PRD with the answers picked."""
        if self._busy or self.phase != "chat" or not self._questions:
            return
        entry = self.query_one("#entry", TextArea)
        answers = self._compile_answers(entry.text)
        entry.text = ""
        if answers:
            self._say(f"[cyan]> {escape(answers)}[/]")
        draft = self._read_draft()
        self._set_phase("finalize")
        self._set_busy(True, "finalizing the PRD with your answers…")
        self._spawn(self._finalize_worker, autodecide=False, draft=draft, answers=answers)

    def action_cowabunga(self) -> None:
        """``^k``: let the model answer whatever is left and move on."""
        if not self._busy and self.phase in _COWABUNGA_TOOLTIPS:
            self._submit("cowabunga")

    def action_edit_draft(self) -> None:
        """``e``: swap the outline for the draft editor (``esc`` swaps back)."""
        if not self.has_class("-asking"):
            return
        editing = not self.has_class("-editing")
        self.remove_class("-context")
        self.set_class(editing, "-editing")
        self.query_one("#draft-cap", Ink).update(cap_markup(editing))
        if editing:
            try:
                self.query_one("#draft-edit", TextArea).focus()
            except Exception:
                pass
            return
        self._refresh_outline()
        self.query_one("#question-form", QuestionForm).focus_page()

    def action_escape(self) -> None:
        """``esc`` closes the editor or the context pane first; otherwise it asks to leave."""
        if self.has_class("-editing"):
            self.action_edit_draft()
            return
        if self.has_class("-context"):
            self.action_question_context()
            return
        self.action_abort()

    def action_jump_question(self, n: int | str) -> None:
        """QuestionIndex row click: jump to that question in the form."""
        try:
            self.query_one("#question-form", QuestionForm).jump_to(int(n))
        except Exception:
            return

    def on_phase_stepper_step_clicked(self, event: PhaseStepper.StepClicked) -> None:
        if self._busy or event.index != _QUESTIONS_STEP or not self._last_questions:
            return
        self._back_to_questions()

    def _back_to_questions(self) -> None:
        """Go back to the questions step, re-filled with the saved answers."""
        self._set_phase("chat")

        async def _swap() -> None:
            form = self.query_one("#question-form", QuestionForm)
            await form.set_questions(self._last_questions)
            form.restore_answers(self._last_answers)

        self.run_worker(_swap, name="questions")  # type: ignore[arg-type]
        self._render_actions("chat")
        self._set_busy(False, "answers · fulfilled · cowabunga")

    def _render_actions(self, phase: str) -> None:
        """Swap the action buttons to match the phase (chat / review / trust / generate)."""
        if phase == "generate":
            btns = [Button("Generate PRD", id="generate", variant="primary")]
        elif phase in ("review", "strategy"):
            btns = [
                Button("Send", id="send", variant="default"),
                Button("Run setup…", id="run-setup", variant="primary"),
                self._cowabunga_button("review"),
            ]
        else:
            btns = [Button("Send", id="send", variant="primary")]
            if phase == "trust":
                btns.append(Button("Send PRD", id="accept", variant="success"))
                btns.append(self._cowabunga_button("trust"))
            else:  # chat
                btns.append(Button("Fulfilled", id="fulfilled", variant="success"))
                btns.append(self._cowabunga_button("chat"))
        if self._failed:
            btns.append(Button("Retry", id="retry", variant="error"))
        bar = self.query_one("#actions", Horizontal)

        async def _swap() -> None:
            # await the removal before mounting, else the old ids collide with the new.
            await bar.remove_children()
            await bar.mount(*btns)

        self.run_worker(_swap, name="actions")  # type: ignore[arg-type]

    @staticmethod
    def _cowabunga_button(phase: str) -> Button:
        """The Cowabunga button, with a tooltip that says what it does now."""
        return Button(
            "Cowabunga",
            id="cowabunga",
            variant="warning",
            tooltip=_COWABUNGA_TOOLTIPS.get(phase, "Let the model decide"),
        )

    def _commit_prd(self, md: str, *, label: str, detail: str = "") -> bool:
        """Write ``prd.md``, version the snapshot, and sync the editor preview."""
        from splinter import prd_session

        text = md.strip()
        if not text:
            return False
        current = (self._read_draft() or self.session.read("prd.md")).strip()
        if current and current != text and not prd_session.should_accept_prd_update(current, text):
            self._say(
                "[yellow]Rejected PRD update — content shrank too much. "
                "Previous version kept (see prd-versions/ in analyze).[/]"
            )
            return False
        prd_session.save_prd_version(self.session, text, label=label, detail=detail)
        self.session.write("prd.md", text if text.endswith("\n") else text + "\n")
        self._write_source_prd(text)
        try:
            editor = self.query_one("#draft-edit", TextArea)
        except Exception:
            editor = None
        if editor is not None:
            self._prd_base = text
            editor.text = text
        return True

    def _set_preview(self, md: str, *, label: str = "update", detail: str = "") -> None:
        self._commit_prd(md, label=label, detail=detail)

    def _write_source_prd(self, md: str) -> None:
        if not self._source_prd_path:
            return
        try:
            from pathlib import Path as _Path

            _Path(self._source_prd_path).write_text(md)
        except OSError as exc:
            self._say(f"[red]ERROR: {escape(f'cannot write PRD: {exc}')}[/]")

    def _focus_instructions(self) -> None:
        if self.mode == "task":
            self.call_after_refresh(lambda: self.query_one("#task-path", Input).focus())
            return
        self.query_one("#instructions", TextArea).focus()

    def _fail(self, msg: str) -> None:
        """A model error keeps the screen open: show it in red and offer Retry."""
        self._say(f"[red]ERROR: {escape(msg)}[/]")
        self._failed = True
        if self.phase == "init" and not self._generating:
            self.phase = "chat"
        self._render_actions("generate" if self._generating else self.phase)
        self._set_busy(False, "retry, or keep typing")

    def _spawn(self, fn: Any, **kw: Any) -> None:
        from functools import partial

        self._failed = False
        self._last_job = (fn, kw)
        waiting = self.query_one(WaitingRoom)
        self.run_worker(partial(waiting.run_job, fn, **kw), thread=True, name="prd")

    def _retry(self) -> None:
        if self._busy or self._last_job is None:
            return
        fn, kw = self._last_job
        self._say("[dim]retrying…[/]")
        self._set_busy(True, "retrying…")
        self._spawn(fn, **kw)

    def _grounding(self, text: str) -> str:
        """Return grounding string from cached localization, or "" if --no-ground."""
        if self.no_ground:
            return ""
        from splinter import prd_session
        from splinter.models.roster import load_ladder

        return prd_session.ground_localization(self.session, load_ladder(), text)

    # --- workers (run off the UI thread) ---
    def _questions_worker(self) -> None:
        from splinter import prd_session

        try:
            turn = prd_session.open_questions(
                self._initial_prd,
                strategy=self.strategy,
                localization=self._grounding(self._initial_prd),
            )
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(self._fail, f"PRD model: {exc}")
            return
        self.session.log_llm_usage(turn.model, turn.tokens, turn.cost)
        self.call_from_thread(self._after_questions, turn.text, turn.session_id)

    def _refine_worker(self, answers: str, draft: str) -> None:
        from splinter import prd_session

        seed = draft or None if not self.claude_session else None
        try:
            turn = prd_session.refine(
                answers,
                resume=self.claude_session,
                prd_text=seed,
                current_prd=draft,
                localization=self._grounding(draft),
            )
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(self._fail, f"PRD model: {exc}")
            return
        self.session.log_llm_usage(turn.model, turn.tokens, turn.cost)
        self.call_from_thread(self._after_refine, turn.text, turn.session_id)

    def _finalize_worker(self, autodecide: bool, draft: str, answers: str = "") -> None:
        from splinter import prd_session

        seed = draft or None if not self.claude_session else None
        try:
            turn = prd_session.finalize(
                resume=self.claude_session,
                strategy=self.strategy,
                autodecide=autodecide,
                answers=answers,
                prd_text=seed,
                current_prd=draft,
                localization=self._grounding(draft or self._initial_prd),
            )
            prd = prd_session.ensure_frontmatter(
                turn.text, description=self._desc, strategy=self.strategy
            )
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(self._fail, f"PRD model: {exc}")
            return
        self.session.log_llm_usage(turn.model, turn.tokens, turn.cost)
        self.call_from_thread(self._after_finalize, prd, turn.session_id)

    def _revise_worker(self, instructions: str, draft: str) -> None:
        from splinter import prd_session

        seed = draft or None if not self.claude_session else None
        try:
            turn = prd_session.revise_final(instructions, resume=self.claude_session, prd_text=seed)
            prd = prd_session.ensure_frontmatter(
                turn.text, description=self._desc, strategy=self.strategy
            )
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(self._fail, f"PRD model: {exc}")
            return
        self.session.log_llm_usage(turn.model, turn.tokens, turn.cost)
        self.call_from_thread(self._after_revise, prd, turn.session_id)

    def _generate_worker(self, instructions: str) -> None:
        from splinter import prd_session

        try:
            turn = prd_session.generate_prd(
                instructions,
                strategy=self.strategy,
                localization=self._grounding(instructions),
            )
            prd = prd_session.ensure_frontmatter(
                turn.text, description=self._desc or "feature", strategy=self.strategy
            )
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(self._fail, f"PRD model: {exc}")
            return
        self.session.log_llm_usage(turn.model, turn.tokens, turn.cost)
        self.call_from_thread(self._after_generate, prd, turn.session_id)

    # --- worker callbacks (back on the UI thread) ---
    def _after_questions(self, questions: str, sid: str) -> None:
        from splinter import prd_session

        self.claude_session = sid
        prd_session.log_phase(self.session, "clarify")
        self._set_phase("chat")
        self._save_state()
        self._set_preview(self._initial_prd, label="initial")
        self._say(escape(questions))
        self._set_questions(parse_questions(questions))
        self._render_actions("chat")
        self._set_busy(False, "answers · fulfilled · cowabunga")

    def _after_refine(self, draft: str, sid: str) -> None:
        from splinter import prd_session

        self.claude_session = sid
        prd_session.log_phase(self.session, "refine")
        self._set_phase("chat")
        self._save_state()
        self._set_preview(prd_session.extract_working_draft(draft), label="refine")
        self._set_questions(extract_open_questions(draft))
        self._render_actions("chat")
        self._set_busy(False, "answers · fulfilled · cowabunga")

    def _after_finalize(self, prd: str, sid: str) -> None:
        from splinter import prd_session

        self.claude_session = sid
        self.final_prd = prd
        n_stories = len(prd_session.user_story_titles(prd))
        prd_session.log_phase(self.session, "finalize", f"{n_stories} stories")
        self._set_preview(prd, label="finalize", detail=f"{n_stories} stories")
        if self.cowabunga:
            self._begin_run(autopick=True)
            return
        self._to_strategy_phase()

    def _after_revise(self, prd: str, sid: str) -> None:
        from splinter import prd_session

        self.claude_session = sid
        self.final_prd = prd
        prd_session.log_phase(self.session, "revise")
        self._set_phase("review")
        self._save_state()
        self._set_preview(prd, label="revise")
        self._show_stories()
        self._show_final_eval_hint()
        self._render_actions("review")
        self._set_busy(False, "accept / gate: <cmds> / changes / cowabunga")

    def _mount_draft_editor(self, content: str) -> None:
        """Replace left pane with editable PRD draft after generation."""

        async def _do() -> None:
            draftpane = self.query_one("#draft-body", Vertical)
            await draftpane.remove_children()
            self._prd_base = content
            edit = TextArea(id="draft-edit", soft_wrap=True, text=content, language="markdown")
            edit.border_title = self._draft_title()
            edit.border_subtitle = ""
            await draftpane.mount(edit)

        self.run_worker(_do(), name="mount-draft")

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Flag the PRD as edited once the user changes the draft."""
        if event.text_area.id != "draft-edit":
            return
        edited = bool(self._prd_base) and event.text_area.text != self._prd_base
        event.text_area.border_subtitle = "edited" if edited else ""
        if self.has_class("-asking"):
            self._refresh_outline()

    def _after_generate(self, prd: str, sid: str) -> None:
        from splinter import prd_session

        self.claude_session = sid
        self.final_prd = prd
        self._generating = False
        n_stories = len(prd_session.user_story_titles(prd))
        prd_session.log_phase(self.session, "generate", f"{n_stories} stories")
        self._set_preview(prd, label="generate", detail=f"{n_stories} stories")
        self._mount_draft_editor(prd)
        # Enter Q&A phase on the generated PRD — user types "fulfilled" to proceed to strategy.
        self._initial_prd = prd
        self._set_phase("chat")
        self._set_composer_visible(True)
        self._render_actions("chat")
        self._set_busy(True, "reading the generated PRD, drafting questions…")
        self._say("[dim]PRD generated. Drafting clarifying questions…[/]")
        self._spawn(self._questions_worker)

    def _show_stories(self) -> None:
        from splinter import prd_session

        titles = prd_session.user_story_titles(self.final_prd)
        if titles:
            self._say("[bold]Tasks:[/]")
            for t in titles:
                self._say(f"  • {escape(t)}")
        else:
            self._say("[yellow]No US-NNN stories found — the PRD runs as a single task.[/]")

    def _show_final_eval_hint(self) -> None:
        """Show configured final_eval entries so the user knows what will run."""
        from splinter.configure import load_config, load_final_eval

        try:
            fe_path = self.session.dir / "final_eval.yaml"
            if fe_path.exists():
                import yaml as _yaml

                _fe_cfg = _yaml.safe_load(fe_path.read_text()) or {}
                entries = load_final_eval(_fe_cfg)
            else:
                entries = load_final_eval(load_config())
        except Exception:
            entries = []
        if entries:
            self._say("[dim]Final eval after tasks:[/]")
            for e in entries:
                self._say(f"  [dim]• {escape(e.name)} ({e.kind})[/]")
        else:
            self._say("[dim]No final eval set — click 'Run setup…' to configure.[/]")

    # --- input dispatch ---
    def _on_generate(self) -> None:
        """Generate button — read instructions and generate PRD."""
        if self._busy:
            return
        instructions = self.query_one("#instructions", TextArea).text.strip()
        if not instructions:
            self._say("[yellow]Please write instructions first.[/]")
            return
        self._set_busy(True, "generating PRD from instructions…")
        self._say("[cyan]Generating PRD from instructions…[/]")
        self._spawn(self._generate_worker, instructions=instructions)

    def action_send(self) -> None:
        """Ctrl+S / Send button — generate/send/accept depending on phase."""
        if self._generating and self.mode == "task":
            self._run_task()
            return
        if self._generating:
            self._on_generate()
            return
        if self.phase == "trust":
            self._accept_trusted()
            return
        entry = self.query_one("#entry", TextArea)
        self._submit(self._compile_answers(entry.text))

    def _command_words(self) -> set[str]:
        """Typed words that bypass the question form (they are commands, not answers)."""
        from splinter import prd_session

        words = {"cowabunga"}
        words.update(
            word for word in ("fulfilled", "done", "ready", "go", "ship", "lgtm")
            if prd_session.is_done(word)
        )
        if self.phase == "review":
            words.update({"accept", "run", "yes", "y", "edit"})
        return words

    def _compile_answers(self, text: str) -> str:
        """Turn the radio selection plus the composer text into one answer."""
        stripped = text.strip()
        if not self._questions or stripped.lower() in self._command_words():
            return stripped
        try:
            form = self.query_one("#question-form", QuestionForm)
        except Exception:
            return stripped
        return form.compile_answer(stripped)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "send":
            self.action_send()
        elif bid == "retry":
            self._retry()
        elif bid == "generate":
            self._on_generate()
        elif bid == "accept":
            if self.phase == "trust":
                self._accept_trusted()
            else:
                self._submit("accept")
        elif bid == "run-setup":
            self._open_run_setup_modal()
        elif bid in ("fulfilled", "cowabunga", "run"):
            self._submit(bid)
        elif bid == "set-final-eval":
            self._open_final_eval_modal()

    def _submit(self, raw: str) -> None:
        if self._busy:
            return
        text = raw.strip()
        self.query_one("#entry", TextArea).text = ""
        if not text:
            return
        self._say(f"[cyan]> {escape(text)}[/]")
        handler = {
            "chat": self._on_chat,
            "strategy": self._on_review,
            "review": self._on_review,
            "trust": self._on_trust,
        }.get(self.phase)
        if handler:
            handler(text)

    def _on_chat(self, text: str) -> None:
        from splinter import prd_session

        draft = self._read_draft()
        if prd_session.is_cowabunga(text):
            self._set_phase("finalize")
            self._set_busy(True, "cowabunga — finalizing…")
            self._spawn(self._finalize_worker, autodecide=True, draft=draft)
        elif prd_session.is_done(text):
            self._set_phase("finalize")
            self._set_busy(True, "finalizing the PRD…")
            self._spawn(self._finalize_worker, autodecide=False, draft=draft)
        else:
            self._set_busy(True, "incorporating your answers…")
            self._spawn(self._refine_worker, answers=text, draft=draft)

    def _on_review(self, text: str) -> None:
        from splinter import prd_session
        from splinter.strategy_pick import canonical_strategy_name

        # Let the user set the mechanical gate for this run, e.g.
        #   gate: npm run lint; npm test
        # Stored per-session; the planner only auto-detects when none is set.
        if text.lower().startswith("gate:"):
            from splinter.agents import gate

            checks = gate.parse_gate_spec(text.split(":", 1)[1], "unknown")
            gate.save_gate_checks(self.session.dir, checks)
            if checks:
                self._say("[green]Gate set:[/] " + ", ".join(c["cmd"] for c in checks))
            else:
                self._say("[yellow]Gate disabled — no mechanical checks this run.[/]")
            msg = "accept / gate: <cmds> / final_eval: ask_user|<cmd>|none / cowabunga"
            self._set_busy(False, msg)
            return

        if text.lower().startswith("final_eval:"):
            import yaml as _yaml

            spec = text.split(":", 1)[1].strip()
            fe_path = self.session.dir / "final_eval.yaml"
            if spec.lower() == "none":
                fe_path.write_text(_yaml.dump({"final_eval": []}, default_flow_style=False))
                self._say("[yellow]Final eval disabled for this run.[/]")
            elif spec.lower() == "ask_user":
                fe_path.write_text(
                    _yaml.dump(
                        {"final_eval": [{"name": "review", "kind": "ask_user"}]},
                        default_flow_style=False,
                    )
                )
                self._say("[green]Final eval set:[/] ask_user (manual review after run)")
            else:
                fe_path.write_text(
                    _yaml.dump(
                        {"final_eval": [{"name": spec.split()[0], "kind": "command", "cmd": spec}]},
                        default_flow_style=False,
                    )
                )
                self._say(f"[green]Final eval set:[/] {spec}")
            msg = "accept / gate: <cmds> / final_eval: ask_user|<cmd>|none / cowabunga"
            self._set_busy(False, msg)
            return

        if prd_session.is_cowabunga(text):
            self._begin_run(auto=True)
            return
        if text.lower() in {"accept", "run", "yes", "go", "y"}:
            self._begin_run()
            return
        canonical = canonical_strategy_name(text)
        if canonical is not None:
            self.strategy = canonical
            prd_session.log_phase(self.session, "strategy", self.strategy)
            self.final_prd = _set_fm_strategy(self._read_draft(), self.strategy)
            self.phase = "review"
            self._save_state()
            self._set_preview(self.final_prd, label="strategy", detail=self.strategy)
            self._say(f"[green]Strategy set:[/] {self.strategy}")
            self._set_busy(False, "accept / gate: <cmds> / changes / cowabunga")
            return
        self._set_busy(True, "applying your changes…")
        self._spawn(self._revise_worker, instructions=text, draft=self._read_draft())

    def _open_final_eval_modal(self) -> None:
        """Open the Set Final Eval modal and persist the user's choice."""
        from splinter.tui_run_setup import save_final_eval_choice

        def _on_result(result: dict[str, str | None] | None) -> None:
            if result is None:
                return
            kind_label = save_final_eval_choice(self.session.dir, result)
            self._say(f"[green]Final eval set:[/] {kind_label}")
            self._show_final_eval_hint()

        self.push_screen(_FinalEvalModal(), _on_result)

    def _open_run_setup_modal(self) -> None:
        """Open RunSetupModal for strategy, parallel, gate, and final-eval selection."""
        from splinter import prd_session
        from splinter.tui_run_setup import RunSetup, RunSetupModal

        draft = self._read_draft()
        if self.strategy:
            draft = _set_fm_strategy(draft, self.strategy)
        modal = RunSetupModal(
            draft,
            self.session,
            parallel=self.run_kwargs.get("parallel"),
            on_note=self._say,
        )

        def _after(setup: RunSetup | None) -> None:
            if setup is None:
                try:
                    self.query_one("#entry", TextArea).focus()
                except Exception:
                    pass
                return
            self.strategy = setup.strategy
            prd_session.log_phase(self.session, "strategy", self.strategy)
            self.final_prd = _set_fm_strategy(self._read_draft(), self.strategy)
            self.run_kwargs["parallel"] = bool(setup.parallel)
            self._finish_run()

        self.push_screen(modal, _after)

    # --- finish ---
    def _should_ask_parallel(self) -> bool:
        """Prompt for parallel only when it was left unspecified on the CLI and the
        run can actually parallelize: a multi-task strategy, 2+ stories, and git
        worktree support. `--parallel`/`--no-parallel` and cowabunga skip the ask."""
        from splinter.tui_run_setup import should_ask_parallel

        return should_ask_parallel(
            self.final_prd,
            self.strategy,
            self.run_kwargs.get("parallel"),
        )

    def _begin_run(self, autopick: bool = False, auto: bool = False) -> None:
        from splinter.strategy_pick import canonical_strategy_name, recommend_strategy

        draft = self._read_draft()
        if autopick or not self.strategy:
            rec, _ = recommend_strategy(draft)
            self.strategy = canonical_strategy_name(self.strategy or "") or rec
            draft = _set_fm_strategy(draft, self.strategy)
        self.final_prd = draft
        if self.run_kwargs.get("parallel") is None:
            self.run_kwargs["parallel"] = False
        self._finish_run()

    def _finish_run(self) -> None:
        from splinter import prd_session

        prd_session.log_phase(self.session, "run", self.strategy or "cascade")
        self._commit_prd(self.final_prd, label="run", detail=self.strategy or "cascade")
        self.session.update_index(
            f"# Session {self.session.id}\n- prd: prd.md\n- strategy: {self.strategy}\n"
        )
        mode = " · parallel" if self.run_kwargs.get("parallel") else ""
        self._say(f"[green]▶ running with strategy '{self.strategy}'{mode}…[/]")
        self._set_phase("run")
        self._save_state()
        self.exit(0)

    def action_abort(self) -> None:
        # Confirm first — the draft is recoverable, but a stray ESC shouldn't nuke the run.
        if isinstance(self.screen, ConfirmQuit):
            return  # dialog already up

        def _decide(leave: bool | None) -> None:
            if leave:
                from splinter import procreg

                self._say(
                    "[yellow]Leaving — resume later with:[/] "
                    f"[bold]uv run splinter resume {self.session.id}[/]"
                )
                procreg.terminate_all()
                self.exit(None)

        self.push_screen(ConfirmQuit(self.session.id), _decide)

    def get_system_commands(self, screen: Any) -> Iterable[SystemCommand]:
        yield _find_shortcuts_cmd(screen, self)
        yield SystemCommand("Theme", "Change the current theme", self.action_change_theme)
        if self._maximized:
            yield SystemCommand("Minimize", "Restore default layout", self.action_toggle_maximize)
        else:
            yield SystemCommand("Maximize", "Maximize right panel", self.action_toggle_maximize)
        yield SystemCommand(
            "Screenshot",
            "Save an SVG screenshot of the current screen",
            lambda: self.set_timer(0.1, self.deliver_screenshot),
        )
        yield SystemCommand("Quit", "Quit the application", self.action_quit)

    def action_toggle_maximize(self) -> None:
        self._maximized = not self._maximized

    def watch__maximized(self, val: bool) -> None:
        self.set_class(val, "--maximized")


def _prd_run_kwargs(prd_path: str, session: Session, run_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Build canonical run_kwargs from finalized PRD and strategy."""
    fm, _ = _fm_block(session.read("prd.md"))
    strategy = str(fm.get("strategy") or "") or "cascade"
    return {
        **run_kwargs,
        "strategy": strategy,
        "prd_path": prd_path,
        "task_path": None,
    }


def run_prd_interactive(run_kwargs: dict[str, Any]) -> int:
    """Refine the PRD in a TUI, then execute it in-app and return exit code."""
    from splinter.memory.session import new_session_id
    from splinter.prd_session import prd_session_is_resumable

    session = Session(new_session_id())
    app = PrdSessionApp(session, run_kwargs)
    result = app.run()
    handoff = getattr(app, "task_handoff", None)
    if result == 0 and isinstance(handoff, str) and handoff:
        task_kwargs = {**run_kwargs, "prd_path": None, "task_path": handoff}
        if getattr(app, "task_plan", False) is True:
            from splinter import tui_plan

            planned = tui_plan.PlanSessionApp(session, task_kwargs).run()
            return tui_plan._handoff(session, task_kwargs, planned)
        return run_with_tui(task_kwargs, session=session)
    if result is None:
        # Abandoned with no runnable PRD (a stub passes is_empty but has no
        # user stories) — delete instead of littering a dead refining session.
        if not prd_session_is_resumable(session):
            delete_session(session.id)
        elif session.dir.exists():
            print(f"PRD session aborted — resume later: uv run splinter resume {session.id}")
        return 0
    if isinstance(result, int) and result == 0:
        status = session.read_status()
        if status.get("phase") == "run":
            prd_path = str(session.dir / "prd.md")
            final_run_kwargs = _prd_run_kwargs(prd_path, session, run_kwargs)
            return run_with_tui(final_run_kwargs, session=session)
    if isinstance(result, int):
        return result
    return 0


def _resume_prd(session: Session, status: dict[str, Any]) -> int:
    """Re-enter a PRD refinement; on finalize, run it in-app and return exit code."""
    saved_strategy = status.get("strategy")
    run_kwargs: dict[str, Any] = {
        "strategy": saved_strategy if saved_strategy and saved_strategy != "?" else None,
        "prd_path": str(session.dir / "prd.md"),
        "task_path": None,
        "effort": None,
        "budget": None,
        "max_iterations": 5,
        "cowabunga": False,
        "resume": True,
        "parallel": bool(status.get("parallel", False)),
    }
    from splinter.prd_session import prd_session_is_resumable

    result = PrdSessionApp(session, run_kwargs).run()
    if result is None:
        if not prd_session_is_resumable(session):
            delete_session(session.id)  # nothing runnable to resume — don't litter
            return 0
        print(f"PRD session aborted — resume later: uv run splinter resume {session.id}")
        return 0
    if isinstance(result, int) and result == 0:
        new_status = session.read_status()
        if new_status.get("phase") == "run":
            prd_path = str(session.dir / "prd.md")
            final_run_kwargs = _prd_run_kwargs(prd_path, session, run_kwargs)
            return run_with_tui(final_run_kwargs, session=session)
    if isinstance(result, int):
        return result
    return 0


#: Which artifact a given stage produces — dropped to redo that stage on rollback.
_STAGE_ARTIFACT = {"localize": "knowledge/localization.md", "run": "knowledge/plan.md"}


def _resume_run(session: Session, status: dict[str, Any], *, reset: bool = False) -> int:
    """Re-enter a failed/interrupted pipeline run.

    - ``reset``: ignore all artifacts, re-run from the head (fresh localize + plan).
    - critical failure: roll the failing stage back (drop its artifact, redo it).
    - transient failure (provider/network blip): keep everything, continue.
    """
    saved_strategy = status.get("strategy")
    source = str(status.get("source") or "")
    prd_path: str | None = None
    task_path: str | None = None
    if source.endswith((".yaml", ".yml")):
        task_path = source
    elif session.read("prd.md").strip():
        prd_path = str(session.dir / "prd.md")
    elif source:
        prd_path = source
    else:
        print(f"session {session.id}: no PRD or task input recorded — cannot resume run.")
        return 1

    def _num(key: str) -> Any:
        val = status.get(key)
        return val if val not in (None, "") else None

    fail_class = str(status.get("fail_class") or "")
    stage = str(status.get("stage") or "")
    if reset:
        # events.jsonl is append-only, so a reset (unlike a normal resume) must
        # drop it explicitly — otherwise the discarded attempt's cost/runs would
        # keep summing into Trace.from_jsonl alongside the fresh one.
        (session.dir / "events.jsonl").unlink(missing_ok=True)
        print(f"resetting run {session.id} — re-running from the head.")
    elif fail_class == "critical":
        artifact = _STAGE_ARTIFACT.get(stage)
        if artifact:
            path = session.dir / artifact
            path.unlink(missing_ok=True)
            print(f"critical failure at stage '{stage}' — rolling back, redoing it.")
        else:
            print(f"critical failure at stage '{stage}' — redoing run.")
    else:
        print(f"resuming run {session.id} (reusing localization + plan)…")

    run_kwargs: dict[str, Any] = {
        "strategy": saved_strategy if saved_strategy and saved_strategy != "?" else None,
        "prd_path": prd_path,
        "task_path": task_path,
        "effort": _num("effort"),
        "budget": _num("budget"),
        "max_iterations": int(status.get("max_iterations") or 5),
        "cowabunga": False,
        "resume": not reset,
        "parallel": bool(status.get("parallel", False)),
        "max_concurrency": _num("max_concurrency"),
    }
    return run_with_tui(run_kwargs, session=session)


def resume_session(session_id: str | None, *, reset: bool = False) -> int:
    """Resume any session: PRD refinement, or a failed/interrupted pipeline run.

    ``reset`` forces a run to re-run from the head (fresh localize + plan).
    """
    from splinter.analyze import _run_state
    from splinter.memory.session import list_sessions
    from splinter.prd_session import prune_dead_prd_sessions

    prune_dead_prd_sessions()  # drop abandoned empty refinements before resolving
    resumable_run = {"FAILED", "INTERRUPTED", "PAUSED", "AWAITING_USER", "AWAITING_VALIDATION"}
    sessions = list_sessions()

    sid = session_id
    if sid is None:

        def _has_run_checkpoint(s: Session) -> bool:
            events_path = s.dir / "events.jsonl"
            return bool(
                s.read("run_checkpoint.json").strip()
                or s.read("checkpoint.json").strip()
                or (events_path.exists() and events_path.stat().st_size > 0)
            )

        # Prefer unfinished/paused runs (checkpoint continuity) over PRD refinement.
        for cand in sessions:
            sc = Session(cand)
            if _run_state(sc) in resumable_run or _has_run_checkpoint(sc):
                sid = cand
                break
        if sid is None:
            for cand in sessions:
                if Session(cand).read_status().get("state") == "refining":
                    sid = cand
                    break
        if sid is None:
            print("no resumable session found (none refining, failed, or interrupted).")
            return 1

    if sid not in sessions:
        print(f"no such session: {sid}")
        return 1

    session = Session(sid)
    status = session.read_status()
    if status.get("state") == "refining":
        if status.get("source") == "task":
            from splinter import tui_plan

            return tui_plan._resume_plan(session, status)
        return _resume_prd(session, status)

    state = _run_state(session)
    if state == "RUNNING":
        print(f"session {sid} is still running — not resumable while its process is alive.")
        return 1
    if state in ("COMPLETED", "DONE"):
        summary = format_run_completion(session)
        print(f"session {sid} finished ({summary}). Opening analyze.")
        AnalyzeApp(session).run()
        return 0
    if state in resumable_run:
        return _resume_run(session, status, reset=reset)

    print(f"session {sid} is not resumable (state: {state}).")
    return 1
