"""Distilled overview for the live run screen.

``splinter.analyze.render_overview`` stays the verbose, file-by-file view used by
``splinter analyze`` and the print paths. This module answers the *live* question
in two seconds — is it progressing, what is running right now, do I need to act —
and is what :class:`splinter.tui.RunApp` renders in its left pane.

Layout, top to bottom: a one-line header, a NOW card (only while the run is
live), the STEPS list, a TASKS table for multi-task strategies, a WAVES list for
the agents strategy and the PRD trajectory line. Unknown values are omitted
rather than printed as ``?``.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from rich.markup import escape

from splinter.analyze import (
    _has_final_eval_artifacts,
    _plan_files,
    _prd_phases,
)
from splinter.memory.session import Session
from splinter.obs.trace import RunEntry
from splinter.run_overview_agents import render_waves
from splinter.run_snapshot import (
    _FILE_RE,
    Snapshot,
    TaskRow,
    Verdict,
    blocked_task_ids,
    cost_split,
    current_task_no,
    fmt_span,
    last_verdict,
    latest_review_note,
    loop_verdicts,
    pre_run_models,
    snapshot,
    stage_label,
    task_rows,
    task_status,
    tier_effort,
    trace_span,
    verdict_from_note,
)

_STATE_EMOJI = {
    "RUNNING": "🟡",
    "COMPLETED": "🟢",
    "DONE": "🟢",
    "FAILED": "🔴",
    "INTERRUPTED": "🟠",
    "PAUSED": "🟠",
    "AWAITING_USER": "🟣",
    "AWAITING_VALIDATION": "🔍",
    "UNKNOWN": "⚪",
}

_STATE_COLOR = {
    "RUNNING": "yellow",
    "COMPLETED": "green",
    "DONE": "green",
    "FAILED": "red",
    "INTERRUPTED": "dark_orange",
    "PAUSED": "dark_orange",
    "AWAITING_USER": "magenta",
    "AWAITING_VALIDATION": "cyan",
    "UNKNOWN": "white",
}

#: States where the NOW card is meaningful — the run is live or waiting on the user.
LIVE_STATES = ("RUNNING", "AWAITING_USER", "AWAITING_VALIDATION")
AWAITING_STATES = ("AWAITING_USER", "AWAITING_VALIDATION")

#: Task status glyphs for the TASKS table.
_TASK_ICON = {
    "pending": ("○", "grey50"),
    "running": ("▶", "yellow"),
    "pass": ("✓", "green"),
    "failed": ("✗", "red"),
    "blocked": ("⏸", "yellow"),
}

#: Verdict colours reused from the analyze views (PASS/RETRY/ESCALATE/…).
_VERDICT_COLOR = {
    "PASS": "green",
    "RETRY": "yellow",
    "ESCALATE": "yellow",
    "JUMP_PREMIUM": "magenta",
    "ASK_USER": "cyan",
    "FAIL": "red",
}

#: Column widths of the TASKS table (visible characters).
_COL_NUM = 3
_COL_TITLE = 32
_COL_TIER = 3
_COL_ITERS = 3
_COL_VERDICT = 7

#: Longest verdict detail shown in the NOW card.
_DETAIL_LIMIT = 110

#: The PRD trajectory stays on one line — phases past the budget are dropped.
_PRD_LINE_BUDGET = 60

_Snapshot = Snapshot
_Verdict = Verdict
_TaskRow = TaskRow
_snapshot = snapshot
_pre_run_models = pre_run_models
_cost_split = cost_split
_fmt_span = fmt_span
_trace_span = trace_span
_tier_effort = tier_effort
_stage_label = stage_label
_last_verdict = last_verdict
_loop_verdicts = loop_verdicts
_latest_review_note = latest_review_note
_verdict_from_note = verdict_from_note
_task_rows = task_rows
_task_status = task_status
_current_task_no = current_task_no
_blocked_task_ids = blocked_task_ids


def render_run_overview(session: Session, state: str) -> str:
    """Render the live run overview as Rich markup."""
    snap = _snapshot(session, state)
    rows = _task_rows(snap)
    lines: list[str] = [_header_line(snap)]
    breakdown = _cost_breakdown_line(snap)
    if breakdown:
        lines.append(breakdown)
    lines.extend(_now_lines(snap, rows))
    lines.extend(_steps_lines(snap, rows))
    lines.extend(_tasks_lines(rows))
    lines.extend(render_waves(snap.session, str(snap.status.get("strategy", "") or "")))
    prd_line = _prd_line(snap)
    if prd_line:
        lines.append(prd_line)
    return "\n".join(lines)


# --- header -----------------------------------------------------------------


def _header_line(snap: _Snapshot) -> str:
    emoji = _STATE_EMOJI.get(snap.state, "⚪")
    color = _STATE_COLOR.get(snap.state, "white")
    parts = [f"[{color}]{emoji} {snap.state}[/]"]
    strategy = str(snap.status.get("strategy", "") or "")
    if strategy:
        parts.append(f"[b]{strategy}[/]")
    cost = _cost_part(snap)
    if cost:
        parts.append(cost)
    elapsed = _elapsed(snap)
    if elapsed:
        parts.append(f"[dim]⏱[/] {elapsed}")
    return "  ·  ".join(parts)


def _cost_part(snap: _Snapshot) -> str:
    """Total spend: run + pre-run (PRD, planner) + jev."""
    if not snap.entries and not snap.pre_run_models:
        return ""
    run_cost, pre_run, jev_cost, _ = _cost_split(snap)
    return f"[green]${run_cost + pre_run + jev_cost:.4f}[/]"


def _cost_breakdown_line(snap: _Snapshot) -> str:
    """Dim line under the header splitting the total; empty when it is all run spend."""
    if not snap.pre_run_models:
        return ""
    run_cost, pre_run, jev_cost, jev_calls = _cost_split(snap)
    parts = [f"run ${run_cost:.4f}"]
    if pre_run:
        parts.append(f"pre-run ${pre_run:.4f}")
    if jev_cost or jev_calls:
        calls = f" ×{jev_calls}" if jev_calls else ""
        parts.append(f"jev ${jev_cost:.4f}{calls}")
    return "[dim]  " + " · ".join(parts) + "[/]"


def _elapsed(snap: _Snapshot) -> str:
    started = str(snap.status.get("started_at") or snap.status.get("started") or "")
    if started and snap.state in LIVE_STATES:
        secs = _seconds_since(started)
        if secs is not None:
            return _fmt_span(secs)
    span = _trace_span(snap.entries)
    if span <= 0:
        return ""
    return _fmt_span(span)


def _seconds_since(stamp: str) -> float | None:
    try:
        then = datetime.fromisoformat(stamp)
    except (ValueError, TypeError):
        return None
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - then).total_seconds()


# --- NOW card ---------------------------------------------------------------


def _now_lines(snap: _Snapshot, rows: list[_TaskRow]) -> list[str]:
    if snap.state not in LIVE_STATES:
        return []
    stage = _stage_label(str(snap.status.get("stage", "")))
    entry = _current_entry(snap, stage)
    verdict = _last_verdict(snap)

    lines = ["[bold]NOW[/]"]
    if snap.state in AWAITING_STATES:
        lines.append("  [bold cyan]waiting for you[/] — answer in the dialog")
    task_line = _task_label(rows)
    if task_line:
        lines.append(f"  {task_line}")
    detail = _now_detail(snap, stage, entry)
    if detail:
        lines.append(f"  {detail}")
    if verdict.name:
        color = _VERDICT_COLOR.get(verdict.name, "white")
        line = f"  last: [{color}]{verdict.name}[/]"
        if verdict.detail:
            detail = verdict.detail.lstrip("—–-: ").strip()
            if detail:
                line += f" — {escape(_truncate(detail, _DETAIL_LIMIT))}"
        lines.append(line)
    return lines


def _now_detail(snap: _Snapshot, stage: str, entry: RunEntry | None) -> str:
    parts: list[str] = []
    if stage:
        parts.append(f"[yellow]{stage}[/]")
    if entry is not None:
        effort = _tier_effort(entry.tier, entry.model)
        label = f"{entry.model}@{effort}" if effort else entry.model
        parts.append(f"[b]{label}[/]")
    iteration = _iteration_label(snap, entry)
    if iteration:
        parts.append(f"[dim]{iteration}[/]")
    elapsed = _stage_elapsed(snap)
    if elapsed:
        parts.append(f"[dim]{elapsed} in stage[/]")
    return "  [dim]·[/]  ".join(parts)


def _iteration_label(snap: _Snapshot, entry: RunEntry | None) -> str:
    if entry is None:
        return ""
    max_iters = str(snap.status.get("max_iterations", "") or "")
    if max_iters and max_iters != "?":
        return f"iter {entry.iteration}/{max_iters}"
    return f"iter {entry.iteration}"


def _stage_elapsed(snap: _Snapshot) -> str:
    updated = str(snap.status.get("updated", "") or "")
    if not updated:
        return ""
    secs = _seconds_since(updated)
    if secs is None or secs < 0:
        return ""
    return _fmt_span(secs)


def _current_entry(snap: _Snapshot, stage: str) -> RunEntry | None:
    """Most recent model call for the current stage — the one that is working."""
    if not snap.entries:
        return None
    role = {"plan": "plan", "run": "run", "eval": "eval", "final_eval": "eval"}.get(stage)
    matching = [e for e in snap.entries if role and e.role == role]
    pool = matching or snap.entries
    return pool[-1] if pool else None


# --- verdict ----------------------------------------------------------------


# --- STEPS ------------------------------------------------------------------


def _steps_lines(snap: _Snapshot, rows: list[_TaskRow]) -> list[str]:
    stage = _stage_label(str(snap.status.get("stage", "")))
    running = snap.state == "RUNNING"
    completed = snap.state in ("COMPLETED", "DONE")

    lines = ["[bold]STEPS[/]"]
    lines.append(
        _step_line(
            bool(snap.localization),
            running and stage == "localize",
            "localize",
            _localize_detail(snap.localization),
        )
    )
    lines.append(
        _step_line(
            bool(snap.plan),
            running and stage == "plan",
            "plan",
            _plan_detail(snap),
        )
    )
    lines.extend(_run_step_lines(snap, rows, stage, running, completed))
    lines.extend(_eval_step_lines(snap, running, completed))
    lines.extend(_final_eval_step_lines(snap, stage, running))
    return lines


def _run_step_lines(
    snap: _Snapshot,
    rows: list[_TaskRow],
    stage: str,
    running: bool,
    completed: bool,
) -> list[str]:
    current = _running_row(rows)
    iters = current.iters if current else 0
    if completed:
        detail = "done"
    elif iters:
        max_iters = str(snap.status.get("max_iterations", "") or "")
        head = f"iter {iters}/{max_iters}" if max_iters and max_iters != "?" else f"iter {iters}"
        detail = f"{head} · {current.tier}" if current and current.tier else head
    else:
        detail = ""
    current_step = running and stage == "run" and not completed
    return [_step_line(completed or bool(iters), current_step, "run", detail)]


def _eval_step_lines(snap: _Snapshot, running: bool, completed: bool) -> list[str]:
    verdict = _last_verdict(snap)
    detail = f"last: {verdict.name}" if verdict.name else ""
    done = verdict.name == "PASS" or completed
    current = running and bool(verdict.name) and verdict.name != "PASS" and not completed
    return [_step_line(done, current, "eval", detail)]


def _final_eval_step_lines(snap: _Snapshot, stage: str, running: bool) -> list[str]:
    if not _has_final_eval_artifacts(
        snap.session, final_eval_md=snap.final_eval_md, status=snap.status
    ):
        return []
    completed = snap.state in ("COMPLETED", "DONE")
    passed = snap.status.get("final_eval_passed")
    summary = str(snap.status.get("final_eval_summary", "") or "").strip()
    awaiting = snap.state == "AWAITING_VALIDATION"
    if passed:
        detail = "approved"
    elif awaiting:
        detail = "awaiting review"
    elif snap.final_eval_md or summary:
        detail = "failed"
    else:
        detail = ""
    done = bool(passed) or (completed and bool(snap.final_eval_md))
    current = awaiting or (running and stage == "final_eval")
    return [_step_line(done, current, "final_eval", detail)]


def _step_line(done: bool, current: bool, name: str, detail: str = "") -> str:
    if current:
        icon, color = "▶", "yellow"
    elif done:
        icon, color = "✓", "green"
    else:
        icon, color = "○", "grey50"
    line = f"  [{color}]{icon}[/] [{color}]{name:<9}[/]"
    if detail:
        line += f"[dim] ·[/] {detail}"
    return line


def _localize_detail(localization: str) -> str:
    files = {m.group(1).strip() for m in _FILE_RE.finditer(localization)}
    return f"{len(files)} files" if files else ""


def _plan_detail(snap: _Snapshot) -> str:
    if not snap.plan.strip():
        return ""
    steps = len(re.findall(r"^\s*\d+\.", snap.plan, re.MULTILINE))
    plans = _plan_files(snap.session)
    if len(plans) > 1:
        return f"{len(plans)} plans · {steps} steps"
    return f"{steps} steps"


# --- TASKS ------------------------------------------------------------------


def _running_row(rows: list[_TaskRow]) -> _TaskRow | None:
    for row in rows:
        if row.status == "running":
            return row
    return None


def _task_label(rows: list[_TaskRow]) -> str:
    """``Task 3/8 · <title>`` for the task the run is on."""
    running = [row for row in rows if row.status == "running"]
    if not running:
        return ""
    row = running[0]
    total = f"/{len(rows)}" if len(rows) > 1 else ""
    label = f"[b]Task {row.number}{total}[/]"
    title = _truncate(row.title, 44)
    if title:
        label += f" · {title}"
    if len(running) > 1:
        label += f"  [dim](+{len(running) - 1} more running)[/]"
    return label


def _tasks_lines(rows: list[_TaskRow]) -> list[str]:
    if len(rows) <= 1:
        return []
    lines = ["[bold]TASKS[/]"]
    lines.extend(_task_row_line(row) for row in rows)
    return lines


def _task_row_line(row: _TaskRow) -> str:
    icon, color = _TASK_ICON.get(row.status, ("·", "white"))
    title = _truncate(row.title, _COL_TITLE)
    verdict = row.verdict
    verdict_color = _VERDICT_COLOR.get(verdict, "white")
    return (
        f"{row.number:>{_COL_NUM}}  [{color}]{icon}[/]  "
        f"{title:<{_COL_TITLE}}  "
        f"{row.tier:<{_COL_TIER}}{row.iters:>{_COL_ITERS}}  "
        f"[{verdict_color}]{verdict:<{_COL_VERDICT}}[/]"
    )


def _truncate(text: str, width: int) -> str:
    clean = " ".join(text.replace("[", "(").replace("]", ")").split())
    if len(clean) <= width:
        return clean
    return clean[: width - 1] + "…"


# --- PRD trajectory ---------------------------------------------------------


def _prd_line(snap: _Snapshot) -> str:
    """The PRD trajectory as one dim line — truncated so it never wraps."""
    phases = _prd_phases(snap.phases_md)
    names = [escape(name) for name, _ in phases if name]
    if not names:
        return ""
    shown: list[str] = []
    for name in names:
        if shown and _prd_width(shown + [name]) > _PRD_LINE_BUDGET:
            shown.append("…")
            break
        shown.append(name)
    return "[dim]prd[/]  " + " [dim]→[/] ".join(shown)


def _prd_width(names: list[str]) -> int:
    """Visible width of the PRD line for ``names`` (label + separators)."""
    return 5 + sum(len(name) for name in names) + 3 * (len(names) - 1)
