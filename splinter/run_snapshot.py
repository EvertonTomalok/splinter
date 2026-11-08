"""Shared read-side snapshot and parsing helpers for run views."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from splinter.analyze import _task_iters
from splinter.memory.session import Session
from splinter.models.roster import Ladder
from splinter.obs.trace import RunEntry, Trace

_FILE_RE = re.compile(r"^file:\s*(.+)$", re.MULTILINE)
_BLOCKED_HEAD_RE = re.compile(r"^##\s+blocked tasks", re.IGNORECASE)
_BLOCKED_ITEM_RE = re.compile(r"^-\s+(\S+?):")
_VERDICT_LINE_RE = re.compile(r"^-\s*verdict:\s*(\w+)\s*(.*)$", re.MULTILINE)
_REVIEW_FILE_RE = re.compile(r"^task(\d+)-review-iter-(\d+)\.md$")


@dataclass(frozen=True)
class Snapshot:
    """Everything the overview reads, captured once per render."""

    session: Session
    state: str
    status: dict[str, Any]
    loop_md: str
    localization: str
    plan: str
    phases_md: str
    final_eval_md: str
    entries: list[RunEntry]
    pre_run_models: dict[str, Any]


@dataclass(frozen=True)
class Verdict:
    name: str = ""
    detail: str = ""


@dataclass(frozen=True)
class TaskRow:
    number: int
    title: str
    status: str = "pending"
    tier: str = ""
    iters: int = 0
    verdict: str = ""


def snapshot(session: Session, state: str) -> Snapshot:
    trace = Trace.from_jsonl(session)
    return Snapshot(
        session=session,
        state=state,
        status=session.read_status(),
        loop_md=session.read("loop.md"),
        localization=session.read("knowledge/localization.md"),
        plan=session.read("knowledge/plan.md"),
        phases_md=session.read("prd_phases.md"),
        final_eval_md=session.read("final_eval.md"),
        entries=list(trace.entries),
        pre_run_models=pre_run_models(session),
    )


def pre_run_models(session: Session) -> dict[str, Any]:
    models = session.read_pre_run_usage().get("models", {})
    return models if isinstance(models, dict) else {}


def cost_split(snap: Snapshot) -> tuple[float, float, float, int]:
    """(run, pre-run without jev, jev, jev calls) spend of the session."""
    run_cost = sum(entry.cost for entry in snap.entries)
    pre_run = jev_cost = 0.0
    jev_calls = 0
    for name, entry in snap.pre_run_models.items():
        if not isinstance(entry, dict):
            continue
        cost = float(entry.get("cost", 0.0) or 0.0)
        if str(name).startswith("jev:"):
            jev_cost += cost
            jev_calls += int(entry.get("calls", 0) or 0)
        else:
            pre_run += cost
    return run_cost, pre_run, jev_cost, jev_calls


def fmt_span(seconds: float) -> str:
    """Human span: ``45s`` · ``2m14s`` · ``1h05m`` · ``3d04h``."""
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


def trace_span(entries: list[RunEntry]) -> float:
    stamps = [entry.ts for entry in entries if entry.ts]
    if len(stamps) < 2:
        return 0.0
    try:
        parsed = sorted(datetime.fromisoformat(stamp) for stamp in stamps)
    except ValueError:
        return 0.0
    return (parsed[-1] - parsed[0]).total_seconds()


def tier_effort(level: int, model: str, ladder: Ladder | None = None) -> str:
    """Reasoning effort configured for the tier that ran ``model``."""
    if ladder is None:
        try:
            from splinter.models.roster import load_ladder

            ladder = load_ladder()
        except Exception:
            return ""
    try:
        return ladder.model_variant(level, model) or ladder.tier_variant(level) or ""
    except Exception:
        return ""


def stage_label(stage: str) -> str:
    label = (stage or "").strip().lower()
    if not label:
        return ""
    if label in ("filter", "localize"):
        return "localize"
    if label in ("gate", "run"):
        return "run"
    return label


def last_verdict(snap: Snapshot) -> Verdict:
    note = latest_review_note(snap.session)
    if note is not None:
        return note
    verdicts = loop_verdicts(snap.loop_md)
    if not verdicts:
        return Verdict()
    name, detail = verdicts[-1]
    return Verdict(name, detail)


def loop_verdicts(loop_md: str) -> list[tuple[str, str]]:
    return [
        (match.group(1), match.group(2).strip()) for match in _VERDICT_LINE_RE.finditer(loop_md)
    ]


def latest_review_note(session: Session) -> Verdict | None:
    """Newest per-iteration review note — carries the verdict and its reason."""
    kdir = session.dir / "knowledge"
    if not kdir.exists():
        return None
    best: tuple[int, int] = (-1, -1)
    chosen: Verdict | None = None
    for path in kdir.glob("task*-review-iter-*.md"):
        match = _REVIEW_FILE_RE.match(path.name)
        if not match:
            continue
        key = (int(match.group(2)), int(match.group(1)))
        if key <= best:
            continue
        verdict = verdict_from_note(path.read_text())
        if verdict is not None:
            best = key
            chosen = verdict
    return chosen


def verdict_from_note(text: str) -> Verdict | None:
    name = ""
    detail = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- decision:"):
            name = stripped.split(":", 1)[1].strip().upper()
        elif stripped.startswith("- reason:") and not detail:
            detail = stripped.split(":", 1)[1].strip()
    if not name:
        return None
    return Verdict(name, detail)


def task_rows(
    snap: Snapshot,
    parsed_groups: list[tuple[int, str, list[tuple[int, str, str]]]] | None = None,
) -> list[TaskRow]:
    groups = (
        parsed_groups
        if parsed_groups is not None
        else _task_iters(snap.loop_md, session=snap.session)
    )
    blocked = blocked_task_ids(snap.loop_md)
    current = current_task_no(snap, groups)
    rows: list[TaskRow] = []
    for number, title, iters in groups:
        raw_verdict = iters[-1][2] if iters else ""
        raw_tier = iters[-1][1] if iters else ""
        verdict = "" if raw_verdict == "?" else raw_verdict
        tier = "" if raw_tier.endswith("?") else raw_tier
        rows.append(
            TaskRow(
                number=number,
                title=title,
                status=task_status(number, title, verdict, iters, blocked, current, snap),
                tier=tier,
                iters=len(iters),
                verdict=verdict,
            )
        )
    rows.sort(key=lambda row: row.number)
    return rows


def task_status(
    number: int,
    title: str,
    verdict: str,
    iters: list[tuple[int, str, str]],
    blocked: set[str],
    current: int | None,
    snap: Snapshot,
) -> str:
    if verdict == "PASS":
        return "pass"
    if blocked and title.split(":", 1)[0].strip() in blocked:
        return "blocked"
    if not iters:
        return "pending"
    if not verdict:
        return "running"
    if verdict in ("RETRY", "ESCALATE", "JUMP_PREMIUM"):
        active = current == number and snap.state in (
            "RUNNING",
            "AWAITING_USER",
            "AWAITING_VALIDATION",
        )
        return "running" if active else "failed"
    return "failed"


def current_task_no(
    snap: Snapshot, groups: list[tuple[int, str, list[tuple[int, str, str]]]]
) -> int | None:
    """The task the run is on right now (in-flight first, else the status index)."""
    for number, _title, iters in groups:
        if iters and iters[-1][2] == "?":
            return number
    raw = snap.status.get("task_index")
    try:
        return int(raw) + 1 if isinstance(raw, (int, float, str)) else None
    except (TypeError, ValueError):
        return None


def blocked_task_ids(loop_md: str) -> set[str]:
    ids: set[str] = set()
    in_section = False
    for line in loop_md.splitlines():
        stripped = line.strip()
        if stripped.startswith("##"):
            in_section = bool(_BLOCKED_HEAD_RE.match(stripped))
            continue
        if not in_section:
            continue
        match = _BLOCKED_ITEM_RE.match(stripped)
        if match:
            ids.add(match.group(1))
    return ids
