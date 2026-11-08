"""Waves section of the live run overview (agents strategy).

The ``agents`` strategy writes its progress into ``agents-run/`` and into the
session event log:

* ``agents-run/README.md``     — ``- Wave 1: B-01, B-02``
* ``agents-run/STATE.md``      — ``- merged B-01 (mcode) into <branch>``
* ``agents-run/checkpoint.json`` — ``{"completed": ["B-01"]}``
* ``events.md``                — ``[AGENTS] brief B-01 engine=mcode …``

This module turns those files into the compact per-brief rows the run screen
shows. Every file is optional: a missing one simply hides what it carries.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace

from splinter.configure import configured_agents_options
from splinter.memory.session import Session

_AGENTS_EVENT_RE = re.compile(r"^\[AGENTS\] brief (\S+)(.*)$")
_WAVE_RE = re.compile(r"^-\s*Wave\s+(\d+):\s*(.+)$")
_STATE_MERGED_RE = re.compile(r"^-\s*merged\s+(\S+)\s+\(([^)]+)\)")
_STATE_CONFLICT_RE = re.compile(r"^-\s*CONFLICT\s+(\S+)\s+\(([^)]+)\)")


@dataclass(frozen=True)
class BriefRow:
    """One brief of the agents run and what happened to it so far."""

    ident: str
    engine: str = ""
    fix_round: int = 0
    merged: bool = False
    assumed: bool = False
    conflict: bool = False
    rerouted_from: str = ""


def render_waves(session: Session, strategy: str) -> list[str]:
    """Wave rows for the overview — empty for other strategies or missing data."""
    if strategy.lower() not in ("agents", "splinter"):
        return []
    waves = parse_waves(session)
    if not waves:
        return []
    max_fix = max_fix_rounds()
    lines = ["[bold]WAVES[/]"]
    for wave_no, briefs in waves:
        lines.append(f"  [b]Wave {wave_no}[/]")
        lines.extend(f"    {brief_line(brief, max_fix)}" for brief in briefs)
    return lines


def parse_waves(session: Session) -> list[tuple[int, list[BriefRow]]]:
    """``[(wave number, [brief, …]), …]`` — parsed from the agents-run files."""
    wave_ids = _wave_ids(session)
    briefs = _brief_rows(session)
    if not wave_ids and not briefs:
        return []
    waves: list[tuple[int, list[BriefRow]]] = []
    claimed: set[str] = set()
    for wave_no, ids in wave_ids:
        rows = [briefs[i] for i in ids if i in briefs]
        claimed.update(ids)
        waves.append((wave_no, rows))
    leftovers = [b for ident, b in briefs.items() if ident not in claimed]
    if leftovers:
        waves.append((len(waves) + 1, leftovers))
    return waves


def max_fix_rounds() -> int:
    """Fix rounds before the orchestrator takes over: one per burned attempt but the last."""
    return max(configured_agents_options().max_attempts - 1, 0)


def brief_line(brief: BriefRow, max_fix: int) -> str:
    """``B-04  mcode  fix 1/2  ↺ from agy`` — the brief's one-line status."""
    parts = [f"[b]{brief.ident}[/]"]
    if brief.engine:
        parts.append(brief.engine)
    if brief.fix_round:
        parts.append(f"fix {brief.fix_round}/{max_fix}")
    if brief.rerouted_from and brief.rerouted_from != brief.engine:
        parts.append(f"↺ from {brief.rerouted_from}")
    if brief.merged:
        parts.append("[green]✓ merged[/]")
    elif brief.assumed:
        parts.append("[yellow]assumed[/]")
    elif brief.conflict:
        parts.append("[red]✗ conflict[/]")
    return "  ".join(parts)


def _wave_ids(session: Session) -> list[tuple[int, list[str]]]:
    text = session.read("agents-run/README.md")
    waves: list[tuple[int, list[str]]] = []
    for line in text.splitlines():
        m = _WAVE_RE.match(line.strip())
        if not m:
            continue
        ids = [part.strip() for part in m.group(2).split(",") if part.strip()]
        waves.append((int(m.group(1)), ids))
    return waves


def _brief_rows(session: Session) -> dict[str, BriefRow]:
    from splinter.obs.events import load_log_events

    rows: dict[str, BriefRow] = {}
    for event in load_log_events(session):
        raw = str(event.payload.get("raw") or event.payload.get("message") or "")
        m = _AGENTS_EVENT_RE.match(raw.strip())
        if not m:
            continue
        ident = m.group(1)
        row = _apply_agents_event(rows.get(ident) or BriefRow(ident=ident), m.group(2))
        rows[ident] = row
    for ident in _completed_briefs(session):
        row = rows.get(ident) or BriefRow(ident=ident)
        rows[ident] = replace(row, merged=True)
    for ident, engine, kind in _state_briefs(session):
        row = rows.get(ident) or BriefRow(ident=ident)
        rows[ident] = replace(
            row,
            engine=row.engine or engine,
            merged=row.merged or kind == "merged",
            conflict=kind == "conflict",
        )
    return rows


def _apply_agents_event(row: BriefRow, rest: str) -> BriefRow:
    engine_m = re.search(r"engine=(\S+)", rest)
    engine = engine_m.group(1) if engine_m else ""
    if "infra failure" in rest:
        failed = engine or row.engine
        return replace(row, engine=failed, rerouted_from=failed)
    current, origin = _engine_after(row, engine)
    if "fix round" in rest:
        round_m = re.search(r"fix round (\d+)", rest)
        round_no = int(round_m.group(1)) if round_m else 0
        return replace(row, engine=current, rerouted_from=origin, fix_round=round_no)
    if "merged=" in rest:
        merged = "merged=True" in rest
        return replace(
            row, engine=current, rerouted_from=origin,
            merged=merged or row.merged, conflict=not merged,
        )
    if "assumed by orchestrator" in rest:
        return replace(row, engine=current, rerouted_from=origin, assumed=True)
    return row


def _engine_after(row: BriefRow, engine: str) -> tuple[str, str]:
    """(current engine, reroute origin) after an event naming ``engine``."""
    if not engine:
        return row.engine, row.rerouted_from
    if row.rerouted_from == engine:
        return engine, ""
    return engine, row.rerouted_from


def _completed_briefs(session: Session) -> set[str]:
    raw = session.read("agents-run/checkpoint.json")
    if not raw.strip():
        return set()
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return set()
    completed = data.get("completed") if isinstance(data, dict) else None
    if not isinstance(completed, list):
        return set()
    return {str(item) for item in completed}


def _state_briefs(session: Session) -> list[tuple[str, str, str]]:
    """``(brief id, engine, kind)`` for every merge/conflict line in STATE.md."""
    text = session.read("agents-run/STATE.md")
    out: list[tuple[str, str, str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        merged = _STATE_MERGED_RE.match(stripped)
        if merged:
            out.append((merged.group(1), merged.group(2), "merged"))
            continue
        conflict = _STATE_CONFLICT_RE.match(stripped)
        if conflict:
            out.append((conflict.group(1), conflict.group(2), "conflict"))
    return out
