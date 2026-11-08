"""Refines PRD tasks into parallel workhorse briefs for the agents strategy."""

from __future__ import annotations

import json
import logging
import re
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any

from splinter.agents.runner import Task
from splinter.enums import Effort
from splinter.memory.session import Session
from splinter.models.roster import Ladder
from splinter.obs.trace import Trace
from splinter.providers.dispatch import run_text
from splinter.templating import render, section

log = logging.getLogger("splinter.refiner")


@dataclass
class Brief:
    """One workhorse-sized unit of work derived from a PRD task."""

    id: str
    goal: str = ""
    files: list[str] = field(default_factory=list)
    sketch: str = ""
    acceptance: list[str] = field(default_factory=list)
    deps: list[str] = field(default_factory=list)
    effort: str = ""
    task_id: str = ""


def _sanitize_brief_id(value: str) -> str:
    """Coerce an id into one safe path component; unusable input degrades to "b"."""
    cleaned = re.sub(r"[^a-z0-9-]+", "-", value.lower())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned or "b"


def validate_briefs(briefs: list[Brief]) -> None:
    """Validate unique non-empty ids, resolvable deps, and acyclicity.

    Raises ValueError on the first violation.
    """
    ids: set[str] = set()
    for brief in briefs:
        if not brief.id:
            raise ValueError("brief with empty id")
        if brief.id in ids:
            raise ValueError(f"duplicate brief id {brief.id!r}")
        ids.add(brief.id)

    for brief in briefs:
        for dep in brief.deps:
            if dep not in ids:
                raise ValueError(f"brief {brief.id!r} references unknown dep {dep!r}")

    in_degree: dict[str, int] = {b.id: 0 for b in briefs}
    adj: dict[str, list[str]] = {b.id: [] for b in briefs}
    for brief in briefs:
        for dep in brief.deps:
            adj[dep].append(brief.id)
            in_degree[brief.id] += 1

    queue: deque[str] = deque(bid for bid in in_degree if in_degree[bid] == 0)
    visited = 0
    while queue:
        bid = queue.popleft()
        visited += 1
        for nxt in adj[bid]:
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)

    if visited != len(briefs):
        raise ValueError("dependency cycle detected among briefs")


def brief_waves(briefs: list[Brief]) -> list[list[Brief]]:
    """Group briefs into dependency waves (Kahn levels, stable input order).

    Briefs referencing unknown dep ids land in the last wave; a cycle collapses
    everything into a single wave — both are defensive paths that
    :func:`validate_briefs` normally prevents.
    """
    if not briefs:
        return []
    ids = {b.id for b in briefs}
    leveled = [b for b in briefs if all(d in ids for d in b.deps)]
    deferred = [b for b in briefs if any(d not in ids for d in b.deps)]
    remaining = {b.id: {d for d in b.deps if d in ids} for b in leveled}
    waves: list[list[Brief]] = []
    pending = set(remaining)
    while pending:
        wave = [b for b in leveled if b.id in pending and not remaining[b.id]]
        if not wave:
            return [list(briefs)]
        waves.append(wave)
        done = {b.id for b in wave}
        pending -= done
        for bid in pending:
            remaining[bid] -= done
    if deferred:
        waves.append(deferred)
    return waves


def brief_to_task(brief: Brief) -> Task:
    """Adapt a Brief into the Task shape consumed by the DAG scheduler."""
    return Task(
        description=brief.goal,
        acceptance="\n".join(brief.acceptance),
        effort=brief.effort or Effort.NORMAL,
        reasoning_effort="auto",
        target_files=list(brief.files),
        id=brief.id,
        deps=brief.deps or None,
    )


def _as_lines(value: object) -> list[str]:
    """Coerce an LLM-supplied field into a list of non-empty lines."""
    if isinstance(value, str):
        return [ln.strip() for ln in value.splitlines() if ln.strip()]
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


def _strip_fences(text: str) -> str:
    """Drop markdown code fences an LLM may wrap around the JSON payload."""
    stripped = text.strip()
    if stripped.startswith("```"):
        nl = stripped.find("\n")
        if nl != -1:
            stripped = stripped[nl + 1 :]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    return stripped.strip()


def _valid_effort(value: object) -> str:
    """Normalise an effort bucket; anything outside :class:`Effort` becomes ""."""
    text = str(value or "").strip().lower()
    return text if text in {e.value for e in Effort} else ""


def _inherit_effort(briefs: list[Brief], tasks: list[Task]) -> list[Brief]:
    """Briefs without an effort take their originating task's effort (else ``normal``)."""
    by_id = {t.id: _valid_effort(t.effort) for t in tasks if t.id}
    return [
        b if b.effort else replace(b, effort=by_id.get(b.task_id) or Effort.NORMAL)
        for b in briefs
    ]


def _brief_from_item(item: dict[str, Any]) -> Brief:
    return Brief(
        id=_sanitize_brief_id(str(item.get("id") or "")),
        goal=str(item.get("goal") or ""),
        files=_as_lines(item.get("files")),
        sketch=str(item.get("sketch") or ""),
        acceptance=_as_lines(item.get("acceptance")),
        deps=[_sanitize_brief_id(d) for d in _as_lines(item.get("deps"))],
        effort=_valid_effort(item.get("effort")),
        task_id=str(item.get("task") or ""),
    )


def _fallback_briefs(tasks: list[Task]) -> list[Brief]:
    """One Brief per task, preserving input order and dependencies."""
    briefs: list[Brief] = []
    for n, task in enumerate(tasks):
        briefs.append(
            Brief(
                id=_sanitize_brief_id(task.id or f"t{n}"),
                goal=task.description,
                files=list(task.target_files or []),
                acceptance=[ln.strip() for ln in task.acceptance.splitlines() if ln.strip()],
                task_id=task.id,
                deps=[_sanitize_brief_id(dep) for dep in (task.deps or [])],
                effort=_valid_effort(task.effort),
            )
        )
    return briefs


def _tasks_block(tasks: list[Task]) -> str:
    lines: list[str] = []
    for n, task in enumerate(tasks, start=1):
        tid = task.id or f"t{n}"
        lines.append(f"- {tid} (effort: {task.effort})")
        lines.append(f"  description: {task.description}")
        for ln in task.acceptance.splitlines():
            if ln.strip():
                lines.append(f"  acceptance: {ln.strip()}")
        if task.target_files:
            lines.append(f"  target_files: {', '.join(task.target_files)}")
    return "\n".join(lines)


def _refine_via_llm(
    tasks: list[Task],
    ladder: Ladder,
    prd_text: str,
    trace: Trace | None,
    task_index: int,
    orchestrator_notes: str,
) -> list[Brief]:
    prompt = render(
        "refine",
        tasks=section("Tasks", _tasks_block(tasks)),
        prd=section("PRD", prd_text),
        orchestrator_notes_section=orchestrator_notes,
    )
    raw = run_text(
        prompt,
        ladder.prd_model,
        variant=ladder.prd_effort,
        output_format="json",
        timeout=ladder.prd_timeout,
        trace=trace,
        task_index=task_index,
        role="prd",
        pool=ladder.step_pool("prd"),
    )
    if not raw or not raw.strip():
        raise ValueError("empty refinement output")
    data = json.loads(_strip_fences(raw))
    if isinstance(data, dict):
        items = data.get("briefs")
    elif isinstance(data, list):
        items = data
    else:
        items = None
    if not isinstance(items, list):
        raise ValueError("refinement output has no briefs list")
    briefs = [_brief_from_item(item) for item in items if isinstance(item, dict)]
    if not briefs:
        raise ValueError("refinement output parsed to zero briefs")
    return briefs


def refine_briefs(
    tasks: list[Task],
    ladder: Ladder,
    *,
    prd_text: str = "",
    session: Session | None = None,
    trace: Trace | None = None,
    task_index: int = 0,
    orchestrator_notes: str = "",
) -> list[Brief]:
    """Refine PRD tasks into workhorse briefs through the refine template.

    Any LLM, parse, or validation failure degrades to one Brief per task
    (1:1, input order, empty deps). Pure: nothing is written to the session —
    ``session`` is accepted for orchestrator symmetry only; the caller
    persists the brief table.
    """
    if tasks:
        try:
            briefs = _inherit_effort(
                _refine_via_llm(tasks, ladder, prd_text, trace, task_index, orchestrator_notes),
                tasks,
            )
            validate_briefs(briefs)
            return briefs
        except Exception as exc:
            log.warning("refine_briefs: LLM refinement failed (%s) — 1:1 fallback", exc)
    briefs = _fallback_briefs(tasks)
    validate_briefs(briefs)
    return briefs
