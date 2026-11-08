"""Refines PRD tasks into parallel workhorse briefs for the agents strategy."""

from __future__ import annotations

import json
import logging
import re
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from threading import Lock
from typing import Any

from splinter.agents.runner import Task
from splinter.agents.story_order import with_story_deps
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


def sanitize_brief_id(value: str) -> str:
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
        id=sanitize_brief_id(str(item.get("id") or "")),
        goal=str(item.get("goal") or ""),
        files=_as_lines(item.get("files")),
        sketch=str(item.get("sketch") or ""),
        acceptance=_as_lines(item.get("acceptance")),
        deps=[sanitize_brief_id(d) for d in _as_lines(item.get("deps"))],
        effort=_valid_effort(item.get("effort")),
        task_id=str(item.get("task") or ""),
    )


def _fallback_briefs(tasks: list[Task]) -> list[Brief]:
    """One Brief per task, preserving input order and dependencies."""
    briefs: list[Brief] = []
    for n, task in enumerate(tasks):
        briefs.append(
            Brief(
                id=sanitize_brief_id(task.id or f"t{n}"),
                goal=task.description,
                files=list(task.target_files or []),
                acceptance=[ln.strip() for ln in task.acceptance.splitlines() if ln.strip()],
                task_id=task.id,
                deps=[sanitize_brief_id(dep) for dep in (task.deps or [])],
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
        context = _clip_context(task.filtered_context)
        if context:
            lines.append("  code_context: |")
            lines.extend(f"    {ln}" for ln in context.splitlines())
    return "\n".join(lines)


#: Code notes kept per task in the refine prompt (the filter already read the files).
_CONTEXT_MAX = 2500


def _clip_context(text: str) -> str:
    body = text.strip()
    return body if len(body) <= _CONTEXT_MAX else body[: _CONTEXT_MAX - 1].rstrip() + "…"


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
        ladder.planner_model,
        variant=ladder.planner_effort,
        output_format="json",
        timeout=ladder.prd_timeout,
        trace=trace,
        task_index=task_index,
        role="refine",
        pool=ladder.step_pool("planner"),
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
    reuse_progress: bool = False,
) -> list[Brief]:
    """Refine PRD tasks into workhorse briefs through the refine template.

    One model call per story, several at once; a story whose call fails gets
    one brief of its own. Each finished story is written to
    :data:`PROGRESS_FILE` so the run screen shows the plan as it forms; the
    caller persists the final brief table.
    """
    if not tasks:
        return []
    tasks = with_story_deps(
        tasks,
        prd_text,
        lambda prompt: run_text(
            prompt,
            ladder.planner_model,
            variant=ladder.planner_effort,
            output_format="json",
            timeout=ladder.prd_timeout,
            trace=trace,
            task_index=task_index,
            role="refine",
            pool=ladder.step_pool("planner"),
        ),
    )
    prd_head = _clip_prd(prd_text)
    log.info(
        "refine · %d stories into briefs · %s, one call per story",
        len(tasks),
        ladder.planner_model,
    )
    done: dict[int, list[Brief]] = (
        _saved_progress(session, tasks) if reuse_progress and session is not None else {}
    )
    if done:
        log.info("refine · resume · %d/%d stories already split", len(done), len(tasks))
    progress_lock = Lock()

    def report(n: int, briefs: list[Brief]) -> None:
        with progress_lock:
            done[n] = briefs
            if session is None:
                return
            payload = {
                "total": len(tasks),
                "done": [
                    {"task": tasks[i].id, "briefs": [asdict(b) for b in done[i]]}
                    for i in sorted(done)
                ],
            }
            session.write(PROGRESS_FILE, json.dumps(payload))

    def one(n: int, task: Task) -> list[Brief]:
        if n in done:
            return done[n]
        try:
            briefs = _inherit_effort(
                _refine_via_llm([task], ladder, prd_head, trace, task_index, orchestrator_notes),
                [task],
            )
            validate_briefs(briefs)
        except Exception as exc:
            log.warning("refine · %s failed (%s) — one brief for the story", task.id, exc)
            briefs = _fallback_briefs([task])
            for brief in briefs:
                brief.deps = []
        log.info("refine · %s → %d brief(s)", task.id or f"t{n}", len(briefs))
        report(n, briefs)
        return briefs

    workers = max(1, min(len(tasks), _REFINE_WORKERS))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        per_task = list(pool.map(one, range(len(tasks)), tasks))
    briefs = _join_story_briefs(tasks, per_task)
    try:
        validate_briefs(briefs)
    except ValueError as exc:
        log.warning("refine · joined briefs invalid (%s) — 1:1 fallback", exc)
        briefs = _fallback_briefs(tasks)
        validate_briefs(briefs)
    return briefs


#: Briefs of the stories refined so far, for the run screen while refine runs.
PROGRESS_FILE = "agents-run/briefs-partial.json"
#: Stories refined at once; each is its own model call.
_REFINE_WORKERS = 6
#: PRD text sent with each story: its opening (goals, scope, rules), not all of it.
_PRD_HEAD = 6000


def _saved_progress(session: Session, tasks: list[Task]) -> dict[int, list[Brief]]:
    """Stories a killed refine already split, by task index, from :data:`PROGRESS_FILE`."""
    try:
        data = json.loads(session.read(PROGRESS_FILE) or "{}")
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict) or data.get("total") != len(tasks):
        return {}
    index = {task.id: n for n, task in enumerate(tasks) if task.id}
    found: dict[int, list[Brief]] = {}
    for story in data.get("done") or []:
        if not isinstance(story, dict) or story.get("task") not in index:
            continue
        try:
            briefs = [Brief(**item) for item in story.get("briefs") or []]
        except TypeError:
            continue
        if briefs:
            found[index[story["task"]]] = briefs
    return found


def _clip_prd(prd_text: str) -> str:
    text = prd_text.strip()
    return text if len(text) <= _PRD_HEAD else text[:_PRD_HEAD].rstrip() + "\n…"


def _join_story_briefs(tasks: list[Task], per_task: list[list[Brief]]) -> list[Brief]:
    """One brief table from per-story results: unique ids, story deps carried over.

    Ids that clash with another story's get the story id as prefix. A story's
    first briefs (no deps inside the story) wait for every brief of the
    stories it depends on.
    """
    seen: dict[str, int] = {}
    for briefs in per_task:
        for brief in briefs:
            seen[brief.id] = seen.get(brief.id, 0) + 1
    by_task: dict[str, list[str]] = {}
    joined: list[Brief] = []
    for n, (task, briefs) in enumerate(zip(tasks, per_task)):
        prefix = sanitize_brief_id(task.id or f"t{n}")
        rename = {
            b.id: (f"{prefix}-{b.id}" if seen[b.id] > 1 and not b.id.startswith(prefix) else b.id)
            for b in briefs
        }
        story_ids = [rename[b.id] for b in briefs]
        by_task[task.id or f"t{n}"] = story_ids
        by_task[prefix] = story_ids
        for brief in briefs:
            deps = [rename[d] for d in brief.deps if d in rename]
            joined.append(
                replace(brief, id=rename[brief.id], deps=deps, task_id=brief.task_id or task.id)
            )
    for n, task in enumerate(tasks):
        upstream = [
            bid
            for dep in (task.deps or [])
            for bid in by_task.get(dep, by_task.get(sanitize_brief_id(dep), []))
        ]
        if not upstream:
            continue
        mine = set(by_task.get(task.id or f"t{n}", []))
        for i, brief in enumerate(joined):
            if brief.id in mine and not brief.deps:
                joined[i] = replace(brief, deps=list(dict.fromkeys(upstream)))
    return joined
