"""Read a task plan folder: ``tasks/<plan>/README.md`` plus its ``T01-….md`` files.

The README is the orchestrator's entry point: decisions, run order, gates and
handoff notes, and a task table whose rows link each task file. Every linked row
becomes one :class:`Task`; its id is the link text, its deps come from the
"Depends on" column and its effort from the "Model" column when the table has one.
Rows without a link (orchestrator steps such as merges or reviews) are not tasks;
deps on them are dropped. The README itself is kept for the planner and the
refiner as the plan index (:data:`PLAN_INDEX_FILE` in the session).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from splinter.agents.runner import Task
from splinter.enums import Effort
from splinter.memory.session import Session
from splinter.task_file import read_task_mapping

__all__ = [
    "PLAN_INDEX_FILE",
    "PlanIndex",
    "is_plan_index",
    "load_plan_index",
    "seed_task_plans",
    "task_as_plan",
]

PLAN_INDEX_FILE = "knowledge/plan-index.md"

_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+\.md)\)")
_TASK_ID = re.compile(r"\b([A-Z]{1,3}\d+[a-z]?)\b")
_RANGE = re.compile(r"\b([A-Z]{1,3})(\d+)\s*[–—-]\s*(?:\1)?(\d+)\b")
_ALL_DEPS = frozenset({"all", "todas", "todos", "tudo"})
_DEPS_HEADERS = ("depends", "depende", "deps")
_MODEL_HEADERS = ("model", "modelo", "effort")


@dataclass(frozen=True)
class PlanIndex:
    """A plan folder read into tasks, plus the README text for the orchestrator."""

    path: Path
    text: str
    tasks: tuple[Task, ...]


def is_plan_index(path: Path) -> bool:
    """True for a plan folder or a README whose table links task files."""
    readme = _readme_of(path)
    if readme is None:
        return False
    try:
        return bool(_task_rows(readme.read_text()))
    except OSError:
        return False


def load_plan_index(path: Path) -> PlanIndex:
    """Read the README (or the folder holding it) into ordered tasks.

    Raises ``ValueError`` when no row links a task file or a linked file is
    missing, and ``OSError`` when the README cannot be read.
    """
    readme = _readme_of(path)
    if readme is None:
        raise ValueError(f"no README.md plan index at {path}")
    text = readme.read_text()
    rows = _task_rows(text)
    if not rows:
        raise ValueError(f"{readme} has no task table linking T-files")
    ids = [row.task_id for row in rows]
    tasks = tuple(_task_from_row(readme.parent, row, ids) for row in rows)
    return PlanIndex(path=readme, text=text, tasks=tasks)


def task_as_plan(task: Task, index: Path | None = None) -> str:
    """A task file is already a plan: its text, plus where the plan index lives."""
    parts = [task.description.strip()]
    if index is not None:
        parts.append(f"Plan index (decisions, file ownership, handoffs): `{index}`")
    return "\n\n".join(parts)


def seed_task_plans(session: Session, tasks: Sequence[Task], index: Path | None = None) -> int:
    """Write each task's own text as ``knowledge/plan-N.md`` so the run skips the planner.

    A plan already on disk (resume, or one written by id in plan mode) is kept.
    Returns how many plans were written.
    """
    written = 0
    for i, task in enumerate(tasks):
        name = f"knowledge/plan-{i + 1}.md"
        by_id = f"knowledge/plan-{task.id}.md" if task.id else ""
        if session.read(name).strip() or (by_id and session.read(by_id).strip()):
            continue
        session.write(name, f"# Plan\n\n{task_as_plan(task, index)}\n")
        written += 1
    return written


@dataclass(frozen=True)
class _Row:
    task_id: str
    file: str
    deps: str
    model: str


def _readme_of(path: Path) -> Path | None:
    if path.is_dir():
        candidate = path / "README.md"
        return candidate if candidate.is_file() else None
    if path.name.lower() == "readme.md" and path.is_file():
        return path
    return None


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _column(header: list[str], names: tuple[str, ...]) -> int | None:
    for index, cell in enumerate(header):
        if cell.lower().startswith(names):
            return index
    return None


def _task_rows(text: str) -> list[_Row]:
    rows: list[_Row] = []
    header: list[str] = []
    for line in text.splitlines():
        if not line.lstrip().startswith("|"):
            header = []
            continue
        cells = _cells(line)
        if not header:
            header = cells
            continue
        link = _LINK.search(cells[0]) if cells else None
        if link is None:
            continue
        deps_at = _column(header, _DEPS_HEADERS)
        model_at = _column(header, _MODEL_HEADERS)
        rows.append(
            _Row(
                task_id=link.group(1).strip(),
                file=link.group(2),
                deps=_cell_at(cells, deps_at),
                model=_cell_at(cells, model_at),
            )
        )
    return rows


def _cell_at(cells: list[str], index: int | None) -> str:
    if index is None or index >= len(cells):
        return ""
    return cells[index]


def _task_from_row(folder: Path, row: _Row, ids: list[str]) -> Task:
    file = folder / row.file
    if not file.is_file():
        raise ValueError(f"task {row.task_id} links a missing file: {file}")
    data = read_task_mapping(file)
    return Task(
        description=f"{row.task_id}: {data['description']}",
        acceptance=str(data.get("acceptance", "")),
        effort=_effort(row.model),
        id=row.task_id,
        deps=_deps(row.deps, row.task_id, ids) or None,
    )


def _deps(cell: str, task_id: str, ids: list[str]) -> list[str]:
    earlier = ids[: ids.index(task_id)]
    if cell.strip().lower() in _ALL_DEPS:
        return earlier
    wanted: list[str] = []
    for prefix, start, end in _RANGE.findall(cell):
        width = len(start)
        wanted.extend(f"{prefix}{n:0{width}d}" for n in range(int(start), int(end) + 1))
    wanted.extend(_TASK_ID.findall(_RANGE.sub("", cell)))
    known = [dep for dep in dict.fromkeys(wanted) if dep in ids and dep != task_id]
    return known


def _effort(model: str) -> str:
    text = model.lower()
    if "xhigh" in text or "max" in text:
        return Effort.CRITICAL
    if "high" in text or "opus" in text:
        return Effort.HARD
    if "low" in text or "haiku" in text or "flash" in text:
        return Effort.TRIVIAL
    return Effort.NORMAL
