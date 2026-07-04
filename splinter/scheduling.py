"""DAG scheduler and concurrency helpers for parallel task execution."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from enum import Enum

from splinter.agents.runner import Task


class TaskState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"


class DagScheduler:
    """Tracks task states and exposes the ready set for parallel dispatch.

    ready() returns tasks whose deps all PASSED and are still PENDING.
    mark_failed() transitively blocks dependents; independent tasks keep running.
    """

    def __init__(self, tasks: list[Task]) -> None:
        self._tasks: dict[str, Task] = {t.id: t for t in tasks if t.id}
        self._states: dict[str, TaskState] = {
            tid: TaskState.PENDING for tid in self._tasks
        }
        self._blocked_reasons: dict[str, str] = {}

    def ready(self) -> list[Task]:
        result: list[Task] = []
        for tid, state in self._states.items():
            if state != TaskState.PENDING:
                continue
            task = self._tasks[tid]
            if self._deps_passed(task):
                result.append(task)
        return result

    def _deps_passed(self, task: Task) -> bool:
        return all(
            self._states.get(dep) == TaskState.PASSED
            for dep in (task.deps or [])
            if dep in self._tasks
        )

    def mark_running(self, task_id: str) -> None:
        if task_id in self._states:
            self._states[task_id] = TaskState.RUNNING

    def mark_passed(self, task_id: str) -> None:
        if task_id in self._states:
            self._states[task_id] = TaskState.PASSED

    def mark_failed(self, task_id: str) -> None:
        if task_id not in self._states:
            return
        self._states[task_id] = TaskState.FAILED
        self._block_dependents(task_id, f"dep {task_id!r} failed")

    def _block_dependents(self, failed_id: str, reason: str) -> None:
        for tid, task in self._tasks.items():
            if self._states[tid] == TaskState.PENDING and failed_id in (task.deps or []):
                self._states[tid] = TaskState.BLOCKED
                self._blocked_reasons[tid] = reason
                self._block_dependents(tid, f"dep {tid!r} blocked ({reason})")

    def state(self, task_id: str) -> TaskState:
        return self._states.get(task_id, TaskState.PENDING)

    def blocked_reason(self, task_id: str) -> str:
        return self._blocked_reasons.get(task_id, "")

    def is_done(self) -> bool:
        return all(
            s in (TaskState.PASSED, TaskState.FAILED, TaskState.BLOCKED)
            for s in self._states.values()
        )

    def has_running(self) -> bool:
        return TaskState.RUNNING in self._states.values()

    def all_states(self) -> dict[str, TaskState]:
        return dict(self._states)


def topo_order(tasks: list[Task]) -> list[Task] | None:
    """Topological sort; returns None on cycle (Kahn's algorithm)."""
    from collections import deque

    id_to_task: dict[str, Task] = {t.id: t for t in tasks if t.id}
    task_ids = set(id_to_task)
    prd_order = [t.id for t in tasks if t.id]

    in_degree: dict[str, int] = {tid: 0 for tid in task_ids}
    adj: dict[str, list[str]] = {tid: [] for tid in task_ids}

    for task in tasks:
        if not task.id:
            continue
        for dep in task.deps or []:
            if dep in task_ids:
                adj[dep].append(task.id)
                in_degree[task.id] += 1

    queue: deque[str] = deque(
        sorted((tid for tid in task_ids if in_degree[tid] == 0), key=prd_order.index)
    )
    result: list[Task] = []
    while queue:
        tid = queue.popleft()
        result.append(id_to_task[tid])
        for nxt in sorted(adj[tid], key=lambda x: prd_order.index(x)):
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)

    if len(result) != len(task_ids):
        return None

    no_id = [t for t in tasks if not t.id]
    return result + no_id


def default_max_concurrency() -> int:
    """CPU-based default: max(1, cpu_count - 1)."""
    return max(1, (os.cpu_count() or 2) - 1)


@dataclass
class BudgetPool:
    """Thread-safe global budget pool shared across all parallel tasks."""

    _budget: float | None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _extra_cost: float = 0.0

    def add_cost(self, cost: float) -> None:
        with self._lock:
            self._extra_cost += cost

    def exhausted(self, base_cost: float = 0.0) -> bool:
        if self._budget is None:
            return False
        with self._lock:
            return (base_cost + self._extra_cost) >= self._budget

    @property
    def total_extra(self) -> float:
        with self._lock:
            return self._extra_cost
