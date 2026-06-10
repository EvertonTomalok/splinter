"""Donatello — the ``cascade`` multi-task dependency-ordered strategy.

Flow: topological sort on task.deps, then run each task in order with
per-task checkpoint persistence. A crash mid-run resumes at the first
un-checkpointed task. Budget exhaustion stops the cascade cleanly.

Inherits the full per-task Run → Gate → Eval loop from DirectStrategy.
"""

from __future__ import annotations

import json
import logging
from collections import deque

from splinter.agents.runner import RunResult, Task
from splinter.memory.knowledge import KnowledgeStore
from splinter.memory.session import Session
from splinter.models.roster import Ladder
from splinter.obs.trace import Trace
from splinter.strategies.direct import DirectStrategy
from splinter.strategies.registry import register

log = logging.getLogger("splinter.loop")


@register
class CascadeStrategy(DirectStrategy):
    name = "cascade"
    aliases = ["donatello"]

    def execute(
        self,
        tasks: list[Task],
        session: Session,
        ladder: Ladder,
        *,
        effort: str | None = None,
        budget: float | None = None,
        max_iterations: int = 5,
        localization: str = "",
        eval_skill: str | None = None,
        cowabunga: bool = False,
        resume: bool = False,
        claude_runner_fallback: bool = False,
        user_guidance: str | None = None,
        jump_premium: bool = False,
    ) -> list[RunResult]:
        ordered = self._topo_sort(tasks)

        existing_trace = session.read("trace.md")
        if resume and existing_trace.strip():
            trace = Trace.from_markdown(existing_trace)
        else:
            trace = Trace()

        knowledge = KnowledgeStore(session)
        results: list[RunResult] = []
        done = self._load_checkpoint(session) if resume else set()

        if done:
            log.info("cascade resume: %d task(s) already checkpointed", len(done))

        for i, task in enumerate(ordered):
            if task.id and task.id in done:
                log.info("resume: skip %s (checkpointed)", task.id)
                continue

            session.set_status(
                "running",
                stage="run",
                task_index=i,
                task_total=len(ordered),
                task=task.description.splitlines()[0][:80],
            )
            session.append(
                "loop.md",
                f"# Task {i + 1}/{len(ordered)}: "
                f"{task.description.splitlines()[0][:80]}\n\n",
            )

            result = self._run_task_loop(
                task,
                session,
                ladder,
                trace,
                knowledge,
                task_index=i,
                effort=effort,
                budget=budget,
                max_iterations=max_iterations,
                localization=localization,
                eval_skill=eval_skill,
                cowabunga=cowabunga,
                resume=False,
            )

            if result is not None:
                results.append(result)
                if task.id:
                    done.add(task.id)
                    self._save_checkpoint(session, done)

            if budget is not None and trace.total_cost >= budget:
                session.append("loop.md", f"## Budget exhausted (${trace.total_cost:.4f})\n")
                break

        session.set_status("running", task_index=len(ordered), task_total=len(ordered))
        session.write("trace.md", trace.summary())
        return results

    @staticmethod
    def _topo_sort(tasks: list[Task]) -> list[Task]:
        """Kahn's algorithm over task.deps. Cycle → warn + fallback to original order."""
        id_to_task: dict[str, Task] = {t.id: t for t in tasks if t.id}
        task_ids: set[str] = set(id_to_task)

        in_degree: dict[str, int] = {tid: 0 for tid in task_ids}
        adj: dict[str, list[str]] = {tid: [] for tid in task_ids}

        for task in tasks:
            if not task.id:
                continue
            for dep in (task.deps or []):
                if dep in task_ids:
                    adj[dep].append(task.id)
                    in_degree[task.id] += 1

        queue: deque[str] = deque(
            tid for tid in task_ids if in_degree[tid] == 0
        )
        # preserve PRD order among ties
        prd_order = [t.id for t in tasks if t.id]
        queue = deque(sorted(queue, key=lambda tid: prd_order.index(tid)))

        result: list[Task] = []
        while queue:
            tid = queue.popleft()
            result.append(id_to_task[tid])
            for nxt in sorted(adj[tid], key=lambda x: prd_order.index(x)):
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)

        if len(result) != len(task_ids):
            log.warning("cascade: dependency cycle detected — falling back to PRD order")
            return list(tasks)

        # tasks without ids are appended at end in original order
        no_id = [t for t in tasks if not t.id]
        return result + no_id

    @staticmethod
    def _load_checkpoint(session: Session) -> set[str]:
        raw = session.read("checkpoint.json")
        if not raw.strip():
            return set()
        try:
            data: dict[str, list[str]] = json.loads(raw)
            return set(data.get("completed", []))
        except (json.JSONDecodeError, AttributeError):
            return set()

    @staticmethod
    def _save_checkpoint(session: Session, done_ids: set[str]) -> None:
        session.write("checkpoint.json", json.dumps({"completed": sorted(done_ids)}))
