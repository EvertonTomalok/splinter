"""Donatello — the ``adaptive`` per-task cost-routing strategy.

Flow: topological sort (like cascade), then for each task pick the *cheapest*
ladder tier capable of handling its estimated effort. A soft budget is maintained
across the session: when the target is exceeded, escalation is suppressed but tasks
continue running at their current tier until completion.

Budget precedence: CLI ``--budget`` arg → session/config ``defaults.budget``.
"""

from __future__ import annotations

import logging

from splinter.agents.runner import RunResult, Task
from splinter.configure import configured_budget
from splinter.memory.knowledge import KnowledgeStore
from splinter.memory.session import Session
from splinter.models.pricing import estimate_tier_cost
from splinter.models.roster import Ladder
from splinter.obs.trace import Trace
from splinter.strategies.cascade import CascadeStrategy
from splinter.strategies.registry import register

log = logging.getLogger("splinter.loop")


@register
class AdaptiveStrategy(CascadeStrategy):
    name = "adaptive"
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
        effective_budget = budget if budget is not None else configured_budget()
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
            log.info("adaptive resume: %d task(s) already checkpointed", len(done))

        self._run_plan_phase(ordered, session, ladder, localization, trace=trace)

        for i, task in enumerate(ordered):
            if task.id and task.id in done:
                log.info("resume: skip %s (checkpointed)", task.id)
                continue

            task_effort = effort or task.effort
            routed_tier = self._route_tier(task_effort, ladder)
            log.info(
                "adaptive: task %d/%d effort=%s → T%d (%s)",
                i + 1,
                len(ordered),
                task_effort,
                routed_tier,
                ladder.tier_by_level(routed_tier).name,
            )

            session.set_status(
                "running",
                stage="run",
                task_index=i,
                task_total=len(ordered),
                task=task.description.splitlines()[0][:80],
            )
            session.append(
                "loop.md",
                f"# Task {i + 1}/{len(ordered)}: {task.description.splitlines()[0][:80]}\n\n",
            )

            result = self._run_task_loop(
                task,
                session,
                ladder,
                trace,
                knowledge,
                task_index=i,
                effort=effort,
                budget=effective_budget,
                max_iterations=max_iterations,
                localization=localization,
                eval_skill=eval_skill,
                cowabunga=cowabunga,
                resume=False,
                soft_budget=True,
                start_tier_override=routed_tier,
            )

            if result is not None:
                results.append(result)
                if task.id:
                    done.add(task.id)
                    self._save_checkpoint(session, done)

        session.set_status("running", task_index=len(ordered), task_total=len(ordered))
        session.write("trace.md", trace.summary())
        return results

    @staticmethod
    def _route_tier(effort: str, ladder: Ladder) -> int:
        """Pick cheapest capable tier for the given effort level.

        Capable = tier level >= effort_mapping floor. Among candidates, pick
        by lowest estimated cost (see splinter.models.pricing).
        """
        em = ladder.effort_mapping(effort)
        floor = em.start_tier if em is not None else 0
        candidates = [t for t in ladder.tiers if t.level >= floor]
        if not candidates:
            return floor
        return min(candidates, key=lambda t: estimate_tier_cost(t, effort)).level
