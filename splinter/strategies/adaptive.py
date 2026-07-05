"""Donatello — the ``adaptive`` per-task cost-routing strategy.

Flow: topological sort (like cascade), then for each task pick the cheapest tier
that fits within its effort-weighted share of the remaining session budget. When
the floor tier is affordable, it is used (same as cascade start). When budget is
tight, tasks are down-routed below the floor to reduce cost, accepting higher
escalation risk. Critical effort is never down-routed.

Budget precedence: CLI ``--budget`` arg → session/config ``defaults.budget``.
"""

from __future__ import annotations

import logging

from splinter.agents.runner import RunResult, Task
from splinter.configure import configured_budget, configured_soft_budget
from splinter.enums import Effort
from splinter.memory.knowledge import KnowledgeStore
from splinter.memory.session import Session
from splinter.models.pricing import estimate_tier_cost
from splinter.models.roster import Ladder
from splinter.obs.trace import Trace
from splinter.strategies.cascade import CascadeStrategy
from splinter.strategies.direct import TaskOutcome
from splinter.strategies.registry import register

log = logging.getLogger("splinter.loop")

_EFFORT_WEIGHTS: dict[str, int] = {
    Effort.TRIVIAL: 1,
    Effort.NORMAL: 2,
    Effort.HARD: 3,
    Effort.CRITICAL: 4,
}


def _effort_weight(effort: str) -> int:
    return _EFFORT_WEIGHTS.get(effort, 2)


@register
class AdaptiveStrategy(CascadeStrategy):
    name = "adaptive"
    aliases = ["donatello"]
    _log_prefix: str = "adaptive"
    _log_routing_detail: bool = True

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
        skip_planner: bool = False,
        skip_eval: bool = False,
        force_replan: bool = False,
        parallel: bool = False,
        max_concurrency: int | None = None,
    ) -> list[RunResult]:
        effective_budget = budget if budget is not None else configured_budget()
        effective_soft_budget = configured_soft_budget()
        ordered = self._topo_sort(tasks)

        trace = Trace.from_jsonl(session) if resume else Trace(session=session)

        knowledge = KnowledgeStore(session)
        results: list[RunResult] = []
        done = self._load_checkpoint(session) if resume else set()

        if done:
            log.info("adaptive resume: %d task(s) already checkpointed", len(done))

        self._run_plan_phase(
            ordered,
            session,
            ladder,
            localization,
            trace=trace,
            skip_planner=skip_planner,
            resume=resume,
            force_replan=force_replan,
        )

        if parallel and len(ordered) > 1:
            tier_overrides = self._compute_tier_overrides(
                ordered, done, effort, effective_budget, ladder, trace
            )
            results = self._run_parallel_dag(
                ordered,
                session,
                ladder,
                trace,
                knowledge,
                done=done,
                effort=effort,
                budget=effective_budget,
                max_iterations=max_iterations,
                localization=localization,
                eval_skill=eval_skill,
                cowabunga=cowabunga,
                skip_planner=skip_planner,
                skip_eval=skip_eval,
                max_concurrency=max_concurrency,
                start_tier_overrides=tier_overrides,
            )
            session.set_status("running", task_index=len(ordered), task_total=len(ordered))
            return results

        for i, task in enumerate(ordered):
            if task.id and task.id in done:
                log.info("resume: skip %s (checkpointed)", task.id)
                continue

            if (
                effective_budget is not None
                and not effective_soft_budget
                and trace.total_cost >= effective_budget
            ):
                session.append("loop.md", f"## Budget exhausted (${trace.total_cost:.4f})\n")
                break

            task_effort = effort or task.effort

            remaining_budget = (
                None if effective_budget is None else max(0.0, effective_budget - trace.total_cost)
            )
            remaining_efforts = [
                effort or t.effort for t in ordered[i:] if not (t.id and t.id in done)
            ]
            routed_tier = self._route_tier(task_effort, ladder, remaining_budget, remaining_efforts)

            if self._log_routing_detail:
                em = ladder.effort_mapping(task_effort)
                floor = em.start_tier if em is not None else 0
                down_routed = routed_tier < floor
                log.info(
                    "%s: task %d/%d effort=%s → T%d (%s) [budget=%s floor=T%d down_routed=%s]",
                    self._log_prefix,
                    i + 1,
                    len(ordered),
                    task_effort,
                    routed_tier,
                    ladder.tier_by_level(routed_tier).name,
                    f"{remaining_budget:.4f}" if remaining_budget is not None else "n/a",
                    floor,
                    down_routed,
                )
            else:
                log.info(
                    "%s: task %d/%d effort=%s → T%d (%s)",
                    self._log_prefix,
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
                task=task.description.splitlines()[0],
            )
            session.append(
                "loop.md",
                f"# Task {i + 1}/{len(ordered)}: {task.description.splitlines()[0]}\n\n",
            )

            outcome = TaskOutcome()
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
                localization=task.filtered_context or localization,
                eval_skill=eval_skill,
                cowabunga=cowabunga,
                resume=False,
                soft_budget=effective_soft_budget,
                start_tier_override=routed_tier,
                skip_planner=skip_planner,
                skip_eval=skip_eval,
                force_replan=force_replan,
                outcome=outcome,
            )

            if result is not None:
                results.append(result)
                # Checkpoint only a genuine PASS — see CascadeStrategy._run_sequential.
                if task.id and outcome.passed:
                    done.add(task.id)
                    self._save_checkpoint(session, done)

        session.set_status("running", task_index=len(ordered), task_total=len(ordered))
        return results

    def _compute_tier_overrides(
        self,
        ordered: list[Task],
        done: set[str],
        effort: str | None,
        effective_budget: float | None,
        ladder: Ladder,
        trace: Trace,
    ) -> dict[str, int]:
        overrides: dict[str, int] = {}
        remaining_budget = (
            None if effective_budget is None else max(0.0, effective_budget - trace.total_cost)
        )
        remaining = [t for t in ordered if not (t.id and t.id in done)]
        remaining_efforts = [effort or t.effort for t in remaining]
        for i, task in enumerate(remaining):
            task_effort = effort or task.effort
            routed = self._route_tier(task_effort, ladder, remaining_budget, remaining_efforts[i:])
            if task.id:
                overrides[task.id] = routed
        return overrides

    @staticmethod
    def _route_tier(
        effort: str,
        ladder: Ladder,
        remaining_budget: float | None = None,
        remaining_efforts: list[str] | None = None,
    ) -> int:
        """Budget-aware start-tier selection.

        Without a budget: returns the effort floor (capability baseline).
        With a budget: computes an effort-weighted per-task share and routes to
        the floor if affordable; otherwise down-routes to the cheapest affordable
        tier (level may be below floor). Critical effort is hard-floored — never
        down-routed regardless of share.
        """
        em = ladder.effort_mapping(effort)
        floor = em.start_tier if em is not None else 0

        if remaining_budget is None:
            return floor

        if effort == Effort.CRITICAL:
            return floor

        efforts = remaining_efforts or [effort]
        total_w = sum(_effort_weight(e) for e in efforts)
        per_task_share = remaining_budget * _effort_weight(effort) / total_w

        floor_tier = ladder.tier_by_level(floor)
        if estimate_tier_cost(floor_tier, effort) <= per_task_share:
            return floor

        candidates = [t for t in ladder.tiers if estimate_tier_cost(t, effort) <= per_task_share]
        if candidates:
            return min(candidates, key=lambda t: estimate_tier_cost(t, effort)).level
        return min(t.level for t in ladder.tiers)
