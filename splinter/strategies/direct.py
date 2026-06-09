"""Raphael — the ``direct`` single-task strategy.

Flow: plan **once**, then *run is the loop*. Each iteration runs the task at the
current tier and is judged by the evaluator:

* **PASS** -> done.
* **fail, same model** -> re-run in the *same* opencode session with the
  evaluator's corrections (cheapest retry; the model keeps its context).
* **fail again** -> escalate one tier. The higher model starts a fresh session
  and receives the corrections *plus the session knowledge memory* directly — the
  original plan is reused, never regenerated.

The per-iteration Run -> Gate -> Eval pipeline is a Chain of Responsibility (see
:mod:`splinter.strategies.stages`); this class owns only the retry/escalate policy.
"""

from __future__ import annotations

import logging

from splinter.agents.runner import RunResult, Task
from splinter.memory.knowledge import KnowledgeStore
from splinter.memory.session import Session
from splinter.models.roster import Ladder
from splinter.obs.trace import Trace
from splinter.providers import claude_cli
from splinter.strategies.base import Strategy
from splinter.strategies.registry import register
from splinter.strategies.stages import (
    EvalStage,
    GateStage,
    IterationContext,
    RunStage,
    build_chain,
)
from splinter.templating import render, section

log = logging.getLogger("splinter.loop")

#: Ceiling tier — ``opus-4.8``; the loop stops escalating here.
MAX_TIER = 4
#: Consecutive failures at one tier before escalating to the next.
FAILS_BEFORE_ESCALATE = 2


@register
class DirectStrategy(Strategy):
    name = "direct"
    aliases = ["raphael"]

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
    ) -> list[RunResult]:
        trace = Trace()
        knowledge = KnowledgeStore(session)
        results: list[RunResult] = []

        for task in tasks:
            result = self._run_task_loop(
                task,
                session,
                ladder,
                trace,
                knowledge,
                effort=effort,
                budget=budget,
                max_iterations=max_iterations,
                localization=localization,
            )
            if result is not None:
                results.append(result)

        session.write("trace.md", trace.summary())
        return results

    def _run_task_loop(
        self,
        task: Task,
        session: Session,
        ladder: Ladder,
        trace: Trace,
        knowledge: KnowledgeStore,
        *,
        effort: str | None,
        budget: float | None,
        max_iterations: int,
        localization: str,
    ) -> RunResult | None:
        tier = self._start_tier(task, ladder)

        # The plan is produced once and reused for every iteration and escalation.
        log.info("planning with %s (once)", ladder.planner_model)
        plan = _make_plan(task, ladder, localization)
        session.write("plan.md", f"# Plan\n\n{plan}\n")

        chain = build_chain(RunStage(), GateStage(), EvalStage())
        eval_history: list[str] = []
        oc_session: str | None = None
        corrections = ""
        consecutive_fails = 0
        last_result: RunResult | None = None

        for iteration in range(1, max_iterations + 1):
            ctx = IterationContext(
                task=task,
                plan=plan,
                tier=tier,
                iteration=iteration,
                ladder=ladder,
                session=session,
                trace=trace,
                knowledge=knowledge,
                localization=localization,
                effort_override=effort,
                oc_session=oc_session,
                corrections=corrections,
                eval_history=eval_history,
            )
            chain.handle(ctx)

            # Persist the trace each iteration so `analyze` sees live cost/runs.
            session.write("trace.md", trace.summary())

            last_result = ctx.run_result
            oc_session = ctx.oc_session

            if not ctx.gate_passed:
                consecutive_fails += 1
                if consecutive_fails >= FAILS_BEFORE_ESCALATE:
                    if tier >= MAX_TIER:
                        session.append("loop.md", f"## Max tier reached (T{tier}), stopping.\n\n")
                        return ctx.run_result

                    tier += 1
                    log.info("escalating to T%d (gate failures)", tier)
                    session.append("loop.md", f"## Escalate to tier {tier} (gate failures)\n\n")
                    oc_session = None
                    consecutive_fails = 0
                    corrections = ""
                continue

            verdict = ctx.verdict
            assert verdict is not None and ctx.run_result is not None

            if verdict.passed:
                log.info("task PASSED at T%d after %d iteration(s)", tier, iteration)
                return ctx.run_result

            consecutive_fails += 1
            # Same-model retry reuses the live session; corrections alone suffice.
            corrections = verdict.corrections

            if consecutive_fails >= FAILS_BEFORE_ESCALATE:
                if tier >= MAX_TIER:
                    log.warning("max tier T%d reached — stopping", tier)
                    session.append("loop.md", f"## Max tier reached (T{tier}), stopping.\n\n")
                    return ctx.run_result

                tier += 1
                log.info("escalating to T%d (plan reused)", tier)
                session.append("loop.md", f"## Escalate to tier {tier} (plan reused)\n\n")
                # Fresh model, fresh session: hand it corrections + the knowledge memory.
                oc_session = None
                consecutive_fails = 0
                corrections = _correction_context(knowledge, verdict.corrections)

            if budget is not None and trace.total_cost >= budget:
                session.append("loop.md", f"## Budget exhausted (${trace.total_cost:.4f})\n")
                return ctx.run_result

        return last_result

    @staticmethod
    def _start_tier(task: Task, ladder: Ladder) -> int:
        em = ladder.effort_mapping(task.effort)
        return em.start_tier if em is not None else task.suggested_tier


def _correction_context(knowledge: KnowledgeStore, latest: str) -> str:
    """Bundle the latest corrections with all accumulated session knowledge notes.

    Used only when escalating to a fresh model that has no live session context.
    """
    parts: list[str] = []
    if latest:
        parts.append(latest)

    notes = [
        f"### {topic}\n{knowledge.read_note(topic)}".strip()
        for topic in knowledge.list_notes()
    ]
    if notes:
        parts.append("## Session Knowledge\n" + "\n\n".join(notes))

    return "\n\n".join(parts)


def _make_plan(task: Task, ladder: Ladder, localization: str) -> str:
    prompt = render(
        "plan",
        task_section=section("Task", task.description),
        acceptance_section=section("Acceptance Criteria", task.acceptance),
        code_context_section=section("Code Context", localization),
    )
    result = claude_cli.run(prompt, ladder.planner_model, effort=ladder.planner_effort)
    return result.text
