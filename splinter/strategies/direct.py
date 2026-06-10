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

from splinter import prd_session
from splinter.agents.evaluator import Evaluator
from splinter.agents.runner import RunResult, Task
from splinter.memory.knowledge import KnowledgeStore
from splinter.memory.session import Session
from splinter.models.roster import Ladder
from splinter.obs.trace import Trace
from splinter.providers.dispatch import run_text
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
        cowabunga: bool = False,
        resume: bool = False,
    ) -> list[RunResult]:
        trace = Trace()
        knowledge = KnowledgeStore(session)
        results: list[RunResult] = []

        start_index = self._resume_start_index(session, tasks) if resume else 0
        if start_index:
            log.info("resume: %d task(s) already done — restarting at task %d/%d",
                     start_index, start_index + 1, len(tasks))

        for i, task in enumerate(tasks):
            if i < start_index:
                continue
            # Persist progress *before* running task i so a crash here resumes here.
            session.set_status(
                "running", stage="run", task_index=i, task_total=len(tasks),
                task=task.description.splitlines()[0][:80],
            )
            session.append(
                "loop.md",
                f"# Task {i + 1}/{len(tasks)}: "
                f"{task.description.splitlines()[0][:80]}\n\n",
            )
            # Reuse the persisted plan only when resuming the exact single task we
            # stopped on; a multi-task resume starts its target task from a fresh plan.
            task_resume = resume and i == start_index and len(tasks) == 1
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
                cowabunga=cowabunga,
                resume=task_resume,
            )
            if result is not None:
                results.append(result)

        # All tasks done — record it so a stray resume is a no-op, not a re-run.
        session.set_status("running", task_index=len(tasks), task_total=len(tasks))
        session.write("trace.md", trace.summary())
        return results

    @staticmethod
    def _resume_start_index(session: Session, tasks: list[Task]) -> int:
        """How many leading tasks are already finished.

        The PRD is the source of truth: a task whose user story has all its
        acceptance-criteria boxes ticked is done. Falls back to the positional
        ``task_index`` in status when the PRD carries no checkboxes (e.g. a single
        ``--task`` yaml run).
        """
        prd = session.read("prd.md")
        done = prd_session.completed_story_ids(prd) if prd.strip() else set()
        if not done:
            return int(session.read_status().get("task_index") or 0)
        start = 0
        for task in tasks:
            sid = prd_session.story_id(task.description)
            if sid and sid in done:
                start += 1
            else:
                break
        return start

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
        cowabunga: bool = False,
        resume: bool = False,
    ) -> RunResult | None:
        tier = self._start_tier(task, ladder)

        # The plan is produced once and reused for every iteration and escalation.
        # On resume, reuse the persisted plan instead of regenerating (it's expensive
        # and deterministic for the task).
        existing_plan = session.read("knowledge/plan.md").strip()
        if resume and existing_plan:
            log.info("resume: reusing existing plan")
            # plan.md is stored as "# Plan\n\n<plan>"; drop the header to match a fresh plan.
            plan = existing_plan[len("# Plan"):].lstrip("\n") \
                if existing_plan.startswith("# Plan") else existing_plan
        else:
            log.info("planning with %s (once)", ladder.planner_model)
            plan = _make_plan(task, ladder, localization)
            # Plan is model-consumed memory → knowledge store, not a loose root file.
            session.write("knowledge/plan.md", f"# Plan\n\n{plan}\n")

        chain = build_chain(RunStage(), GateStage(), EvalStage())
        evaluator = Evaluator(ladder)
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

            action = evaluator.next_action(
                verdict, tier, max_tier=MAX_TIER, cowabunga=cowabunga
            )

            if action.stop:
                if action.ask_user:
                    log.warning("iter %d · ASK_USER — needs human input, stopping task",
                                iteration)
                    session.append(
                        "loop.md",
                        f"## ASK_USER (iter {iteration}) — needs human input\n"
                        f"{verdict.reason}\n\n",
                    )
                    knowledge.write_note(f"ask-user-iter-{iteration}", verdict.reason)
                elif verdict.passed:
                    log.info("task PASSED at T%d after %d iteration(s)", tier, iteration)
                    _mark_story_done(session, task)
                else:
                    log.warning("max tier T%d reached — stopping", tier)
                    session.append("loop.md", f"## Max tier reached (T{tier}), stopping.\n\n")
                return ctx.run_result

            if action.next_tier != tier:
                tier = action.next_tier
                log.info("escalating to T%d (plan reused)", tier)
                session.append("loop.md", f"## Escalate to tier {tier} (plan reused)\n\n")
                oc_session = None
                consecutive_fails = 0
                corrections = _correction_context(knowledge, verdict.corrections)
            else:
                consecutive_fails += 1
                corrections = verdict.corrections

                if consecutive_fails >= FAILS_BEFORE_ESCALATE:
                    if tier >= MAX_TIER:
                        log.warning("max tier T%d reached — stopping", tier)
                        session.append(
                            "loop.md", f"## Max tier reached (T{tier}), stopping.\n\n"
                        )
                        return ctx.run_result

                    tier += 1
                    log.info("escalating to T%d (plan reused)", tier)
                    session.append(
                        "loop.md", f"## Escalate to tier {tier} (plan reused)\n\n"
                    )
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


def _mark_story_done(session: Session, task: Task) -> None:
    """Tick the task's user-story acceptance boxes in ``prd.md`` once it PASSes.

    The PRD is the durable progress record: completed stories show ``- [x]`` so
    both resume and the human can see what is finished. No-op for task-yaml runs
    (no PRD / no ``US-NNN`` id) and when nothing changed.
    """
    sid = prd_session.story_id(task.description)
    if not sid:
        return
    prd = session.read("prd.md")
    if not prd.strip():
        return
    updated = prd_session.mark_story_done(prd, sid)
    if updated != prd:
        session.write("prd.md", updated)
        log.info("PRD: %s acceptance criteria checked off (done)", sid)


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
    return run_text(
        prompt, ladder.planner_model, variant=ladder.planner_effort,
        timeout=ladder.planner_timeout,
    )
