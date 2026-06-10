"""Raphael — the ``direct`` single-task strategy.

Flow: plan **once**, then *run is the loop*. Each iteration runs the task at the
current tier; the mechanical gate records a pass/fail (never a veto) and a
frontier-LLM evaluator judges the code generation against the task. The evaluator
alone decides what happens next:

* **PASS** -> done.
* **fix, same model** -> re-run in the *same* session with the evaluator's
  corrections plus the gate output (so the model understands what broke). Cheapest
  retry — both the runner and the evaluator keep their conversation context. A
  gate failure is normal and lands here, not in an escalation.
* **change the runner** (ESCALATE / JUMP_PREMIUM) -> bump the tier; the new model
  AND a fresh quality eval start from clean sessions, with the corrections plus the
  session knowledge memory. The original plan is reused, never regenerated.

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
from splinter.skills import resolve_eval_skill
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

#: Ceiling tier — ``last-resort`` (sonnet @ max); the loop stops escalating here.
MAX_TIER = 5


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
        eval_skill: str | None = None,
        cowabunga: bool = False,
        resume: bool = False,
    ) -> list[RunResult]:
        existing_trace = session.read("trace.md")
        if resume and existing_trace.strip():
            trace = Trace.from_markdown(existing_trace)
        else:
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
                task_index=i,
                effort=effort,
                budget=budget,
                max_iterations=max_iterations,
                localization=localization,
                eval_skill=eval_skill,
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
        task_index: int = 0,
        effort: str | None,
        budget: float | None,
        max_iterations: int,
        localization: str,
        eval_skill: str | None = None,
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
            plan = existing_plan[len("# Plan"):].lstrip("\n") \
                if existing_plan.startswith("# Plan") else existing_plan
            if not session.read(f"knowledge/plan-{task_index + 1}.md").strip():
                session.write(f"knowledge/plan-{task_index + 1}.md", f"# Plan\n\n{plan}\n")
        else:
            log.info("planning with %s (once)", ladder.planner_model)
            plan = _make_plan(task, ladder, localization)
            session.write("knowledge/plan.md", f"# Plan\n\n{plan}\n")
            session.write(f"knowledge/plan-{task_index + 1}.md", f"# Plan\n\n{plan}\n")

        resolved = resolve_eval_skill(eval_skill or task.eval_skill)
        chain = build_chain(RunStage(), GateStage(), EvalStage(resolved_skill=resolved))
        evaluator = Evaluator(ladder)
        eval_history: list[str] = []
        # The runner and the evaluator each keep a provider session that lives as
        # long as the runner model is unchanged. When the eval decides to change
        # the runner (escalate), both are reset so the new model — and a fresh
        # quality eval — start from a clean conversation.
        oc_session: str | None = None
        eval_session: str | None = None
        corrections = ""
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
                eval_session=eval_session,
                corrections=corrections,
                eval_history=eval_history,
                task_index=task_index,
            )
            chain.handle(ctx)

            session.write("trace.md", trace.summary())

            last_result = ctx.run_result
            oc_session = ctx.oc_session
            eval_session = ctx.eval_session

            verdict = ctx.verdict
            assert verdict is not None and ctx.run_result is not None

            # The evaluator owns the verdict. A gate failure never short-circuits
            # it and never escalates on its own — it is fed back to the runner as
            # corrections. The runner changes ONLY when the eval says so.
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

            retry_notes = _retry_corrections(verdict.corrections, ctx.gate_output)

            if action.next_tier != tier:
                # Eval judged this runner incapable → change it. Fresh runner AND
                # eval sessions: the new model gets a clean slate and the quality
                # eval re-judges its output from scratch.
                tier = action.next_tier
                log.info("escalating to T%d (eval changed the runner; fresh sessions)", tier)
                session.append(
                    "loop.md", f"## Escalate to tier {tier} (eval changed the runner)\n\n"
                )
                oc_session = None
                eval_session = None
                corrections = _correction_context(knowledge, retry_notes)
            else:
                # Same runner, same session — let it understand and fix what's wrong.
                corrections = retry_notes

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


def _retry_corrections(corrections: str, gate_output: str) -> str:
    """Combine evaluator corrections + gate output into the retry context string."""
    parts = []
    if corrections:
        parts.append(corrections)
    if gate_output:
        parts.append(f"## Gate Output\n{gate_output}")
    return "\n\n".join(parts)


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
