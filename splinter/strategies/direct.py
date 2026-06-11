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

import json
import logging
from dataclasses import asdict, dataclass

from splinter import prd_session
from splinter.agents.evaluator import PREMIUM_TIER, Evaluator
from splinter.agents.runner import RunResult, Task
from splinter.enums import Decision
from splinter.memory.knowledge import KnowledgeStore
from splinter.memory.session import Session
from splinter.models.roster import Ladder
from splinter.obs.agentic import agentic_scope, record_exchange
from splinter.obs.trace import Trace
from splinter.providers.dispatch import run_text
from splinter.skills import resolve_eval_skill
from splinter.strategies.base import AskUserPause, Strategy
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

_CHECKPOINT_FILE = "run_checkpoint.json"


@dataclass
class RunCheckpoint:
    tier: int
    iteration: int
    task_index: int
    oc_session: str | None
    eval_session: str | None
    corrections: str
    eval_history: list[str]
    reason: str
    gate_output: str


def _save_checkpoint(session: Session, cp: RunCheckpoint) -> None:
    session.write(_CHECKPOINT_FILE, json.dumps(asdict(cp), indent=2))


def _load_checkpoint(session: Session) -> RunCheckpoint | None:
    raw = session.read(_CHECKPOINT_FILE).strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return RunCheckpoint(
        tier=int(data.get("tier", 0)),
        iteration=int(data.get("iteration", 1)),
        task_index=int(data.get("task_index", 0)),
        oc_session=data.get("oc_session") or None,
        eval_session=data.get("eval_session") or None,
        corrections=str(data.get("corrections", "")),
        eval_history=[str(x) for x in data.get("eval_history", [])],
        reason=str(data.get("reason", "")),
        gate_output=str(data.get("gate_output", "")),
    )


def _clear_checkpoint(session: Session) -> None:
    p = session.dir / _CHECKPOINT_FILE
    if p.exists():
        p.unlink()


def _pause_for_user(
    *,
    session: Session,
    knowledge: KnowledgeStore,
    task_index: int,
    iteration: int,
    tier: int,
    reason: str,
    corrections: str,
    gate_output: str,
    oc_session: str | None,
    eval_session: str | None,
    eval_history: list[str],
) -> None:
    retry_notes = _retry_corrections(corrections, gate_output)
    log.warning("iter %d · needs human input — pausing task", iteration)
    session.append(
        "loop.md",
        f"## ASK_USER (iter {iteration}) — needs human input\n{reason}\n\n",
    )
    knowledge.write_note(f"ask-user-iter-{iteration}", reason)
    _save_checkpoint(
        session,
        RunCheckpoint(
            tier=tier,
            iteration=iteration,
            task_index=task_index,
            oc_session=oc_session,
            eval_session=eval_session,
            corrections=retry_notes,
            eval_history=list(eval_history),
            reason=reason,
            gate_output=gate_output,
        ),
    )
    raise AskUserPause(
        reason=reason,
        corrections=corrections,
        tier=tier,
        iteration=iteration,
        task_index=task_index,
    )


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
        claude_runner_fallback: bool = False,
        user_guidance: str | None = None,
        jump_premium: bool = False,
    ) -> list[RunResult]:
        existing_trace = session.read("trace.md")
        if resume and existing_trace.strip():
            trace = Trace.from_markdown(existing_trace)
        else:
            trace = Trace()
        knowledge = KnowledgeStore(session)
        results: list[RunResult] = []

        checkpoint = (
            _load_checkpoint(session)
            if resume and session.read_status().get("state") == "awaiting_user"
            else None
        )
        start_index = (
            checkpoint.task_index
            if checkpoint is not None
            else (self._resume_start_index(session, tasks) if resume else 0)
        )
        if start_index:
            log.info(
                "resume: %d task(s) already done — restarting at task %d/%d",
                start_index,
                start_index + 1,
                len(tasks),
            )

        for i, task in enumerate(tasks):
            if i < start_index:
                continue
            # Persist progress *before* running task i so a crash here resumes here.
            session.set_status(
                "running",
                stage="run",
                task_index=i,
                task_total=len(tasks),
                task=task.description.splitlines()[0][:80],
            )
            session.append(
                "loop.md",
                f"# Task {i + 1}/{len(tasks)}: {task.description.splitlines()[0][:80]}\n\n",
            )
            task_resume = (checkpoint is not None and i == checkpoint.task_index) or (
                resume and i == start_index and len(tasks) == 1
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
                resume=task_resume,
                checkpoint=checkpoint if task_resume else None,
                user_guidance=user_guidance if task_resume else None,
                jump_premium=jump_premium if task_resume else False,
            )
            if result is not None:
                results.append(result)
            checkpoint = None

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

    def _plan_all_tasks(
        self,
        tasks: list[Task],
        session: Session,
        ladder: Ladder,
        localization: str,
    ) -> None:
        """Pre-generate plans for all tasks; reuses existing files on resume."""
        for i, task in enumerate(tasks):
            task_plan_file = f"knowledge/plan-{i + 1}.md"
            if session.read(task_plan_file).strip():
                log.info("plan exists for task %d — reusing", i + 1)
                continue
            try:
                log.info("planning task %d with %s", i + 1, ladder.planner_model)
                task_loc = session.read(f"knowledge/localization-{i + 1}.md")
                code_ctx = "\n\n".join(
                    filter(None, [task_loc, task.filtered_context or localization])
                )
                with agentic_scope(session, "plan", i, 0):
                    plan = _make_plan(task, ladder, code_ctx, session=session)
                session.write(task_plan_file, f"# Plan\n\n{plan}\n")
                if i == 0:
                    session.write("knowledge/plan.md", f"# Plan\n\n{plan}\n")
            except Exception as e:
                # swallow: planner unavailable; _run_task_loop plans per-task as fallback
                log.warning("bulk planning skipped for task %d: %s", i + 1, e)

    def _run_plan_phase(
        self,
        tasks: list[Task],
        session: Session,
        ladder: Ladder,
        localization: str,
    ) -> None:
        session.set_status("running", stage="plan")
        self._plan_all_tasks(tasks, session, ladder, localization)
        session.set_status("running", stage="run")

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
        soft_budget: bool = False,
        start_tier_override: int | None = None,
        checkpoint: RunCheckpoint | None = None,
        user_guidance: str | None = None,
        jump_premium: bool = False,
    ) -> RunResult | None:
        if checkpoint is not None:
            _clear_checkpoint(session)
            tier = checkpoint.tier
            if jump_premium:
                tier = max(tier, PREMIUM_TIER)
                oc_session: str | None = None
                eval_session: str | None = None
            else:
                oc_session = checkpoint.oc_session
                eval_session = checkpoint.eval_session
            corrections = checkpoint.corrections
            if user_guidance:
                corrections = f"{corrections}\n\n## User guidance\n{user_guidance}".strip()
            eval_history = list(checkpoint.eval_history)
            start_iteration = checkpoint.iteration
        else:
            tier = (
                start_tier_override
                if start_tier_override is not None
                else self._start_tier(task, ladder)
            )
            oc_session = None
            eval_session = None
            corrections = ""
            eval_history = []
            start_iteration = 1

        # Reuse any existing plan — opus calls are expensive and the plan is
        # deterministic for this task. Always check the task-specific file first.
        task_plan_file = f"knowledge/plan-{task_index + 1}.md"
        existing_plan = (
            session.read(task_plan_file).strip() or session.read("knowledge/plan.md").strip()
        )
        if existing_plan:
            log.info("plan exists for task %d — reusing (skipping planner)", task_index + 1)
            plan = (
                existing_plan[len("# Plan") :].lstrip("\n")
                if existing_plan.startswith("# Plan")
                else existing_plan
            )
            if not session.read(task_plan_file).strip():
                session.write(task_plan_file, f"# Plan\n\n{plan}\n")
        else:
            log.info("planning with %s (once)", ladder.planner_model)
            code_ctx = task.filtered_context or localization
            with agentic_scope(session, "plan", task_index, 0):
                plan = _make_plan(task, ladder, code_ctx, session=session)
            session.write("knowledge/plan.md", f"# Plan\n\n{plan}\n")
            session.write(task_plan_file, f"# Plan\n\n{plan}\n")

        resolved = resolve_eval_skill(eval_skill or task.eval_skill)
        chain = build_chain(RunStage(), GateStage(), EvalStage(resolved_skill=resolved))
        evaluator = Evaluator(ladder)
        last_result: RunResult | None = None

        for iteration in range(start_iteration, max_iterations + 1):
            ctx = IterationContext(
                task=task,
                plan=plan,
                tier=tier,
                iteration=iteration,
                ladder=ladder,
                session=session,
                trace=trace,
                knowledge=knowledge,
                localization=task.filtered_context or localization,
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
            action = evaluator.next_action(verdict, tier, max_tier=MAX_TIER, cowabunga=cowabunga)

            if action.stop:
                if verdict.decision == Decision.ASK_USER and cowabunga:
                    return ctx.run_result
                if action.ask_user:
                    _pause_for_user(
                        session=session,
                        knowledge=knowledge,
                        task_index=task_index,
                        iteration=iteration,
                        tier=tier,
                        reason=verdict.reason,
                        corrections=verdict.corrections,
                        gate_output=ctx.gate_output,
                        oc_session=oc_session,
                        eval_session=eval_session,
                        eval_history=eval_history,
                    )
                if verdict.passed:
                    log.info("task PASSED at T%d after %d iteration(s)", tier, iteration)
                    _mark_story_done(session, task)
                    return ctx.run_result
                reason = (
                    f"Stopped at T{tier} without PASS "
                    f"(verdict: {verdict.decision}). "
                    f"Use Jump Premium or answer to continue."
                )
                session.append("loop.md", f"## Max tier reached (T{tier}), stopping.\n\n")
                _pause_for_user(
                    session=session,
                    knowledge=knowledge,
                    task_index=task_index,
                    iteration=iteration,
                    tier=tier,
                    reason=reason,
                    corrections=verdict.corrections,
                    gate_output=ctx.gate_output,
                    oc_session=oc_session,
                    eval_session=eval_session,
                    eval_history=eval_history,
                )

            if (
                verdict.decision == Decision.JUMP_PREMIUM
                and action.next_tier == tier
                and not cowabunga
            ):
                reason = (
                    f"Eval requests JUMP_PREMIUM but already at T{tier}. "
                    f"Use Jump Premium to switch models or answer to steer."
                )
                _pause_for_user(
                    session=session,
                    knowledge=knowledge,
                    task_index=task_index,
                    iteration=iteration,
                    tier=tier,
                    reason=reason,
                    corrections=verdict.corrections,
                    gate_output=ctx.gate_output,
                    oc_session=oc_session,
                    eval_session=eval_session,
                    eval_history=eval_history,
                )

            retry_notes = _retry_corrections(verdict.corrections, ctx.gate_output)

            over_budget = budget is not None and trace.total_cost >= budget

            if over_budget and not soft_budget:
                session.append("loop.md", f"## Budget exhausted (${trace.total_cost:.4f})\n")
                return ctx.run_result

            if action.next_tier != tier:
                if over_budget and soft_budget:
                    # Soft cap: continue at current tier, suppress escalation.
                    log.info("over soft budget — capping escalation at T%d", tier)
                    session.append(
                        "loop.md", f"## Over budget — capping at T{tier} (soft budget)\n\n"
                    )
                    corrections = retry_notes
                else:
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

        if last_result is not None and not _story_done(session, task):
            sid = prd_session.story_id(task.description)
            if sid:
                _pause_for_user(
                    session=session,
                    knowledge=knowledge,
                    task_index=task_index,
                    iteration=max_iterations,
                    tier=tier,
                    reason=(
                        f"Task {sid} did not PASS after {max_iterations} iteration(s) at T{tier}. "
                        f"Use Jump Premium or answer to continue."
                    ),
                    corrections="",
                    gate_output="",
                    oc_session=oc_session,
                    eval_session=eval_session,
                    eval_history=eval_history,
                )
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


def _story_done(session: Session, task: Task) -> bool:
    sid = prd_session.story_id(task.description)
    if not sid:
        return False
    prd = session.read("prd.md")
    if not prd.strip():
        return False
    return sid in prd_session.completed_story_ids(prd)


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
        f"### {topic}\n{knowledge.read_note(topic)}".strip() for topic in knowledge.list_notes()
    ]
    if notes:
        parts.append("## Session Knowledge\n" + "\n\n".join(notes))

    return "\n\n".join(parts)


def _make_plan(task: Task, ladder: Ladder, code_ctx: str, session: object = None) -> str:
    from splinter.templating import load_standards

    prompt = render(
        "plan",
        task_section=section("Task", task.description),
        acceptance_section=section("Acceptance Criteria", task.acceptance),
        code_context_section=section("Code Context", code_ctx),
        standards_section=section("Code Conventions", load_standards()),
    )
    plan = run_text(
        prompt,
        ladder.planner_model,
        variant=ladder.planner_effort,
        timeout=ladder.planner_timeout,
        session=session,
    )
    record_exchange(prompt, plan, model=ladder.planner_model)
    return plan
