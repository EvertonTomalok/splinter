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
from typing import Any

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
from splinter.strategies.base import AskUserPause, EvalVerdict, GracefulPause, Strategy
from splinter.strategies.registry import register
from splinter.strategies.stages import (
    EvalStage,
    GateStage,
    IterationContext,
    RunStage,
    build_chain,
    build_chain_from,
)
from splinter.templating import render, section

log = logging.getLogger("splinter.loop")

#: Ceiling tier — ``last-resort`` (sonnet @ max); the loop stops escalating here.
MAX_TIER = 5

#: Max iterations on the same model tier before forcing escalation.
MAX_TRIES_PER_MODEL = 3

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
    stage: str = ""
    run_result: dict[str, Any] | None = None
    gate_passed: bool = True
    verdict: dict[str, Any] | None = None


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
        stage=str(data.get("stage", "")),
        run_result=data.get("run_result") or None,
        gate_passed=bool(data.get("gate_passed", True)),
        verdict=data.get("verdict") or None,
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


def _deserialize_run_result(data: dict[str, Any]) -> RunResult:
    return RunResult(
        text=str(data.get("text", "")),
        model=str(data.get("model", "")),
        tier=int(data.get("tier", 0)),
        tokens={str(k): int(v) for k, v in (data.get("tokens") or {}).items()},
        cost=float(data.get("cost", 0.0)),
        raw=data.get("raw") or {},
        opencode_session=data.get("opencode_session") or None,
    )


def _deserialize_verdict(data: dict[str, Any]) -> EvalVerdict:
    return EvalVerdict(
        decision=str(data.get("decision", "")),
        reason=str(data.get("reason", "")),
        corrections=str(data.get("corrections", "")),
        raw=str(data.get("raw", "")),
        eval_session=data.get("eval_session") or None,
        cost=float(data.get("cost", 0.0)),
        tokens={str(k): int(v) for k, v in (data.get("tokens") or {}).items()},
    )


def _pause_graceful(
    *,
    session: Session,
    knowledge: KnowledgeStore,
    task_index: int,
    iteration: int,
    tier: int,
    stage: str,
    corrections: str,
    gate_output: str,
    run_result: RunResult | None,
    gate_passed: bool,
    verdict: EvalVerdict | None,
    oc_session: str | None,
    eval_session: str | None,
    eval_history: list[str],
) -> None:
    reason = f"Paused by user at '{stage}' boundary (iter {iteration}). Resume to continue."
    log.info("graceful pause requested — pausing at stage '%s' after iter %d", stage, iteration)
    session.append(
        "loop.md",
        f"## Graceful pause (iter {iteration}) — boundary before {stage}\n{reason}\n\n",
    )
    knowledge.write_note(f"graceful-pause-iter-{iteration}", reason)

    run_result_dict: dict[str, Any] | None = None
    if run_result is not None:
        try:
            run_result_dict = asdict(run_result)
        except Exception:
            pass

    verdict_dict: dict[str, Any] | None = None
    if verdict is not None:
        try:
            verdict_dict = asdict(verdict)
        except Exception:
            pass

    _save_checkpoint(
        session,
        RunCheckpoint(
            tier=tier,
            iteration=iteration,
            task_index=task_index,
            oc_session=oc_session,
            eval_session=eval_session,
            corrections=corrections,
            eval_history=list(eval_history),
            reason=reason,
            gate_output=gate_output,
            stage=stage,
            run_result=run_result_dict,
            gate_passed=gate_passed,
            verdict=verdict_dict,
        ),
    )
    raise GracefulPause(
        reason=reason,
        corrections=corrections,
        tier=tier,
        iteration=iteration,
        task_index=task_index,
        stage=stage,
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
        skip_planner: bool = False,
        skip_eval: bool = False,
    ) -> list[RunResult]:
        existing_trace = session.read("trace.md")
        if resume and existing_trace.strip():
            trace = Trace.from_markdown(existing_trace)
        else:
            trace = Trace()
        knowledge = KnowledgeStore(session)

        if not tasks:
            return []

        # Raphael is single-shot: the pipeline collapsed every PRD story into one
        # task, so there is exactly one Run → Gate → Eval loop to drive here.
        task = tasks[0]
        n_stories = _story_count(session)

        checkpoint = (
            _load_checkpoint(session)
            if resume and session.read_status().get("state") in {"awaiting_user", "paused"}
            else None
        )

        session.set_status(
            "running",
            stage="run",
            task_index=0,
            task_total=1,
            task=task.description.splitlines()[0],
        )
        session.append("loop.md", _single_task_header(session, task, n_stories))

        task_resume = checkpoint is not None or resume
        result = self._run_task_loop(
            task,
            session,
            ladder,
            trace,
            knowledge,
            task_index=0,
            effort=effort,
            budget=budget,
            max_iterations=max_iterations,
            localization=localization,
            eval_skill=eval_skill,
            cowabunga=cowabunga,
            resume=task_resume,
            checkpoint=checkpoint,
            user_guidance=user_guidance if task_resume else None,
            jump_premium=jump_premium if task_resume else False,
            skip_planner=skip_planner,
            skip_eval=skip_eval,
        )
        results = [result] if result is not None else []

        # Done — record it so a stray resume is a no-op, not a re-run.
        session.set_status("running", task_index=1, task_total=1)
        session.write("trace.md", trace.summary())
        return results

    def _plan_all_tasks(
        self,
        tasks: list[Task],
        session: Session,
        ladder: Ladder,
        localization: str,
        trace: object = None,
        skip_planner: bool = False,
    ) -> None:
        """Pre-generate plans for all tasks; reuses existing files on resume."""
        for i, task in enumerate(tasks):
            task_plan_file = f"knowledge/plan-{i + 1}.md"
            if session.read(task_plan_file).strip():
                log.info("plan exists for task %d — reusing", i + 1)
                continue
            if skip_planner:
                log.info("plan skipped for task %d (skip_planner)", i + 1)
                continue
            try:
                log.info("planning task %d with %s", i + 1, ladder.planner_model)
                task_loc = session.read(f"knowledge/localization-{i + 1}.md")
                prev_rounds = session.read("knowledge/previous_rounds.md")
                code_ctx = "\n\n".join(
                    filter(None, [prev_rounds, task_loc, task.filtered_context or localization])
                )
                with agentic_scope(session, "plan", i, 0):
                    plan = _make_plan(
                        task,
                        ladder,
                        code_ctx,
                        session=session,
                        trace=trace,
                        iteration=0,
                        tier=0,
                        task_index=i,
                    )
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
        trace: object = None,
        skip_planner: bool = False,
    ) -> None:
        session.set_status("running", stage="plan")
        self._plan_all_tasks(
            tasks, session, ladder, localization, trace=trace, skip_planner=skip_planner
        )
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
        skip_planner: bool = False,
        skip_eval: bool = False,
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

        corrections = _merge_guidance(corrections, user_guidance)

        if user_guidance:
            log.info("user guidance injected: %s…", user_guidance[:120])
            session.append("events.md", f"[USER GUIDANCE] {user_guidance}\n")

        task_plan_file = f"knowledge/plan-{task_index + 1}.md"

        from_ask_user = checkpoint is not None and not checkpoint.stage
        if from_ask_user and corrections.strip():
            # Resuming after ASK_USER (skill/command/user): corrections = skill findings
            # + user guidance. That IS the plan — what to fix is already known.
            log.info("ASK_USER resume — using corrections as plan (skill output + user guidance)")
            plan = corrections
        else:
            # Reuse any existing plan — opus calls are expensive and the plan is
            # deterministic for this task. Always check the task-specific file first.
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
            elif skip_planner:
                log.info("planner skipped by user — using corrections/guidance as plan")
                plan = corrections or user_guidance or ""
            else:
                log.info("planning with %s (once)", ladder.planner_model)
                prev_rounds = session.read("knowledge/previous_rounds.md")
                code_ctx = "\n\n".join(
                    filter(None, [prev_rounds, task.filtered_context or localization])
                )
                with agentic_scope(session, "plan", task_index, 0):
                    plan = _make_plan(
                        task,
                        ladder,
                        code_ctx,
                        session=session,
                        trace=trace,
                        iteration=0,
                        tier=tier,
                        task_index=task_index,
                    )
                session.write("knowledge/plan.md", f"# Plan\n\n{plan}\n")
                session.write(task_plan_file, f"# Plan\n\n{plan}\n")

        resolved = resolve_eval_skill(eval_skill or task.eval_skill)
        evaluator = Evaluator(ladder)
        last_result: RunResult | None = None
        resume_stage = checkpoint.stage if checkpoint is not None else ""
        tier_tries = 0

        for iteration in range(start_iteration, max_iterations + 1):
            tier_tries += 1

            live_cmd = session.pop_live_commands()
            if live_cmd:
                log.info("live directive received from TUI — injecting into corrections")
                session.append(
                    "loop.md",
                    f"## Live directive (iter {iteration})\n{live_cmd}\n\n",
                )
                corrections = _merge_guidance(corrections, live_cmd)

            if iteration == start_iteration and resume_stage:
                chain = build_chain_from(
                    resume_stage, RunStage(), GateStage(), EvalStage(resolved_skill=resolved)
                )
            else:
                chain = build_chain(RunStage(), GateStage(), EvalStage(resolved_skill=resolved))

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
                skip_eval=skip_eval,
            )

            if iteration == start_iteration and resume_stage and checkpoint is not None:
                if checkpoint.run_result is not None:
                    ctx.run_result = _deserialize_run_result(checkpoint.run_result)
                ctx.gate_passed = checkpoint.gate_passed
                ctx.gate_output = checkpoint.gate_output
                if checkpoint.verdict is not None:
                    ctx.verdict = _deserialize_verdict(checkpoint.verdict)

            chain.handle(ctx)

            session.write("trace.md", trace.summary())

            last_result = ctx.run_result
            oc_session = ctx.oc_session
            eval_session = ctx.eval_session

            if ctx.pause_at_stage:
                _pause_graceful(
                    session=session,
                    knowledge=knowledge,
                    task_index=task_index,
                    iteration=iteration,
                    tier=tier,
                    stage=ctx.pause_at_stage,
                    corrections=corrections,
                    gate_output=ctx.gate_output,
                    run_result=ctx.run_result,
                    gate_passed=ctx.gate_passed,
                    verdict=ctx.verdict,
                    oc_session=oc_session,
                    eval_session=eval_session,
                    eval_history=eval_history,
                )

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
                    tier_tries = 0
                    log.info("escalating to T%d (eval changed the runner; fresh sessions)", tier)
                    session.append(
                        "loop.md", f"## Escalate to tier {tier} (eval changed the runner)\n\n"
                    )
                    oc_session = None
                    eval_session = None
                    corrections = _correction_context(knowledge, retry_notes)
            elif (
                tier_tries >= MAX_TRIES_PER_MODEL
                and tier < MAX_TIER
                and not (over_budget and soft_budget)
            ):
                # Per-model try limit reached — force escalate regardless of eval verdict.
                tier += 1
                tier_tries = 0
                log.info(
                    "max tries/model (%d) at T%d — forcing escalation to T%d",
                    MAX_TRIES_PER_MODEL,
                    tier - 1,
                    tier,
                )
                session.append(
                    "loop.md",
                    f"## Max tries/model ({MAX_TRIES_PER_MODEL}) at T{tier - 1}"
                    f" — escalating to T{tier}\n\n",
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


def _story_count(session: Session) -> int:
    """How many ``US-NNN`` stories the session PRD carries (0 for a task-yaml run)."""
    prd = session.read("prd.md")
    return len(prd_session.user_story_titles(prd)) if prd.strip() else 0


def _feature_name(session: Session, task: Task) -> str:
    """Feature name for the single-shot loop header: PRD frontmatter ``feature`` if
    present, else the task description's first line."""
    prd = session.read("prd.md")
    if prd.strip():
        from splinter.agents.planner import _parse_frontmatter

        fm, _ = _parse_frontmatter(prd)
        feat = fm.get("feature")
        if feat:
            return str(feat)
    first = task.description.strip().splitlines()
    return first[0] if first else "task"


def _single_task_header(session: Session, task: Task, n_stories: int) -> str:
    """``# Task: <feature> (<N> stories)`` — no ``1/N`` counter (single-shot)."""
    feature = _feature_name(session, task)
    suffix = f" ({n_stories} stories)" if n_stories > 1 else ""
    return f"# Task: {feature}{suffix}\n\n"


def _mark_story_done(session: Session, task: Task) -> None:
    """Tick acceptance boxes in ``prd.md`` once the task PASSes.

    The PRD is the durable progress record: completed stories show ``- [x]`` so
    both resume and the human can see what is finished. For a normal story task we
    tick that story's boxes; for the raphael single-shot task (no leading ``US-NNN``
    id) the verdict is holistic, so every story is ticked. No-op for task-yaml runs
    (no PRD) and when nothing changed.
    """
    prd = session.read("prd.md")
    if not prd.strip():
        return
    sid = prd_session.story_id(task.description)
    if sid:
        updated = prd_session.mark_story_done(prd, sid)
        label = f"{sid} acceptance criteria checked off (done)"
    else:
        updated = prd_session.mark_all_stories_done(prd)
        label = "all stories' acceptance criteria checked off (holistic PASS)"
    if updated != prd:
        session.write("prd.md", updated)
        log.info("PRD: %s", label)


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


def _merge_guidance(corrections: str, guidance: str | None) -> str:
    g = (guidance or "").strip()
    if not g:
        return corrections
    block = f"## User guidance\n{g}"
    if block in corrections:
        return corrections
    return f"{corrections}\n\n{block}".strip() if corrections else block


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


def _make_plan(
    task: Task,
    ladder: Ladder,
    code_ctx: str,
    session: object = None,
    trace: object = None,
    iteration: int = 0,
    tier: int = 0,
    task_index: int = 0,
) -> str:
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
        trace=trace,
        iteration=iteration,
        tier=tier,
        task_index=task_index,
        role="plan",
    )
    record_exchange(prompt, plan, model=ladder.planner_model)
    return plan
