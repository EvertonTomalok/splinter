"""Chain-of-Responsibility stages for one iteration of the orchestration loop.

A single iteration is Run -> Gate -> Eval. The gate is NOT a veto: it runs the
mechanical checks and records the result, but never short-circuits the chain. The
evaluator is a frontier LLM that judges the actual code generation against the
task — it is the authority on quality and always runs, with the gate result as a
secondary signal. The per-task orchestration (retry vs. escalate, budget,
replanning policy) lives in the strategy that drives this chain.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from splinter.agents.evaluator import Evaluator
from splinter.agents.gate import run_gate, task_languages
from splinter.agents.runner import RunResult, Task, run_task
from splinter.memory.knowledge import KnowledgeStore
from splinter.memory.session import Session
from splinter.models.roster import Ladder
from splinter.obs.agentic import agentic_scope, load_agentic_events, record_gate_marker
from splinter.obs.trace import Trace
from splinter.skills import ResolvedSkill
from splinter.strategies.base import EvalVerdict

log = logging.getLogger("splinter.loop")


@dataclass
class IterationContext:
    """Mutable state threaded through the stages of a single iteration."""

    task: Task
    plan: str
    tier: int
    iteration: int
    ladder: Ladder
    session: Session
    trace: Trace
    knowledge: KnowledgeStore
    localization: str = ""
    effort_override: str | None = None
    oc_session: str | None = None
    #: Evaluator's provider session — continued across same-runner retries.
    eval_session: str | None = None
    corrections: str = ""
    eval_history: list[str] = field(default_factory=list)
    task_index: int = 0
    skip_eval: bool = False
    # produced by the stages
    run_result: RunResult | None = None
    gate_passed: bool = True
    gate_detail: str = ""
    gate_output: str = ""
    verdict: EvalVerdict | None = None
    pause_at_stage: str | None = None
    _loop_lines: list[str] = field(default_factory=list)
    _structure_flushed: bool = field(default=False)

    def flush_structure(self) -> None:
        """Write the iteration header to ``loop.md`` (idempotent, at RUN entry)."""
        if self._structure_flushed:
            return
        self.session.append("loop.md", f"## Iteration {self.iteration}\n")
        self._structure_flushed = True

    def flush_loop(self) -> None:
        """Write the accumulated iteration log to ``loop.md`` (append-only)."""
        self.session.append("loop.md", "\n".join(self._loop_lines) + "\n\n")


class Stage(ABC):
    """A link in the iteration chain."""

    name: str = ""

    def __init__(self) -> None:
        self._next: Stage | None = None

    def set_next(self, nxt: Stage) -> Stage:
        self._next = nxt
        return nxt

    def handle(self, ctx: IterationContext) -> IterationContext:
        if self.process(ctx):
            if self._next is not None:
                from splinter import procreg

                if procreg.stop_requested():
                    ctx.pause_at_stage = self._next.name
                    return ctx
                return self._next.handle(ctx)
        return ctx

    @abstractmethod
    def process(self, ctx: IterationContext) -> bool:
        """Run this stage; return ``True`` to pass control to the next link."""
        ...


def build_chain(*stages: Stage) -> Stage:
    """Link ``stages`` head-to-tail and return the head."""
    for current, nxt in zip(stages, stages[1:]):
        current.set_next(nxt)
    return stages[0]


def build_chain_from(start: str | None, *stages: Stage) -> Stage:
    """Link stages head-to-tail; return the stage whose name == start (or head if None)."""
    for current, nxt in zip(stages, stages[1:]):
        current.set_next(nxt)
    if start is None:
        return stages[0]
    for stage in stages:
        if stage.name == start:
            return stage
    return stages[0]


def _render_actions(task_index: int, iteration: int, session: Session) -> str:
    """Render captured actions for this task/iteration as markdown.

    Returns empty string if no actions found.
    """
    events = load_agentic_events(session)
    actions = [
        e
        for e in events
        if e.task_index == task_index
        and e.iteration == iteration
        and e.kind in {"tool_use", "text"}
    ]
    if not actions:
        return ""
    lines = ["## Actions"]
    for e in actions:
        summary = e.extra.get("summary", "")
        if summary:
            lines.append(f"- {summary}")
    return "\n".join(lines) + "\n"


class RunStage(Stage):
    """Execute the task with the current tier's model, recording the run."""

    name = "run"

    def process(self, ctx: IterationContext) -> bool:
        from splinter.agents.runner import resolve_model, resolve_variant

        ctx.flush_structure()
        mode = "fixing" if ctx.corrections else "implementing"
        model_id, _ = resolve_model(ctx.tier, ctx.ladder)
        variant = resolve_variant(ctx.task, ctx.effort_override, ctx.ladder, ctx.tier)
        log.info(
            "iter %d · T%d · %s with %s (variant=%s, %s)",
            ctx.iteration,
            ctx.tier,
            mode,
            model_id,
            variant,
            "same session" if ctx.oc_session else "new session",
        )
        log.info("  ▸ task: %s", ctx.task.description.splitlines()[0])
        if ctx.corrections and "[USER GUIDANCE]" in ctx.corrections:
            guidance_line = next(
                (line for line in ctx.corrections.splitlines() if "[USER GUIDANCE]" in line), ""
            )
            if guidance_line:
                log.info("  ▸ %s", guidance_line.strip())
        with agentic_scope(ctx.session, "run", ctx.task_index, ctx.iteration):
            result = run_task(
                ctx.task,
                ctx.plan,
                ctx.tier,
                ctx.ladder,
                effort_override=ctx.effort_override,
                localization=ctx.localization,
                corrections=ctx.corrections,
                opencode_session=ctx.oc_session,
                trace=ctx.trace,
                iteration=ctx.iteration,
                task_index=ctx.task_index,
            )
        log.info(
            "iter %d · ran %s · tokens=%s · $%.4f",
            ctx.iteration,
            result.model,
            result.tokens,
            result.cost,
        )
        ctx.run_result = result
        if ctx.oc_session is None and result.opencode_session:
            ctx.oc_session = result.opencode_session

        # Persist the runner's raw output so `analyze` can expand it per iteration.
        actions_md = _render_actions(ctx.task_index, ctx.iteration, ctx.session)
        ctx.session.write(
            f"runs/iter-{ctx.iteration}.md",
            f"# Run output — iteration {ctx.iteration}\n"
            f"- model: {result.model} (tier {ctx.tier})\n"
            f"- tokens: {result.tokens}\n"
            f"- cost: ${result.cost:.4f}\n\n"
            f"{actions_md}"
            f"{result.text}\n",
        )

        ctx._loop_lines += [
            f"- model: {result.model} (tier {ctx.tier})",
            f"- session: {ctx.oc_session or 'new'}",
            f"- tokens: {result.tokens}",
            f"- cost: ${result.cost:.4f}",
        ]
        if ctx.corrections:
            ctx._loop_lines.append(f"- corrections applied: {ctx.corrections[:200]}")
        return True


class GateStage(Stage):
    """Run the deterministic checks and record the result — never a veto.

    The gate is a mechanical signal, not the judge. A failure is normal and gets
    fed back to the runner (so it understands what's wrong) and to the evaluator
    (as secondary context); the chain always continues to ``EvalStage``, which
    owns the quality verdict.
    """

    name = "gate"

    def process(self, ctx: IterationContext) -> bool:
        with agentic_scope(ctx.session, "gate", ctx.task_index, ctx.iteration):
            try:
                langs = task_languages(ctx.task)
                result = run_gate(session_dir=ctx.session.dir, languages=langs)
            except Exception:
                # Gate unavailable in this project — nothing to record, eval decides.
                return True

            ctx.gate_passed = result.passed
            if not result.passed:
                failed = [name for name, passed, _ in result.checks if not passed]
                ctx.gate_detail = f"failed: {', '.join(failed)}"
                # Keep the actual failing output so the runner can see what broke.
                ctx.gate_output = "\n\n".join(
                    f"### {name}\n{out}".rstrip()
                    for name, passed, out in result.checks
                    if not passed and out
                )
            log.info(
                "iter %d · gate %s%s",
                ctx.iteration,
                "PASS" if result.passed else "FAIL",
                f" ({ctx.gate_detail})" if not result.passed else "",
            )

            ctx._loop_lines.append(
                f"- gate: {'PASS' if result.passed else 'FAIL'} {ctx.gate_detail}".rstrip()
            )
            record_gate_marker()
            return True


class EvalStage(Stage):
    """Judge the run output against acceptance criteria and persist the verdict."""

    name = "eval"

    def __init__(
        self,
        evaluator: Evaluator | None = None,
        resolved_skill: ResolvedSkill | None = None,
    ) -> None:
        super().__init__()
        self._evaluator = evaluator
        self._resolved_skill = resolved_skill

    def process(self, ctx: IterationContext) -> bool:
        assert ctx.run_result is not None

        if ctx.skip_eval:
            from splinter.enums import Decision

            ctx.verdict = EvalVerdict(
                decision=Decision.PASS,
                reason="eval skipped by user",
                corrections="",
            )
            ctx._loop_lines.append("- verdict: PASS (eval skipped)")
            ctx.flush_loop()
            ctx.session.append(
                "eval.md",
                f"### Iter {ctx.iteration}: PASS (skipped)\n**Reason:** eval skipped by user\n\n",
            )
            return True

        evaluator = self._evaluator or Evaluator(ctx.ladder)
        eval_effort = evaluator.eval_effort_for(ctx.tier)

        with agentic_scope(ctx.session, "eval", ctx.task_index, ctx.iteration):
            verdict = evaluator.judge(
                ctx.task,
                ctx.run_result.text,
                eval_model=ctx.ladder.eval_model,
                eval_effort=eval_effort,
                plan=ctx.plan,
                previous_evals="\n".join(ctx.eval_history[-2:]),
                eval_skill=self._resolved_skill,
                gate_passed=ctx.gate_passed,
                gate_detail=ctx.gate_detail,
                session=ctx.eval_session,
                timeout=ctx.ladder.eval_timeout,
                trace=ctx.trace,
                iteration=ctx.iteration,
                tier=ctx.tier,
                task_index=ctx.task_index,
            )
        ctx.verdict = verdict
        ctx.eval_session = verdict.eval_session
        log.info(
            "iter %d · eval %s — %s · $%.4f",
            ctx.iteration,
            verdict.decision,
            verdict.reason,
            verdict.cost,
        )
        ctx.eval_history.append(
            f"Iter {ctx.iteration} [{verdict.decision}]: "
            f"{verdict.reason} | Corrections: {verdict.corrections}"
        )

        ctx._loop_lines.append(f"- verdict: {verdict.decision} — {verdict.reason}")
        if verdict.corrections:
            ctx._loop_lines.append(f"- corrections: {verdict.corrections}")
        ctx.flush_loop()

        ctx.session.append(
            "eval.md",
            f"### Iter {ctx.iteration}: {verdict.decision}\n"
            f"**Reason:** {verdict.reason}\n"
            f"**Corrections:** {verdict.corrections}\n"
            f"{verdict.raw}\n\n",
        )

        # Persist the full review to the knowledge store every iteration so the
        # runner (and any escalated model) can read the review history, not just
        # the latest corrections.
        review = (
            f"# Review · iter {ctx.iteration} (T{ctx.tier})\n"
            f"- decision: {verdict.decision}\n"
            f"- reason: {verdict.reason}\n"
        )
        if verdict.corrections:
            review += f"\n## Corrections\n{verdict.corrections}\n"
        ctx.knowledge.write_note(f"review-iter-{ctx.iteration}", review)
        return True


def _evaluate(
    task: Task,
    run_output: str,
    eval_model: str,
    eval_effort: str,
    *,
    previous_evals: str = "",
    timeout: int | None = None,
) -> EvalVerdict:
    from splinter.models.roster import Ladder

    ladder = Ladder(
        tiers=[],
        effort_map={},
        eval_model=eval_model,
        eval_effort=eval_effort,
        planner_model="",
        planner_effort="",
        localizer_recall_model="",
        localizer_recall_large_model="",
        localizer_precision_model="",
    )
    evaluator = Evaluator(ladder)
    return evaluator.judge(
        task,
        run_output,
        eval_model=eval_model,
        eval_effort=eval_effort,
        previous_evals=previous_evals,
        timeout=timeout,
    )


def _parse_verdict(text: str) -> EvalVerdict:
    return Evaluator._parse_verdict(text)
