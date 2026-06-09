"""Chain-of-Responsibility stages for one iteration of the orchestration loop.

A single iteration is Run -> Gate -> Eval. The gate is a short-circuit link: when
mechanical checks fail there is no point paying for an LLM eval, so ``GateStage``
returns ``False`` and the chain stops before ``EvalStage`` runs. The per-task
orchestration (retry vs. escalate, budget, replanning policy) lives in the
strategy that drives this chain, not in the stages themselves.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from splinter.agents.gate import run_gate
from splinter.agents.runner import RunResult, Task, run_task
from splinter.enums import Decision, Variant
from splinter.memory.knowledge import KnowledgeStore
from splinter.memory.session import Session
from splinter.models.roster import Ladder
from splinter.obs.trace import Trace, log_run
from splinter.providers.dispatch import run_text
from splinter.strategies.base import EvalVerdict
from splinter.templating import render, section

log = logging.getLogger("splinter.loop")

#: Tier at/above which the evaluator is run at maximum effort.
PREMIUM_TIER = 3
MAX_EVAL_EFFORT = Variant.HIGH


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
    corrections: str = ""
    eval_history: list[str] = field(default_factory=list)
    # produced by the stages
    run_result: RunResult | None = None
    gate_passed: bool = True
    gate_detail: str = ""
    verdict: EvalVerdict | None = None
    _loop_lines: list[str] = field(default_factory=list)

    def flush_loop(self) -> None:
        """Write the accumulated iteration log to ``loop.md`` (append-only)."""
        self.session.append("loop.md", "\n".join(self._loop_lines) + "\n\n")


class Stage(ABC):
    """A link in the iteration chain."""

    def __init__(self) -> None:
        self._next: Stage | None = None

    def set_next(self, nxt: Stage) -> Stage:
        self._next = nxt
        return nxt

    def handle(self, ctx: IterationContext) -> IterationContext:
        if self.process(ctx) and self._next is not None:
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


class RunStage(Stage):
    """Execute the task with the current tier's model, recording the run."""

    def process(self, ctx: IterationContext) -> bool:
        mode = "fixing" if ctx.corrections else "implementing"
        log.info("iter %d · T%d · %s (%s)", ctx.iteration, ctx.tier, mode,
                 "same session" if ctx.oc_session else "new session")
        result = run_task(
            ctx.task,
            ctx.plan,
            ctx.tier,
            ctx.ladder,
            effort_override=ctx.effort_override,
            localization=ctx.localization,
            corrections=ctx.corrections,
            opencode_session=ctx.oc_session,
        )
        log.info("iter %d · ran %s · tokens=%s · $%.4f",
                 ctx.iteration, result.model, result.tokens, result.cost)
        ctx.run_result = result
        log_run(ctx.trace, result, ctx.iteration)
        if ctx.oc_session is None and result.opencode_session:
            ctx.oc_session = result.opencode_session

        # Persist the runner's raw output so `analyze` can expand it per iteration.
        ctx.session.write(
            f"runs/iter-{ctx.iteration}.md",
            f"# Run output — iteration {ctx.iteration}\n"
            f"- model: {result.model} (tier {ctx.tier})\n"
            f"- tokens: {result.tokens}\n"
            f"- cost: ${result.cost:.4f}\n\n"
            f"{result.text}\n",
        )

        ctx._loop_lines += [
            f"## Iteration {ctx.iteration}",
            f"- model: {result.model} (tier {ctx.tier})",
            f"- session: {ctx.oc_session or 'new'}",
            f"- tokens: {result.tokens}",
            f"- cost: ${result.cost:.4f}",
        ]
        if ctx.corrections:
            ctx._loop_lines.append(f"- corrections applied: {ctx.corrections[:200]}")
        return True


class GateStage(Stage):
    """Run deterministic checks; short-circuit the chain on failure."""

    def process(self, ctx: IterationContext) -> bool:
        try:
            result = run_gate()
        except Exception:
            # Gate unavailable in this project — treat as a pass and let eval decide.
            return True

        ctx.gate_passed = result.passed
        if not result.passed:
            failed = [name for name, passed, _ in result.checks if not passed]
            ctx.gate_detail = f"failed: {', '.join(failed)}"
        log.info("iter %d · gate %s%s", ctx.iteration,
                 "PASS" if result.passed else "FAIL",
                 f" ({ctx.gate_detail})" if not result.passed else "")

        ctx._loop_lines.append(
            f"- gate: {'PASS' if result.passed else 'FAIL'} {ctx.gate_detail}".rstrip()
        )
        if not ctx.gate_passed:
            ctx._loop_lines.append(f"- verdict: RETRY (gate failed: {ctx.gate_detail})")
            ctx.flush_loop()
            return False
        return True


class EvalStage(Stage):
    """Judge the run output against acceptance criteria and persist the verdict."""

    def process(self, ctx: IterationContext) -> bool:
        assert ctx.run_result is not None  # guaranteed by RunStage running first

        eval_effort = ctx.ladder.eval_effort
        if ctx.tier >= PREMIUM_TIER:
            eval_effort = MAX_EVAL_EFFORT

        verdict = _evaluate(
            ctx.task,
            ctx.run_result.text,
            ctx.ladder.eval_model,
            eval_effort,
            previous_evals="\n".join(ctx.eval_history[-2:]),
            timeout=ctx.ladder.eval_timeout,
        )
        ctx.verdict = verdict
        log.info("iter %d · eval %s — %s", ctx.iteration, verdict.decision, verdict.reason)
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

        if not verdict.passed:
            # Persist corrections to session knowledge so an escalated model can read them.
            ctx.knowledge.write_note(
                f"corrections-iter-{ctx.iteration}", verdict.corrections
            )
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
    prompt = render(
        "eval",
        task_section=section("Task", task.description),
        acceptance_section=section("Acceptance Criteria", task.acceptance),
        output_section=section("Implementation Output", run_output),
        previous_evals_section=section("Previous Eval Feedback", previous_evals),
    )
    text = run_text(prompt, eval_model, variant=eval_effort, timeout=timeout)
    return _parse_verdict(text)


def _parse_verdict(text: str) -> EvalVerdict:
    text = text.strip()
    upper = text.upper()

    decision: str = Decision.RETRY
    for candidate in Decision:
        if candidate.value in upper:
            decision = candidate
            break

    reason = text
    corrections = ""
    for line in text.splitlines():
        prefix = line.upper().strip()
        if prefix.startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
        elif prefix.startswith("CORRECTIONS:"):
            corrections = line.split(":", 1)[1].strip()

    if not corrections and decision != Decision.PASS:
        corrections = reason

    return EvalVerdict(decision=decision, reason=reason, corrections=corrections, raw=text)
