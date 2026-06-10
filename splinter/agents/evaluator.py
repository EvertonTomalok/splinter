"""Cross-family evaluator: judges a run and emits exactly one of 5 actions.

The evaluator is provider-agnostic — it delegates the LLM call to
:func:`splinter.providers.dispatch.run_text`, which routes by model id.
Tier-climb policy lives here so strategies share one implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

from splinter.agents.runner import Task
from splinter.enums import Decision, Variant
from splinter.models.roster import Ladder
from splinter.providers.dispatch import run_text
from splinter.strategies.base import EvalVerdict
from splinter.templating import render, section

PREMIUM_TIER = 3
MAX_EVAL_EFFORT = Variant.HIGH

_DECISION_PRIORITY: tuple[Decision, ...] = (
    Decision.JUMP_PREMIUM,
    Decision.ASK_USER,
    Decision.ESCALATE,
    Decision.PASS,
)


@dataclass(frozen=True)
class EvalAction:
    decision: str
    next_tier: int
    ask_user: bool = False
    stop: bool = False


class Evaluator:
    """Standalone cross-family evaluator with tier-climb logic."""

    def __init__(
        self,
        ladder: Ladder,
        *,
        premium_tier: int = PREMIUM_TIER,
        max_eval_effort: str = MAX_EVAL_EFFORT,
    ) -> None:
        self.ladder = ladder
        self.premium_tier = premium_tier
        self.max_eval_effort = max_eval_effort

    def eval_effort_for(self, tier: int) -> str:
        if tier >= self.premium_tier:
            return self.max_eval_effort
        return self.ladder.eval_effort

    def judge(
        self,
        task: Task,
        run_output: str,
        *,
        eval_model: str | None = None,
        eval_effort: str | None = None,
        previous_evals: str = "",
        timeout: int | None = None,
    ) -> EvalVerdict:
        model = eval_model or self.ladder.eval_model
        effort = eval_effort or self.ladder.eval_effort
        prompt = render(
            "eval",
            task_section=section("Task", task.description),
            acceptance_section=section("Acceptance Criteria", task.acceptance),
            output_section=section("Implementation Output", run_output),
            previous_evals_section=section("Previous Eval Feedback", previous_evals),
        )
        text = run_text(prompt, model, variant=effort, timeout=timeout)
        return self._parse_verdict(text)

    @staticmethod
    def _parse_verdict(text: str) -> EvalVerdict:
        text = text.strip()
        upper = text.upper()

        decision: str = Decision.RETRY
        for candidate in _DECISION_PRIORITY:
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

    def next_action(
        self,
        verdict: EvalVerdict,
        tier: int,
        *,
        max_tier: int,
        cowabunga: bool = False,
    ) -> EvalAction:
        if verdict.passed:
            return EvalAction(decision=Decision.PASS, next_tier=tier, stop=True)

        if verdict.decision == Decision.ASK_USER:
            if cowabunga:
                return EvalAction(decision=Decision.ASK_USER, next_tier=tier, stop=True)
            return EvalAction(
                decision=Decision.ASK_USER, next_tier=tier, ask_user=True, stop=True
            )

        if verdict.decision == Decision.JUMP_PREMIUM:
            target = max(tier, self.premium_tier)
            return EvalAction(decision=Decision.JUMP_PREMIUM, next_tier=target)

        if verdict.decision == Decision.ESCALATE:
            if tier >= max_tier:
                if cowabunga:
                    return EvalAction(decision=Decision.ESCALATE, next_tier=tier, stop=True)
                return EvalAction(
                    decision=Decision.ASK_USER, next_tier=tier, ask_user=True, stop=True
                )
            return EvalAction(decision=Decision.ESCALATE, next_tier=tier + 1)

        return EvalAction(decision=Decision.RETRY, next_tier=tier)
