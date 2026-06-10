"""Cross-family evaluator: judges a run and emits exactly one of 5 actions.

The evaluator is provider-agnostic — it delegates the LLM call to
:func:`splinter.providers.dispatch.run_text`, which routes by model id.
Tier-climb policy lives here so strategies share one implementation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from splinter.agents.runner import Task
from splinter.enums import Decision, Variant
from splinter.models.roster import Ladder
from splinter.providers.dispatch import run_provider_session
from splinter.skills import ResolvedSkill
from splinter.strategies.base import EvalVerdict
from splinter.templating import render, section

PREMIUM_TIER = 3
MAX_EVAL_EFFORT = Variant.HIGH

#: Matches a decision token as a whole word (``_`` counts as a word char, so
#: ``JUMP_PREMIUM`` stays intact). Used to pull the verdict out of the VERDICT line.
_DECISION_RE = re.compile(r"\b(JUMP_PREMIUM|ASK_USER|ESCALATE|RETRY|PASS)\b")


@dataclass(frozen=True)
class EvalAction:
    decision: Decision
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
        return self.ladder.eval_effort

    def judge(
        self,
        task: Task,
        run_output: str,
        *,
        eval_model: str | None = None,
        eval_effort: str | None = None,
        plan: str = "",
        previous_evals: str = "",
        eval_skill: ResolvedSkill | None = None,
        gate_passed: bool = True,
        gate_detail: str = "",
        session: str | None = None,
        timeout: int | None = None,
    ) -> EvalVerdict:
        """Judge the code generation against the task with a frontier LLM.

        The evaluator is the authority on quality — it reads the actual output and
        the acceptance criteria. The mechanical ``gate`` result is passed only as a
        secondary signal: a gate failure is normal and usually fixable, never a
        reason to skip this judgment. ``session`` resumes a prior eval conversation
        (same runner); the returned verdict carries the (possibly new) session id.
        """
        from splinter.templating import load_standards

        model = eval_model or self.ladder.eval_model
        effort = eval_effort or self.ladder.eval_effort
        skill_section_text = ""
        if eval_skill is not None:
            skill_section_text = section("Eval Skill", eval_skill.body)
        gate_text = "PASS" if gate_passed else f"FAIL — {gate_detail or 'mechanical checks failed'}"
        plan_section_text = section("Implementation Plan", plan) if plan else ""
        prompt = render(
            "eval",
            task_section=section("Task", task.description),
            acceptance_section=section("Acceptance Criteria", task.acceptance),
            plan_section=plan_section_text,
            output_section=section("Implementation Output", run_output),
            gate_section=section("Mechanical Gate Result", gate_text),
            previous_evals_section=section("Previous Eval Feedback", previous_evals),
            skill_section=skill_section_text,
            standards_section=section("Code Conventions", load_standards()),
        )
        response, sid = run_provider_session(
            prompt, model, variant=effort, session=session, timeout=timeout
        )
        verdict = self._parse_verdict(response.text)
        return replace(verdict, eval_session=sid, cost=response.cost, tokens=response.tokens)

    @staticmethod
    def _parse_verdict(text: str) -> EvalVerdict:
        text = text.strip()

        # Read the decision from the VERDICT line ONLY, and take the *first*
        # decision token on it — never a priority scan. The implementation output
        # (and the reason itself) routinely names other decisions in prose — e.g.
        # "no need to escalate", or literally the option list
        # "(PASS/RETRY/ESCALATE/JUMP_PREMIUM/ASK_USER)" when the task is the
        # evaluator itself — and a whole-text/priority scan would latch onto
        # JUMP_PREMIUM and escalate a run that actually passed. Absent a VERDICT
        # line we default to RETRY rather than inferring escalation from prose.
        decision: Decision = Decision.RETRY
        for line in text.splitlines():
            if line.upper().strip().startswith("VERDICT:"):
                value = line.split(":", 1)[1]
                m = _DECISION_RE.search(value.upper())
                if m is not None:
                    decision = Decision(m.group(1))
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
        """Map a verdict to exactly one of the 5 actions.

        The five are exhaustive and mutually exclusive:
        PASS, ASK_USER, JUMP_PREMIUM, ESCALATE, and RETRY (the default 5th,
        returned when no other branch matches). ESCALATE advances ``tier + 1``;
        when the ladder is exhausted (``tier >= max_tier``) and ``cowabunga`` is
        off, the action surfaces as ASK_USER instead.
        """
        if verdict.passed:
            return EvalAction(decision=Decision.PASS, next_tier=tier, stop=True)

        if verdict.decision == Decision.ASK_USER:
            if cowabunga:
                return EvalAction(decision=Decision.ASK_USER, next_tier=tier, stop=True)
            return EvalAction(decision=Decision.ASK_USER, next_tier=tier, ask_user=True, stop=True)

        if verdict.decision == Decision.JUMP_PREMIUM:
            target = min(max(tier, self.premium_tier), max_tier)
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
