from __future__ import annotations

from unittest.mock import patch

from splinter.agents.evaluator import Evaluator
from splinter.agents.runner import Task
from splinter.enums import Decision
from splinter.models.roster import Ladder, load_ladder
from splinter.strategies.base import EvalVerdict


def _ladder() -> Ladder:
    return load_ladder()


def _evaluator(ladder: Ladder | None = None) -> Evaluator:
    return Evaluator(ladder or _ladder())


# --- _parse_verdict: each of the 5 decisions --------------------------------


def test_parse_pass() -> None:
    v = Evaluator._parse_verdict(
        "VERDICT: PASS\nREASON: all good\nCORRECTIONS: none"
    )
    assert v.decision == Decision.PASS
    assert v.passed
    assert v.reason == "all good"


def test_parse_retry() -> None:
    v = Evaluator._parse_verdict(
        "VERDICT: RETRY\nREASON: missing import\nCORRECTIONS: add import os"
    )
    assert v.decision == Decision.RETRY
    assert not v.passed
    assert v.corrections == "add import os"


def test_parse_escalate() -> None:
    v = Evaluator._parse_verdict(
        "VERDICT: ESCALATE\nREASON: too complex\nCORRECTIONS: rewrite parser"
    )
    assert v.decision == Decision.ESCALATE
    assert not v.passed


def test_parse_jump_premium() -> None:
    v = Evaluator._parse_verdict(
        "VERDICT: JUMP_PREMIUM\nREASON: needs opus\nCORRECTIONS: full rewrite"
    )
    assert v.decision == Decision.JUMP_PREMIUM
    assert not v.passed


def test_parse_ask_user() -> None:
    v = Evaluator._parse_verdict(
        "VERDICT: ASK_USER\nREASON: ambiguous spec\nCORRECTIONS: clarify requirements"
    )
    assert v.decision == Decision.ASK_USER
    assert not v.passed


def test_parse_default_is_retry() -> None:
    v = Evaluator._parse_verdict("some random text with no verdict keyword")
    assert v.decision == Decision.RETRY


def test_parse_exactly_one_decision() -> None:
    v = Evaluator._parse_verdict(
        "VERDICT: JUMP_PREMIUM\nREASON: not a PASS situation\nCORRECTIONS: escalate"
    )
    assert v.decision == Decision.JUMP_PREMIUM


def test_parse_pass_with_escalate_in_body() -> None:
    v = Evaluator._parse_verdict(
        "VERDICT: PASS\nREASON: implementation is solid, no need to escalate\nCORRECTIONS: none"
    )
    assert v.decision == Decision.PASS


def test_parse_verdict_line_with_option_list_takes_first_token() -> None:
    # When the task IS the evaluator, the model often echoes the full option list
    # on the verdict line. The FIRST token (PASS) is the verdict — a priority scan
    # would wrongly latch onto JUMP_PREMIUM and escalate a passing run.
    v = Evaluator._parse_verdict(
        "VERDICT: PASS — one of PASS/RETRY/ESCALATE/JUMP_PREMIUM/ASK_USER\n"
        "REASON: all 5 criteria verified independently\nCORRECTIONS: none"
    )
    assert v.decision == Decision.PASS


def test_parse_no_verdict_line_does_not_escalate_from_prose() -> None:
    # No VERDICT line → default RETRY, never infer escalation from a keyword in prose.
    v = Evaluator._parse_verdict("I think we should JUMP_PREMIUM here, it's hard.")
    assert v.decision == Decision.RETRY


# --- eval_effort_for --------------------------------------------------------


def test_eval_effort_for_below_premium() -> None:
    ladder = _ladder()
    ev = _evaluator(ladder)
    assert ev.eval_effort_for(0) == ladder.eval_effort
    assert ev.eval_effort_for(2) == ladder.eval_effort


def test_eval_effort_for_at_premium() -> None:
    ev = _evaluator()
    assert ev.eval_effort_for(3) == "high"


def test_eval_effort_for_above_premium() -> None:
    ev = _evaluator()
    assert ev.eval_effort_for(4) == "high"


# --- next_action: tier-climb logic ------------------------------------------


def test_next_action_pass() -> None:
    ev = _evaluator()
    v = EvalVerdict(decision=Decision.PASS, reason="ok")
    action = ev.next_action(v, tier=1, max_tier=4)
    assert action.decision == Decision.PASS
    assert action.stop
    assert action.next_tier == 1


def test_next_action_retry_same_tier() -> None:
    ev = _evaluator()
    v = EvalVerdict(decision=Decision.RETRY, reason="fix it", corrections="add import")
    action = ev.next_action(v, tier=1, max_tier=4)
    assert action.decision == Decision.RETRY
    assert action.next_tier == 1
    assert not action.stop


def test_next_action_escalate_advances_tier() -> None:
    ev = _evaluator()
    v = EvalVerdict(decision=Decision.ESCALATE, reason="too hard", corrections="rewrite")
    action = ev.next_action(v, tier=1, max_tier=4)
    assert action.decision == Decision.ESCALATE
    assert action.next_tier == 2
    assert not action.stop


def test_next_action_escalate_at_max_tier_asks_user() -> None:
    ev = _evaluator()
    v = EvalVerdict(decision=Decision.ESCALATE, reason="too hard", corrections="rewrite")
    action = ev.next_action(v, tier=4, max_tier=4, cowabunga=False)
    assert action.decision == Decision.ASK_USER
    assert action.ask_user
    assert action.stop


def test_next_action_escalate_at_max_tier_cowabunga_stops() -> None:
    ev = _evaluator()
    v = EvalVerdict(decision=Decision.ESCALATE, reason="too hard", corrections="rewrite")
    action = ev.next_action(v, tier=4, max_tier=4, cowabunga=True)
    assert action.decision == Decision.ESCALATE
    assert action.stop
    assert not action.ask_user


def test_next_action_jump_premium() -> None:
    ev = _evaluator()
    v = EvalVerdict(decision=Decision.JUMP_PREMIUM, reason="needs premium")
    action = ev.next_action(v, tier=0, max_tier=4)
    assert action.decision == Decision.JUMP_PREMIUM
    assert action.next_tier == 3
    assert not action.stop


def test_next_action_jump_premium_already_at_premium() -> None:
    ev = _evaluator()
    v = EvalVerdict(decision=Decision.JUMP_PREMIUM, reason="needs premium")
    action = ev.next_action(v, tier=3, max_tier=4)
    assert action.next_tier == 3


def test_next_action_jump_premium_clamped_to_max_tier() -> None:
    ev = Evaluator(_ladder(), premium_tier=5)
    v = EvalVerdict(decision=Decision.JUMP_PREMIUM, reason="needs premium")
    action = ev.next_action(v, tier=1, max_tier=4)
    assert action.decision == Decision.JUMP_PREMIUM
    assert action.next_tier == 4


def test_next_action_ask_user_surfaces() -> None:
    ev = _evaluator()
    v = EvalVerdict(decision=Decision.ASK_USER, reason="ambiguous")
    action = ev.next_action(v, tier=1, max_tier=4, cowabunga=False)
    assert action.decision == Decision.ASK_USER
    assert action.ask_user
    assert action.stop


def test_next_action_ask_user_cowabunga_stops() -> None:
    ev = _evaluator()
    v = EvalVerdict(decision=Decision.ASK_USER, reason="ambiguous")
    action = ev.next_action(v, tier=1, max_tier=4, cowabunga=True)
    assert action.decision == Decision.ASK_USER
    assert action.stop
    assert not action.ask_user


# --- judge: cross-family (provider-agnostic) --------------------------------


def test_judge_calls_run_text_with_injected_model() -> None:
    ladder = _ladder()
    ev = Evaluator(ladder)
    task = Task(description="test task", acceptance="must work")

    with patch("splinter.agents.evaluator.run_text", return_value=(
        "VERDICT: PASS\nREASON: ok\nCORRECTIONS: none"
    )) as mock_run:
        verdict = ev.judge(
            task, "some output",
            eval_model="opencode-go/test-model",
            eval_effort="low",
        )

    assert verdict.passed
    mock_run.assert_called_once()
    call_args = mock_run.call_args
    assert call_args.args[1] == "opencode-go/test-model"
    assert call_args.kwargs["variant"] == "low"


def test_judge_uses_ladder_defaults() -> None:
    ladder = _ladder()
    ev = Evaluator(ladder)
    task = Task(description="test task", acceptance="must work")

    with patch("splinter.agents.evaluator.run_text", return_value=(
        "VERDICT: RETRY\nREASON: fix\nCORRECTIONS: do better"
    )) as mock_run:
        ev.judge(task, "some output")

    mock_run.assert_called_once()
    call_args = mock_run.call_args
    assert call_args.args[1] == ladder.eval_model
    assert call_args.kwargs["variant"] == ladder.eval_effort


# --- back-compat shim -------------------------------------------------------


def test_stages_parse_verdict_shim() -> None:
    from splinter.strategies.stages import _parse_verdict

    v = _parse_verdict("VERDICT: ESCALATE\nREASON: hard\nCORRECTIONS: rewrite")
    assert v.decision == Decision.ESCALATE
