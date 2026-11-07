"""Acceptance mode parsing and check_acceptance routing (all I/O mocked)."""

from __future__ import annotations

from pathlib import Path

import pytest

from splinter.agents import acceptance
from splinter.agents.acceptance import (
    AcceptanceMode,
    AcceptanceResult,
    check_acceptance,
    parse_acceptance_mode,
)
from splinter.agents.gate import GateResult
from splinter.enums import Decision
from splinter.models.roster import Ladder


def _ladder() -> Ladder:
    return Ladder(
        tiers=[],
        effort_map={},
        eval_model="eval-model",
        eval_effort="high",
        planner_model="planner-model",
        planner_effort="medium",
        localizer_recall_model="recall-model",
        localizer_recall_large_model="recall-large-model",
        localizer_precision_model="precision-model",
    )


def _mock_gate(
    monkeypatch: pytest.MonkeyPatch,
    passed: bool = True,
    output: str = "ruff failed hard",
) -> None:
    def fake_run_gate(
        project_dir: str = ".",
        session_dir: str | Path | None = None,
        languages: set[str] | None = None,
    ) -> GateResult:
        checks = [] if passed else [("ruff", False, output)]
        return GateResult(passed=passed, checks=checks)

    monkeypatch.setattr(acceptance, "run_gate", fake_run_gate)


def _mock_run_text(
    monkeypatch: pytest.MonkeyPatch,
    response: str | Exception = "",
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def fake_run_text(prompt: str, model: str, **kwargs: object) -> str:
        calls.append({"prompt": prompt, "model": model, **kwargs})
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(acceptance, "run_text", fake_run_text)
    return calls


class TestParseMode:
    def test_blank_defaults_to_gate_eval(self) -> None:
        assert parse_acceptance_mode(None) is AcceptanceMode.GATE_EVAL
        assert parse_acceptance_mode("") is AcceptanceMode.GATE_EVAL
        assert parse_acceptance_mode("   ") is AcceptanceMode.GATE_EVAL

    def test_exact_and_underscore_aliases(self) -> None:
        assert parse_acceptance_mode("gate") is AcceptanceMode.GATE
        assert parse_acceptance_mode("gate+eval") is AcceptanceMode.GATE_EVAL
        assert parse_acceptance_mode("gate_eval") is AcceptanceMode.GATE_EVAL
        assert parse_acceptance_mode("gate+orchestrator") is AcceptanceMode.GATE_ORCHESTRATOR
        assert parse_acceptance_mode("gate_orchestrator") is AcceptanceMode.GATE_ORCHESTRATOR

    def test_case_insensitive_and_whitespace(self) -> None:
        assert parse_acceptance_mode("GATE") is AcceptanceMode.GATE
        assert parse_acceptance_mode("  Gate+Eval ") is AcceptanceMode.GATE_EVAL
        assert parse_acceptance_mode("GATE_ORCHESTRATOR") is AcceptanceMode.GATE_ORCHESTRATOR

    def test_invalid_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            parse_acceptance_mode("bogus")
        with pytest.raises(ValueError):
            parse_acceptance_mode("gate+magic")


class TestGateMode:
    def test_gate_pass_never_calls_llm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_gate(monkeypatch)
        calls = _mock_run_text(monkeypatch, "VERDICT: PASS")
        result = check_acceptance(
            goal="ship it",
            acceptance=["must add the endpoint"],
            mode=AcceptanceMode.GATE,
        )
        assert result == AcceptanceResult(
            passed=True, mode=AcceptanceMode.GATE, gate_passed=True
        )
        assert result.verdict is None
        assert calls == []

    def test_gate_fail_reports_truncated_detail(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _mock_gate(monkeypatch, passed=False, output="boom " * 200)
        calls = _mock_run_text(monkeypatch, "VERDICT: PASS")
        result = check_acceptance(
            goal="ship it",
            acceptance=[],
            mode=AcceptanceMode.GATE,
        )
        assert result.passed is False
        assert result.gate_passed is False
        assert result.detail.startswith("ruff:")
        assert len(result.detail) <= 400
        assert calls == []


class TestGateEval:
    def test_verdict_pass(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_gate(monkeypatch)
        calls = _mock_run_text(monkeypatch, "all lines satisfied\nVERDICT: PASS")
        result = check_acceptance(
            goal="ship it",
            acceptance=["must add the endpoint"],
            mode=AcceptanceMode.GATE_EVAL,
            diff="+def endpoint():",
        )
        assert result.passed is True
        assert result.gate_passed is True
        assert result.detail == ""
        assert result.verdict is not None
        assert result.verdict.decision == Decision.PASS
        assert len(calls) == 1

    def test_prompt_carries_goal_acceptance_and_diff(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _mock_gate(monkeypatch)
        calls = _mock_run_text(monkeypatch, "VERDICT: PASS")
        check_acceptance(
            goal="ship the feature",
            acceptance=["must add the endpoint", "must add tests"],
            mode=AcceptanceMode.GATE_EVAL,
            diff="+def endpoint(): pass",
        )
        prompt = str(calls[0]["prompt"])
        assert "ship the feature" in prompt
        assert "must add the endpoint" in prompt
        assert "must add tests" in prompt
        assert "+def endpoint(): pass" in prompt

    def test_verdict_fail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_gate(monkeypatch)
        _mock_run_text(monkeypatch, "missing tests for the endpoint\nVERDICT: FAIL")
        result = check_acceptance(
            goal="ship it",
            acceptance=["must add tests"],
            mode=AcceptanceMode.GATE_EVAL,
            diff="+def endpoint():",
        )
        assert result.passed is False
        assert result.gate_passed is True
        assert result.verdict is not None
        assert result.verdict.decision == Decision.RETRY
        assert "missing tests" in result.detail

    def test_malformed_verdict_token_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _mock_gate(monkeypatch)
        _mock_run_text(monkeypatch, "VERDICT: MAYBE")
        result = check_acceptance(
            goal="ship it",
            acceptance=["must add the endpoint"],
            mode=AcceptanceMode.GATE_EVAL,
            diff="+x",
        )
        assert result.passed is False
        assert result.verdict is not None
        assert result.verdict.decision == Decision.RETRY

    def test_missing_verdict_line_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _mock_gate(monkeypatch)
        _mock_run_text(monkeypatch, "all good, no explicit verdict")
        result = check_acceptance(
            goal="ship it",
            acceptance=["must add the endpoint"],
            mode=AcceptanceMode.GATE_EVAL,
            diff="+x",
        )
        assert result.passed is False
        assert result.verdict is not None
        assert result.verdict.decision == Decision.RETRY
        assert "VERDICT" in result.detail

    def test_gate_fail_short_circuits_without_llm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _mock_gate(monkeypatch, passed=False)
        calls = _mock_run_text(monkeypatch, "VERDICT: PASS")
        result = check_acceptance(
            goal="ship it",
            acceptance=["must add the endpoint"],
            mode=AcceptanceMode.GATE_EVAL,
        )
        assert result.passed is False
        assert result.gate_passed is False
        assert result.verdict is None
        assert calls == []

    def test_run_text_exception_fails_not_crashes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _mock_gate(monkeypatch)
        _mock_run_text(monkeypatch, RuntimeError("provider down"))
        result = check_acceptance(
            goal="ship it",
            acceptance=["must add the endpoint"],
            mode=AcceptanceMode.GATE_EVAL,
        )
        assert result.passed is False
        assert result.gate_passed is True
        assert "provider down" in result.detail
        assert result.verdict is None

    def test_routes_eval_model_role_and_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _mock_gate(monkeypatch)
        calls = _mock_run_text(monkeypatch, "VERDICT: PASS")
        ladder = _ladder()
        check_acceptance(
            goal="g",
            acceptance=["a"],
            mode=AcceptanceMode.GATE_EVAL,
            ladder=ladder,
            task_index=3,
        )
        call = calls[0]
        assert call["model"] == "eval-model"
        assert call["variant"] == "low"
        assert call["role"] == "eval"
        assert call["timeout"] == ladder.eval_timeout
        assert call["output_format"] == "json"
        assert call["task_index"] == 3

    def test_defaults_without_ladder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_gate(monkeypatch)
        calls = _mock_run_text(monkeypatch, "VERDICT: PASS")
        check_acceptance(goal="g", acceptance=["a"], mode=AcceptanceMode.GATE_EVAL)
        call = calls[0]
        assert call["model"] == "sonnet"
        assert call["variant"] == "low"
        assert call["timeout"] is None


class TestGateOrchestrator:
    def test_routes_planner_model_and_orchestrator_role(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _mock_gate(monkeypatch)
        calls = _mock_run_text(monkeypatch, "VERDICT: PASS")
        ladder = _ladder()
        ladder.step_pools = {"planner": ["planner-model@medium", "sonnet@low"]}
        result = check_acceptance(
            goal="g",
            acceptance=["a"],
            mode=AcceptanceMode.GATE_ORCHESTRATOR,
            ladder=ladder,
        )
        call = calls[0]
        assert call["model"] == "planner-model"
        assert call["variant"] == "medium"
        assert call["role"] == "orchestrator"
        assert call["pool"] == ladder.step_pool("planner")
        assert call["timeout"] == ladder.planner_timeout
        assert result.passed is True

    def test_defaults_without_ladder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_gate(monkeypatch)
        calls = _mock_run_text(monkeypatch, "VERDICT: FAIL")
        result = check_acceptance(
            goal="g",
            acceptance=["a"],
            mode=AcceptanceMode.GATE_ORCHESTRATOR,
        )
        call = calls[0]
        assert call["model"] == "sonnet"
        assert call["variant"] == "low"
        assert call["timeout"] is None
        assert result.passed is False
        assert result.verdict is not None
        assert result.verdict.decision == Decision.RETRY
