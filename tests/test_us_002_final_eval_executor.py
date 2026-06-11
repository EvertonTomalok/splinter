"""US-002: Final eval executor — kind dispatch and exit-code mapping."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from splinter.agents.final_eval import (
    run_all_final_evals,
    run_final_eval,
)
from splinter.configure import FinalEvalEntry
from splinter.enums import Decision, FinalEvalKind
from splinter.providers.base import ProviderResponse

# ── helpers ───────────────────────────────────────────────────────────────────

def _entry(kind: FinalEvalKind, **kw) -> FinalEvalEntry:
    return FinalEvalEntry(name="test", kind=kind, **kw)


def _task(description: str = "do thing", acceptance: str = "thing done"):
    from splinter.agents.runner import Task
    return Task(id="t1", description=description, acceptance=acceptance)


# ── command kind ──────────────────────────────────────────────────────────────

class TestCommandDispatch:
    def test_exit_zero_maps_to_pass(self) -> None:
        entry = _entry(FinalEvalKind.COMMAND, cmd="true")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="ok\n", stderr="")
            result = run_final_eval(entry)
        assert result.passed is True
        assert result.verdict is not None
        assert result.verdict.decision == Decision.PASS

    def test_nonzero_exit_maps_to_fail(self) -> None:
        entry = _entry(FinalEvalKind.COMMAND, cmd="false")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error msg")
            result = run_final_eval(entry)
        assert result.passed is False
        assert result.verdict is not None
        assert result.verdict.decision == Decision.RETRY

    def test_exit_code_2_also_fails(self) -> None:
        entry = _entry(FinalEvalKind.COMMAND, cmd="mypy .")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=2, stdout="", stderr="type errors")
            result = run_final_eval(entry)
        assert result.passed is False

    def test_output_captured_in_result(self) -> None:
        entry = _entry(FinalEvalKind.COMMAND, cmd="ruff check")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="all good", stderr="")
            result = run_final_eval(entry)
        assert "all good" in result.output

    def test_timeout_maps_to_fail(self) -> None:
        import subprocess as _subprocess
        entry = _entry(FinalEvalKind.COMMAND, cmd="sleep 9999")
        with patch("subprocess.run", side_effect=_subprocess.TimeoutExpired("sleep", 1)):
            result = run_final_eval(entry)
        assert result.passed is False
        assert "timed out" in result.output

    def test_command_not_found_maps_to_fail(self) -> None:
        entry = _entry(FinalEvalKind.COMMAND, cmd="nonexistent_cmd_xyz")
        with patch("subprocess.run", side_effect=FileNotFoundError("no such file")):
            result = run_final_eval(entry)
        assert result.passed is False
        assert "command not found" in result.output

    def test_corrections_empty_on_pass(self) -> None:
        entry = _entry(FinalEvalKind.COMMAND, cmd="pytest")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="passed", stderr="")
            result = run_final_eval(entry)
        assert result.verdict is not None
        assert result.verdict.corrections == ""

    def test_corrections_set_on_fail(self) -> None:
        entry = _entry(FinalEvalKind.COMMAND, cmd="mypy splinter")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error: bad type")
            result = run_final_eval(entry)
        assert result.verdict is not None
        assert result.verdict.corrections != ""


# ── skill (llm) kind ──────────────────────────────────────────────────────────

class TestSkillDispatch:
    def _mock_response(self, text: str) -> tuple[ProviderResponse, None]:
        return (
            ProviderResponse(text=text, tokens={"input": 10, "output": 5}, cost=0.01),
            None,
        )

    def test_pass_verdict_in_response(self) -> None:
        entry = _entry(FinalEvalKind.SKILL, skill="review")
        task = _task()
        with patch("splinter.skills.resolve_eval_skill", return_value=None), \
             patch("splinter.providers.dispatch.run_provider_session",
                   return_value=self._mock_response("VERDICT: PASS\nLooks good.")):
            result = run_final_eval(entry, task=task)
        assert result.passed is True
        assert result.verdict is not None
        assert result.verdict.decision == Decision.PASS

    def test_retry_verdict_in_response(self) -> None:
        entry = _entry(FinalEvalKind.SKILL, skill="review")
        task = _task()
        with patch("splinter.skills.resolve_eval_skill", return_value=None), \
             patch("splinter.providers.dispatch.run_provider_session",
                   return_value=self._mock_response("VERDICT: RETRY\nMissing test.")):
            result = run_final_eval(entry, task=task)
        assert result.passed is False
        assert result.verdict.decision == Decision.RETRY

    def test_cost_and_tokens_propagated(self) -> None:
        entry = _entry(FinalEvalKind.SKILL, skill="review")
        task = _task()
        with patch("splinter.skills.resolve_eval_skill", return_value=None), \
             patch("splinter.providers.dispatch.run_provider_session",
                   return_value=self._mock_response("VERDICT: PASS")):
            result = run_final_eval(entry, task=task)
        assert result.cost == pytest.approx(0.01)
        assert result.tokens == {"input": 10, "output": 5}

    def test_provider_exception_returns_fail(self) -> None:
        entry = _entry(FinalEvalKind.SKILL, skill="review")
        with patch("splinter.skills.resolve_eval_skill", return_value=None), \
             patch("splinter.providers.dispatch.run_provider_session",
                   side_effect=RuntimeError("API error")):
            result = run_final_eval(entry, task=_task())
        assert result.passed is False
        assert "API error" in result.output

    def test_missing_skill_still_runs(self) -> None:
        from splinter.skills import ResolvedSkill
        entry = _entry(FinalEvalKind.SKILL, skill="nonexistent")
        missing = ResolvedSkill(name="nonexistent", description="", body="", missing=True)
        with patch("splinter.skills.resolve_eval_skill", return_value=missing), \
             patch("splinter.providers.dispatch.run_provider_session",
                   return_value=self._mock_response("VERDICT: PASS")):
            result = run_final_eval(entry, task=_task())
        assert result.passed is True

    def test_model_from_entry_used(self) -> None:
        entry = _entry(FinalEvalKind.SKILL, skill="review", model="opus")
        calls: list[tuple] = []

        def fake_run(prompt, model, **kw):
            calls.append((prompt, model))
            return self._mock_response("VERDICT: PASS")

        with patch("splinter.skills.resolve_eval_skill", return_value=None), \
             patch("splinter.providers.dispatch.run_provider_session", side_effect=fake_run):
            run_final_eval(entry, task=_task())
        assert calls[0][1] == "opus"


# ── cursor kind ───────────────────────────────────────────────────────────────

class TestCursorDispatch:
    def _mock_cursor_result(self, text: str):
        from splinter.providers.cursor import CursorResult
        return CursorResult(text=text, raw={"returncode": 0})

    def test_pass_verdict(self) -> None:
        entry = _entry(FinalEvalKind.CURSOR)
        with patch("splinter.providers.cursor.run",
                   return_value=self._mock_cursor_result("VERDICT: PASS\nAll criteria met.")):
            result = run_final_eval(entry, task=_task())
        assert result.passed is True
        assert result.verdict.decision == Decision.PASS

    def test_retry_verdict(self) -> None:
        entry = _entry(FinalEvalKind.CURSOR)
        with patch("splinter.providers.cursor.run",
                   return_value=self._mock_cursor_result("VERDICT: RETRY\nTests missing.")):
            result = run_final_eval(entry, task=_task())
        assert result.passed is False
        assert result.verdict.decision == Decision.RETRY

    def test_cursor_exception_returns_fail(self) -> None:
        entry = _entry(FinalEvalKind.CURSOR)
        with patch("splinter.providers.cursor.run",
                   side_effect=RuntimeError("cursor exited 1: not found")):
            result = run_final_eval(entry, task=_task())
        assert result.passed is False
        assert "cursor exited 1" in result.output


# ── cursor provider unit tests ────────────────────────────────────────────────

class TestCursorProvider:
    def test_run_passes_prompt_to_subprocess(self) -> None:
        from splinter.procreg import CompletedProcess
        from splinter.providers.cursor import run as cursor_run

        completed = CompletedProcess(returncode=0, stdout="ok", stderr="")
        with patch("splinter.providers.cursor.run_subprocess", return_value=completed) as mock_sub:
            result = cursor_run("hello")
        assert result.text == "ok"
        cmd = mock_sub.call_args[0][0]
        assert "cursor" in cmd
        assert "hello" in cmd

    def test_nonzero_exit_raises(self) -> None:
        from splinter.procreg import CompletedProcess
        from splinter.providers.cursor import run as cursor_run

        completed = CompletedProcess(returncode=1, stdout="", stderr="err")
        with patch("splinter.providers.cursor.run_subprocess", return_value=completed):
            with pytest.raises(RuntimeError, match="cursor exited 1"):
                cursor_run("hello")

    def test_provider_class_wraps_run(self) -> None:
        from splinter.providers.cursor import CursorProvider, CursorResult

        provider = CursorProvider()
        fake = CursorResult(text="VERDICT: PASS", raw={"returncode": 0})
        with patch("splinter.providers.cursor.run", return_value=fake):
            resp = provider.run("prompt", "cursor")
        assert "PASS" in resp.text


# ── unknown kind raises ───────────────────────────────────────────────────────

class TestUnknownKind:
    def test_unknown_kind_raises_value_error(self) -> None:
        entry = FinalEvalEntry(name="bad", kind="unknown")  # type: ignore[arg-type]
        with pytest.raises((ValueError, KeyError)):
            run_final_eval(entry)


# ── run_all_final_evals ───────────────────────────────────────────────────────

class TestRunAllFinalEvals:
    def _pass_entry(self, name: str) -> FinalEvalEntry:
        return FinalEvalEntry(name=name, kind=FinalEvalKind.COMMAND, cmd="true")

    def _fail_entry(self, name: str) -> FinalEvalEntry:
        return FinalEvalEntry(name=name, kind=FinalEvalKind.COMMAND, cmd="false")

    def test_all_pass_returns_all_results(self) -> None:
        entries = [self._pass_entry("a"), self._pass_entry("b")]
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
            results = run_all_final_evals(entries)
        assert len(results) == 2
        assert all(r.passed for r in results)

    def test_fail_fast_stops_after_first_failure(self) -> None:
        entries = [self._fail_entry("a"), self._pass_entry("b")]
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="fail")
            results = run_all_final_evals(entries, fail_fast=True)
        assert len(results) == 1
        assert results[0].name == "a"

    def test_no_fail_fast_runs_all(self) -> None:
        entries = [self._fail_entry("a"), self._pass_entry("b")]
        side_effects = [
            MagicMock(returncode=1, stdout="", stderr="fail"),
            MagicMock(returncode=0, stdout="ok", stderr=""),
        ]
        with patch("subprocess.run", side_effect=side_effects):
            results = run_all_final_evals(entries, fail_fast=False)
        assert len(results) == 2

    def test_empty_entries_returns_empty(self) -> None:
        results = run_all_final_evals([])
        assert results == []
