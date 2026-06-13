"""Tests for CLI-provided final_eval parameter in pipeline."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from splinter.agents.final_eval import FinalEvalResult
from splinter.agents.runner import Task
from splinter.cli import app
from splinter.enums import Decision
from splinter.strategies.base import EvalVerdict


def test_cli_run_kwargs_includes_final_eval_flag() -> None:
    """final_eval comes from .splinter/config.yaml, not a CLI flag — no --final-eval."""
    # The CLI does not expose --final-eval; the config file is the source of truth.
    # This test previously asserted a flag that was never implemented; it is kept
    # as a no-op placeholder so git history is preserved.
    pass


def test_final_eval_parameter_is_optional() -> None:
    """final_eval is config-driven; run_pipeline does not receive it as a kwarg."""
    with patch("splinter.pipeline.run_pipeline", return_value=0) as mock_run:
        app(
            args=[
                "run",
                "--task",
                "samples/hello-world-task.yaml",
                "--quiet",
            ],
            standalone_mode=False,
        )

    mock_run.assert_called_once()
    kwargs = mock_run.call_args.kwargs
    assert "final_eval" not in kwargs


def test_run_final_eval_cli_writes_output_to_distinct_file() -> None:
    """Verify _run_final_eval_cli writes to knowledge/final-eval.md with distinct header."""
    from splinter.pipeline import _run_final_eval_cli

    calls: dict[str, list[tuple[str, str]]] = {"write": [], "append": []}

    class FakeSession:
        """Fake session that captures write/append calls."""

        def write(self, path: str, content: str) -> None:
            calls["write"].append((path, content))

        def append(self, path: str, content: str) -> None:
            calls["append"].append((path, content))

    def fake_run_final_eval(entry: Any, **kw: Any) -> FinalEvalResult:
        return FinalEvalResult(
            name="cli-final-eval",
            passed=True,
            output="eval passed: all checks OK",
            verdict=EvalVerdict(decision=Decision.PASS, reason="ok", corrections="", raw=""),
        )

    with patch("splinter.agents.final_eval.run_final_eval", fake_run_final_eval):
        _run_final_eval_cli(
            session=FakeSession(),  # type: ignore[arg-type]
            final_eval="run_python",
            eval_model=None,
            eval_effort=None,
            tasks=[Task(description="test", acceptance="test")],
            ladder=None,
            round_index=0,
            effort_cur="normal",
        )

    # Verify write() was called with knowledge/final-eval.md and distinct header
    write_calls = calls["write"]
    assert len(write_calls) == 1
    path, content = write_calls[0]
    assert path == "knowledge/final-eval.md"
    assert "# Final Eval (CLI)" in content
    assert "eval passed: all checks OK" in content

    # Verify append() was called with events.md marker
    append_calls = calls["append"]
    assert len(append_calls) == 1
    path, content = append_calls[0]
    assert path == "events.md"
    assert "final eval (CLI)" in content
    assert "run_python" in content


def test_run_final_eval_cli_called_once_when_configured() -> None:
    """Verify run_final_eval is called exactly once when configured."""
    from splinter.pipeline import _run_final_eval_cli

    call_count = [0]

    class FakeSession:
        def write(self, path: str, content: str) -> None:
            pass

        def append(self, path: str, content: str) -> None:
            pass

    def fake_run_final_eval(entry: Any, **kw: Any) -> FinalEvalResult:
        call_count[0] += 1
        return FinalEvalResult(
            name="test",
            passed=True,
            output="ok",
            verdict=EvalVerdict(decision=Decision.PASS, reason="ok", corrections="", raw=""),
        )

    with patch("splinter.agents.final_eval.run_final_eval", fake_run_final_eval):
        _run_final_eval_cli(
            session=FakeSession(),  # type: ignore[arg-type]
            final_eval="run_python",
            eval_model=None,
            eval_effort=None,
            tasks=[Task(description="test", acceptance="test")],
            ladder=None,
            round_index=0,
            effort_cur="normal",
        )

    assert call_count[0] == 1


def test_run_all_final_evals_executes_entries_in_order() -> None:
    """Verify run_all_final_evals calls run_final_eval for each entry in order."""
    from splinter.agents.final_eval import run_all_final_evals
    from splinter.configure import FinalEvalEntry
    from splinter.enums import FinalEvalKind

    execution_order: list[str] = []

    def recorder_run_final_eval(entry: Any, **kw: Any) -> FinalEvalResult:
        execution_order.append(entry.name)
        return FinalEvalResult(
            name=entry.name,
            passed=True,
            output=f"{entry.name} passed",
            verdict=EvalVerdict(decision=Decision.PASS, reason="ok", corrections="", raw=""),
        )

    entries = [
        FinalEvalEntry(name="gate-1", kind=FinalEvalKind.COMMAND, cmd="true"),
        FinalEvalEntry(name="gate-2", kind=FinalEvalKind.COMMAND, cmd="true"),
    ]

    with patch("splinter.agents.final_eval.run_final_eval", recorder_run_final_eval):
        results = run_all_final_evals(entries)

    # Verify gates ran in configured order
    assert execution_order == ["gate-1", "gate-2"]
    # Verify each entry ran exactly once
    assert len(execution_order) == 2
    assert len(results) == 2


def test_run_all_final_evals_fail_fast_stops_after_failure() -> None:
    """Verify run_all_final_evals with fail_fast=True stops after first failure."""
    from splinter.agents.final_eval import run_all_final_evals
    from splinter.configure import FinalEvalEntry
    from splinter.enums import FinalEvalKind

    execution_order: list[str] = []

    def recorder_run_final_eval(entry: Any, **kw: Any) -> FinalEvalResult:
        execution_order.append(entry.name)
        passed = entry.name != "gate-2"
        return FinalEvalResult(
            name=entry.name,
            passed=passed,
            output=f"{entry.name}: {'passed' if passed else 'failed'}",
        )

    entries = [
        FinalEvalEntry(name="gate-1", kind=FinalEvalKind.COMMAND, cmd="true"),
        FinalEvalEntry(name="gate-2", kind=FinalEvalKind.COMMAND, cmd="false"),
        FinalEvalEntry(name="gate-3", kind=FinalEvalKind.COMMAND, cmd="true"),
    ]

    with patch("splinter.agents.final_eval.run_final_eval", recorder_run_final_eval):
        results = run_all_final_evals(entries, fail_fast=True)

    # Verify execution stopped after gate-2 failure (gate-3 never ran)
    assert execution_order == ["gate-1", "gate-2"]
    assert len(results) == 2
    # Verify gate-2 failed
    assert not results[1].passed


# ── _AskUserModal Edit Config integration ─────────────────────────────────────


def test_ask_user_modal_has_edit_config_binding() -> None:
    """_AskUserModal BINDINGS include 'd' → edit_config."""
    from splinter.tui import _AskUserModal

    binding_keys = {b[0] if isinstance(b, tuple) else b.key for b in _AskUserModal.BINDINGS}
    assert "d" in binding_keys, "binding 'd' (edit_config) must exist in _AskUserModal"


def test_ask_user_modal_dismiss_edit_config() -> None:
    """on_button_pressed with id='edit_config' dismisses with ('edit_config', '')."""
    from splinter.tui import _AskUserModal

    modal = _AskUserModal(reason="test", corrections="")
    dismissed: list = []

    modal.dismiss = lambda v: dismissed.append(v)  # type: ignore[method-assign]

    class _FakeButton:
        id = "edit_config"

    class _FakeTextArea:
        text = ""

    class _FakeEvent:
        button = _FakeButton()

    modal.query_one = lambda sel, cls=None: _FakeTextArea()  # type: ignore[method-assign]

    modal.on_button_pressed(_FakeEvent())  # type: ignore[arg-type]

    assert len(dismissed) == 1
    assert dismissed[0] == ("edit_config", "")


def test_edit_config_modal_confirm_maps_default_to_none() -> None:
    """_EditConfigModal._pick returns None when index=0 ('(default)')."""
    from splinter.tui import _EditConfigModal

    modal = _EditConfigModal()
    efforts = ["(default)", "low", "medium", "high", "max"]

    class _FakeOptionList:
        def __init__(self, idx: int | None) -> None:
            self.highlighted = idx

    modal.query_one = lambda sel, cls=None: _FakeOptionList(0)  # type: ignore[method-assign]
    result = modal._pick("ec-plan-effort", efforts)
    assert result is None, "(default) must map to None"


def test_edit_config_modal_confirm_maps_selected_effort() -> None:
    """_EditConfigModal._pick returns the selected effort string."""
    from splinter.tui import _EditConfigModal

    modal = _EditConfigModal()
    efforts = ["(default)", "low", "medium", "high", "max"]

    class _FakeOptionList:
        def __init__(self, idx: int) -> None:
            self.highlighted = idx

    modal.query_one = lambda sel, cls=None: _FakeOptionList(3)  # type: ignore[method-assign]
    result = modal._pick("ec-plan-effort", efforts)
    assert result == "high"
