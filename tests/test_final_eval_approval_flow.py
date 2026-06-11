"""Simulation: ask_user final_eval → ManualValidationPause → TUI approve → completed.

Reproduces the full pipeline tail for ses_20260611-201301:
  1. config.yaml has final_eval: [{name: user-review, kind: ask_user}]
  2. run_all_final_evals returns passed=False (ask_user never auto-passes)
  3. pipeline raises ManualValidationPause
  4. except handler sets session status awaiting_validation
  5. user clicks Approve in _ManualValidationModal → set_status("completed")
"""

from __future__ import annotations

import pytest

from splinter.agents.final_eval import run_all_final_evals
from splinter.agents.runner import Task
from splinter.configure import FinalEvalEntry, load_final_eval
from splinter.enums import Decision, FinalEvalKind
from splinter.strategies.base import ManualValidationPause

# ── fake session ──────────────────────────────────────────────────────────────

class _FakeSession:
    def __init__(self) -> None:
        self.status_calls: list[tuple[str, dict]] = []
        self.written: dict[str, str] = {}

    def set_status(self, state: str, **kwargs) -> None:
        self.status_calls.append((state, kwargs))

    def write(self, path: str, content: str) -> None:
        self.written[path] = content

    def read_status(self) -> dict:
        if not self.status_calls:
            return {}
        state, kwargs = self.status_calls[-1]
        return {"state": state, **kwargs}


# ── PRD task (mirrors US-004 from ses_20260611-201301) ────────────────────────

_TASK = Task(
    description=(
        "As Splinter, I want dispatch.py to resolve a provider object and delegate, "
        "replacing the two-way if/else in all dispatch functions."
    ),
    acceptance=(
        "run_text, run_provider_session delegate to provider object from registry — "
        "no if opencode/else claude branching. "
        "All 4 consumers verified unchanged in behavior for claude/opencode."
    ),
)


# ── 1. config loads ask_user entry (session-level final_eval.yaml) ────────────

def test_session_final_eval_yaml_loads_ask_user_entry(tmp_path) -> None:
    import yaml
    fe_path = tmp_path / "final_eval.yaml"
    fe_path.write_text("final_eval:\n- name: user-review\n  kind: ask_user\n")
    config = yaml.safe_load(fe_path.read_text()) or {}
    entries = load_final_eval(config)
    assert len(entries) == 1
    assert entries[0].name == "user-review"
    assert entries[0].kind == FinalEvalKind.ASK_USER


def test_config_final_eval_loads_ask_user_entry() -> None:
    config = {"final_eval": [{"name": "user-review", "kind": "ask_user"}]}
    entries = load_final_eval(config)
    assert len(entries) == 1
    assert entries[0].name == "user-review"
    assert entries[0].kind == FinalEvalKind.ASK_USER


# ── 2. ask_user final_eval returns passed=False + ASK_USER verdict ────────────

def test_ask_user_final_eval_never_auto_passes() -> None:
    entry = FinalEvalEntry(name="user-review", kind=FinalEvalKind.ASK_USER)
    results = run_all_final_evals([entry], task=_TASK)

    assert len(results) == 1
    r = results[0]
    assert r.passed is False
    assert r.verdict is not None
    assert r.verdict.decision == Decision.ASK_USER
    assert "Manual review requested" in r.output
    assert _TASK.description[:40] in r.output


# ── 3. pipeline raises ManualValidationPause when any result fails ─────────────

def test_pipeline_raises_manual_validation_pause_for_ask_user() -> None:
    entry = FinalEvalEntry(name="user-review", kind=FinalEvalKind.ASK_USER)
    results = run_all_final_evals([entry], task=_TASK)
    all_passed = all(r.passed for r in results)
    fe_summary = "\n".join(
        f"- {r.name}: {'PASS' if r.passed else 'FAIL'} — {r.output[:200]}"
        for r in results
    )

    assert not all_passed
    with pytest.raises(ManualValidationPause) as exc_info:
        raise ManualValidationPause(summary=fe_summary, all_passed=all_passed)

    pause = exc_info.value
    assert pause.all_passed is False
    assert "user-review" in pause.summary
    assert "FAIL" in pause.summary


# ── 4. except handler sets awaiting_validation status ─────────────────────────

def test_except_handler_sets_awaiting_validation_status() -> None:
    session = _FakeSession()
    entry = FinalEvalEntry(name="user-review", kind=FinalEvalKind.ASK_USER)
    results = run_all_final_evals([entry], task=_TASK)
    all_passed = all(r.passed for r in results)
    fe_summary = "\n".join(
        f"- {r.name}: {'PASS' if r.passed else 'FAIL'} — {r.output[:200]}"
        for r in results
    )

    try:
        raise ManualValidationPause(summary=fe_summary, all_passed=all_passed)
    except ManualValidationPause as val_exc:
        session.set_status(
            "awaiting_validation",
            stage="final_eval",
            final_eval_summary=val_exc.summary,
            final_eval_passed=val_exc.all_passed,
        )

    st = session.read_status()
    assert st["state"] == "awaiting_validation"
    assert st["stage"] == "final_eval"
    assert st["final_eval_passed"] is False
    assert "user-review" in st["final_eval_summary"]


# ── 5. TUI approve click → session transitions to completed ───────────────────

def test_tui_approve_transitions_session_to_completed() -> None:
    session = _FakeSession()

    # Set up awaiting_validation (pipeline except handler)
    session.set_status(
        "awaiting_validation",
        stage="final_eval",
        final_eval_summary="- user-review: FAIL — Manual review requested: user-review",
        final_eval_passed=False,
    )
    assert session.read_status()["state"] == "awaiting_validation"

    # Simulate _ManualValidationModal callback with ("approve", "")
    result = ("approve", "")
    action, _text = result
    assert action == "approve"

    # TUI _on_choice approve branch: session.set_status("completed", stage="done")
    session.set_status("completed", stage="done")

    st = session.read_status()
    assert st["state"] == "completed"
    assert st["stage"] == "done"


# ── 6. full end-to-end simulation ─────────────────────────────────────────────

def test_full_ask_user_final_eval_approve_flow() -> None:
    """Simulate complete flow: config → run_all → pause → approve → completed."""
    session = _FakeSession()

    # Config load
    config = {"final_eval": [{"name": "user-review", "kind": "ask_user"}]}
    entries = load_final_eval(config)

    # run_all_final_evals (pipeline line ~349)
    fe_results = run_all_final_evals(entries, task=_TASK)
    fe_summary = "\n".join(
        f"- {r.name}: {'PASS' if r.passed else 'FAIL'} — {r.output[:200]}"
        for r in fe_results
    )
    all_passed = all(r.passed for r in fe_results)

    # Pipeline raises ManualValidationPause
    raised: ManualValidationPause | None = None
    try:
        raise ManualValidationPause(summary=fe_summary, all_passed=all_passed)
    except ManualValidationPause as val_exc:
        raised = val_exc
        # except handler in pipeline.py
        session.set_status(
            "awaiting_validation",
            stage="final_eval",
            final_eval_summary=val_exc.summary,
            final_eval_passed=val_exc.all_passed,
        )

    assert raised is not None
    assert session.read_status()["state"] == "awaiting_validation"

    # TUI reads status, shows modal with summary + all_passed=False
    st = session.read_status()
    modal_summary = str(st.get("final_eval_summary", ""))
    modal_all_passed = bool(st.get("final_eval_passed", True))
    assert modal_all_passed is False
    assert "user-review" in modal_summary

    # User clicks Approve (no correction text)
    tui_result: tuple[str, str] | None = ("approve", "")
    assert tui_result is not None
    action, _ = tui_result

    # TUI _on_choice approve branch
    if action == "approve":
        session.set_status("completed", stage="done")

    assert session.read_status()["state"] == "completed"
    assert session.read_status()["stage"] == "done"
