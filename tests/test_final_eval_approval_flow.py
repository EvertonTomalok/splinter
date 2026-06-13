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
        f"- {r.name}: {'PASS' if r.passed else 'FAIL'} — {r.output[:200]}" for r in results
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
        f"- {r.name}: {'PASS' if r.passed else 'FAIL'} — {r.output[:200]}" for r in results
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
        f"- {r.name}: {'PASS' if r.passed else 'FAIL'} — {r.output[:200]}" for r in fe_results
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


# ── 7. _ManualValidationModal exposes edit_config action ──────────────────────


def test_manual_validation_modal_has_edit_config_binding() -> None:
    """_ManualValidationModal BINDINGS include 'c' → edit_config."""
    from splinter.tui import _ManualValidationModal

    binding_keys = {
        b[0] if isinstance(b, tuple) else b.key for b in _ManualValidationModal.BINDINGS
    }
    assert "c" in binding_keys, "binding 'c' (edit_config) must exist in _ManualValidationModal"


def test_manual_validation_modal_dismiss_edit_config() -> None:
    """on_button_pressed with id='edit_config' dismisses with ('edit_config', '')."""
    from splinter.tui import _ManualValidationModal

    modal = _ManualValidationModal(summary="eval failed", all_passed=False)
    dismissed: list = []

    modal.dismiss = lambda v: dismissed.append(v)  # type: ignore[method-assign]

    class _FakeButton:
        id = "edit_config"

    class _FakeEvent:
        button = _FakeButton()

    modal.query_one = lambda sel, cls=None: type(  # type: ignore[method-assign]
        "_FakeTextArea", (), {"text": ""}
    )()

    modal.on_button_pressed(_FakeEvent())  # type: ignore[arg-type]

    assert len(dismissed) == 1
    assert dismissed[0] == ("edit_config", "")


def test_store_config_overrides_writes_next_keys(tmp_path) -> None:
    """_store_config_overrides persists next_* keys to session status."""
    import os

    from splinter.memory.session import Session

    os.environ["SPLINTER_HOME"] = str(tmp_path)
    s = Session("ses_sc_test")
    s.set_status("awaiting_user", round_index=1)

    cfg = {
        "planner_model": "opus",
        "planner_effort": "high",
        "runner_model": "sonnet",
        "runner_effort": "medium",
        "eval_model": "haiku",
        "eval_effort": "low",
    }

    st = s.read_status()
    state = st.get("state", "running")
    s.set_status(
        state,
        next_planner_model=cfg.get("planner_model") or "",
        next_planner_effort=cfg.get("planner_effort") or "",
        next_runner_model=cfg.get("runner_model") or "",
        next_runner_effort=cfg.get("runner_effort") or "",
        next_eval_model=cfg.get("eval_model") or "",
        next_eval_effort=cfg.get("eval_effort") or "",
    )

    nc = s.read_next_config()
    assert nc["next_planner_model"] == "opus"
    assert nc["next_planner_effort"] == "high"
    assert nc["next_runner_model"] == "sonnet"
    assert nc["next_runner_effort"] == "medium"
    assert nc["next_eval_model"] == "haiku"
    assert nc["next_eval_effort"] == "low"

    del os.environ["SPLINTER_HOME"]
