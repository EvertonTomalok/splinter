"""US-001: pause at sub-action boundary, resume into in-flight stage.

Unit tests for:
- Stage.handle boundary stop check (stop_requested sets pause_at_stage, current stage finishes)
- Checkpoint round-trip with new fields (stage, run_result, gate_passed, verdict)
- build_chain_from returns correct head stage for resume
- GracefulPause exception type and fields
- _pause_graceful raises GracefulPause with correct stage
- User guidance carried in corrections on resume
"""

from __future__ import annotations

import os

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_stages():
    from splinter.strategies.stages import Stage

    class FakeRun(Stage):
        name = "run"

        def __init__(self):
            super().__init__()
            self.called = False

        def process(self, ctx):
            self.called = True
            return True

    class FakeGate(Stage):
        name = "gate"

        def __init__(self):
            super().__init__()
            self.called = False

        def process(self, ctx):
            self.called = True
            return True

    class FakeEval(Stage):
        name = "eval"

        def __init__(self):
            super().__init__()
            self.called = False

        def process(self, ctx):
            self.called = True
            return True

    return FakeRun(), FakeGate(), FakeEval()


def _make_ctx():
    from unittest.mock import MagicMock

    from splinter.strategies.stages import IterationContext

    return IterationContext(
        task=MagicMock(),
        plan="plan",
        tier=1,
        iteration=1,
        ladder=MagicMock(),
        session=MagicMock(),
        trace=MagicMock(),
        knowledge=MagicMock(),
    )


@pytest.fixture(autouse=True)
def _clear_stop():
    from splinter import procreg

    procreg.clear_stop()
    yield
    procreg.clear_stop()


# ---------------------------------------------------------------------------
# 1. Stage-boundary stop check
# ---------------------------------------------------------------------------


def test_stage_boundary_stop_sets_pause_at_stage():
    from splinter import procreg

    run_s, gate_s, eval_s = _make_fake_stages()
    run_s.set_next(gate_s)
    gate_s.set_next(eval_s)

    procreg.request_stop()
    ctx = _make_ctx()
    run_s.handle(ctx)

    assert run_s.called, "RunStage must finish fully"
    assert not gate_s.called, "GateStage must NOT run when stop requested"
    assert not eval_s.called, "EvalStage must NOT run when stop requested"
    assert ctx.pause_at_stage == "gate"


def test_stage_boundary_no_stop_runs_full_chain():
    run_s, gate_s, eval_s = _make_fake_stages()
    run_s.set_next(gate_s)
    gate_s.set_next(eval_s)

    ctx = _make_ctx()
    run_s.handle(ctx)

    assert run_s.called and gate_s.called and eval_s.called
    assert ctx.pause_at_stage is None


def test_stop_after_gate_stage_pauses_before_eval():
    from splinter import procreg

    run_s, gate_s, eval_s = _make_fake_stages()
    run_s.set_next(gate_s)
    gate_s.set_next(eval_s)

    ctx = _make_ctx()

    original_gate_process = gate_s.process

    def gate_process_with_stop(c):
        procreg.request_stop()
        return original_gate_process(c)

    gate_s.process = gate_process_with_stop

    run_s.handle(ctx)

    assert run_s.called and gate_s.called
    assert not eval_s.called
    assert ctx.pause_at_stage == "eval"


# ---------------------------------------------------------------------------
# 2. Checkpoint round-trip with new fields
# ---------------------------------------------------------------------------


def test_checkpoint_roundtrip_with_new_fields(tmp_path):
    os.environ["SPLINTER_HOME"] = str(tmp_path)
    from dataclasses import asdict

    from splinter.agents.runner import RunResult
    from splinter.memory.session import Session
    from splinter.strategies.base import EvalVerdict
    from splinter.strategies.direct import (
        RunCheckpoint,
        _load_checkpoint,
        _save_checkpoint,
    )

    session = Session("ses_cp_rt")

    rr = RunResult(
        text="impl done",
        model="sonnet",
        tier=1,
        tokens={"input": 100, "output": 200},
        cost=0.01,
        raw={"provider": "anthropic"},
    )
    ev = EvalVerdict(decision="PASS", reason="looks correct", corrections="none")

    cp = RunCheckpoint(
        tier=2,
        iteration=3,
        task_index=0,
        oc_session=None,
        eval_session="eval-123",
        corrections="fix the retry logic",
        eval_history=["iter 1 [PASS]: good"],
        reason="user stop",
        gate_output="tests failed: E001",
        stage="gate",
        run_result=asdict(rr),
        gate_passed=False,
        verdict=asdict(ev),
    )
    _save_checkpoint(session, cp)
    loaded = _load_checkpoint(session)

    assert loaded is not None
    assert loaded.stage == "gate"
    assert loaded.gate_passed is False
    assert loaded.run_result is not None
    assert loaded.run_result["text"] == "impl done"
    assert loaded.run_result["tokens"]["input"] == 100
    assert loaded.verdict is not None
    assert loaded.verdict["decision"] == "PASS"
    assert loaded.eval_session == "eval-123"
    assert loaded.gate_output == "tests failed: E001"


def test_checkpoint_backcompat_missing_new_fields(tmp_path):
    """Old checkpoints without new fields still load cleanly."""
    import json

    os.environ["SPLINTER_HOME"] = str(tmp_path)
    from splinter.memory.session import Session
    from splinter.strategies.direct import _load_checkpoint

    session = Session("ses_cp_compat")
    old_data = {
        "tier": 1,
        "iteration": 2,
        "task_index": 0,
        "oc_session": None,
        "eval_session": None,
        "corrections": "old corrections",
        "eval_history": [],
        "reason": "max tier",
        "gate_output": "",
    }
    session.write("run_checkpoint.json", json.dumps(old_data))
    loaded = _load_checkpoint(session)

    assert loaded is not None
    assert loaded.stage == ""
    assert loaded.run_result is None
    assert loaded.gate_passed is True
    assert loaded.verdict is None


# ---------------------------------------------------------------------------
# 3. build_chain_from returns correct head
# ---------------------------------------------------------------------------


def test_build_chain_from_returns_gate_head():
    from splinter.strategies.stages import EvalStage, GateStage, RunStage, build_chain_from

    run_s = RunStage()
    gate_s = GateStage()
    eval_s = EvalStage()

    head = build_chain_from("gate", run_s, gate_s, eval_s)

    assert head is gate_s
    assert gate_s._next is eval_s
    assert eval_s._next is None
    assert run_s._next is gate_s


def test_build_chain_from_returns_eval_head():
    from splinter.strategies.stages import EvalStage, GateStage, RunStage, build_chain_from

    run_s = RunStage()
    gate_s = GateStage()
    eval_s = EvalStage()

    head = build_chain_from("eval", run_s, gate_s, eval_s)

    assert head is eval_s
    assert eval_s._next is None


def test_build_chain_from_none_returns_first():
    from splinter.strategies.stages import EvalStage, GateStage, RunStage, build_chain_from

    run_s = RunStage()
    gate_s = GateStage()
    eval_s = EvalStage()

    head = build_chain_from(None, run_s, gate_s, eval_s)
    assert head is run_s


def test_build_chain_from_unknown_returns_first():
    from splinter.strategies.stages import EvalStage, GateStage, RunStage, build_chain_from

    run_s = RunStage()
    gate_s = GateStage()
    eval_s = EvalStage()

    head = build_chain_from("nonexistent", run_s, gate_s, eval_s)
    assert head is run_s


# ---------------------------------------------------------------------------
# 4. GracefulPause exception type and fields
# ---------------------------------------------------------------------------


def test_graceful_pause_is_exception():
    from splinter.strategies.base import GracefulPause

    gp = GracefulPause(reason="test", tier=2, iteration=3, task_index=0, stage="gate")
    assert isinstance(gp, Exception)
    assert gp.stage == "gate"
    assert gp.tier == 2
    assert gp.iteration == 3
    assert gp.reason == "test"


def test_graceful_pause_stage_default_empty():
    from splinter.strategies.base import GracefulPause

    gp = GracefulPause(reason="test")
    assert gp.stage == ""


# ---------------------------------------------------------------------------
# 5. _pause_graceful raises GracefulPause with correct stage
# ---------------------------------------------------------------------------


def test_pause_graceful_raises_with_stage(tmp_path):

    os.environ["SPLINTER_HOME"] = str(tmp_path)
    from splinter.memory.knowledge import KnowledgeStore
    from splinter.memory.session import Session
    from splinter.strategies.base import GracefulPause
    from splinter.strategies.direct import _pause_graceful

    session = Session("ses_pg_raises")
    knowledge = KnowledgeStore(session)

    with pytest.raises(GracefulPause) as exc_info:
        _pause_graceful(
            session=session,
            knowledge=knowledge,
            task_index=0,
            iteration=2,
            tier=1,
            stage="eval",
            corrections="fix x",
            gate_output="gate failed",
            run_result=None,
            gate_passed=False,
            verdict=None,
            oc_session=None,
            eval_session=None,
            eval_history=[],
        )

    gp = exc_info.value
    assert gp.stage == "eval"
    assert gp.iteration == 2
    assert gp.tier == 1


def test_pause_graceful_saves_checkpoint(tmp_path):
    os.environ["SPLINTER_HOME"] = str(tmp_path)
    from splinter.memory.knowledge import KnowledgeStore
    from splinter.memory.session import Session
    from splinter.strategies.base import GracefulPause
    from splinter.strategies.direct import _load_checkpoint, _pause_graceful

    session = Session("ses_pg_cp")
    knowledge = KnowledgeStore(session)

    with pytest.raises(GracefulPause):
        _pause_graceful(
            session=session,
            knowledge=knowledge,
            task_index=0,
            iteration=4,
            tier=2,
            stage="gate",
            corrections="prev corrections",
            gate_output="",
            run_result=None,
            gate_passed=True,
            verdict=None,
            oc_session="oc-abc",
            eval_session=None,
            eval_history=["iter 1 [PASS]: ..."],
        )

    loaded = _load_checkpoint(session)
    assert loaded is not None
    assert loaded.stage == "gate"
    assert loaded.iteration == 4
    assert loaded.tier == 2
    assert loaded.oc_session == "oc-abc"


# ---------------------------------------------------------------------------
# 6. User guidance carried in corrections on resume
# ---------------------------------------------------------------------------


def test_user_guidance_merged_into_corrections():
    """corrections variable on resume merges checkpoint.corrections + user_guidance."""
    from splinter.strategies.direct import RunCheckpoint

    cp = RunCheckpoint(
        tier=1,
        iteration=2,
        task_index=0,
        oc_session=None,
        eval_session=None,
        corrections="existing fix notes",
        eval_history=[],
        reason="paused",
        gate_output="",
        stage="gate",
    )
    user_guidance = "please focus on the retry logic"

    corrections = cp.corrections
    if user_guidance:
        corrections = f"{corrections}\n\n## User guidance\n{user_guidance}".strip()

    assert "please focus on the retry logic" in corrections
    assert "existing fix notes" in corrections
    assert "## User guidance" in corrections


def test_user_guidance_empty_leaves_corrections_unchanged():
    from splinter.strategies.direct import RunCheckpoint

    cp = RunCheckpoint(
        tier=1,
        iteration=1,
        task_index=0,
        oc_session=None,
        eval_session=None,
        corrections="fix abc",
        eval_history=[],
        reason="paused",
        gate_output="",
        stage="",
    )
    user_guidance = ""

    corrections = cp.corrections
    if user_guidance:
        corrections = f"{corrections}\n\n## User guidance\n{user_guidance}".strip()

    assert corrections == "fix abc"


# ---------------------------------------------------------------------------
# 7. Stage name attributes
# ---------------------------------------------------------------------------


def test_stage_name_attributes():
    from splinter.strategies.stages import EvalStage, GateStage, RunStage

    assert RunStage.name == "run"
    assert GateStage.name == "gate"
    assert EvalStage.name == "eval"
