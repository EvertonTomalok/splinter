from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from splinter.agents.acceptance import AcceptanceMode
from splinter.agents.runner import RunResult, Task
from splinter.memory.knowledge import KnowledgeStore
from splinter.memory.session import Session
from splinter.models.roster import load_ladder
from splinter.obs.trace import Trace
from splinter.strategies.base import EvalVerdict
from splinter.strategies.stages import EvalStage, IterationContext


@pytest.fixture
def tmp_session(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Session:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    return Session("test-session")


class _ForbiddenJudge:
    eval_effort_for = staticmethod(lambda tier: "high")

    def judge(self, *args: object, **kwargs: object) -> EvalVerdict:
        raise AssertionError("the eval model must not run in this mode")


def _ctx(session: Session, mode: AcceptanceMode) -> IterationContext:
    return IterationContext(
        task=Task(description="the goal", acceptance="first line\nsecond line"),
        plan="plan",
        tier=1,
        iteration=1,
        ladder=load_ladder(),
        session=session,
        trace=Trace(),
        knowledge=KnowledgeStore(session),
        run_result=RunResult(text="done", model="runner", tier=1, tokens={}, cost=0.0, raw={}),
        acceptance=mode,
    )


@pytest.fixture
def no_diff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("splinter.strategies.stages.worktree_diff", lambda path: "the diff")


def _reply(monkeypatch: pytest.MonkeyPatch, text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake(prompt: str, model: str, **kwargs: Any) -> str:
        calls.append({"prompt": prompt, "model": model, **kwargs})
        return text

    monkeypatch.setattr("splinter.agents.acceptance.run_text", fake)
    return calls


def test_gate_only_passes_without_any_model(tmp_session: Session) -> None:
    ctx = _ctx(tmp_session, AcceptanceMode.GATE)

    EvalStage(evaluator=_ForbiddenJudge()).process(ctx)  # type: ignore[arg-type]

    assert ctx.verdict is not None and ctx.verdict.passed


@pytest.mark.usefixtures("no_diff")
def test_orchestrator_mode_never_calls_the_eval_model(
    tmp_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _reply(monkeypatch, "looks right\nVERDICT: PASS")
    ctx = _ctx(tmp_session, AcceptanceMode.GATE_ORCHESTRATOR)

    EvalStage(evaluator=_ForbiddenJudge()).process(ctx)  # type: ignore[arg-type]

    assert ctx.verdict is not None and ctx.verdict.passed
    assert calls[0]["role"] == "orchestrator"
    assert calls[0]["model"] == ctx.ladder.planner_model
    assert "the diff" in calls[0]["prompt"] and "second line" in calls[0]["prompt"]


@pytest.mark.usefixtures("no_diff")
def test_orchestrator_rejection_becomes_a_retry_with_its_reason(
    tmp_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    _reply(monkeypatch, "the second line is missing\nVERDICT: FAIL")
    ctx = _ctx(tmp_session, AcceptanceMode.GATE_ORCHESTRATOR)

    EvalStage(evaluator=_ForbiddenJudge()).process(ctx)  # type: ignore[arg-type]

    assert ctx.verdict is not None and not ctx.verdict.passed
    assert "second line is missing" in ctx.verdict.corrections
    assert "second line is missing" in (tmp_session.dir / "eval.md").read_text()


@pytest.mark.usefixtures("no_diff")
def test_a_crashed_orchestrator_is_a_retry_not_a_crash(
    tmp_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: object, **kwargs: object) -> str:
        raise RuntimeError("provider down")

    monkeypatch.setattr("splinter.agents.acceptance.run_text", boom)
    ctx = _ctx(tmp_session, AcceptanceMode.GATE_ORCHESTRATOR)

    EvalStage(evaluator=_ForbiddenJudge()).process(ctx)  # type: ignore[arg-type]

    assert ctx.verdict is not None and not ctx.verdict.passed
    assert "provider down" in ctx.verdict.reason


def test_a_diff_that_cannot_be_read_is_a_retry(
    tmp_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(path: object) -> str:
        raise OSError("no git")

    monkeypatch.setattr("splinter.strategies.stages.worktree_diff", broken)
    ctx = _ctx(tmp_session, AcceptanceMode.GATE_ORCHESTRATOR)

    EvalStage(evaluator=_ForbiddenJudge()).process(ctx)  # type: ignore[arg-type]

    assert ctx.verdict is not None and not ctx.verdict.passed


def test_default_mode_still_uses_the_eval_model(tmp_session: Session) -> None:
    class _Judge:
        eval_effort_for = staticmethod(lambda tier: "high")
        called = False

        def judge(self, *args: object, **kwargs: object) -> EvalVerdict:
            _Judge.called = True
            return EvalVerdict(decision="PASS", reason="ok", corrections="", raw="")

    ctx = _ctx(tmp_session, AcceptanceMode.GATE_EVAL)

    EvalStage(evaluator=_Judge()).process(ctx)  # type: ignore[arg-type]

    assert _Judge.called
