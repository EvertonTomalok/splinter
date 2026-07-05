"""Tests for stage trace recording and action rendering."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from splinter.memory.session import Session
from splinter.obs.agentic import AgenticEvent, agentic_scope, append_jsonl, record_action
from splinter.strategies.stages import _render_actions


@pytest.fixture
def tmp_session(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Session:
    """Create a session in a temporary directory."""
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    return Session("test-session")


def test_render_actions_empty(tmp_session: Session) -> None:
    """_render_actions returns empty string when no actions found."""
    result = _render_actions(task_index=0, iteration=1, session=tmp_session)
    assert result == ""


def test_render_actions_filters_by_task_iteration(tmp_session: Session) -> None:
    """_render_actions filters events by task_index and iteration."""
    # Add events for different task/iteration combos
    event1 = AgenticEvent(
        task_index=0,
        iteration=1,
        provider="claude",
        model="",
        kind="tool_use",
        ts="2026-06-11T00:00:00Z",
        extra={"summary": "🔧 Edit task0-iter1"},
    )
    event2 = AgenticEvent(
        task_index=0,
        iteration=2,
        provider="claude",
        model="",
        kind="tool_use",
        ts="2026-06-11T00:00:01Z",
        extra={"summary": "🔧 Edit task0-iter2"},
    )
    event3 = AgenticEvent(
        task_index=1,
        iteration=1,
        provider="claude",
        model="",
        kind="tool_use",
        ts="2026-06-11T00:00:02Z",
        extra={"summary": "🔧 Edit task1-iter1"},
    )
    append_jsonl(tmp_session, event1)
    append_jsonl(tmp_session, event2)
    append_jsonl(tmp_session, event3)

    # Only task 0, iteration 1 should render
    result = _render_actions(task_index=0, iteration=1, session=tmp_session)
    assert "task0-iter1" in result
    assert "task0-iter2" not in result
    assert "task1-iter1" not in result


def test_render_actions_includes_tool_use_and_text(tmp_session: Session) -> None:
    """_render_actions includes both tool_use and text kind events."""
    event1 = AgenticEvent(
        task_index=0,
        iteration=1,
        provider="claude",
        model="",
        kind="tool_use",
        ts="2026-06-11T00:00:00Z",
        extra={"summary": "🔧 Edit /file.py"},
    )
    event2 = AgenticEvent(
        task_index=0,
        iteration=1,
        provider="claude",
        model="",
        kind="text",
        ts="2026-06-11T00:00:01Z",
        extra={"summary": "💬 Solution is ready"},
    )
    append_jsonl(tmp_session, event1)
    append_jsonl(tmp_session, event2)

    result = _render_actions(task_index=0, iteration=1, session=tmp_session)
    assert "## Actions" in result
    assert "🔧 Edit /file.py" in result
    assert "💬 Solution is ready" in result


def test_render_actions_markdown_format(tmp_session: Session) -> None:
    """_render_actions formats output as markdown list."""
    event = AgenticEvent(
        task_index=0,
        iteration=1,
        provider="claude",
        model="",
        kind="tool_use",
        ts="2026-06-11T00:00:00Z",
        extra={"summary": "🔧 Write /new/file.ts"},
    )
    append_jsonl(tmp_session, event)

    result = _render_actions(task_index=0, iteration=1, session=tmp_session)
    assert result.startswith("## Actions\n")
    assert "- 🔧 Write /new/file.ts" in result
    assert result.endswith("\n")


def test_render_actions_skips_non_action_kinds(tmp_session: Session) -> None:
    """_render_actions ignores events with kind not in tool_use/text."""
    event1 = AgenticEvent(
        task_index=0,
        iteration=1,
        provider="claude",
        model="opus",
        kind="run",  # Not tool_use or text
        ts="2026-06-11T00:00:00Z",
        extra={"summary": "🔧 Edit /file.py"},
    )
    append_jsonl(tmp_session, event1)

    result = _render_actions(task_index=0, iteration=1, session=tmp_session)
    assert result == ""


def test_render_actions_multiple_events_ordering(tmp_session: Session) -> None:
    """_render_actions renders multiple events in order."""
    events = [
        AgenticEvent(
            task_index=0,
            iteration=1,
            provider="claude",
            model="",
            kind="tool_use",
            ts=f"2026-06-11T00:00:{i:02d}Z",
            extra={"summary": f"🔧 Action {i}"},
        )
        for i in range(3)
    ]
    for event in events:
        append_jsonl(tmp_session, event)

    result = _render_actions(task_index=0, iteration=1, session=tmp_session)
    lines = result.strip().split("\n")
    assert len(lines) == 4  # Header + 3 items
    assert lines[0] == "## Actions"
    assert "Action 0" in lines[1]
    assert "Action 1" in lines[2]
    assert "Action 2" in lines[3]


def test_record_action_inside_scope_with_render(tmp_session: Session) -> None:
    """record_action inside scope creates events that render correctly."""
    with agentic_scope(tmp_session, "run", 0, 1):
        record_action("tool_use", "🔧 Edit /path/to/file.py")
        record_action("text", "💬 Changes applied successfully")

    result = _render_actions(task_index=0, iteration=1, session=tmp_session)
    assert "## Actions" in result
    assert "🔧 Edit /path/to/file.py" in result
    assert "💬 Changes applied successfully" in result


def test_eval_stage_logs_model_and_effort_before_judge(
    tmp_session: Session,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from splinter.agents.runner import RunResult, Task
    from splinter.memory.knowledge import KnowledgeStore
    from splinter.models.roster import load_ladder
    from splinter.obs.trace import Trace
    from splinter.strategies.base import EvalVerdict
    from splinter.strategies.stages import EvalStage, IterationContext

    class _FakeEvaluator:
        def eval_effort_for(self, tier: int) -> str:
            return "max"

        def judge(self, *args: object, **kwargs: object) -> EvalVerdict:
            return EvalVerdict(
                decision="PASS",
                reason="ok",
                corrections="",
                raw="",
                eval_session="ev-1",
            )

    ladder = load_ladder()
    ctx = IterationContext(
        task=Task(description="task", acceptance="accept"),
        plan="plan",
        tier=0,
        iteration=1,
        ladder=ladder,
        session=tmp_session,
        trace=Trace(),
        knowledge=KnowledgeStore(tmp_session),
        run_result=RunResult(
            text="result",
            model="runner",
            tier=0,
            tokens={},
            cost=0.0,
            raw={},
        ),
    )

    with caplog.at_level(logging.INFO, logger="splinter.loop"):
        EvalStage(evaluator=_FakeEvaluator()).process(ctx)

    assert any(
        "evaluating with" in rec.message
        and ladder.eval_model in rec.message
        and "effort=max" in rec.message
        for rec in caplog.records
    )


class TestCostReconciliation:
    def test_trace_total_cost_matches_summary_parse(self) -> None:
        from splinter.analyze import _trace_metrics
        from splinter.obs.trace import RunEntry, Trace

        trace = Trace()
        trace.entries.append(
            RunEntry(
                model="m",
                tier=1,
                iteration=1,
                tokens={"input": 100, "output": 50},
                cost=0.0100,
                latency_s=0.0,
                task=0,
            )
        )
        trace.entries.append(
            RunEntry(
                model="m",
                tier=0,
                iteration=2,
                tokens={"input": 80, "output": 40},
                cost=0.0200,
                latency_s=0.0,
                task=0,
                role="eval",
            )
        )
        trace.entries.append(
            RunEntry(
                model="m",
                tier=2,
                iteration=3,
                tokens={"input": 120, "output": 60},
                cost=0.0300,
                latency_s=0.0,
                task=0,
            )
        )

        md = trace.summary()
        metrics = _trace_metrics(md)

        assert float(metrics["cost"]) == pytest.approx(trace.total_cost, abs=1e-6)
        assert trace.total_cost == pytest.approx(0.0600)

    def test_log_run_then_trace_parse_agrees(self) -> None:
        from splinter.agents.runner import RunResult
        from splinter.analyze import _trace_metrics
        from splinter.obs.trace import Trace, log_run

        trace = Trace()
        log_run(
            trace,
            RunResult(text="a", model="m", tier=1, tokens={"input": 10}, cost=0.0400, raw={}),
            iteration=1,
            task=0,
        )
        log_run(
            trace,
            RunResult(text="b", model="m", tier=2, tokens={"input": 20}, cost=0.0250, raw={}),
            iteration=2,
            task=0,
        )

        md = trace.summary()
        metrics = _trace_metrics(md)

        assert float(metrics["cost"]) == pytest.approx(trace.total_cost, abs=1e-6)
        assert trace.total_cost == pytest.approx(0.0650)
