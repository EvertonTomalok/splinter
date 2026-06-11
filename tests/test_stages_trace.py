"""Tests for stage trace recording and action rendering."""

from __future__ import annotations

from pathlib import Path

import pytest

from splinter.memory.session import Session
from splinter.obs.agentic import AgenticEvent, agentic_scope, append_jsonl, record_action
from splinter.strategies.stages import _render_actions


@pytest.fixture
def tmp_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Session:
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
        tokens={},
        cost=0.0,
        ts="2026-06-11T00:00:00Z",
        extra={"summary": "🔧 Edit task0-iter1"},
    )
    event2 = AgenticEvent(
        task_index=0,
        iteration=2,
        provider="claude",
        model="",
        kind="tool_use",
        tokens={},
        cost=0.0,
        ts="2026-06-11T00:00:01Z",
        extra={"summary": "🔧 Edit task0-iter2"},
    )
    event3 = AgenticEvent(
        task_index=1,
        iteration=1,
        provider="claude",
        model="",
        kind="tool_use",
        tokens={},
        cost=0.0,
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
        tokens={},
        cost=0.0,
        ts="2026-06-11T00:00:00Z",
        extra={"summary": "🔧 Edit /file.py"},
    )
    event2 = AgenticEvent(
        task_index=0,
        iteration=1,
        provider="claude",
        model="",
        kind="text",
        tokens={},
        cost=0.0,
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
        tokens={},
        cost=0.0,
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
        tokens={"input": 100, "output": 50},
        cost=0.01,
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
            tokens={},
            cost=0.0,
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
