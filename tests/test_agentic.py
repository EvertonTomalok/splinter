"""Tests for agentic event metric API and verbatim exchange recording."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from splinter.memory.session import Session
from splinter.obs.agentic import (
    AgenticEvent,
    agentic_scope,
    append_jsonl,
    load_agentic_events,
    read_events,
    record_action,
    record_exchange,
    record_gate_marker,
)
from splinter.providers.claude_cli import _event_summaries


@pytest.fixture
def tmp_session(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Session:
    """Create a session in a temporary directory."""
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    return Session("test-session")


def test_round_trip_single_event(tmp_session: Session) -> None:
    """Write one event, read it back."""
    event = AgenticEvent(
        task_index=0,
        iteration=1,
        provider="opencode",
        model="opencode-go/gpt-4",
        kind="localize",
        tokens={"input": 100, "output": 50},
        cost=0.01,
        ts="2026-06-10T12:00:00Z",
        extra={},
    )
    append_jsonl(tmp_session, event)

    loaded = load_agentic_events(tmp_session)
    assert len(loaded) == 1
    assert loaded[0] == event


def test_separate_files_per_task_index(
    tmp_session: Session,
) -> None:
    """Events for different task_index values land in separate files."""
    event0 = AgenticEvent(
        task_index=0,
        iteration=1,
        provider="opencode",
        model="opencode-go/gpt-4",
        kind="plan",
        tokens={"input": 200, "output": 100},
        cost=0.02,
        ts="2026-06-10T12:00:00Z",
        extra={},
    )
    event1 = AgenticEvent(
        task_index=1,
        iteration=1,
        provider="claude",
        model="opus",
        kind="run",
        tokens={"input": 300, "output": 150},
        cost=0.03,
        ts="2026-06-10T12:00:01Z",
        extra={},
    )

    append_jsonl(tmp_session, event0)
    append_jsonl(tmp_session, event1)

    trace_dir = tmp_session.dir / "trace"
    assert (trace_dir / "agentic-0.jsonl").exists()
    assert (trace_dir / "agentic-1.jsonl").exists()

    loaded = load_agentic_events(tmp_session)
    assert len(loaded) == 2
    assert event0 in loaded
    assert event1 in loaded


def test_malformed_jsonl_skipped(tmp_session: Session) -> None:
    """Malformed/truncated lines are skipped; valid lines still load."""
    tmp_session._ensure_dir()
    trace_dir = tmp_session.dir / "trace"
    trace_dir.mkdir(parents=True, exist_ok=True)

    file_path = trace_dir / "agentic-0.jsonl"
    line1 = json.dumps(
        {
            "task_index": 0,
            "iteration": 1,
            "provider": "opencode",
            "model": "gpt-4",
            "kind": "localize",
            "tokens": {"input": 100, "output": 50},
            "cost": 0.01,
            "ts": "2026-06-10T12:00:00Z",
            "extra": {},
        }
    )
    line3 = json.dumps(
        {
            "task_index": 0,
            "iteration": 2,
            "provider": "claude",
            "model": "opus",
            "kind": "plan",
            "tokens": {"input": 200, "output": 100},
            "cost": 0.02,
            "ts": "2026-06-10T12:00:01Z",
            "extra": {},
        }
    )
    with open(file_path, "w") as f:
        f.write(line1 + "\n")
        f.write("not valid json\n")
        f.write(line3 + "\n")

    loaded = load_agentic_events(tmp_session)
    assert len(loaded) == 2
    assert loaded[0].iteration == 1
    assert loaded[1].iteration == 2


def test_write_error_swallowed(tmp_session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    """Injected write error is swallowed, no exception raised."""
    event = AgenticEvent(
        task_index=0,
        iteration=1,
        provider="opencode",
        model="opencode-go/gpt-4",
        kind="localize",
        tokens={"input": 100, "output": 50},
        cost=0.01,
        ts="2026-06-10T12:00:00Z",
        extra={},
    )

    def mock_open_error(*args: object, **kwargs: object) -> object:
        raise IOError("simulated write error")

    monkeypatch.setattr("builtins.open", mock_open_error)

    append_jsonl(tmp_session, event)


def test_session_with_only_agentic_is_empty(
    tmp_session: Session,
) -> None:
    """Session holding only trace/agentic-*.jsonl is still is_empty."""
    event = AgenticEvent(
        task_index=0,
        iteration=1,
        provider="opencode",
        model="opencode-go/gpt-4",
        kind="localize",
        tokens={"input": 100, "output": 50},
        cost=0.01,
        ts="2026-06-10T12:00:00Z",
        extra={},
    )
    append_jsonl(tmp_session, event)

    assert tmp_session.is_empty()


# ---------------------------------------------------------------------------
# Verbatim exchange recording — AC unit tests
# ---------------------------------------------------------------------------


def test_exchange_stage_task_iteration_correct(tmp_session: Session) -> None:
    with agentic_scope(tmp_session, "run", 3, 1):
        record_exchange("the prompt", "the response", model="m")

    events = read_events(tmp_session, 3)
    assert len(events) == 1
    ev = events[0]
    assert ev.stage == "run"
    assert ev.task_index == 3
    assert ev.iteration == 1


def test_two_iterations_run_eval_events(tmp_session: Session) -> None:
    for iteration in (1, 2):
        with agentic_scope(tmp_session, "run", 0, iteration):
            record_exchange(f"run prompt {iteration}", f"run response {iteration}", model="m")
        with agentic_scope(tmp_session, "eval", 0, iteration):
            record_exchange(f"eval prompt {iteration}", f"eval response {iteration}", model="m")

    events = read_events(tmp_session, 0)
    run_events = [e for e in events if e.stage == "run"]
    eval_events = [e for e in events if e.stage == "eval"]

    assert len(run_events) == 2
    assert len(eval_events) == 2
    assert {e.iteration for e in run_events} == {1, 2}
    assert {e.iteration for e in eval_events} == {1, 2}


def test_gate_marker_empty_prompt_response(tmp_session: Session) -> None:
    with agentic_scope(tmp_session, "gate", 0, 1):
        record_gate_marker()

    events = read_events(tmp_session, 0)
    assert len(events) == 1
    ev = events[0]
    assert ev.stage == "gate"
    assert ev.prompt == ""
    assert ev.response == ""


def test_exchange_verbatim_no_truncation(tmp_session: Session) -> None:
    long_prompt = "A" * 50_000
    long_response = "B" * 50_000

    with agentic_scope(tmp_session, "run", 0, 1):
        record_exchange(long_prompt, long_response, model="m")

    events = read_events(tmp_session, 0)
    assert len(events) == 1
    assert events[0].prompt == long_prompt
    assert events[0].response == long_response


# ---------------------------------------------------------------------------
# Provider action recording — AC unit tests
# ---------------------------------------------------------------------------


def test_event_summaries_tool_use() -> None:
    """Parse assistant message with tool_use block into (kind, summary) pair."""
    line = json.dumps(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Edit",
                        "input": {"file_path": "/path/to/file.py", "old": "a", "new": "b"},
                    }
                ]
            },
        }
    )
    summaries = _event_summaries(line)
    assert len(summaries) == 1
    assert summaries[0][0] == "tool_use"
    assert "Edit" in summaries[0][1]
    assert "/path/to/file.py" in summaries[0][1]


def test_event_summaries_text() -> None:
    """Parse assistant message with text block into (kind, summary) pair."""
    line = json.dumps(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "text",
                        "text": "Here is the solution to your problem",
                    }
                ]
            },
        }
    )
    summaries = _event_summaries(line)
    assert len(summaries) == 1
    assert summaries[0][0] == "text"
    assert "solution" in summaries[0][1]


def test_event_summaries_mixed_content() -> None:
    """Parse assistant message with both tool_use and text blocks."""
    line = json.dumps(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Write",
                        "input": {"file_path": "/new/file.ts"},
                    },
                    {
                        "type": "text",
                        "text": "Created the new file",
                    },
                ]
            },
        }
    )
    summaries = _event_summaries(line)
    assert len(summaries) == 2
    assert summaries[0][0] == "tool_use"
    assert summaries[1][0] == "text"


def test_event_summaries_tool_detail_truncation() -> None:
    """Tool details truncated to ≤90 chars."""
    long_path = "a" * 200
    line = json.dumps(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Edit",
                        "input": {"file_path": long_path},
                    }
                ]
            },
        }
    )
    summaries = _event_summaries(line)
    assert len(summaries) == 1
    summary = summaries[0][1]
    assert len(summary) <= 110  # "🔧 Edit " + 90 chars max


def test_event_summaries_text_full() -> None:
    """Text summaries include full content — no truncation."""
    long_text = "B" * 300
    line = json.dumps(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "text",
                        "text": long_text,
                    }
                ]
            },
        }
    )
    summaries = _event_summaries(line)
    assert len(summaries) == 1
    summary = summaries[0][1]
    assert summary == f"💬 {long_text}"


def test_event_summaries_malformed_json() -> None:
    """Malformed JSON returns empty list."""
    assert _event_summaries("not valid json") == []
    assert _event_summaries("{unclosed") == []
    assert _event_summaries("") == []


def test_event_summaries_non_assistant_ignored() -> None:
    """Non-assistant messages return empty list."""
    line = json.dumps({"type": "stream_event", "event": {"type": "other"}})
    assert _event_summaries(line) == []


def test_event_summaries_non_dict() -> None:
    """Non-dict objects return empty list."""
    assert _event_summaries('"just a string"') == []
    assert _event_summaries('["array", "not", "dict"]') == []


def test_record_action_inside_scope(tmp_session: Session) -> None:
    """record_action inside agentic_scope writes AgenticEvent."""
    with agentic_scope(tmp_session, "run", 2, 1):
        record_action("tool_use", "🔧 Edit /path/to/file")
        record_action("text", "💬 Here is the solution")

    loaded = load_agentic_events(tmp_session)
    assert len(loaded) == 2

    tool_event = loaded[0]
    assert tool_event.task_index == 2
    assert tool_event.iteration == 1
    assert tool_event.kind == "tool_use"
    assert tool_event.extra.get("summary") == "🔧 Edit /path/to/file"
    assert tool_event.provider == "claude"
    assert tool_event.model == ""
    assert tool_event.tokens == {}
    assert tool_event.cost == 0.0

    text_event = loaded[1]
    assert text_event.kind == "text"
    assert text_event.extra.get("summary") == "💬 Here is the solution"


def test_record_action_outside_scope(tmp_session: Session) -> None:
    """record_action outside agentic_scope is no-op, no file written."""
    record_action("tool_use", "🔧 Edit /path/to/file")

    loaded = load_agentic_events(tmp_session)
    assert len(loaded) == 0


def test_record_action_custom_provider(tmp_session: Session) -> None:
    """record_action respects custom provider parameter."""
    with agentic_scope(tmp_session, "run", 0, 1):
        record_action("tool_use", "🔧 WebSearch", provider="cursor")

    loaded = load_agentic_events(tmp_session)
    assert len(loaded) == 1
    assert loaded[0].provider == "cursor"


def test_record_action_timestamp_iso_format(tmp_session: Session) -> None:
    """record_action sets ts in ISO 8601 format."""
    with agentic_scope(tmp_session, "run", 0, 1):
        record_action("text", "💬 Action summary")

    loaded = load_agentic_events(tmp_session)
    assert len(loaded) == 1
    ts = loaded[0].ts
    assert "T" in ts
    assert ts.endswith("Z") or "+" in ts
