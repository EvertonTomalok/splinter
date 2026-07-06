"""US-005: Live message + abnormal-termination observability events."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from splinter.memory.session import Session
from splinter.obs.events import (
    AbnormalTerminationEvent,
    LiveMessageFiredEvent,
    LiveMessageQueuedEvent,
    events_path,
)


@pytest.fixture
def session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Session:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    return Session("ses_test_events")


def test_fired_event_constructs_with_expected_fields() -> None:
    evt = LiveMessageFiredEvent(task_no=3, session_id="sid-1", summary="do X")
    assert evt.task_no == 3
    assert evt.session_id == "sid-1"
    assert evt.summary == "do X"
    assert evt.schema_version == 1


def test_queued_event_constructs_with_expected_fields() -> None:
    evt = LiveMessageQueuedEvent(task_no=2, summary="do Y")
    assert evt.task_no == 2
    assert evt.summary == "do Y"
    assert evt.schema_version == 1
    assert not hasattr(evt, "session_id")


def test_abnormal_termination_constructs() -> None:
    evt = AbnormalTerminationEvent(task_no=1, reason="crash")
    assert evt.task_no == 1
    assert evt.reason == "crash"
    assert evt.schema_version == 1


def test_fired_to_event_serializes() -> None:
    evt = LiveMessageFiredEvent(task_no=3, session_id="sid-1", summary="do X").to_event()
    assert evt.type == "live_message_fired"
    assert evt.ts  # non-empty string
    assert evt.payload["task_no"] == 3
    assert evt.payload["session_id"] == "sid-1"
    assert evt.payload["summary"] == "do X"


def test_queued_to_event_serializes() -> None:
    evt = LiveMessageQueuedEvent(task_no=2, summary="do Y").to_event()
    assert evt.type == "live_message_queued"
    assert evt.ts  # non-empty string
    assert evt.payload["task_no"] == 2
    assert evt.payload["summary"] == "do Y"
    assert "session_id" not in evt.payload


def test_abnormal_to_event_serializes() -> None:
    evt = AbnormalTerminationEvent(task_no=1, reason="crash").to_event()
    assert evt.type == "abnormal_termination"
    assert evt.ts  # non-empty string
    assert evt.payload["task_no"] == 1
    assert evt.payload["reason"] == "crash"


def test_emit_appends_jsonl(session: Session) -> None:
    LiveMessageFiredEvent(task_no=3, session_id="sid-1", summary="do X").emit(session)
    path = events_path(session)
    assert path.exists()
    lines = [line for line in path.read_text().split("\n") if line.strip()]
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["type"] == "live_message_fired"
    assert rec["payload"]["session_id"] == "sid-1"


def test_emit_thread_safe_shape(session: Session) -> None:
    LiveMessageFiredEvent(task_no=3, session_id="sid-1", summary="do X").emit(session)
    LiveMessageQueuedEvent(task_no=2, summary="do Y").emit(session)
    path = events_path(session)
    lines = [line for line in path.read_text().split("\n") if line.strip()]
    assert len(lines) == 2
    rec1 = json.loads(lines[0])
    rec2 = json.loads(lines[1])
    assert rec1["type"] == "live_message_fired"
    assert rec2["type"] == "live_message_queued"
    assert json.loads(lines[0])  # ensure valid JSON
    assert json.loads(lines[1])  # ensure valid JSON
