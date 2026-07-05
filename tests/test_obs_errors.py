"""US-003: observability failures surface to ``errors.jsonl``, never crash.

Table-driven: inject write/read/decode failures across the swallow sites and
assert (a) the run is not interrupted and (b) the failure is recorded to the
dedicated ``errors.jsonl`` channel rather than silently discarded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from splinter.memory.session import Session
from splinter.obs.agentic import (
    AgenticEvent,
    append_jsonl,
    load_agentic_events,
    read_events,
)
from splinter.obs.errors import report_obs_error


@pytest.fixture
def sess(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Session:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    return Session("test-obs-errors")


def _errors(session: Session) -> list[dict[str, object]]:
    path = session.dir / "errors.jsonl"
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _event(task_index: int = 0) -> AgenticEvent:
    return AgenticEvent(
        task_index=task_index,
        iteration=1,
        provider="claude",
        model="m",
        kind="text",
        ts="2026-07-05T00:00:00Z",
    )


def test_report_obs_error_writes_record(sess: Session) -> None:
    report_obs_error(sess, "unit.src", "write", ValueError("boom"), detail="d")
    recs = _errors(sess)
    assert len(recs) == 1
    r = recs[0]
    assert r["source"] == "unit.src"
    assert r["op"] == "write"
    assert r["error"] == "ValueError: boom"
    assert r["detail"] == "d"
    assert r["ts"]


def test_report_obs_error_never_raises_on_bad_session(tmp_path: Path) -> None:
    """A reporter that can't even write degrades to a log, never raises."""

    class _Broken:
        dir = tmp_path / "nope"

        def _ensure_dir(self) -> None:
            raise OSError("disk full")

    # Must not raise despite the session being unusable.
    report_obs_error(_Broken(), "unit.src", "write", ValueError("x"))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("source_fn", "op"),
    [
        ("agentic.append_jsonl", "write"),
        ("agentic._persist_exchange", "write"),
    ],
)
def test_write_failure_routed_not_swallowed(
    sess: Session, monkeypatch: pytest.MonkeyPatch, source_fn: str, op: str
) -> None:
    """Injecting a write failure surfaces to errors.jsonl and does not raise."""
    import builtins

    real_open = builtins.open

    def _boom(path: object, *a: object, **k: object) -> object:
        if str(path).endswith(".jsonl") and "errors.jsonl" not in str(path):
            raise OSError("write failed")
        return real_open(path, *a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "open", _boom)

    if source_fn == "agentic.append_jsonl":
        append_jsonl(sess, _event())  # must not raise
    else:
        from splinter.obs.agentic import ExchangeEvent, _persist_exchange

        _persist_exchange(sess, ExchangeEvent(stage="run", task_index=0, iteration=1,
                                              prompt="p", response="r"))

    recs = _errors(sess)
    assert any(r["source"] == source_fn and r["op"] == op for r in recs)


def test_decode_failure_routed_to_errors(sess: Session) -> None:
    """Malformed JSONL lines are recorded (decode), not just warned, and skipped."""
    sess._ensure_dir()
    trace_dir = sess.dir / "trace"
    trace_dir.mkdir(parents=True, exist_ok=True)
    # one valid line + two malformed
    good = json.dumps({
        "task_index": 0, "iteration": 1, "provider": "c", "model": "m",
        "kind": "text", "ts": "t", "schema_version": 1, "extra": {},
    })
    (trace_dir / "agentic-0.jsonl").write_text(good + "\n{bad json\n" + '{"task_index": 0}\n')

    events = load_agentic_events(sess)  # must not raise

    assert len(events) == 1  # only the valid record survives
    recs = _errors(sess)
    assert any(r["source"] == "agentic.load_agentic_events" and r["op"] == "decode" for r in recs)


def test_read_events_decode_failure_routed(sess: Session) -> None:
    sess._ensure_dir()
    agentic_dir = sess.dir / "agentic"
    agentic_dir.mkdir(parents=True, exist_ok=True)
    (agentic_dir / "task-0.jsonl").write_text("{not json\n")

    events = read_events(sess, 0)  # must not raise

    assert events == []
    assert any(r["source"] == "agentic.read_events" and r["op"] == "decode" for r in _errors(sess))


def test_successful_writes_leave_no_errors(sess: Session) -> None:
    """Happy path writes nothing to errors.jsonl."""
    append_jsonl(sess, _event())
    assert load_agentic_events(sess)  # round-trips
    assert _errors(sess) == []
