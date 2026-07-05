from __future__ import annotations

from pathlib import Path

import pytest

from splinter.memory.session import Session


def _descriptor() -> dict[str, object]:
    return {
        "session_id": "sess-abc",
        "model": "claude-opus-4-8",
        "variant": "default",
        "cwd": "/tmp/wt",
        "provider": "claude",
        "iteration": 3,
    }


def test_write_then_read_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_run_live_round_trip")
    d = _descriptor()
    session.write_run_live(1, d)
    assert session.read_run_live(1) == d


def test_read_absent_returns_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_run_live_absent")
    assert session.read_run_live(2) is None


def test_read_after_clear_returns_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_run_live_cleared")
    session.write_run_live(1, _descriptor())
    session.clear_run_live(1)
    assert session.read_run_live(1) is None


def test_clear_missing_is_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_run_live_noop")
    session.clear_run_live(99)


def test_task_scoping_isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_run_live_scoped")
    d1 = _descriptor()
    d2 = {**_descriptor(), "session_id": "sess-def", "iteration": 4}
    session.write_run_live(1, d1)
    assert session.read_run_live(2) is None
    session.write_run_live(2, d2)
    assert session.read_run_live(1) == d1
