"""Regression: a parallel run must resume in parallel.

The bug: ``parallel``/``max_concurrency`` were never persisted to ``status.json``
at run start, so ``_resume_run`` had nothing to restore and every resume fell
back to sequential. These tests lock the persist -> resume round-trip.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from splinter.memory.session import Session
from splinter.tui import _resume_run


def _capture_run_with_tui(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch ``run_with_tui`` (the tail call of ``_resume_run``) to record the
    kwargs it would launch the pipeline with, instead of running anything."""
    captured: dict[str, Any] = {}

    def _fake(run_kwargs: dict[str, Any], session: Session | None = None) -> int:
        captured.update(run_kwargs)
        return 0

    monkeypatch.setattr("splinter.tui.run_with_tui", _fake)
    return captured


def _seed_session(sid: str, status: dict[str, Any]) -> Session:
    session = Session(sid)
    session.write("prd.md", "---\nfeature: x\n---\n\n# PRD\n")
    session.set_status(status.pop("state", "failed"), **status)
    return session


def test_set_status_persists_parallel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``set_status`` round-trips the parallel flag + concurrency cap to disk."""
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("persist_parallel")
    session.set_status("running", parallel=True, max_concurrency=4)

    read = session.read_status()
    assert read["parallel"] is True
    assert read["max_concurrency"] == 4


def test_resume_restores_parallel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A session that ran in parallel resumes in parallel with its cap."""
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    captured = _capture_run_with_tui(monkeypatch)
    session = _seed_session(
        "resume_parallel",
        {
            "state": "failed",
            "strategy": "cascade",
            "max_iterations": 5,
            "parallel": True,
            "max_concurrency": 3,
            "reason": "user_kill",
        },
    )

    rc = _resume_run(session, session.read_status())

    assert rc == 0
    assert captured["parallel"] is True
    assert captured["max_concurrency"] == 3
    assert captured["resume"] is True


def test_resume_default_max_concurrency_when_blank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Blank cap (``""`` = default cpu-1) resumes as ``None``, not ``""``."""
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    captured = _capture_run_with_tui(monkeypatch)
    session = _seed_session(
        "resume_default_cap",
        {"state": "failed", "strategy": "cascade", "parallel": True, "max_concurrency": ""},
    )

    _resume_run(session, session.read_status())

    assert captured["parallel"] is True
    assert captured["max_concurrency"] is None


def test_resume_legacy_session_without_parallel_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-fix sessions (no ``parallel`` in status) resume sequential, no crash."""
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    captured = _capture_run_with_tui(monkeypatch)
    session = _seed_session(
        "resume_legacy",
        {"state": "failed", "strategy": "cascade", "reason": "user_kill"},
    )

    _resume_run(session, session.read_status())

    assert captured["parallel"] is False
    assert captured["max_concurrency"] is None


def test_reset_resume_keeps_parallel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``reset`` re-runs from the head but must still honour parallel mode."""
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    captured = _capture_run_with_tui(monkeypatch)
    session = _seed_session(
        "resume_reset",
        {"state": "failed", "strategy": "cascade", "parallel": True, "max_concurrency": 2},
    )

    _resume_run(session, session.read_status(), reset=True)

    assert captured["parallel"] is True
    assert captured["max_concurrency"] == 2
    assert captured["resume"] is False  # reset re-runs from the head
