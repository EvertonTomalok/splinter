from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from splinter.memory.session import Session
from splinter.run_view import build_run_view
from splinter.run_view_rows import running_state
from splinter.tui_run_chrome import _progress
from tests.fixtures_run_view import FIXTURE_LADDER, build_session


def _parallel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Session:
    session = build_session(tmp_path, monkeypatch, stories=4, live=1, failed=0, asked=0)
    session.write("loop.md", "# Task 1/4: US-001: One\n\n## Iteration 1\n- verdict: PASS\n")
    session.set_status("running", parallel=True, tasks=4, task_index=1)
    session.write_run_live(2, {"provider": "claude", "model": "opus", "task_no": 2})
    session.write_run_live(3, {"provider": "claude", "model": "opus", "task_no": 3})
    return session


def _view(session: Session, state: str) -> Any:
    return build_run_view(session, state, datetime.now(timezone.utc), FIXTURE_LADDER)


def test_every_story_with_a_live_engine_shows_as_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    view = _view(_parallel(tmp_path, monkeypatch), "RUNNING")
    assert view.now == (2, 3)
    assert 2 not in view.next_up and 3 not in view.next_up


def test_a_dead_run_does_not_claim_running_stories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    view = _view(_parallel(tmp_path, monkeypatch), "INTERRUPTED")
    assert view.now == ()


def test_header_counts_the_stories_running_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    view = _view(_parallel(tmp_path, monkeypatch), "RUNNING")
    text = _progress(view, view.stories[1], 5)
    assert "running" in text and "story" not in text
    assert "1/4" in text


def _snap(state: str = "RUNNING", parallel: bool = True) -> Any:
    return SimpleNamespace(state=state, status={"parallel": parallel, "max_iterations": 5})


@pytest.mark.parametrize(
    ("state", "verdict", "iters", "running", "snap", "expected"),
    [
        ("queued", "", 0, True, _snap(), "live"),
        ("queued", "", 0, False, _snap(), "queued"),
        ("fail", "RETRY", 1, False, _snap(), "live"),
        ("fail", "RETRY", 5, False, _snap(), "fail"),
        ("fail", "RETRY", 1, False, _snap(parallel=False), "fail"),
        ("fail", "RETRY", 1, False, _snap(state="FAILED"), "fail"),
        ("fail", "", 0, False, _snap(), "fail"),
        ("pass", "PASS", 2, True, _snap(), "pass"),
        ("asked", "ASK_USER", 1, True, _snap(), "asked"),
    ],
)
def test_running_state(
    state: Any, verdict: str, iters: int, running: bool, snap: Any, expected: str
) -> None:
    assert running_state(state, verdict, iters, running, snap) == expected
