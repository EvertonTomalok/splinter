"""Live-directive routing: per-task queues, no kill+restart.

The TUI queues a directive for a specific running parallel task (or the shared
queue); the runner drains it at the top of the task's next iteration. A directive
scoped to one task must never leak into another task's loop.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from splinter.memory.session import Session


@pytest.fixture
def session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Session:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    return Session("ses_directive")


def test_shared_queue_roundtrip(session: Session) -> None:
    session.queue_live_command("do X")
    # Any task loop, or a single-task loop, drains the shared queue.
    assert "do X" in session.pop_live_commands(task_no=1)
    # One-shot: cleared after popping.
    assert session.pop_live_commands(task_no=1) == ""


def test_scoped_directive_only_reaches_its_task(session: Session) -> None:
    session.queue_live_command("steer task 2", task_no=2)
    # Task 1 drains only the shared queue + its own — never task 2's.
    assert session.pop_live_commands(task_no=1) == ""
    # Task 2 gets it.
    assert "steer task 2" in session.pop_live_commands(task_no=2)
    assert session.pop_live_commands(task_no=2) == ""


def test_task_loop_drains_shared_plus_scoped(session: Session) -> None:
    session.queue_live_command("broadcast")
    session.queue_live_command("just task 3", task_no=3)
    popped = session.pop_live_commands(task_no=3)
    assert "broadcast" in popped
    assert "just task 3" in popped


def test_multiple_directives_same_task_accumulate(session: Session) -> None:
    session.queue_live_command("first", task_no=1)
    session.queue_live_command("second", task_no=1)
    popped = session.pop_live_commands(task_no=1)
    assert "first" in popped
    assert "second" in popped


def test_empty_when_nothing_queued(session: Session) -> None:
    assert session.pop_live_commands() == ""
    assert session.pop_live_commands(task_no=5) == ""
