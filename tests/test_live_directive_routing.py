"""Live-directive routing: per-task queues, no kill+restart.

The TUI queues a directive for a specific running parallel task (or the shared
queue); the runner drains it at the top of the task's next iteration. A directive
scoped to one task must never leak into another task's loop.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from splinter.agents.acceptance import AcceptanceMode
from splinter.memory.session import Session
from splinter.strategies.agents_ctx import BriefCtx


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


def test_pending_directive_stories_lists_each_scoped_queue(session: Session) -> None:
    for story in (2, 7, 9):
        session.queue_live_command(f"steer {story}", task_no=story)
    session.queue_live_command("shared")
    (session.dir / "pending_directive.task-x.txt").write_text("not a story")
    assert session.pending_directive_stories() == {2, 7, 9}


def _ctx(story_count: int) -> BriefCtx:
    return BriefCtx(
        slug="s",
        run_dir=Path("."),
        guard=MagicMock(),
        pool=MagicMock(),
        integration=MagicMock(),
        mode=AcceptanceMode.GATE,
        stall_min=1.0,
        max_attempts=1,
        timeout=1,
        direct=MagicMock(),
        attempts=MagicMock(),
        stories={},
        story_count=story_count,
    )


def test_take_guidance_pops_exactly_the_pending_stories(session: Session) -> None:
    for story in (2, 7, 9):
        session.queue_live_command(f"steer {story}", task_no=story)
    ctx = _ctx(60)
    with patch.object(session, "directive_pending") as per_story_stat:
        ctx.take_guidance(session)
    per_story_stat.assert_not_called()
    assert sorted(ctx.story_guidance) == [2, 7, 9]
    assert session.pending_directive_stories() == set()


def test_take_guidance_leaves_a_story_outside_the_run_alone(session: Session) -> None:
    directive = session.queue_live_command("old run", task_no=99)
    ctx = _ctx(3)
    ctx.take_guidance(session)
    assert ctx.story_guidance == {}
    assert not session.receipt_exists(directive)
