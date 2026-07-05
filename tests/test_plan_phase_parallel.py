"""Plan phase fans out via the US-001 bounded primitive (no one-by-one planning)."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from splinter.agents.runner import Task
from splinter.memory.session import Session
from splinter.models.roster import load_ladder
from splinter.strategies import direct as direct_mod
from splinter.strategies.direct import DirectStrategy


@pytest.fixture
def tmp_session(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Session:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    return Session("test-plan-parallel")


def test_plan_phase_runs_concurrently(
    monkeypatch: pytest.MonkeyPatch, tmp_session: Session
) -> None:
    """Two plans must be in flight at once — a serial loop breaks the barrier."""
    barrier = threading.Barrier(2, timeout=10)

    def fake_make_plan(task: Task, ladder: object, code_ctx: str, **kwargs: object) -> str:
        barrier.wait()
        return f"plan for {task.description}"

    monkeypatch.setattr(direct_mod, "_make_plan", fake_make_plan)

    tasks = [Task(description=f"task-{i}", acceptance="ok") for i in range(2)]

    DirectStrategy()._plan_all_tasks(
        tasks,
        tmp_session,
        load_ladder(),
        localization="",
        max_concurrency=2,
    )

    for i in range(2):
        plan = tmp_session.read(f"knowledge/plan-{i + 1}.md")
        assert f"plan for task-{i}" in plan


def test_plan_phase_reuses_existing_plans(
    monkeypatch: pytest.MonkeyPatch, tmp_session: Session
) -> None:
    """Existing plan files are never regenerated (no planner call at all)."""
    tmp_session.write("knowledge/plan-1.md", "# Plan\n\nold plan\n")
    tmp_session.write("knowledge/plan-2.md", "# Plan\n\nold plan 2\n")

    def fake_make_plan(*args: object, **kwargs: object) -> str:
        raise AssertionError("planner must not be called when plans exist")

    monkeypatch.setattr(direct_mod, "_make_plan", fake_make_plan)

    tasks = [Task(description=f"task-{i}", acceptance="ok") for i in range(2)]

    DirectStrategy()._plan_all_tasks(
        tasks,
        tmp_session,
        load_ladder(),
        localization="",
        max_concurrency=2,
    )

    assert "old plan" in tmp_session.read("knowledge/plan-1.md")
    assert "old plan 2" in tmp_session.read("knowledge/plan-2.md")


def test_plan_phase_resume_ignores_existing_and_defers_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_session: Session
) -> None:
    """On resume the bulk plan phase never plans: existing reused, missing deferred."""
    tmp_session.write("knowledge/plan-1.md", "# Plan\n\nold plan\n")

    def fake_make_plan(*args: object, **kwargs: object) -> str:
        raise AssertionError("planner must not run in bulk phase on resume")

    monkeypatch.setattr(direct_mod, "_make_plan", fake_make_plan)

    tasks = [Task(description=f"task-{i}", acceptance="ok") for i in range(2)]

    DirectStrategy()._plan_all_tasks(
        tasks,
        tmp_session,
        load_ladder(),
        localization="",
        resume=True,
        max_concurrency=2,
    )

    assert "old plan" in tmp_session.read("knowledge/plan-1.md")
    assert not tmp_session.read("knowledge/plan-2.md").strip()


def test_plan_phase_skips_completed_tasks(
    monkeypatch: pytest.MonkeyPatch, tmp_session: Session
) -> None:
    """Checkpointed-done tasks are never planned — even with force_replan."""
    planned: list[str] = []
    lock = threading.Lock()

    def fake_make_plan(task: Task, ladder: object, code_ctx: str, **kwargs: object) -> str:
        with lock:
            planned.append(task.id)
        return "plan"

    monkeypatch.setattr(direct_mod, "_make_plan", fake_make_plan)

    tasks = [
        Task(description="first", acceptance="ok", id="US-001"),
        Task(description="second", acceptance="ok", id="US-002"),
    ]

    DirectStrategy()._plan_all_tasks(
        tasks,
        tmp_session,
        load_ladder(),
        localization="",
        force_replan=True,
        done_ids={"US-001"},
        max_concurrency=2,
    )

    assert planned == ["US-002"]


def test_plan_phase_concurrency_capped(
    monkeypatch: pytest.MonkeyPatch, tmp_session: Session
) -> None:
    """max_concurrency=1 forces serial execution — overlap counter never exceeds 1."""
    lock = threading.Lock()
    active = 0
    max_active = 0

    def fake_make_plan(task: Task, ladder: object, code_ctx: str, **kwargs: object) -> str:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        with lock:
            active -= 1
        return "plan"

    monkeypatch.setattr(direct_mod, "_make_plan", fake_make_plan)

    tasks = [Task(description=f"task-{i}", acceptance="ok") for i in range(3)]

    DirectStrategy()._plan_all_tasks(
        tasks,
        tmp_session,
        load_ladder(),
        localization="",
        max_concurrency=1,
    )

    assert max_active == 1
