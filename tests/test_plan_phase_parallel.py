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


def test_plan_phase_resume_reuses_existing_and_plans_missing_concurrently(
    monkeypatch: pytest.MonkeyPatch, tmp_session: Session
) -> None:
    """Resume: existing plans reused untouched; missing ones planned in bulk,
    concurrently, BEFORE the run phase — never inside per-task workers."""
    tmp_session.write("knowledge/plan-1.md", "# Plan\n\nold plan\n")
    barrier = threading.Barrier(2, timeout=10)
    planned: list[str] = []
    lock = threading.Lock()

    def fake_make_plan(task: Task, ladder: object, code_ctx: str, **kwargs: object) -> str:
        with lock:
            planned.append(task.description)
        barrier.wait()
        return f"plan for {task.description}"

    monkeypatch.setattr(direct_mod, "_make_plan", fake_make_plan)

    tasks = [Task(description=f"task-{i}", acceptance="ok") for i in range(3)]

    DirectStrategy()._plan_all_tasks(
        tasks,
        tmp_session,
        load_ladder(),
        localization="",
        max_concurrency=2,
    )

    assert "old plan" in tmp_session.read("knowledge/plan-1.md")
    assert sorted(planned) == ["task-1", "task-2"]
    assert "plan for task-1" in tmp_session.read("knowledge/plan-2.md")
    assert "plan for task-2" in tmp_session.read("knowledge/plan-3.md")


def test_plan_task_uses_per_task_localization_and_keeps_shared_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_session: Session
) -> None:
    """Per-task planning (cascade): task N>0 must read localization-N.md and must
    NOT clobber knowledge/plan.md (mirror of task 1's plan only)."""
    tmp_session.write("knowledge/plan.md", "# Plan\n\ntask-0 plan\n")
    tmp_session.write("knowledge/localization-2.md", "loc-for-task-2")
    seen_ctx: list[str] = []

    def fake_make_plan(task: Task, ladder: object, code_ctx: str, **kwargs: object) -> str:
        seen_ctx.append(code_ctx)
        return "plan-2-body"

    monkeypatch.setattr(direct_mod, "_make_plan", fake_make_plan)

    task = Task(description="second", acceptance="ok", id="US-002")
    plan = direct_mod._plan_task(
        task,
        tmp_session,
        load_ladder(),
        localization="global-loc",
        task_index=1,
    )

    assert plan == "plan-2-body"
    assert "loc-for-task-2" in seen_ctx[0]
    assert "plan-2-body" in tmp_session.read("knowledge/plan-2.md")
    assert "task-0 plan" in tmp_session.read("knowledge/plan.md")


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
