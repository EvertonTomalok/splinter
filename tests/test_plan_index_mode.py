"""Plan mode on a task plan README: ask across the plan, one plan section per task."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from textual.widgets import RichLog, TextArea

from splinter import plan_session
from splinter.memory.session import Session
from splinter.prd_session import Turn
from splinter.task_plan import load_plan_index

SIZE = (160, 44)
_README = """# Plan

Orchestrator: keep tui.py diffs small.

| Task | What | Model | Depends on |
|---|---|---|---|
| [T01](T01-a.md) | first | Sonnet medium | — |
| [T02](T02-b.md) | second | Opus high | T01 |
"""
_PLAN = "## T01\n1. Add a.py\n\n## T02\n1. Use a.py in b.py\n"


@pytest.fixture
def plan_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path / "home"))
    folder = tmp_path / "tasks" / "plan"
    folder.mkdir(parents=True)
    (folder / "README.md").write_text(_README)
    (folder / "T01-a.md").write_text("# T01\n\nBuild a.\n\n## Acceptance\n- a works\n")
    (folder / "T02-b.md").write_text("# T02\n\nBuild b.\n\n## Acceptance\n- b works\n")
    return folder


@pytest.fixture
def planner(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[dict[str, Any]]]:
    calls: dict[str, list[dict[str, Any]]] = {}

    def fake(name: str, text: str) -> Callable[..., Turn]:
        def _f(*args: Any, **kw: Any) -> Turn:
            calls.setdefault(name, []).append({"args": args, **kw})
            return Turn(text=text, session_id=f"sid-{name}", model="sonnet")

        return _f

    monkeypatch.setattr(plan_session, "open_plan_questions", fake("q", "1. Shared helper?\n A) y"))
    monkeypatch.setattr(plan_session, "refine_plan", fake("refine", "## Decisions\n- y"))
    monkeypatch.setattr(plan_session, "write_plan", fake("write", _PLAN))
    return calls


def test_plan_bundle_has_readme_and_every_task(plan_dir: Path) -> None:
    bundle = plan_session.plan_bundle(load_plan_index(plan_dir))
    assert bundle.startswith("# Task plan: 2 tasks (T01, T02)")
    assert "Orchestrator: keep tui.py diffs small." in bundle
    assert "# T01\n" in bundle and "Acceptance:\n- b works" in bundle


def test_split_task_plans_maps_sections_to_ids() -> None:
    plans = plan_session.split_task_plans(_PLAN + "\n## T09\n1. stray\n", ("T01", "T02"))
    assert plans == {"T01": "1. Add a.py", "T02": "1. Use a.py in b.py"}


def test_save_task_plans_writes_by_id_and_reports_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_plans")
    session.write("knowledge/plan-T03.md", "# Plan\n\nstale\n")
    missing = plan_session.save_task_plans(session, _PLAN, ("T01", "T02", "T03"))
    assert missing == ["T03"]
    assert session.read("knowledge/plan-T01.md") == "# Plan\n\n1. Add a.py\n"
    assert session.read("knowledge/plan-T03.md") == ""
    assert session.read(plan_session.PLAN_ALL_FILE).startswith("# Plan\n\n## T01")
    assert session.read("knowledge/plan.md") == ""


def test_write_plan_asks_for_one_section_per_task(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def _ask(prompt: str, **_: Any) -> Turn:
        seen.append(prompt)
        return Turn(text="x", session_id="s", model="m")

    monkeypatch.setattr(plan_session, "_ask", _ask)
    monkeypatch.setattr(plan_session, "load_standards", lambda: "")
    plan_session.write_plan(
        resume="s", current_task="task", autodecide=False, task_ids=("T01", "T02")
    )
    assert "headed exactly `## <task id>`, in this order: T01, T02" in seen[0]


def test_readme_opens_plan_mode_and_run_saves_plan_per_task(
    plan_dir: Path, planner: dict[str, list[dict[str, Any]]]
) -> None:
    from splinter.tui_plan import PlanSessionApp

    readme = plan_dir / "README.md"
    app = PlanSessionApp(
        Session("ses_idx"), {"task_path": str(readme), "strategy": None, "no_ground": True}
    )

    async def drive() -> None:
        async with app.run_test(size=SIZE) as pilot:
            for _ in range(2):
                await app.workers.wait_for_complete()
                await pilot.pause()
            convo = "\n".join(str(x.text) for x in app.query_one("#convo", RichLog).lines)
            assert "Shared helper?" in convo
            task_edit = app.query_one("#task-edit", TextArea)
            assert task_edit.border_title == "Task · README.md · 2 tasks"
            assert "Build b." in planner["q"][0]["task_md"]
            await pilot.click("#write-plan")
            for _ in range(2):
                await app.workers.wait_for_complete()
                await pilot.pause()
            assert planner["write"][0]["task_ids"] == ("T01", "T02")
            await pilot.click("#run")
            await pilot.pause()

    asyncio.run(drive())
    session = Session("ses_idx")
    assert session.read("knowledge/plan-T02.md") == "# Plan\n\n1. Use a.py in b.py\n"
    status = session.read_status()
    assert status["plan_index"] is True and status["task_path"] == str(readme)
    assert status["strategy"] == "cascade" and status["phase"] == "run"
    assert "Orchestrator: keep tui.py diffs small." in session.read("knowledge/plan-index.md")


def test_handoff_runs_the_readme_not_task_yaml(
    plan_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import splinter.tui as tui
    from splinter.tui_plan import _handoff

    session = Session("ses_hand")
    readme = str(plan_dir / "README.md")
    session.set_status("refining", phase="run", plan_index=True, task_path=readme,
                       strategy="cascade")
    seen: dict[str, Any] = {}

    def _run(kwargs: dict[str, Any], session: Session) -> int:
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(tui, "run_with_tui", _run)
    assert _handoff(session, {"task_path": readme, "strategy": None}, 0) == 0
    assert seen["task_path"] == readme and seen["strategy"] == "cascade"


def test_bulk_plan_phase_adopts_plans_written_by_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import MagicMock, patch

    from splinter.agents.runner import Task
    from splinter.strategies.direct import DirectStrategy

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_adopt")
    session.write("knowledge/plan-T02.md", "# Plan\n\n1. Use a.py in b.py\n")
    tasks = [Task(description="T02: b", acceptance="", id="T02"),
             Task(description="T01: a", acceptance="", id="T01")]
    with patch("splinter.strategies.direct._make_plan", return_value="fresh") as make_plan:
        DirectStrategy()._plan_all_tasks(tasks, session, MagicMock(), "")
    assert session.read("knowledge/plan-1.md") == "# Plan\n\n1. Use a.py in b.py\n"
    assert make_plan.call_count == 1 and make_plan.call_args[0][0] is tasks[1]
