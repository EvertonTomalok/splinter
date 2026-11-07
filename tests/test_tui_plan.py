"""Plan mode TUI (``PlanSessionApp``) driven with every planner call mocked."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from textual.widgets import Button, RichLog, TextArea

from splinter import plan_session
from splinter.memory.session import Session
from splinter.prd_session import Turn

SIZE = (160, 44)


@pytest.fixture
def task_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path / "home"))
    path = tmp_path / "hello.yaml"
    path.write_text(
        yaml.safe_dump(
            {"description": "Add a hello command", "acceptance": "- prints hello", "effort": "hard"}
        )
    )
    return path


@pytest.fixture
def planner(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[dict[str, Any]]]:
    calls: dict[str, list[dict[str, Any]]] = {}

    def fake(name: str, text: str) -> Callable[..., Turn]:
        def _f(*args: Any, **kw: Any) -> Turn:
            calls.setdefault(name, []).append({"args": args, **kw})
            return Turn(text=text, session_id=f"sid-{name}", model="sonnet")

        return _f

    monkeypatch.setattr(plan_session, "open_plan_questions", fake("q", "1. Approach?\n A) x"))
    monkeypatch.setattr(plan_session, "refine_plan", fake("refine", "## Decisions\n- x"))
    monkeypatch.setattr(plan_session, "write_plan", fake("write", "# Plan\n\n1. Edit cli.py"))
    monkeypatch.setattr(plan_session, "revise_plan", fake("revise", "1. Edit tui.py"))
    return calls


def _app(task_file: Path, **extra: Any) -> Any:
    from splinter.tui_plan import PlanSessionApp

    kwargs = {"task_path": str(task_file), "strategy": None, "no_ground": True, **extra}
    return PlanSessionApp(Session("ses_plan"), kwargs)


async def _settle(app: Any, pilot: Any) -> None:
    for _ in range(2):
        await app.workers.wait_for_complete()
        await pilot.pause()


def _convo(app: Any) -> str:
    return "\n".join(str(line.text) for line in app.query_one("#convo", RichLog).lines)


def _buttons(app: Any) -> list[str]:
    return [b.id or "" for b in app.query("#actions Button")]


def test_full_flow_questions_answer_plan_run(
    task_file: Path, planner: dict[str, list[dict[str, Any]]]
) -> None:
    app = _app(task_file)

    async def drive() -> None:
        async with app.run_test(size=SIZE) as pilot:
            await _settle(app, pilot)
            assert "Approach?" in _convo(app)
            task_edit = app.query_one("#task-edit", TextArea)
            assert task_edit.border_title == "Task · hello.yaml"
            assert "Add a hello command" in task_edit.text
            assert app.title == "splinter · plan"
            assert "chat" in app.sub_title
            assert _buttons(app) == ["send", "write-plan", "cowabunga"]

            task_edit.text = "## Task\nAdd a hello command, edited\n\n## Acceptance Criteria\n- ok"
            app.query_one("#entry", TextArea).text = "1A"
            await pilot.click("#send")
            await _settle(app, pilot)
            current_task = planner["refine"][0]["current_task"]
            assert current_task.startswith("## Task\nAdd a hello command, ed")
            assert planner["refine"][0]["resume"] == "sid-q"

            await pilot.click("#write-plan")
            await _settle(app, pilot)
            plan_edit = app.query_one("#plan-edit", TextArea)
            assert plan_edit.display and not task_edit.display
            assert plan_edit.border_title == "Plan"
            assert plan_edit.text == "1. Edit cli.py"
            assert planner["write"][0]["autodecide"] is False
            assert _buttons(app) == ["send", "run", "cowabunga"]

    asyncio.run(drive())
    session = Session("ses_plan")
    assert session.read("knowledge/plan.md") == "# Plan\n\n1. Edit cli.py\n"
    data = yaml.safe_load(session.read("task.yaml"))
    assert data["description"] == "Add a hello command, edited" and data["effort"] == "hard"
    status = session.read_status()
    assert status["state"] == "refining" and status["source"] == "task"
    assert status["phase"] == "review" and status["claude_session"] == "sid-write"
    assert "Approach?" in session.read("convo.md")


def _seed_review(task_file: Path) -> Session:
    session = Session("ses_plan")
    plan_session.save_task(session, "Resumed task", "- ok", {"effort": "hard"})
    plan_session.save_plan(session, "1. Saved step")
    session.write("convo.md", "\x00".join(["[cyan]> 1A[/]", "earlier reply"]))
    session.set_status("refining", source="task", phase="review", claude_session="sid-old",
                       task_path=str(session.dir / "task.yaml"), task_name="hello.yaml",
                       strategy="raphael")
    return session


def _resumed_app(session: Session) -> Any:
    from splinter.tui_plan import PlanSessionApp

    return PlanSessionApp(session, {"resume": True, "no_ground": True, "strategy": None,
                                    "task_path": str(session.dir / "task.yaml")})


def test_run_from_review_writes_plan_task_and_status(
    task_file: Path, planner: dict[str, Any]
) -> None:
    session = _seed_review(task_file)
    app = _resumed_app(session)

    async def drive() -> None:
        async with app.run_test(size=SIZE) as pilot:
            await _settle(app, pilot)
            app.query_one("#plan-edit", TextArea).text = "1. Edit cli.py\n2. Add tests"
            await pilot.click("#run")
            await _settle(app, pilot)
        assert app.return_value == 0

    asyncio.run(drive())
    assert session.read("knowledge/plan.md") == "# Plan\n\n1. Edit cli.py\n2. Add tests\n"
    assert session.read("knowledge/plan-1.md") == "# Plan\n\n1. Edit cli.py\n2. Add tests\n"
    data = yaml.safe_load(session.read("task.yaml"))
    assert data == {"effort": "hard", "description": "Resumed task", "acceptance": "- ok"}
    status = session.read_status()
    assert status["phase"] == "run" and status["strategy"] == "raphael"
    assert status["task_path"] == str(session.dir / "task.yaml")


def test_typed_cowabunga_writes_plan_and_runs(
    task_file: Path, planner: dict[str, list[dict[str, Any]]]
) -> None:
    app = _app(task_file, strategy="cascade")

    async def drive() -> None:
        async with app.run_test(size=SIZE) as pilot:
            await _settle(app, pilot)
            app.query_one("#entry", TextArea).text = "cowabunga"
            await pilot.press("ctrl+s")
            await _settle(app, pilot)
        assert app.return_value == 0

    asyncio.run(drive())
    assert planner["write"][0]["autodecide"] is True
    session = Session("ses_plan")
    assert session.read("knowledge/plan.md") == "# Plan\n\n1. Edit cli.py\n"
    assert session.read_status()["strategy"] == "cascade"


def test_review_typed_text_revises_plan(
    task_file: Path, planner: dict[str, list[dict[str, Any]]]
) -> None:
    app = _app(task_file)

    async def drive() -> None:
        async with app.run_test(size=SIZE) as pilot:
            await _settle(app, pilot)
            app.query_one("#entry", TextArea).text = "plan"
            await pilot.click("#send")
            await _settle(app, pilot)
            app.query_one("#entry", TextArea).text = "use tui.py instead"
            await pilot.click("#send")
            await _settle(app, pilot)
            assert planner["revise"][0]["current_plan"] == "1. Edit cli.py"
            assert app.query_one("#plan-edit", TextArea).text == "1. Edit tui.py"
            assert app.phase == "review"

    asyncio.run(drive())


def test_escape_asks_confirm_quit(task_file: Path, planner: dict[str, Any]) -> None:
    from splinter.tui import ConfirmQuit

    app = _app(task_file)

    async def drive() -> None:
        async with app.run_test(size=SIZE) as pilot:
            await _settle(app, pilot)
            await pilot.press("escape")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmQuit)
            await pilot.click("#stay")
            await pilot.pause()
            assert not isinstance(app.screen, ConfirmQuit)
            await pilot.press("escape")
            await pilot.pause()
            await pilot.click("#leave")
            await pilot.pause()
        assert app.return_value is None

    asyncio.run(drive())


def test_model_error_shows_retry_and_keeps_app_open(
    task_file: Path, planner: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts: list[int] = []

    def flaky(*args: Any, **kw: Any) -> Turn:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("429 rate limited")
        return Turn(text="1. Retry question?", session_id="sid-r", model="sonnet")

    monkeypatch.setattr(plan_session, "open_plan_questions", flaky)
    app = _app(task_file)

    async def drive() -> None:
        async with app.run_test(size=SIZE) as pilot:
            await _settle(app, pilot)
            await pilot.pause(0.1)
            assert app.is_running
            assert "429 rate limited" in _convo(app)
            assert "retry" in _buttons(app)
            assert not app.query_one("#entry", TextArea).disabled
            await pilot.click("#retry")
            await _settle(app, pilot)
            assert len(attempts) == 2
            assert "Retry question?" in _convo(app)
            assert "retry" not in _buttons(app)

    asyncio.run(drive())


def test_unreadable_task_file_exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path / "home"))
    app = _app(tmp_path / "missing.yaml")
    app.FATAL_EXIT_DELAY = 0.3

    async def drive() -> None:
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause()
            assert "cannot read task" in _convo(app)
            await pilot.pause(0.5)
        assert app.return_value is None

    asyncio.run(drive())


def test_resume_reenters_review(task_file: Path, planner: dict[str, Any]) -> None:
    session = _seed_review(task_file)
    app = _resumed_app(session)

    async def drive() -> None:
        async with app.run_test(size=SIZE) as pilot:
            await _settle(app, pilot)
            assert app.phase == "review"
            assert app.claude_session == "sid-old"
            assert app.query_one("#plan-edit", TextArea).text == "1. Saved step"
            assert "Resumed task" in app.query_one("#task-edit", TextArea).text
            assert "earlier reply" in _convo(app)
            assert "run" in _buttons(app)
            assert "q" not in planner

    asyncio.run(drive())


def test_buttons_are_clickable_labels(task_file: Path, planner: dict[str, Any]) -> None:
    app = _app(task_file)

    async def drive() -> None:
        async with app.run_test(size=SIZE) as pilot:
            await _settle(app, pilot)
            labels = [str(b.label) for b in app.query("#actions Button").results(Button)]
            assert labels == ["Send (Ctrl+S)", "Write plan", "Cowabunga"]

    asyncio.run(drive())
