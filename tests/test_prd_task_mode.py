"""The ``--prd`` / ``--task`` chips of the draft screen, the task handoff and the diff colours.

No model is called: ``PrdSessionApp._spawn`` is patched, and git runs through a fake runner.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from textual.widgets import TextArea

from splinter.memory.session import Session, new_session_id
from splinter.tui import PrdSessionApp
from splinter.tui_blockword import BlockWord
from splinter.tui_prd_mode import TASK_STEPS, task_source
from splinter.tui_run_views_more import _patch_lines, _stat_lines
from splinter.tui_stepper import PRD_STEPS, PhaseStepper
from splinter.tui_task_pick import TaskPicker, complete, looks_like_path, relative
from splinter.vcs.diff import story_diff, story_worktree


@pytest.fixture
def session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Session:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    return Session(new_session_id())


def _app(session: Session) -> PrdSessionApp:
    app = PrdSessionApp(session, {"prd_path": None, "task_path": None, "cowabunga": False})
    app._spawn = lambda *args, **kwargs: None  # type: ignore[method-assign]
    return app


def test_task_chip_switches_the_draft_to_a_task(session: Session) -> None:
    async def drive() -> None:
        app = _app(session)
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            await app.run_action("mode('task')")
            await pilot.pause()
            assert app.mode == "task"
            assert app.query_one("#stepper", PhaseStepper).steps == TASK_STEPS
            assert app.query_one("#draft-word", BlockWord).word == "TASK"
            await app.run_action("mode('prd')")
            await pilot.pause()
            assert app.query_one("#stepper", PhaseStepper).steps == PRD_STEPS

    asyncio.run(drive())


def test_task_send_hands_the_typed_task_to_the_run(session: Session) -> None:
    async def drive() -> None:
        app = _app(session)
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            await app.run_action("mode('task')")
            await pilot.pause()
            assert not app.query_one("#instructions", TextArea).display
            assert app.focused is app.query_one("#task-path")
            app.query_one(TaskPicker).set_path("rename foo() to bar()")
            app.action_send()
            await pilot.pause()
        assert app.task_handoff == str(session.dir / "task.md")
        assert app.task_plan is False
        assert "rename foo() to bar()" in session.read("task.md")

    asyncio.run(drive())


def test_ctrl_t_plans_the_task_first(session: Session, tmp_path: Path) -> None:
    task = tmp_path / "t.yaml"
    task.write_text("description: x\n")

    async def drive() -> None:
        app = _app(session)
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            await app.run_action("mode('task')")
            await pilot.pause()
            app.query_one(TaskPicker).set_path(str(task))
            await pilot.press("ctrl+t")
            await pilot.pause()
        assert app.task_plan is True
        assert app.task_handoff == str(task.resolve())

    asyncio.run(drive())


def test_task_source_keeps_an_existing_path(session: Session, tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("# plan\n")
    assert task_source(session, f"  {readme}  ") == readme.resolve()
    assert not (session.dir / "task.md").exists()


def test_story_diff_asks_git_for_a_plain_patch(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake(argv: list[str], cwd: Path) -> str:
        calls.append(argv)
        return ""

    story_diff(tmp_path, fake)
    assert calls == [
        ["diff", "--no-color", "--stat"],
        ["diff", "--no-color"],
        ["ls-files", "--others", "--exclude-standard"],
    ]


def test_story_diff_lists_new_files_and_reports_git_errors(tmp_path: Path) -> None:
    def fake(argv: list[str], cwd: Path) -> str:
        if argv[0] == "ls-files":
            return "src/new.py\n"
        return " src/a.py | 2 +-\n" if "--stat" in argv else ""

    diff = story_diff(tmp_path, fake)
    assert diff.stat.splitlines()[0] == " src/new.py | new"

    def broken(argv: list[str], cwd: Path) -> str:
        raise RuntimeError("fatal: not a git repository")

    assert story_diff(tmp_path, broken).error == "fatal: not a git repository"


def test_story_worktree_reads_the_worktree_registry(tmp_path: Path) -> None:
    wt = tmp_path / "wt"
    wt.mkdir()

    class FakeSession:
        def read_worktrees(self) -> dict[str, dict[str, str]]:
            return {"US-007": {"path": str(wt), "branch": "b"}}

        def read_run_live(self, task_no: int) -> None:
            return None

    assert story_worktree(FakeSession(), 3, "US-007") == wt
    assert story_worktree(FakeSession(), 7) == wt
    assert story_worktree(FakeSession(), 4) is None


def test_patch_and_stat_are_coloured() -> None:
    patch = (
        "diff --git a/x.py b/x.py\nindex 1..2 100644\n--- a/x.py\n+++ b/x.py\n"
        "@@ -1,2 +1,2 @@\n keep\n-old\n+new\n"
    )
    text = _patch_lines(patch)
    assert "diff --git" not in text.plain and "index 1..2" not in text.plain
    assert text.plain.startswith("x.py")
    styles = {text.plain[span.start : span.end].strip(): str(span.style) for span in text.spans}
    assert "#d7f58a on #141a08" in styles["+ new"]
    assert "#ff8a80 on #1f0d0b" in styles["- old"]
    assert "strike" not in styles["- old"]
    stat = _stat_lines(" x.py | 3 ++-\n 1 file changed\n")
    marks = {stat.plain[span.start : span.end]: str(span.style) for span in stat.spans}
    assert marks["++"] == "#c6f432" and marks["-"] == "#ff5a4e"


def test_spend_switch_repaints_at_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    from splinter.tui import RunApp
    from tests.fixtures_run_view import FIXTURE_LADDER, build_session

    run_session = build_session(tmp_path, monkeypatch, jev=True)
    run_session.set_status("running", pid=os.getpid())
    monkeypatch.setattr("splinter.models.roster.load_ladder", lambda: FIXTURE_LADDER)
    monkeypatch.setattr(RunApp, "_run_pipeline_worker", lambda self, **kw: None)
    app = RunApp(run_session, {"prd_path": "p.md"})
    painted: list[Any] = []

    async def drive() -> None:
        async with app.run_test(size=(160, 44)) as pilot:
            await app.run_action("open_spend('cost')")
            await pilot.pause()
            original = app._render_view

            def counted() -> None:
                painted.append(app._view_key)
                original()

            monkeypatch.setattr(app, "_render_view", counted)
            await app.run_action("set_spend_by('time')")
            assert app._view_key is not None and "time" in app._view_key

    asyncio.run(drive())
    assert painted


def test_path_completes_and_shows_relative(tmp_path: Path) -> None:
    (tmp_path / "tasks").mkdir()
    (tmp_path / "tasks" / "c2-focus").mkdir()
    (tmp_path / "tasks" / "readme.md").write_text("x")
    assert complete("ta", tmp_path) == "tasks/"
    assert complete("tasks/c", tmp_path) == "tasks/c2-focus/"
    assert complete("tasks/re", tmp_path) == "tasks/readme.md"
    assert complete("nope", tmp_path) == "nope"
    assert relative(tmp_path / "tasks", tmp_path) == "tasks/"
    assert relative(tmp_path / "tasks" / "readme.md", tmp_path) == "tasks/readme.md"


def test_looks_like_path_tells_a_path_from_a_task() -> None:
    assert looks_like_path("tasks/x.yaml")
    assert looks_like_path("plan.md")
    assert looks_like_path("~/plans/")
    assert not looks_like_path("rename foo() to bar()")
    assert not looks_like_path("refactor")


def test_a_missing_path_is_not_run_as_a_task(session: Session) -> None:
    async def drive() -> None:
        app = _app(session)
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            await app.run_action("mode('task')")
            await pilot.pause()
            app.query_one(TaskPicker).set_path("tasks/no-such-file.yaml")
            await pilot.press("enter")
            await pilot.pause()
            assert app.is_running
            assert app.task_handoff is None
            assert not (session.dir / "task.md").exists()

    asyncio.run(drive())


def test_enter_runs_and_tab_moves_to_the_tree(session: Session) -> None:
    async def drive() -> None:
        app = _app(session)
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            await app.run_action("mode('task')")
            await pilot.pause()
            await pilot.press("tab")
            await pilot.pause()
            assert app.focused is app.query_one("#task-tree")
            app.query_one(TaskPicker).focus_path()
            app.query_one(TaskPicker).set_path("rename foo() to bar()")
            await pilot.press("enter")
            await pilot.pause()
        assert app.task_handoff == str(session.dir / "task.md")

    asyncio.run(drive())
