from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from textual.widgets import Static

from tests.fixtures_run_view import FIXTURE_LADDER, build_session

if TYPE_CHECKING:
    import pytest
    from textual.pilot import Pilot

    from splinter.tui import RunApp


def _drive(app: RunApp, body: Callable[[Pilot], Awaitable[None]]) -> None:
    async def run() -> None:
        async with app.run_test(size=(120, 36)) as pilot:
            await pilot.pause()
            await body(pilot)

    asyncio.run(run())


def _app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RunApp:
    from splinter.tui import RunApp

    session = build_session(tmp_path, monkeypatch)
    monkeypatch.setattr("splinter.models.roster.load_ladder", lambda: FIXTURE_LADDER)
    monkeypatch.setattr(RunApp, "_work", lambda self: None)
    return RunApp(session, {})


def _plain(widget: Static) -> str:
    return str(getattr(widget.content, "plain", widget.content))


def _view_text(app: RunApp) -> str:
    static = app.query_one("#view-static", Static)
    console = Console(width=240, color_system=None)
    with console.capture() as capture:
        console.print(static.content)
    return capture.get()


def test_story_selection_drives_focus_and_views(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from splinter.tui_blockword import BlockWord

    app = _app(tmp_path, monkeypatch)

    async def body(pilot: Pilot) -> None:
        await app.run_action("select_story(9)")
        await pilot.pause()
        assert "US-009" in _plain(app.query_one("#sentence", Static))
        assert app.query_one("#block-word", BlockWord).word == "FAIL"
        activity = _view_text(app)
        assert activity.count("iteration ") == 5
        assert "T0 → T1" in activity
        assert "T1 → T2" in activity

        await app.run_action("stage_view('plan')")
        await pilot.pause()
        lines = app.session.read("knowledge/plan-9.md").strip().splitlines()
        plan = next(line for line in lines if line.strip())
        assert plan.lstrip("#* `") in _view_text(app)

        await app.run_action("open_iteration(9,2)")
        await pilot.pause()
        last_action = app.session.read("runs/task-9-iter-2.md").strip().splitlines()[-1]
        text = _view_text(app)
        assert last_action.removeprefix("- 💬").strip()[:30] in text
        assert "think" in text and "fix" in text and "verdict" in text

        await pilot.press("g")
        await pilot.pause()
        assert app.selected_story is None
        assert "US-024" in _plain(app.query_one("#sentence", Static))

    _drive(app, body)


def test_heatmap_click_and_keyboard_navigation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _app(tmp_path, monkeypatch)

    async def body(pilot: Pilot) -> None:
        await pilot.click("#heatmap", offset=(0, 0))
        await pilot.pause()
        assert app.selected_story == 1
        await pilot.press("right")
        await pilot.pause()
        assert app.selected_story == 2
        await pilot.press("left")
        await pilot.pause()
        assert app.selected_story == 1
        await pilot.press("escape")
        await pilot.pause()
        assert app.selected_story is None
        assert app.follow_live

    _drive(app, body)


def test_missing_review_has_an_empty_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _app(tmp_path, monkeypatch)
    (app.session.dir / "knowledge/task8-review-iter-5.md").unlink()

    async def body(pilot: Pilot) -> None:
        await app.run_action("select_story(9)")
        await app.run_action("set_view('eval')")
        await pilot.pause()
        text = _view_text(app)
        assert "iteration 5" in text
        assert "no review yet" in text
        assert text.index("iteration 5") < text.index("iteration 4")

    _drive(app, body)


def test_unchanged_files_are_not_read_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    app = _app(tmp_path, monkeypatch)

    async def body(pilot: Pilot) -> None:
        await app.run_action("select_story(9)")
        await pilot.pause()
        reads: list[Path] = []
        original = Path.read_text

        def counted(path: Path, *args: object, **kwargs: object) -> str:
            if "-iter-" in path.name:
                reads.append(path)
            return original(path, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(Path, "read_text", counted)
        app._refresh()
        await pilot.pause()
        assert reads == []

    _drive(app, body)
