from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

import pytest

from tests.fixtures_run_view import FIXTURE_LADDER, build_session, build_single_session

if TYPE_CHECKING:
    from pathlib import Path

    from pytest import MonkeyPatch
    from textual.pilot import Pilot
    from textual.widgets import Static

    from splinter.memory.session import Session
    from splinter.tui import RunApp


def _plain(widget: Static) -> str:
    rendered = widget.render()
    return str(getattr(rendered, "plain", rendered))


def _drive(app: RunApp, body: Callable[[Pilot], Awaitable[None]], size: tuple[int, int]) -> None:
    async def run() -> None:
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            await body(pilot)

    asyncio.run(run())


def _app(session: Session, monkeypatch: MonkeyPatch) -> RunApp:
    from splinter.tui import RunApp

    monkeypatch.setattr("splinter.models.roster.load_ladder", lambda: FIXTURE_LADDER)
    monkeypatch.setattr(RunApp, "_work", lambda self: None)
    return RunApp(session, {"prd_path": "payments-v2-prd.md"})


def test_run_screen_renders_focus_shell(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Static

    from splinter.tui_blockword import BlockWord

    session = build_session(tmp_path, monkeypatch)
    app = _app(session, monkeypatch)

    async def body(_: Pilot) -> None:
        assert "story 24/58" in app.sub_title
        heatmap = _plain(app.query_one("#heatmap", Static))
        assert heatmap.count("▄") == 58
        overview = _plain(app.query_one("#overview", Static))
        assert "needs you" in overview and "now" in overview and "next" in overview
        assert all(f"US-0{number}" in overview for number in range(25, 30))
        assert app.query_one("#block-word", BlockWord).word == "RUN"

    _drive(app, body, (120, 36))


def test_shell_keeps_log_stop_and_cowabunga(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    pytest.importorskip("textual")
    from textual.widgets import RichLog

    from splinter.tui_run_stop import StopBar

    session = build_session(tmp_path, monkeypatch)
    app = _app(session, monkeypatch)

    async def body(pilot: Pilot) -> None:
        app.write_log("focus shell log", logging.WARNING)
        await app.run_action("set_view('logs')")
        await pilot.pause()
        assert "focus shell log" in list(app.query_one("#log", RichLog).lines)[-1].text
        before = app._kowabunga
        await pilot.press("ctrl+k")
        await pilot.pause()
        assert app._kowabunga != before
        await pilot.press("s")
        await pilot.pause()
        assert app.query_one("#stop-bar", StopBar).display

    _drive(app, body, (120, 36))


def test_single_story_collapses_groups(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Static

    app = _app(build_single_session(tmp_path, monkeypatch), monkeypatch)

    async def body(_: Pilot) -> None:
        assert _plain(app.query_one("#heatmap", Static)).count("▄") == 1
        overview = _plain(app.query_one("#overview", Static))
        assert "now" in overview
        assert "needs you" not in overview and "next" not in overview

    _drive(app, body, (120, 36))


@pytest.mark.parametrize("width", [100, 120])
def test_shell_has_no_horizontal_overflow(
    tmp_path: Path, monkeypatch: MonkeyPatch, width: int
) -> None:
    pytest.importorskip("textual")

    app = _app(build_session(tmp_path, monkeypatch), monkeypatch)

    async def body(_: Pilot) -> None:
        assert all(widget.region.right <= app.size.width for widget in app.query("*"))

    _drive(app, body, (width, 36))
