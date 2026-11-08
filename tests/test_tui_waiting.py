from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest
from textual.widgets import RichLog, Static, TextArea

from splinter.memory.session import Session
from splinter.tui import ConfirmQuit, PrdSessionApp
from splinter.tui_waiting import WaitingRoom


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PrdSessionApp:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    monkeypatch.setattr(PrdSessionApp, "_spawn", lambda *args, **kw: None)
    instance = PrdSessionApp(Session("waiting"), {"no_ground": True})
    instance._model_name = "test/planner"
    return instance


def test_play_switch_pause_logs_and_return_to_questions(app: PrdSessionApp) -> None:
    async def drive() -> None:
        async with app.run_test(size=(100, 38)) as pilot:
            instructions = app.query_one("#instructions", TextArea)
            instructions.text = "Build a calculator"
            app._on_generate()
            await pilot.pause()
            room = app.query_one(WaitingRoom)
            assert room.display and not app.query_one("#session-body").display
            assert room.has_focus
            await pilot.press("2", "space", "up")
            assert room.game.kind == "dino" and room.playing
            await pilot.press("p")
            assert not room.playing and room._frames is None
            await pilot.press("3", "space", "left", "space")
            assert room.game.kind == "ship" and room.game.player == 30
            await pilot.press("q")
            assert room.logs_only and room._frames is None
            assert not room.query_one("#arcade").display
            assert instructions.text == "Build a calculator"
            assert app._busy
            await pilot.click("#wait-next")
            assert not room.logs_only and room.game.kind != "ship"
            await pilot.press("4", "space")
            for cell in sorted(room.game.fleet):
                room.game.cursor = cell
                await pilot.press("space")
            assert room.game.over and not room.playing
            await pilot.press("space")
            assert not room.game.over and room.playing
            app._after_questions("1. Color?\nA) Green\nB) Blue", "session-id")
            await pilot.pause()
            assert not room.display and room._clock is None and room._handler is None
            assert not app._busy and app.query_one("#session-body").display

    asyncio.run(drive())


def test_real_worker_logs_are_scoped_and_markup_is_literal(app: PrdSessionApp) -> None:
    logger = logging.getLogger("splinter.live")
    level = logger.level
    handlers = list(logging.getLogger("splinter").handlers)

    async def drive() -> None:
        async with app.run_test(size=(100, 38)) as pilot:
            app._set_busy(True, "writing plan…")
            room = app.query_one(WaitingRoom)
            logger.info("unrelated worker")

            def job() -> None:
                logger.info("Reading [config].py", extra={"waiting_model": "fallback/model"})

            await asyncio.to_thread(room.run_job, job)
            room._poll()
            await pilot.pause()
            assert room.model == "fallback/model"
            assert str(room.query_one("#wait-latest", Static).content) == "Reading [config].py"
            lines = "\n".join(line.text for line in room.query_one(RichLog).lines)
            assert "[config].py" in lines and "unrelated worker" not in lines
            room._last_event -= 20
            room._poll()
            assert "No new events" in str(room.query_one("#wait-meta", Static).content)
            room.toggle_logs()
            app._set_busy(False)
            app._set_busy(True, "drafting questions…")
            assert room.logs_only

    asyncio.run(drive())
    assert logger.level == level
    assert logging.getLogger("splinter").handlers == handlers


@pytest.mark.parametrize("size", [(80, 32), (80, 24), (60, 18)])
def test_small_terminals_keep_activity_and_cancel_visible(
    app: PrdSessionApp, size: tuple[int, int]
) -> None:
    async def drive() -> None:
        async with app.run_test(size=size) as pilot:
            app._set_busy(True, "generating PRD from instructions…")
            await pilot.pause()
            room = app.query_one(WaitingRoom)
            room.note("Reading " + "file.py " * 20 + "tail-marker")
            await pilot.pause()
            lines = room.query_one("#wait-log", RichLog).lines
            assert any("tail-marker" in line.text for line in lines)
            assert all(len(line.text) <= room.query_one("#wait-log").size.width for line in lines)
            for selector in ("#wait-cancel", "#wait-title", "#wait-meta", "#wait-log"):
                region = room.query_one(selector).region
                assert region.height > 0
                assert region.bottom <= size[1]
                assert region.right <= size[0]
            await pilot.click("#wait-cancel")
            assert isinstance(app.screen, ConfirmQuit)
            assert room._frames is None
            app.screen.dismiss(False)
            await pilot.pause()
            assert app._busy and room.active

    asyncio.run(drive())
