from __future__ import annotations

import asyncio
from unittest.mock import Mock

import pytest
from textual.widgets import Static
from typer.testing import CliRunner

from splinter.arcade_demo import ArcadeDemoApp
from splinter.cli import app as cli
from splinter.tui_waiting import WaitingRoom


def test_demo_plays_and_exits_without_providers_or_subprocesses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocked = Mock(side_effect=AssertionError("Offline demo attempted external I/O"))
    monkeypatch.setattr("splinter.providers.dispatch.get_provider", blocked)
    monkeypatch.setattr("splinter.procreg.run_subprocess", blocked)
    monkeypatch.setattr("subprocess.Popen", blocked)
    monkeypatch.setattr("urllib.request.urlopen", blocked)
    app = ArcadeDemoApp("naval")

    async def drive() -> None:
        async with app.run_test(size=(80, 24)) as pilot:
            room = app.query_one(WaitingRoom)
            await pilot.pause()
            assert room.game.kind == "naval"
            assert room.query_one("#arcade").display
            assert room.query_one("#game-board").region.bottom <= 24
            assert room.query_one("#wait-cancel").region.bottom <= 24
            assert "[SIMULATED]" in str(room.query_one("#wait-latest", Static).content)
            await pilot.press("space", "right", "space")
            assert (1, 0) in room.game.shots
            await pilot.press("3", "space", "left")
            assert room.game.kind == "ship" and room.game.player == 30
            app._next_event()
            assert "Finding files" in str(room.query_one("#wait-latest", Static).content)
            await pilot.press("g")
            assert room.logs_only
            await pilot.click("#wait-cancel")
            assert not room.active and room._handler is None
            assert app._demo_timer is None

    asyncio.run(drive())
    blocked.assert_not_called()


@pytest.mark.parametrize("args, expected", [([], None), (["--game", "naval"], "naval")])
def test_game_demo_cli(
    monkeypatch: pytest.MonkeyPatch, args: list[str], expected: str | None
) -> None:
    games: list[str | None] = []
    monkeypatch.setattr(ArcadeDemoApp, "run", lambda self: games.append(self.initial_game))
    result = CliRunner().invoke(cli, ["game-demo", *args])
    assert result.exit_code == 0, result.output
    assert games == [expected]


def test_game_demo_rejects_unknown_game(monkeypatch: pytest.MonkeyPatch) -> None:
    run = Mock()
    monkeypatch.setattr(ArcadeDemoApp, "run", run)
    result = CliRunner().invoke(cli, ["game-demo", "--game", "unknown"])
    assert result.exit_code == 2
    run.assert_not_called()


@pytest.mark.parametrize("kind", ["pong", "dino", "ship", "naval"])
def test_demo_mouse_controls_start_pause_and_restart(kind: str) -> None:
    app = ArcadeDemoApp(kind)

    async def drive() -> None:
        async with app.run_test(size=(200, 42)) as pilot:
            room = app.query_one(WaitingRoom)
            await pilot.click(f"#game-{kind}")
            await pilot.click("#wait-play")
            assert room.playing and room.has_focus
            await pilot.click("#wait-play")
            assert not room.playing and room._frames is None
            room.game.over = True
            room._draw()
            await pilot.click("#wait-play")
            assert room.playing and not room.game.over
            assert room.query_one("#wait-log").region.bottom <= 42
            assert room.query_one("#wait-log").size.height >= 3
            room._pause()

    asyncio.run(drive())
