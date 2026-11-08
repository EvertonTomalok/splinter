from __future__ import annotations

import asyncio

import pytest
from textual.widgets import Static

from splinter.arcade import Arcade
from splinter.arcade_demo import ArcadeDemoApp
from splinter.tui_naval import NavalLayout, naval_rows
from splinter.tui_waiting import ArcadeBoard, WaitingRoom


@pytest.mark.parametrize("size", [(64, 12), (128, 24)])
def test_every_cell_has_a_distinct_hit_area_and_grid_lines_are_inert(size: tuple[int, int]) -> None:
    layout = NavalLayout.for_size(*size)
    cells = set()
    for y in range(size[1]):
        for x in range(size[0]):
            cell = layout.cell_at(x, y)
            if cell is not None:
                cells.add(cell)
    assert cells == {(x, y) for x in range(8) for y in range(5)}
    for cell in cells:
        cx, cy = layout.center(cell)
        assert layout.cell_at(cx, cy) == cell
        assert layout.cell_at(cx, 0) is None
        assert layout.cell_at(layout.left, cy) is None
        assert layout.cell_at(cx, 1) is None


@pytest.mark.parametrize("size", [(80, 24), (200, 42)])
def test_mouse_fires_hits_misses_and_repeats_without_start_button(size: tuple[int, int]) -> None:
    app = ArcadeDemoApp("naval")

    async def drive() -> None:
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            room = app.query_one(WaitingRoom)
            board = room.query_one(ArcadeBoard)
            layout = NavalLayout.for_size(board.size.width, board.size.height)
            hit = next(iter(room.game.fleet))
            miss = next((x, y) for y in range(5) for x in range(8) if (x, y) not in room.game.fleet)
            await pilot.hover(board, offset=layout.center(hit))
            assert room.game.cursor == hit
            assert not room.game.shots and not room.playing
            await pilot.click(board, offset=layout.center(hit))
            assert room.game.shots == {hit}
            assert room.playing and room.has_focus
            assert "HIT" in str(room.query_one("#game-controls", Static).content)
            await pilot.click(board, offset=layout.center(hit))
            assert len(room.game.shots) == 1
            assert "already tried" in str(room.query_one("#game-controls", Static).content)
            await pilot.click(board, offset=layout.center(miss))
            assert room.game.shots == {hit, miss}
            assert "MISS" in str(room.query_one("#game-controls", Static).content)
            await pilot.click(board, offset=(layout.left, 1))
            assert len(room.game.shots) == 2
            await pilot.press("p")
            assert not room.playing
            await pilot.click(board, offset=layout.center(hit))
            assert room.playing and len(room.game.shots) == 2
            await pilot.resize_terminal(160, 44)
            await pilot.pause()
            layout = NavalLayout.for_size(board.size.width, board.size.height)
            for cell in sorted(room.game.fleet - room.game.shots):
                await pilot.click(board, offset=layout.center(cell))
            assert room.game.over and room.game.result == "You win!"
            assert not room.playing
            previous = set(room.game.shots)
            await pilot.click(board, offset=layout.center((7, 4)))
            assert room.game.shots == previous
            assert "Space restart" in str(room.query_one("#game-controls", Static).content)
            await pilot.press("space")
            assert not room.game.shots and not room.game.over

    asyncio.run(drive())


def test_naval_grid_conceals_fleet_and_marks_a_sunk_ship() -> None:
    game = Arcade("naval")
    layout = NavalLayout.for_size(128, 24)
    rows = naval_rows(game, 128, 24)
    assert len(rows) == 24 and all(row.cell_length == 128 for row in rows)
    for cell in game.fleet:
        x, y = layout.center(cell)
        assert rows[y].text[x] == "~"
    for cell in game.ships[0]:
        game.cursor = cell
        game.key("space")
    assert "SUNK" in game.naval_message
    rows = naval_rows(game, 128, 24)
    for cell in game.ships[0]:
        x, y = layout.center(cell)
        assert rows[y].text[x] == "#"
