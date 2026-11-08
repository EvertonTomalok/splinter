from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby

from rich.segment import Segment
from rich.style import Style
from textual.message import Message
from textual.strip import Strip

from splinter.arcade import Arcade
from splinter.tui_run_palette import AMBER, LIME

SEA = "#0c1820"
WATER = "#88aab8"
RULE = "#526e7a"
_BASE = Style(color=WATER, bgcolor=SEA)
_RULE = Style(color=RULE, bgcolor=SEA)
_AIM = Style(color=LIME, bgcolor="#233a36", bold=True)
_HIT = Style(color=AMBER, bgcolor=SEA, bold=True)
_SUNK = Style(color="#c0cdd3", bgcolor=SEA, bold=True)
_WAVES = Style(color="#66bddd", bgcolor="#102b3b")
_SHIP = Style(color="#88aab8", bgcolor=SEA, bold=True)
_WRECK = ("   ╲│╱   ", "▄▄▄█╳█▄▄▄", " ▀██ ██▀ ")
_VESSEL = ("   │     ", " ▄▄█▄▄▄  ", "  ▀███▀  ")
_SEA_ROWS = (" ≈≈  ≈≈  ", "   ≈≈  ≈ ", " ≈≈  ≈≈  ")


@dataclass(frozen=True)
class NavalLayout:
    left: int
    cell_width: int
    cell_height: int

    @classmethod
    def for_size(cls, width: int, height: int) -> NavalLayout:
        cell_width = 11 if width >= 100 else 5
        return cls((width - 8 * (cell_width + 1) - 1) // 2, cell_width, 3 if height >= 22 else 1)

    def center(self, cell: tuple[int, int]) -> tuple[int, int]:
        x, y = cell
        return (
            self.left + 1 + x * (self.cell_width + 1) + self.cell_width // 2,
            2 + y * (self.cell_height + 1) + self.cell_height // 2,
        )

    def cell_at(self, x: int, y: int) -> tuple[int, int] | None:
        dx, dy = x - self.left, y - 1
        if not (0 < dx < 8 * (self.cell_width + 1) and 0 < dy < 5 * (self.cell_height + 1)):
            return None
        if dx % (self.cell_width + 1) == 0 or dy % (self.cell_height + 1) == 0:
            return None
        return dx // (self.cell_width + 1), dy // (self.cell_height + 1)


class NavalTarget(Message):
    def __init__(self, cell: tuple[int, int], *, fire: bool) -> None:
        super().__init__()
        self.cell = cell
        self.fire = fire


def naval_rows(game: Arcade, width: int, height: int) -> list[Strip]:
    layout = NavalLayout.for_size(width, height)
    canvas = [[(" ", _BASE) for _ in range(width)] for _ in range(height)]

    def put(x: int, y: int, text: str, style: Style = _BASE) -> None:
        for offset, char in enumerate(text):
            if 0 <= x + offset < width and 0 <= y < height:
                canvas[y][x + offset] = (char, style)

    for row in range(6):
        y = 1 + row * (layout.cell_height + 1)
        corners = ("┌", "┬", "┐") if row == 0 else ("└", "┴", "┘") if row == 5 else ("├", "┼", "┤")
        line = corners[0] + corners[1].join(["─" * layout.cell_width] * 8) + corners[2]
        put(layout.left, y, line, _RULE)
    sunk = set().union(*(ship for ship in game.ships if ship <= game.shots))
    for x in range(8):
        cx, _ = layout.center((x, 0))
        put(cx, 0, chr(65 + x), _AIM if x == game.cursor[0] else _BASE)
    for y in range(5):
        _, cy = layout.center((0, y))
        put(layout.left - 2, cy, str(y + 1), _AIM if y == game.cursor[1] else _BASE)
        for dy in range(layout.cell_height):
            row_y = 2 + y * (layout.cell_height + 1) + dy
            for x in range(9):
                put(layout.left + x * (layout.cell_width + 1), row_y, "│", _RULE)
        for x in range(8):
            cell = x, y
            cx, cy = layout.center(cell)
            selected = cell == game.cursor and not game.over
            hit = cell in game.shots and cell in game.fleet
            missed = cell in game.shots and not hit
            revealed = game.over and cell in game.fleet and not hit
            style = (
                _SUNK if cell in sunk else _HIT if hit else _WAVES if missed
                else _SHIP if revealed else _BASE
            )
            if selected:
                style = style + Style(bgcolor="#233a36")
            if hit or selected or revealed or missed:
                for dy in range(layout.cell_height):
                    put(cx - layout.cell_width // 2, cy - layout.cell_height // 2 + dy,
                        " " * layout.cell_width, style)
            if layout.cell_height == 3:
                art = _WRECK if hit else _VESSEL if revealed else _SEA_ROWS if missed else ("·",)
            else:
                art = ("\\_x_/" if hit else "\\___/" if revealed else "≈≈≈" if missed else "·",)
            for dy, line in enumerate(art):
                put(cx - len(line) // 2, cy - len(art) // 2 + dy, line, style)
            if selected and max(len(line) for line in art) < layout.cell_width:
                put(cx - layout.cell_width // 2, cy, "[", _AIM)
                put(cx + layout.cell_width // 2, cy, "]", _AIM)
    if height >= 24:
        fleet = "  ".join("[" + ("#" if ship <= game.shots else "=") * len(ship) + "]"
                          for ship in game.ships)
        legend = f"Fleet {fleet}    ≈ sea   amber wreck: hit   pale wreck: sunk"
        if game.over and not game.fleet <= game.shots:
            legend += "   intact: remaining"
        put((width - len(legend)) // 2, height - 1, legend)
    return [
        Strip([Segment("".join(char for char, _ in cells), style)
               for style, cells in groupby(row, key=lambda item: item[1])], width)
        for row in canvas
    ]
