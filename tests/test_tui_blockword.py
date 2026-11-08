"""Tests for the ANSI Shadow block word and its wipe."""

from __future__ import annotations

import asyncio

import pytest
from rich.style import Style
from rich.text import Span
from textual.app import App, ComposeResult

from splinter.tui_blockword import GLYPHS, SHADOW, BlockWord, render_compact, render_word
from splinter.tui_run_palette import palette_css

#: The design's RUN word: one space between the letters of ``word()``.
_RUN_ROWS = (
    "██████╗  ██╗   ██╗ ███╗   ██╗",
    "██╔══██╗ ██║   ██║ ████╗  ██║",
    "██████╔╝ ██║   ██║ ██╔██╗ ██║",
    "██╔══██╗ ██║   ██║ ██║╚██╗██║",
    "██║  ██║ ╚██████╔╝ ██║ ╚████║",
    "╚═╝  ╚═╝  ╚═════╝  ╚═╝  ╚═══╝",
)

#: Words the run screen shows.
_STAGE_WORDS = (
    "RUN",
    "GATE",
    "EVAL",
    "PLAN",
    "PASS",
    "FAIL",
    "WAIT",
    "ASK",
    "DONE",
    "STOP",
    "FIX",
    "MERGE",
)


class _HostApp(App[None]):
    """App that mounts the block word with the design palette."""

    CSS = palette_css()

    def compose(self) -> ComposeResult:
        yield BlockWord(id="word")


def _is_dim(span: Span) -> bool:
    """Whether a text span is drawn dim (Rich keeps the style as a string)."""
    style = span.style
    if isinstance(style, str):
        style = Style.parse(style)
    return bool(style.dim)


def test_render_word_matches_the_design() -> None:
    rows = render_word("RUN")
    assert len(rows) == 6
    assert len({len(row) for row in rows}) == 1
    assert rows == _RUN_ROWS


def test_every_glyph_has_six_rows_of_equal_width() -> None:
    for letter, rows in GLYPHS.items():
        assert len(rows) == 6, letter
        assert len({len(row) for row in rows}) == 1, letter


def test_unknown_characters_render_as_blank() -> None:
    assert render_word("R?N") == render_word("R N")
    assert render_word("") == ("",) * 6


@pytest.mark.parametrize("word", _STAGE_WORDS)
def test_stage_words_render_as_block_letters(word: str) -> None:
    assert all(char in GLYPHS for char in word)
    rows = render_word(word)
    assert len(rows) == 6
    assert len({len(row) for row in rows}) == 1
    assert all(row.strip() for row in rows)


def test_word_change_wipes_in_from_the_left() -> None:
    app = _HostApp()

    async def main() -> None:
        async with app.run_test(size=(120, 10)) as pilot:
            await pilot.pause()
            word = app.query_one(BlockWord)
            word.word = "GATE"
            await pilot.pause()
            assert str(word.render()) == render_compact("GATE", SHADOW["live"]).plain
            word.word = "EVAL"
            hidden = word.render()
            assert str(hidden) == render_compact("EVAL", SHADOW["live"]).plain
            assert hidden.spans and all(_is_dim(span) for span in hidden.spans)
            await pilot.pause(0.4)
            shown = word.render()
            assert str(shown) == render_compact("EVAL", SHADOW["live"]).plain
            assert not any(_is_dim(span) for span in shown.spans)

    asyncio.run(main())


def test_animation_level_none_swaps_the_word_at_once() -> None:
    app = _HostApp()

    async def main() -> None:
        async with app.run_test(size=(120, 10)) as pilot:
            app.animation_level = "none"
            await pilot.pause()
            word = app.query_one(BlockWord)
            word.word = "GATE"
            await pilot.pause()
            word.word = "EVAL"
            text = word.render()
            assert str(text) == render_compact("EVAL", SHADOW["live"]).plain
            assert not any(_is_dim(span) for span in text.spans)

    asyncio.run(main())


def test_narrow_terminal_falls_back_to_the_plain_word() -> None:
    app = _HostApp()

    async def main() -> None:
        async with app.run_test(size=(20, 10)) as pilot:
            await pilot.pause()
            word = app.query_one(BlockWord)
            word.word = "GATE"
            await pilot.pause()
            assert str(word.render()) == "GATE"

    asyncio.run(main())


def test_role_class_picks_the_colour() -> None:
    app = _HostApp()

    async def main() -> None:
        async with app.run_test(size=(120, 10)) as pilot:
            await pilot.pause()
            word = app.query_one(BlockWord)
            assert word.role == "live"
            assert word.has_class("-live")
            word.set_role("fail")
            await pilot.pause()
            assert word.role == "fail"
            assert word.has_class("-fail")
            assert not word.has_class("-live")
            word.set_role("unknown")
            assert word.role == "live"
            assert word.has_class("-live")

    asyncio.run(main())


def test_compact_word_is_three_rows_of_half_blocks() -> None:
    text = render_compact("RUN", SHADOW["live"])
    rows = text.plain.split("\n")
    assert len(rows) == 3
    assert {len(row) for row in rows} == {34}
    assert set("".join(rows)) <= {"█", "▀", "▄", " "}
