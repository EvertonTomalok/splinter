"""Text helpers of the run screen: story labels, spans and big numerals."""

from __future__ import annotations

import re

from rich.markup import escape
from rich.text import Text

from splinter.run_view import StoryRow

_ID_PREFIX = re.compile(r"^\s*([A-Z][A-Z0-9]*-\d+)\s*[:·—-]?\s*(.*)$")

_DIGITS: dict[str, tuple[str, str, str]] = {
    "0": ("┌┐", "││", "└┘"),
    "1": (" ╷", " │", " ╵"),
    "2": ("╶┐", "┌┘", "└╴"),
    "3": ("╶┐", " ┤", "╶┘"),
    "4": ("╷╷", "└┤", " ╵"),
    "5": ("┌╴", "└┐", "╶┘"),
    "6": ("┌╴", "├┐", "└┘"),
    "7": ("╶┐", " │", " ╵"),
    "8": ("┌┐", "├┤", "└┘"),
    "9": ("┌┐", "└┤", "╶┘"),
    "$": ("┌┼╴", "└┼┐", "╶┼┘"),
    ":": (" ", ":", " "),
    ".": (" ", " ", "·"),
    "-": ("  ", "──", "  "),
}
_TIGHT = {":", "."}


def story_label(row: StoryRow) -> tuple[str, str]:
    """The id and title the design shows: ``US-024`` · ``Retry-After on 429``."""
    match = _ID_PREFIX.match(row.title or "")
    if match:
        return match.group(1), match.group(2).strip()
    return row.id, (row.title or "").strip()


def story_id(row: StoryRow | None) -> str:
    """Just the shown id of a story."""
    return story_label(row)[0] if row is not None else "—"


def hms(seconds: float) -> str:
    """``2:47:18``."""
    total = max(0, int(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02}:{secs:02}"


def money(value: float) -> str:
    return f"${value:.4f}"


def big_rows(text: str) -> tuple[str, str, str]:
    """Draw ``text`` as three rows of thin box-drawing numerals."""
    rows = ["", "", ""]
    previous = ""
    for char in text:
        glyph = _DIGITS.get(char, (" ", char, " "))
        gap = "" if not previous or char in _TIGHT or previous in _TIGHT else " "
        for index in range(3):
            rows[index] += gap + glyph[index]
        previous = char
    return rows[0], rows[1], rows[2]


def big_width(text: str) -> int:
    return len(big_rows(text)[0])


def cell_len(markup: str) -> int:
    """Width of a markup string once rendered."""
    return Text.from_markup(markup).cell_len


def pad(markup: str, width: int) -> str:
    """Pad a markup string with spaces to ``width`` cells."""
    return markup + " " * max(0, width - cell_len(markup))


def clip(text: str, width: int) -> str:
    """Escape ``text`` and cut it to ``width`` cells with an ellipsis."""
    if width <= 0:
        return ""
    if len(text) <= width:
        return escape(text)
    return escape(text[: max(0, width - 1)]) + "…"
