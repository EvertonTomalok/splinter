"""Stage word in ANSI Shadow block letters, with a left-to-right wipe.

On a screen of :data:`TALL_ROWS` rows or more the word is the design's six-row ANSI
Shadow; on a shorter one it folds into :data:`PIXELS`, three half-block rows.

The glyphs are a built-in copy of the design's word table
(``tasks/c2-focus-tui/design/run.html``) — no font dependency, no new package.
The widget never reads the session: the run screen assigns ``word`` and a role
(``live``, ``fail``, ``wait``, ``done``) that picks the colour.

The colours come from the palette variables (``$lime``, ``$red``, ``$amber``,
``$muted``) that ``splinter.tui_run_palette`` writes into this widget's own
stylesheet — Textual resolves a variable only where it is declared, so the
widget carries them. An app may still publish the palette app-wide by adding
``splinter.tui_run_palette.PALETTE`` to ``App.get_css_variables()``.
"""

from __future__ import annotations

import math

from rich.text import Text
from textual.reactive import reactive
from textual.timer import Timer
from textual.widgets import Static

from splinter.tui_run_palette import palette_css

#: The blank glyph: spaces and unknown characters land here.
BLANK: tuple[str, ...] = ("   ", "   ", "   ", "   ", "   ", "   ")

#: One glyph per character: six rows of ANSI Shadow art.
GLYPHS: dict[str, tuple[str, ...]] = {
    "A": (" █████╗ ", "██╔══██╗", "███████║", "██╔══██║", "██║  ██║", "╚═╝  ╚═╝"),
    "C": (" ██████╗", "██╔════╝", "██║     ", "██║     ", "╚██████╗", " ╚═════╝"),
    "D": ("██████╗ ", "██╔══██╗", "██║  ██║", "██║  ██║", "██████╔╝", "╚═════╝ "),
    "E": ("███████╗", "██╔════╝", "█████╗  ", "██╔══╝  ", "███████╗", "╚══════╝"),
    "F": ("███████╗", "██╔════╝", "█████╗  ", "██╔══╝  ", "██║     ", "╚═╝     "),
    "G": (" ██████╗ ", "██╔════╝ ", "██║  ███╗", "██║   ██║", "╚██████╔╝", " ╚═════╝ "),
    "I": ("██╗", "██║", "██║", "██║", "██║", "╚═╝"),
    "K": ("██╗  ██╗", "██║ ██╔╝", "█████╔╝ ", "██╔══██╗", "██║  ██║", "╚═╝  ╚═╝"),
    "L": ("██╗     ", "██║     ", "██║     ", "██║     ", "███████╗", "╚══════╝"),
    "M": ("███╗   ███╗", "████╗ ████║", "██╔████╔██║", "██║╚██╔╝██║", "██║ ╚═╝ ██║", "╚═╝     ╚═╝"),
    "N": ("███╗   ██╗", "████╗  ██║", "██╔██╗ ██║", "██║╚██╗██║", "██║ ╚████║", "╚═╝  ╚═══╝"),
    "O": (" ██████╗ ", "██╔═══██╗", "██║   ██║", "██║   ██║", "╚██████╔╝", " ╚═════╝ "),
    "P": ("██████╗ ", "██╔══██╗", "██████╔╝", "██╔═══╝ ", "██║     ", "╚═╝     "),
    "R": ("██████╗ ", "██╔══██╗", "██████╔╝", "██╔══██╗", "██║  ██║", "╚═╝  ╚═╝"),
    "S": ("███████╗", "██╔════╝", "███████╗", "╚════██║", "███████║", "╚══════╝"),
    "T": ("████████╗", "╚══██╔══╝", "   ██║   ", "   ██║   ", "   ██║   ", "   ╚═╝   "),
    "U": ("██╗   ██╗", "██║   ██║", "██║   ██║", "██║   ██║", "╚██████╔╝", " ╚═════╝ "),
    "V": ("██╗   ██╗", "██║   ██║", "██║   ██║", "╚██╗ ██╔╝", " ╚████╔╝ ", "  ╚═══╝  "),
    "W": ("██╗    ██╗", "██║    ██║", "██║ █╗ ██║", "██║███╗██║", "╚███╔███╔╝", " ╚══╝╚══╝ "),
    "X": ("██╗  ██╗", "╚██╗██╔╝", " ╚███╔╝ ", " ██╔██╗ ", "██╔╝ ██╗", "╚═╝  ╚═╝"),
    " ": BLANK,
}

#: Rows of a glyph.
GLYPH_ROWS = 6

#: Seconds between two steps of the wipe.
_STEP_SECONDS = 0.02

#: Steps of the wipe.
_STEPS = 12

#: Roles the word can wear, in CSS class form.
_ROLES: tuple[str, ...] = ("live", "fail", "wait", "done", "idle")


def render_word(word: str) -> tuple[str, ...]:
    """Render ``word`` as six equal-width lines of block glyphs.

    The word is upper-cased; characters without a glyph become blanks.
    """
    glyphs = [GLYPHS.get(char, BLANK) for char in word.upper()]
    if not glyphs:
        return ("",) * GLYPH_ROWS
    return tuple(" ".join(glyph[row] for glyph in glyphs) for row in range(GLYPH_ROWS))


#: Screen rows from which the word uses the six-row ANSI Shadow letters.
TALL_ROWS = 40

#: Rows of the compact word on screen.
COMPACT_ROWS = 3

#: Shadow tone per role; kept for callers that tint a backdrop.
SHADOW: dict[str, str] = {
    "live": "#4f6114",
    "done": "#4f6114",
    "fail": "#66241f",
    "wait": "#66241f",
    "idle": "#2a2a26",
}

#: Six-pixel-tall letters; two pixel rows fold into one terminal row.
PIXELS: dict[str, tuple[str, ...]] = {
    "A": (".XXX.", "X...X", "X...X", "XXXXX", "X...X", "X...X"),
    "C": (".XXXX", "X....", "X....", "X....", "X....", ".XXXX"),
    "D": ("XXXX.", "X...X", "X...X", "X...X", "X...X", "XXXX."),
    "E": ("XXXXX", "X....", "XXXX.", "X....", "X....", "XXXXX"),
    "F": ("XXXXX", "X....", "XXXX.", "X....", "X....", "X...."),
    "G": (".XXXX", "X....", "X..XX", "X...X", "X...X", ".XXX."),
    "I": ("XXX", ".X.", ".X.", ".X.", ".X.", "XXX"),
    "K": ("X...X", "X..X.", "XXX..", "X..X.", "X...X", "X...X"),
    "L": ("X....", "X....", "X....", "X....", "X....", "XXXXX"),
    "M": ("X...X", "XX.XX", "X.X.X", "X...X", "X...X", "X...X"),
    "N": ("X...X", "XX..X", "X.X.X", "X..XX", "X...X", "X...X"),
    "O": (".XXX.", "X...X", "X...X", "X...X", "X...X", ".XXX."),
    "P": ("XXXX.", "X...X", "X...X", "XXXX.", "X....", "X...."),
    "R": ("XXXX.", "X...X", "X...X", "XXXX.", "X..X.", "X...X"),
    "S": (".XXXX", "X....", ".XXX.", "....X", "....X", "XXXX."),
    "T": ("XXXXX", "..X..", "..X..", "..X..", "..X..", "..X.."),
    "U": ("X...X", "X...X", "X...X", "X...X", "X...X", ".XXX."),
    "V": ("X...X", "X...X", "X...X", "X...X", ".X.X.", "..X.."),
    "W": ("X...X", "X...X", "X...X", "X.X.X", "XX.XX", "X...X"),
    "X": ("X...X", ".X.X.", "..X..", "..X..", ".X.X.", "X...X"),
    " ": ("...", "...", "...", "...", "...", "..."),
}
_HALF = {("X", "X"): "█", ("X", "."): "▀", (".", "X"): "▄"}


def compact_rows(word: str) -> tuple[str, str, str]:
    """``word`` as three rows of half blocks, each pixel two columns wide."""
    rows = ["", "", ""]
    for index, char in enumerate(word.upper()):
        glyph = PIXELS.get(char, PIXELS[" "])
        for row in range(COMPACT_ROWS):
            line = "  " if index else ""
            for top, bottom in zip(glyph[row * 2], glyph[row * 2 + 1], strict=True):
                line += _HALF.get((top, bottom), " ") * 2
            rows[row] += line
    return rows[0], rows[1], rows[2]


def compact_width(word: str) -> int:
    """Columns the on-screen word takes."""
    return len(compact_rows(word)[0])


def render_compact(word: str, shadow: str = "", reveal: int | None = None) -> Text:
    """The on-screen word; columns past ``reveal`` are drawn dim (the wipe)."""
    rows = compact_rows(word)
    width = len(rows[0])
    shown = width if reveal is None else max(0, min(reveal, width))
    text = Text(no_wrap=True)
    for index, row in enumerate(rows):
        if index:
            text.append("\n")
        text.append(row[:shown])
        if shown < width:
            text.append(row[shown:], style="dim")
    return text


def word_width(word: str) -> int:
    """Number of columns the block rendering of ``word`` takes."""
    return len(render_word(word)[0])


class BlockWord(Static):
    """A stage word in block letters, wiped in from the left when it changes."""

    DEFAULT_CSS = (
        palette_css()
        + """
    BlockWord {
        height: auto;
        width: 1fr;
        color: $text;
    }
    BlockWord.-live { color: $lime; }
    BlockWord.-fail { color: $red; }
    BlockWord.-wait { color: $red; }
    BlockWord.-done { color: $lime; }
    BlockWord.-idle { color: $dim; }
    """
    )

    word: reactive[str] = reactive("")

    def __init__(
        self,
        word: str = "",
        *,
        role: str = "live",
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._role = "live"
        self._reveal = 0
        self._timer: Timer | None = None
        self.set_role(role)
        self.word = word

    @property
    def role(self) -> str:
        """The role the word wears."""
        return self._role

    def set_role(self, role: str) -> None:
        """Wear the colour of ``role``: live, fail, wait or done."""
        bare = role.lstrip("-")
        name = bare if bare in _ROLES else "live"
        self._role = name
        self.remove_class(*[f"-{item}" for item in _ROLES])
        self.add_class(f"-{name}")

    def watch_word(self, old: str, new: str) -> None:
        """Show the new word: at once when motion is off, otherwise wiped in."""
        self._stop_timer()
        width = self._width(new)
        if old == "" or not self._motion_enabled():
            self._reveal = width
            self.refresh()
            return
        self._reveal = 0
        self.refresh()
        self._timer = self.set_interval(_STEP_SECONDS, self._step)

    def on_unmount(self) -> None:
        """Stop the wipe when the widget leaves the screen."""
        self._stop_timer()

    def render(self) -> Text:
        """Draw the word: block glyphs, or one plain line when too narrow."""
        word = self.word.strip().upper()
        if not word:
            return Text("")
        tall = self._tall()
        width = word_width(word) if tall else compact_width(word)
        available = self.content_size.width or self.size.width
        if available and available < width:
            return Text(word, style="bold", no_wrap=True)
        if not tall:
            return render_compact(word, SHADOW.get(self._role, SHADOW["live"]), self._reveal)
        reveal = max(0, min(self._reveal, width))
        text = Text(no_wrap=True)
        for index, row in enumerate(render_word(word)):
            if index:
                text.append("\n")
            text.append(row[:reveal])
            if reveal < len(row):
                text.append(row[reveal:], style="dim")
        return text

    def _width(self, word: str) -> int:
        return word_width(word) if self._tall() else compact_width(word)

    def _tall(self) -> bool:
        """Whether the screen has room for the six-row ANSI Shadow letters."""
        if not self.is_mounted:
            return False
        return self.app.size.height >= TALL_ROWS

    def _motion_enabled(self) -> bool:
        """Whether the wipe may run: the app must be up and motion on."""
        if not self.is_mounted:
            return False
        return self.app.animation_level != "none"

    def _step(self) -> None:
        """Reveal the next columns of the wipe."""
        width = self._width(self.word)
        step = max(1, math.ceil(width / _STEPS))
        self._reveal = min(width, self._reveal + step)
        self.refresh()
        if self._reveal >= width:
            self._stop_timer()

    def _stop_timer(self) -> None:
        """Stop the wipe timer when it runs."""
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
