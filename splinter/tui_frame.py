"""Tight one-line frames for the C2 screens, drawn with eighth blocks.

The design frames its segmented controls and answer keys with a 1px line that
hugs the text. A box-drawing frame sits in the middle of its cells, a full row
away from the text. Eighth blocks sit on the cell edge instead: ``▁`` on the
row above, ``▔`` on the row below, ``▕`` and ``▏`` on the columns beside, so
the line touches the framed text the way the design's border does. The corner
cells stay blank: the two lines already meet at the cell corner.
"""

from __future__ import annotations

from rich.cells import cell_len
from rich.markup import escape, render

EDGE = "#2a2a26"
KEY_EDGE = "#55554e"


def frame_markup(inner: str, color: str = EDGE, *, lead: str = "", tail: str = "") -> str:
    """Three markup lines: ``inner`` (one line of markup) framed in ``color``.

    ``lead`` and ``tail`` are markup put before and after the frame on the
    middle line; the rows above and below are padded to line up with them.
    An empty ``color`` leaves the frame in the widget's own CSS ``color``, so a
    ``:focus`` rule can recolour it.
    """
    width = cell_len(render(inner).plain)
    pad = " " * cell_len(render(lead).plain)

    def tone(glyphs: str) -> str:
        return f"[{color}]{glyphs}[/]" if color else glyphs

    top = f"{pad} {tone('▁' * width)} "
    middle = f"{lead}{tone('▕')}{inner}{tone('▏')}{tail}"
    bottom = f"{pad} {tone('▔' * width)} "
    return "\n".join((top, middle, bottom))


def segments_markup(segments: list[tuple[str, str, bool]]) -> str:
    """``[label](action)`` chips side by side; the active one is a light chip."""
    chips = []
    for label, action, active in segments:
        tone = "#0a0a09 on #ecebe4" if active else "#8f8e85"
        chips.append(f"[@click={action}][{tone}] {escape(label)} [/][/]")
    return "".join(chips)
