"""Colour tokens of the run screen, as Textual CSS variables.

The values come from the C2 design (``tasks/c2-focus-tui/design/run.html``).
Widgets name the variables (``$lime``, ``$red`` …) in their stylesheets and
the hex values (``INK``, ``LIME`` …) in Rich markup, so the palette lives in
one place.

Textual resolves a variable only where it is declared, so an app that wants
the palette in its own widgets' stylesheets publishes it through
``App.get_css_variables``::

    def get_css_variables(self) -> dict[str, str]:
        return palette_variables(super().get_css_variables())
"""

from __future__ import annotations

from collections.abc import Mapping

from textual.reactive import Reactive
from textual.widgets import Static

BG = "#0a0a09"
INK = "#ecebe4"
MUTED = "#8f8e85"
DIM = "#7a7972"
LIME = "#c6f432"
AMBER = "#ffb020"
RED = "#ff5a4e"
LINE = "#1d1d1a"
LINE2 = "#2a2a26"
EDGE = "#3a3a35"
HOVER = "#141412"
SELECTED = "#1a1a17"
TRACK = "#161614"
GREY_BAR = "#6b6b62"
LIVE_LINE = "#3a4a14"
FAIL_BG = "#170b09"
FAIL_LINE = "#5a1f1a"
ORCH_BG = "#1a1708"
DEEP_RED = "#8a1408"
ADD_FG = "#d7f58a"
ADD_BG = "#141a08"
DEL_FG = "#ff8a80"
DEL_BG = "#1f0d0b"

#: Design tokens: variable name → hex colour.
PALETTE: dict[str, str] = {
    "focus-bg": BG,
    "ink": INK,
    "lime": LIME,
    "red": RED,
    "amber": AMBER,
    "muted": MUTED,
    "dim": DIM,
    "line": LINE,
    "line2": LINE2,
    "edge": EDGE,
    "hover": HOVER,
    "selected": SELECTED,
    "live-line": LIVE_LINE,
    "fail-bg": FAIL_BG,
    "fail-line": FAIL_LINE,
    "deep-red": DEEP_RED,
}

#: Verdict → colour, shared by the activity rows, the iteration bars and eval.
VERDICT_COLOR: dict[str, str] = {
    "PASS": LIME,
    "RETRY": AMBER,
    "ESCALATE": AMBER,
    "FAIL": RED,
    "ASKED": RED,
    "ASK_USER": RED,
}


class Ink(Static):
    """A Static whose clickable spans keep their own colours (no link styling)."""

    auto_links: Reactive[bool] = Reactive(False)


def verdict_label(verdict: str) -> str:
    """The verdict as the design writes it: an escalation is a RETRY."""
    return {"ESCALATE": "RETRY", "ASK_USER": "ASKED"}.get(verdict, verdict)


def palette_css() -> str:
    """Return the palette as Textual CSS variable declarations."""
    return "\n".join(f"${name}: {value};" for name, value in PALETTE.items())


def palette_variables(base: Mapping[str, str]) -> dict[str, str]:
    """Merge the palette into the CSS variables an app already has."""
    return {**base, **PALETTE}
