"""The C2 look shared by the PRD and plan screens: theme, top bar and flat controls.

The design (``tasks/c2-focus-tui/design``) is a near-black page, light text, one lime
accent and amber for attention. :data:`SPLINTER_THEME` maps it onto Textual's theme
variables, so stock widgets (footer, text areas, scrollbars, buttons) pick it up;
:data:`SHELL_CSS` flattens the blue default controls into the design's plain text keys.
"""

from __future__ import annotations

from typing import Any

from rich.markup import escape
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.theme import Theme
from textual.widget import Widget
from textual.widgets import Static

from splinter.tui_run_palette import PALETTE, Ink

BG = PALETTE["focus-bg"]
LIME = PALETTE["lime"]
AMBER = PALETTE["amber"]
RED = PALETTE["red"]
MUTED = PALETTE["muted"]
DIM = PALETTE["dim"]
TEXT = "#ecebe4"
LINE = "#1d1d1a"
EDGE = "#2a2a26"
HOVER = "#121210"


def _theme() -> Theme:
    return Theme(
        name="splinter",
        primary=LIME,
        secondary=MUTED,
        accent=AMBER,
        warning=AMBER,
        error=RED,
        success=LIME,
        foreground=TEXT,
        background=BG,
        surface=BG,
        panel=HOVER,
        boost=HOVER,
        dark=True,
        variables={
            "text-muted": MUTED,
            "border": EDGE,
            "border-blurred": LINE,
            "footer-background": BG,
            "footer-key-foreground": LIME,
            "footer-key-background": BG,
            "footer-description-foreground": MUTED,
            "footer-description-background": BG,
            "scrollbar": LINE,
            "scrollbar-hover": DIM,
            "scrollbar-active": MUTED,
            "scrollbar-background": BG,
            "scrollbar-background-hover": BG,
            "scrollbar-background-active": BG,
            "scrollbar-corner-color": BG,
            "block-cursor-background": LIME,
            "block-cursor-foreground": BG,
            "input-selection-background": f"{LIME} 30%",
            "link-style": "none",
            "link-background": "initial",
            "link-color": TEXT,
            "link-style-hover": "bold not underline",
            "link-background-hover": HOVER,
            "link-color-hover": "#ffffff",
        },
    )


def use_splinter_theme(app: App[Any]) -> None:
    """Register the C2 theme on ``app`` and switch to it."""
    theme = _theme()
    app.register_theme(theme)
    app.theme = theme.name


def crumb_markup(mode: str, label: str) -> str:
    """``splinter / --prd --task  <file>``: the active mode is a light chip, both click."""
    chips = []
    for name in ("prd", "task"):
        tone = f"{BG} on {TEXT}" if name == mode else f"{MUTED} on {HOVER}"
        chips.append(f"[@click=app.mode('{name}')][{tone}] --{name} [/][/]")
    file_part = f"  [{MUTED}]{escape(label)}[/]" if label else ""
    return f"[b {LIME}]splinter[/] [{DIM}]/[/] {''.join(chips)}{file_part}"


class ShellBar(Horizontal):
    """Top bar: breadcrumb on the left, the phase stepper on the right."""

    DEFAULT_CSS = f"""
    ShellBar {{
        height: 3;
        padding: 1 2 0 2;
        background: {BG};
        border-bottom: solid {LINE};
    }}
    ShellBar #shell-crumb {{
        width: 1fr;
        height: 1;
    }}
    ShellBar PhaseStepper {{
        width: auto;
        padding: 0;
    }}
    """

    def __init__(self, crumb: str, right: Widget, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self._crumb = crumb
        self._right = right

    def compose(self) -> ComposeResult:
        yield Ink(self._crumb, id="shell-crumb")
        yield self._right

    def set_crumb(self, crumb: str) -> None:
        """Replace the breadcrumb text."""
        self._crumb = crumb
        for crumb_widget in self.query("#shell-crumb").results(Static):
            crumb_widget.update(crumb)


SHELL_CSS = f"""
    Screen {{
        background: {BG};
        color: {TEXT};
    }}
    Footer {{
        background: {BG};
        border-top: solid {LINE};
        height: 2;
    }}
    FooterKey {{
        background: {BG};
    }}
    FooterKey .footer-key--key {{
        color: {LIME};
        background: {BG};
        text-style: bold;
    }}
    FooterKey .footer-key--description {{
        color: {MUTED};
        background: {BG};
    }}
    FooterKey:hover {{
        background: {HOVER};
    }}
    TextArea {{
        background: {BG};
        color: {TEXT};
    }}
    TextArea .text-area--cursor-line {{
        background: {HOVER};
    }}
    TextArea .text-area--cursor {{
        background: {LIME};
        color: {BG};
    }}
    RichLog {{
        background: {BG};
    }}
    Button {{
        height: 1;
        min-width: 0;
        width: auto;
        border: none;
        padding: 0 1;
        margin: 0 2 0 0;
        background: {BG};
        color: {TEXT};
        text-style: none;
    }}
    Button:hover {{
        background: {HOVER};
        color: #ffffff;
        border: none;
    }}
    Button:focus {{
        text-style: bold;
        background-tint: {LIME} 0%;
    }}
    Button.-active {{
        background: {HOVER};
        border: none;
        tint: {BG} 0%;
    }}
    Button.-primary {{
        background: {LIME};
        color: {BG};
        text-style: bold;
    }}
    Button.-primary:hover {{
        background: #d6ff52;
        color: {BG};
    }}
    Button.-success {{
        color: {LIME};
    }}
    Button.-warning {{
        color: {AMBER};
    }}
    Button.-error {{
        color: {RED};
    }}
    Tooltip {{
        background: {HOVER};
        color: {TEXT};
        border: round {EDGE};
    }}
"""
