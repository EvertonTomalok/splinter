"""Header breadcrumb and footer keys of the run screen."""

from __future__ import annotations

from rich.markup import escape
from textual.app import ComposeResult
from textual.containers import Horizontal

from splinter.run_view import RunView, StoryRow
from splinter.tui_run_palette import DIM, INK, LIME, MUTED, Ink
from splinter.tui_run_text import story_label

FOOTER_KEYS: tuple[tuple[str, str, str], ...] = (
    ("i", "instruct", "app.instruct"),
    ("d", "diff", "app.open_diff"),
    ("$", "spend", "app.open_spend('cost')"),
    ("q", "quit", "app.quit_or_stop"),
)


def key_markup(key: str, label: str, action: str) -> str:
    return f"[@click={action}][b {LIME}]{escape(key)}[/] [{INK}]{escape(label)}[/][/]"


def cowabunga_markup(on: bool, error: bool = False) -> str:
    """The ``^k cowabunga off`` key, which also shows the mode."""
    if error:
        return key_markup("^k", "cowabunga · unreadable — off", "app.toggle_kowabunga")
    return key_markup("^k", f"cowabunga {'on' if on else 'off'}", "app.toggle_kowabunga")


class RunHeader(Horizontal):
    """``splinter / strategy / file / story`` on the left, progress on the right."""

    DEFAULT_CSS = """
    RunHeader {
        height: 2;
        padding: 0 3;
        border-bottom: solid $line;
    }
    RunHeader #crumbs { width: 1fr; height: 1; }
    RunHeader #progress { width: auto; height: 1; }
    """

    def compose(self) -> ComposeResult:
        yield Ink(id="crumbs")
        yield Ink(id="progress")

    def show(
        self,
        view: RunView,
        story: int | None,
        strategy: str,
        source: str,
        max_iterations: int,
    ) -> None:
        row = next((item for item in view.stories if item.number == story), None)
        live = next((item for item in view.stories if item.number == view.live_story), None)
        self.query_one("#crumbs", Ink).update(_crumbs(row, live, strategy, source))
        self.query_one("#progress", Ink).update(_progress(view, row, max_iterations))


def _crumbs(row: StoryRow | None, live: StoryRow | None, strategy: str, source: str) -> str:
    sep = f" [{DIM}]/[/] "
    parts = [f"[b {LIME}]splinter[/]", f"[{MUTED}]{escape(strategy)}[/]"]
    if source:
        parts.append(f"[{MUTED}]{escape(source)}[/]")
    if row is not None:
        ident, title = story_label(row)
        parts.append(f"[@click=app.follow_live][b {INK}]{escape(ident)} {escape(title)}[/][/]")
    text = sep.join(parts)
    if row is not None and live is not None and row.number != live.number:
        back = f"← back to live {escape(story_label(live)[0])}"
        text += f"  [@click=app.follow_live][{INK} on #1d1d1a] {back} [{DIM}]esc[/] [/][/]"
    return text


def _progress(view: RunView, row: StoryRow | None, max_iterations: int) -> str:
    total = len(view.stories)
    number = row.number if row is not None else 0
    count = len(view.iterations_by_story.get(number, ()))
    iteration = f"{count}/{max(max_iterations, count)}" if count else "—"
    return (
        f"[{MUTED}]story[/] [b {INK}]{number}/{total}[/] [{DIM}]·[/] "
        f"[{MUTED}]iteration[/] [b {INK}]{iteration}[/]"
    )


class FooterKeys(Horizontal):
    """The key hints of the design; every hint is clickable."""

    DEFAULT_CSS = """
    FooterKeys {
        height: 2;
        padding: 0 3;
        border-top: solid $line;
    }
    FooterKeys Ink { width: auto; height: 1; margin-right: 2; }
    FooterKeys #footer-fill { width: 1fr; margin-right: 0; }
    FooterKeys #footer-clock { margin-right: 0; color: $dim; }
    """

    def compose(self) -> ComposeResult:
        yield Ink(key_markup("s", "stop", "app.stop"), id="footer-stop")
        yield Ink(cowabunga_markup(False), id="kowabunga-ind")
        for key, label, action in FOOTER_KEYS:
            yield Ink(key_markup(key, label, action), classes="footer-key")
        yield Ink(id="footer-fill")
        yield Ink(id="footer-clock")
