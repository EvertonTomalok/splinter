"""Scroll panes that follow the tail only while the reader sits there.

A refresh never moves the reader. While the pane is at its bottom, new lines
keep it there; once the reader scrolls up the position holds, and new content
shows a "↓ new" bar under the pane. Clicking the bar (or scrolling back down)
pins the pane to the tail again.
"""

from __future__ import annotations

from typing import Any, Self

from textual.containers import VerticalScroll
from textual.message import Message
from textual.widget import Widget
from textual.widgets import RichLog

from splinter.tui_run_palette import LIME, MUTED, Ink

NEW_BELOW = (
    f"[@click=app.follow_tail('{{target}}')][b {LIME}]↓ new below[/] "
    f"[{MUTED}]· click to follow the tail[/][/]"
)


class TailChanged(Message):
    """A follow pane reached its tail (``following``) or got content below the reader."""

    def __init__(self, pane: Widget, following: bool) -> None:
        super().__init__()
        self.pane = pane
        self.following = following


def _at_tail(widget: Widget) -> bool:
    return widget.scroll_offset.y >= widget.max_scroll_y - 1


class _Follow:
    following: bool = True

    def _track(self, widget: Widget, new_value: float) -> None:
        tail = new_value >= widget.max_scroll_y - 1
        if tail != self.following:
            self.following = tail
            if tail:
                widget.post_message(TailChanged(widget, True))

    def to_tail(self, widget: Widget) -> None:
        self.following = True
        widget.scroll_end(animate=False, immediate=True)


class FollowScroll(_Follow, VerticalScroll):
    """A ``VerticalScroll`` that remembers whether the reader is at the tail."""

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
        self._track(self, new_value)

    def new_below(self) -> bool:
        """Whether there is content under the reader's position."""
        return not _at_tail(self)

    def follow(self) -> None:
        self.to_tail(self)


class FollowLog(_Follow, RichLog):
    """A ``RichLog`` that scrolls to new lines only while the reader is at the end."""

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
        self._track(self, new_value)

    def write(
        self,
        content: Any,
        width: int | None = None,
        expand: bool = False,
        shrink: bool = True,
        scroll_end: bool | None = None,
        animate: bool = False,
    ) -> Self:
        stick = self.following if scroll_end is None else scroll_end
        if not stick:
            self.post_message(TailChanged(self, False))
        return super().write(content, width, expand, shrink, stick, animate)

    def follow(self) -> None:
        self.to_tail(self)


def new_bar(target: str, bar_id: str) -> Ink:
    """The hidden "↓ new below" bar for the pane whose id is ``target``."""
    bar = Ink(NEW_BELOW.format(target=target), id=bar_id, classes="new-below")
    bar.display = False
    return bar
