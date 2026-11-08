"""Follow panes: a refresh never moves a reader who scrolled up."""

from __future__ import annotations

import asyncio

from textual.app import App, ComposeResult
from textual.widgets import Static

from splinter.tui_follow import FollowLog, FollowScroll, TailChanged


class _Panes(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[str | None, bool]] = []

    def compose(self) -> ComposeResult:
        yield FollowScroll(Static("\n".join(map(str, range(80))), id="body"), id="scroll")
        yield FollowLog(id="log")

    def on_tail_changed(self, event: TailChanged) -> None:
        self.events.append((event.pane.id, event.following))


def test_log_holds_a_reader_who_scrolled_up_and_follows_again_on_request() -> None:
    async def run() -> None:
        app = _Panes()
        async with app.run_test(size=(60, 20)) as pilot:
            log = app.query_one(FollowLog)
            for n in range(60):
                log.write(f"line {n}")
            await pilot.pause()
            assert log.is_vertical_scroll_end and log.following
            log.scroll_home(animate=False, immediate=True)
            await pilot.pause()
            assert not log.following
            log.write("new line")
            await pilot.pause()
            assert log.scroll_offset.y == 0
            assert ("log", False) in app.events
            log.follow()
            await pilot.pause()
            assert log.is_vertical_scroll_end and log.following

    asyncio.run(run())


def test_scroll_keeps_position_when_content_grows_and_reports_the_tail() -> None:
    async def run() -> None:
        app = _Panes()
        async with app.run_test(size=(60, 20)) as pilot:
            scroll = app.query_one(FollowScroll)
            scroll.scroll_to(y=10, animate=False, immediate=True)
            await pilot.pause()
            assert not scroll.following and scroll.new_below()
            app.query_one("#body", Static).update("\n".join(map(str, range(120))))
            await pilot.pause()
            assert scroll.scroll_offset.y == 10
            scroll.scroll_end(animate=False, immediate=True)
            await pilot.pause()
            assert scroll.following and not scroll.new_below()
            assert ("scroll", True) in app.events
            scroll.scroll_home(animate=False, immediate=True)
            scroll.follow()
            await pilot.pause()
            assert scroll.following and not scroll.new_below()

    asyncio.run(run())
