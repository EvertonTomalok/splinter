from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


def test_long_overview_shows_scrollbar(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
    import pytest

    pytest.importorskip("textual")

    from textual.containers import VerticalScroll
    from textual.widgets import Static

    from splinter.memory.session import Session
    from splinter.tui import RunApp

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_us001_long")
    session.update_index("# us001 long\n")
    session.set_status("running", strategy="raphael", tasks=1, stage="run")

    def patched_refresh(self: RunApp) -> None:
        tall_content = "\n".join([f"Line {i}" for i in range(200)])
        self.query_one("#overview", Static).update(tall_content)

    monkeypatch.setattr(RunApp, "_refresh", patched_refresh)

    def fake_pipeline(**kwargs: object) -> int:
        return 0

    monkeypatch.setattr("splinter.pipeline.run_pipeline", fake_pipeline)

    async def drive() -> None:
        app = RunApp(session, {})
        async with app.run_test() as pilot:
            await pilot.pause()
            for _ in range(20):
                await pilot.pause(0.05)
                if app.workers and all(w.state.name in ("SUCCESS", "ERROR") for w in app.workers):
                    break
            await pilot.pause()
            scroll = app.query_one("#overview-scroll", VerticalScroll)
            max_scroll = scroll.max_scroll_y
            assert max_scroll > 0, f"scrollbar not present — max_scroll_y = {max_scroll}"
            await pilot.press("q")

    asyncio.run(drive())


def test_short_overview_no_scrollbar(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
    import pytest

    pytest.importorskip("textual")

    from textual.containers import VerticalScroll
    from textual.widgets import Static

    from splinter.memory.session import Session
    from splinter.tui import RunApp

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_us001_short")
    session.update_index("# us001 short\n")
    session.set_status("running", strategy="raphael", tasks=1, stage="run")

    def patched_refresh(self: RunApp) -> None:
        short_content = "Short line 1\nShort line 2\nShort line 3"
        self.query_one("#overview", Static).update(short_content)

    monkeypatch.setattr(RunApp, "_refresh", patched_refresh)

    def fake_pipeline(**kwargs: object) -> int:
        return 0

    monkeypatch.setattr("splinter.pipeline.run_pipeline", fake_pipeline)

    async def drive() -> None:
        app = RunApp(session, {})
        async with app.run_test() as pilot:
            await pilot.pause()
            for _ in range(20):
                await pilot.pause(0.05)
                if app.workers and all(w.state.name in ("SUCCESS", "ERROR") for w in app.workers):
                    break
            await pilot.pause()
            scroll = app.query_one("#overview-scroll", VerticalScroll)
            max_scroll = scroll.max_scroll_y
            assert max_scroll == 0, (
                f"scrollbar present but should not be — max_scroll_y = {max_scroll}"
            )
            await pilot.press("q")

    asyncio.run(drive())


def test_scroll_reaches_top_and_bottom(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
    import pytest

    pytest.importorskip("textual")

    from textual.containers import VerticalScroll
    from textual.widgets import Static

    from splinter.memory.session import Session
    from splinter.tui import RunApp

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_us001_topbot")
    session.update_index("# us001 topbot\n")
    session.set_status("running", strategy="raphael", tasks=1, stage="run")

    def patched_refresh(self: RunApp) -> None:
        tall_content = "\n".join([f"Line {i}" for i in range(200)])
        self.query_one("#overview", Static).update(tall_content)

    monkeypatch.setattr(RunApp, "_refresh", patched_refresh)

    def fake_pipeline(**kwargs: object) -> int:
        return 0

    monkeypatch.setattr("splinter.pipeline.run_pipeline", fake_pipeline)

    async def drive() -> None:
        app = RunApp(session, {})
        async with app.run_test() as pilot:
            await pilot.pause()
            for _ in range(20):
                await pilot.pause(0.05)
                if app.workers and all(w.state.name in ("SUCCESS", "ERROR") for w in app.workers):
                    break
            await pilot.pause()
            scroll = app.query_one("#overview-scroll", VerticalScroll)
            max_scroll = scroll.max_scroll_y

            assert scroll.scroll_offset.y == 0, "should start at top"

            await pilot.press("end")
            await pilot.pause()
            assert scroll.scroll_offset.y == max_scroll, (
                f"end key should reach bottom — {scroll.scroll_offset.y} != {max_scroll}"
            )

            await pilot.press("home")
            await pilot.pause()
            assert scroll.scroll_offset.y == 0, "home key should return to top"

            await pilot.press("q")

    asyncio.run(drive())


def test_pagedown_pageup_scroll(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
    import pytest

    pytest.importorskip("textual")

    from textual.containers import VerticalScroll
    from textual.widgets import Static

    from splinter.memory.session import Session
    from splinter.tui import RunApp

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_us001_pageupdown")
    session.update_index("# us001 pageupdown\n")
    session.set_status("running", strategy="raphael", tasks=1, stage="run")

    def patched_refresh(self: RunApp) -> None:
        tall_content = "\n".join([f"Line {i}" for i in range(200)])
        self.query_one("#overview", Static).update(tall_content)

    monkeypatch.setattr(RunApp, "_refresh", patched_refresh)

    def fake_pipeline(**kwargs: object) -> int:
        return 0

    monkeypatch.setattr("splinter.pipeline.run_pipeline", fake_pipeline)

    async def drive() -> None:
        app = RunApp(session, {})
        async with app.run_test() as pilot:
            await pilot.pause()
            for _ in range(20):
                await pilot.pause(0.05)
                if app.workers and all(w.state.name in ("SUCCESS", "ERROR") for w in app.workers):
                    break
            await pilot.pause()
            scroll = app.query_one("#overview-scroll", VerticalScroll)

            start_offset = scroll.scroll_offset.y
            assert start_offset == 0, "should start at top"

            await pilot.press("pagedown")
            await pilot.pause()
            after_pagedown = scroll.scroll_offset.y
            assert after_pagedown > start_offset, (
                f"pagedown should scroll — {after_pagedown} <= {start_offset}"
            )

            await pilot.press("pageup")
            await pilot.pause()
            after_pageup = scroll.scroll_offset.y
            assert after_pageup < after_pagedown, (
                f"pageup should scroll back — {after_pageup} >= {after_pagedown}"
            )

            await pilot.press("q")

    asyncio.run(drive())


def test_run_keybindings_not_shadowed(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
    import pytest

    pytest.importorskip("textual")

    from textual.widgets import Static

    from splinter.memory.session import Session
    from splinter.tui import RunApp

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_us001_keybind")
    session.update_index("# us001 keybind\n")
    session.set_status("running", strategy="raphael", tasks=1, stage="run")

    def patched_refresh(self: RunApp) -> None:
        tall_content = "\n".join([f"Line {i}" for i in range(200)])
        self.query_one("#overview", Static).update(tall_content)

    monkeypatch.setattr(RunApp, "_refresh", patched_refresh)

    actions_called: list[str] = []

    async def patched_pause_graceful(self: RunApp) -> None:
        actions_called.append("pause_graceful")

    async def patched_pause_kill(self: RunApp) -> None:
        actions_called.append("pause_kill")

    monkeypatch.setattr(RunApp, "action_pause_graceful", patched_pause_graceful)
    monkeypatch.setattr(RunApp, "action_pause_kill", patched_pause_kill)

    def fake_pipeline(**kwargs: object) -> int:
        return 0

    monkeypatch.setattr("splinter.pipeline.run_pipeline", fake_pipeline)

    async def drive() -> None:
        app = RunApp(session, {})
        async with app.run_test() as pilot:
            await pilot.pause()
            for _ in range(20):
                await pilot.pause(0.05)
                if app.workers and all(w.state.name in ("SUCCESS", "ERROR") for w in app.workers):
                    break
            await pilot.pause()

            await pilot.press("p")
            await pilot.pause()

            await pilot.press("escape")
            await pilot.pause()

            await pilot.press("q")

    asyncio.run(drive())

    assert "pause_graceful" in actions_called, (
        f"p key did not trigger pause_graceful — got: {actions_called}"
    )
    assert "pause_kill" in actions_called, (
        f"k key did not trigger pause_kill — got: {actions_called}"
    )


def test_log_pane_unchanged(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
    import pytest

    pytest.importorskip("textual")

    from textual.containers import VerticalScroll
    from textual.widgets import RichLog, Static

    from splinter.memory.session import Session
    from splinter.tui import RunApp

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_us001_logpane")
    session.update_index("# us001 logpane\n")
    session.set_status("running", strategy="raphael", tasks=1, stage="run")

    def fake_pipeline(**kwargs: object) -> int:
        return 0

    monkeypatch.setattr("splinter.pipeline.run_pipeline", fake_pipeline)

    async def drive() -> None:
        app = RunApp(session, {})
        async with app.run_test() as pilot:
            await pilot.pause()
            for _ in range(20):
                await pilot.pause(0.05)
                if app.workers and all(w.state.name in ("SUCCESS", "ERROR") for w in app.workers):
                    break
            await pilot.pause()

            overview_scroll = app.query_one("#overview-scroll", VerticalScroll)
            overview_static = app.query_one("#overview", Static)
            log = app.query_one("#log", RichLog)

            assert isinstance(overview_scroll, VerticalScroll), (
                "overview-scroll should be VerticalScroll"
            )
            assert isinstance(overview_static, Static), "overview should be Static"
            assert isinstance(log, RichLog), "log should be RichLog"

            await pilot.press("q")

    asyncio.run(drive())
