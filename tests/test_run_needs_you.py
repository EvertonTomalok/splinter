from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

    from splinter.memory.session import Session
    from splinter.tui import RunApp


def _session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Session:
    from tests.fixtures_run_view import build_session

    return build_session(tmp_path, monkeypatch)


def _app(session: Session, monkeypatch: pytest.MonkeyPatch) -> RunApp:
    from splinter.tui import RunApp

    monkeypatch.setattr(RunApp, "_work", lambda self: None)
    return RunApp(session, {})


def test_asked_banner_option_uses_the_modal_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pytest

    pytest.importorskip("textual")
    from textual.widgets import Static

    from splinter.tui import RunApp
    from splinter.tui_run_banner import NeedsYouBanner

    session = _session(tmp_path, monkeypatch)
    session.set_status(
        "awaiting_user",
        tasks=58,
        stage="run",
        task_index=16,
        ask_reason="A. x\nB. y",
    )
    app = _app(session, monkeypatch)
    choices: list[tuple[str, str] | None] = []

    def handle(
        self: RunApp,
        result: tuple[str, str] | None,
        reopen: Callable[[str], None],
    ) -> None:
        choices.append(result)

    monkeypatch.setattr(RunApp, "_handle_ask_choice", handle)

    async def drive() -> None:
        async with app.run_test(size=(120, 36)) as pilot:
            await pilot.pause()
            await pilot.pause()
            banner = app.query_one("#needs-you", NeedsYouBanner)
            assert banner.display
            assert "A. x" in str(banner.query_one(".nb-title", Static).content)
            await pilot.click("#nb-opt-B")
            await pilot.pause()

    asyncio.run(drive())
    assert choices == [("answer", "B. y")]


def test_validation_banner_approve_uses_the_modal_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pytest

    pytest.importorskip("textual")

    from splinter.tui import RunApp
    from splinter.tui_run_banner import NeedsYouBanner

    session = _session(tmp_path, monkeypatch)
    session.set_status(
        "awaiting_validation",
        tasks=58,
        stage="final_eval",
        task_index=23,
        final_eval_summary="Review the implementation",
        final_eval_passed=False,
    )
    app = _app(session, monkeypatch)
    choices: list[tuple[str, str] | None] = []

    def handle(
        self: RunApp,
        result: tuple[str, str] | None,
        reopen: Callable[[str], None],
    ) -> None:
        choices.append(result)

    monkeypatch.setattr(RunApp, "_handle_validation_choice", handle)

    async def drive() -> None:
        async with app.run_test(size=(120, 36)) as pilot:
            await pilot.pause()
            await pilot.pause()
            assert app.query_one("#needs-you", NeedsYouBanner).display
            await pilot.click("#nb-approve")
            await pilot.pause()

    asyncio.run(drive())
    assert choices == [("approve", "")]


def test_failed_story_retry_queues_guidance_and_restarts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pytest

    pytest.importorskip("textual")
    from textual.widgets import Input

    session = _session(tmp_path, monkeypatch)
    app = _app(session, monkeypatch)
    killed: list[bool] = []
    monkeypatch.setattr(app, "_do_pause_kill", lambda: killed.append(True))

    async def drive() -> None:
        async with app.run_test(size=(120, 36)) as pilot:
            await pilot.pause()
            app.select_story(9)
            app._refresh()
            await pilot.pause()
            app.query_one("#nb-guidance", Input).value = "Apply the exemption"
            await pilot.click("#nb-retry")
            await pilot.pause()
            assert killed == []
            await pilot.press("y")
            await pilot.pause()

    asyncio.run(drive())
    assert killed == [True]
    queued = session.dir / "pending_directive.task-9.txt"
    assert "Apply the exemption" in queued.read_text()


def test_stop_bar_pause_uses_the_existing_pause_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pytest

    pytest.importorskip("textual")
    from textual.containers import Horizontal

    from splinter.tui_run_stop import StopBar

    session = _session(tmp_path, monkeypatch)
    app = _app(session, monkeypatch)
    calls: list[str] = []
    monkeypatch.setattr("splinter.procreg.request_stop", lambda: calls.append("pause"))

    async def drive() -> None:
        async with app.run_test(size=(120, 36)) as pilot:
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()
            assert app.query_one("#stop-bar", StopBar).display
            assert not app.query_one("#footer-bar", Horizontal).display
            await pilot.press("p")
            await pilot.pause()
            assert not app.query_one("#stop-bar", StopBar).display
            assert app.query_one("#footer-bar", Horizontal).display

    asyncio.run(drive())
    assert calls == ["pause"]


def test_stop_bar_escape_restores_the_footer_without_stopping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pytest

    pytest.importorskip("textual")
    from textual.containers import Horizontal

    from splinter.tui_run_stop import StopBar

    session = _session(tmp_path, monkeypatch)
    app = _app(session, monkeypatch)
    calls: list[str] = []
    monkeypatch.setattr("splinter.procreg.request_stop", lambda: calls.append("pause"))

    async def drive() -> None:
        async with app.run_test(size=(120, 36)) as pilot:
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert not app.query_one("#stop-bar", StopBar).display
            assert app.query_one("#footer-bar", Horizontal).display

    asyncio.run(drive())
    assert calls == []


def test_short_screen_uses_the_ask_modal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pytest

    pytest.importorskip("textual")

    from splinter.tui import _AskUserModal

    session = _session(tmp_path, monkeypatch)
    session.set_status(
        "awaiting_user",
        tasks=58,
        stage="run",
        task_index=16,
        ask_reason="A. x\nB. y",
    )
    app = _app(session, monkeypatch)

    async def drive() -> None:
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.pause()
            assert isinstance(app.screen, _AskUserModal)

    asyncio.run(drive())


def test_an_empty_answer_is_not_sent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import pytest

    pytest.importorskip("textual")
    from textual.widgets import Input

    from splinter.tui import RunApp
    from splinter.tui_run_banner import NeedsYouBanner

    session = _session(tmp_path, monkeypatch)
    session.set_status(
        "awaiting_user", tasks=58, stage="run", task_index=16, ask_reason="Which port?"
    )
    app = _app(session, monkeypatch)
    choices: list[tuple[str, str] | None] = []

    def handle(
        self: RunApp,
        result: tuple[str, str] | None,
        reopen: Callable[[str], None],
    ) -> None:
        choices.append(result)

    monkeypatch.setattr(RunApp, "_handle_ask_choice", handle)

    async def drive() -> None:
        async with app.run_test(size=(120, 36)) as pilot:
            await pilot.pause()
            await pilot.pause()
            banner = app.query_one("#needs-you", NeedsYouBanner)
            banner.query_one("#nb-submit").press()
            await pilot.pause()
            assert app.focused is banner.query_one("#nb-answer", Input)
            banner.query_one("#nb-answer", Input).value = "8080"
            banner.query_one("#nb-submit").press()
            await pilot.pause()

    asyncio.run(drive())
    assert choices == [("answer", "8080")]
