from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

    from splinter.memory.session import Session
    from splinter.tui import RunApp


def _session(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch", sid: str) -> "Session":
    from splinter.memory.session import Session

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session(sid)
    session.update_index("# stop modal\n")
    return session


def _running_session(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> "Session":
    session = _session(tmp_path, monkeypatch, "ses_stop_running")
    session.set_status(
        "running", strategy="raphael", tasks=1, stage="run", pid=os.getpid()
    )
    return session


def _idle_session(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> "Session":
    session = _session(tmp_path, monkeypatch, "ses_stop_idle")
    session.set_status("completed", strategy="raphael", tasks=1, stage="done")
    return session


def _drive(app: "RunApp", body: object, size: tuple[int, int] = (120, 24)) -> None:
    """Run ``app`` headless, letting ``body`` (a coroutine fn) drive the pilot."""

    async def run() -> None:
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            await pilot.pause(0.05)
            await body(pilot)  # type: ignore[operator]

    asyncio.run(run())


def test_stop_key_opens_the_stop_bar(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    import pytest

    pytest.importorskip("textual")

    from textual.containers import Horizontal

    from splinter.tui import RunApp
    from splinter.tui_run_stop import StopBar

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    monkeypatch.setattr(RunApp, "_work", lambda self: None)
    session = _running_session(tmp_path, monkeypatch)

    async def body(pilot: object) -> None:
        await pilot.press("s")  # type: ignore[attr-defined]
        await pilot.pause()
        assert app.query_one("#stop-bar", StopBar).display
        assert not app.query_one("#footer-bar", Horizontal).display

    app = RunApp(session, {})
    _drive(app, body, size=(160, 44))


def test_stop_key_opens_the_stop_menu_on_a_short_screen(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    import pytest

    pytest.importorskip("textual")

    from splinter.tui import RunApp
    from splinter.tui_run_stop import StopMenuModal

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    monkeypatch.setattr(RunApp, "_work", lambda self: None)
    session = _running_session(tmp_path, monkeypatch)

    async def body(pilot: object) -> None:
        await pilot.press("s")  # type: ignore[attr-defined]
        await pilot.pause()
        assert isinstance(app.screen, StopMenuModal)
        assert app.screen.query_one("#stop-pause")

    app = RunApp(session, {})
    _drive(app, body)


def test_pause_button_requests_a_graceful_pause(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    import pytest

    pytest.importorskip("textual")

    from splinter.tui import RunApp

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    monkeypatch.setattr(RunApp, "_work", lambda self: None)
    session = _running_session(tmp_path, monkeypatch)

    calls: list[str] = []
    monkeypatch.setattr("splinter.procreg.request_stop", lambda: calls.append("pause"))
    monkeypatch.setattr("splinter.procreg.terminate_all", lambda: calls.append("terminate"))

    async def body(pilot: object) -> None:
        await pilot.press("s")  # type: ignore[attr-defined]
        await pilot.pause()
        await pilot.click("#stop-pause")  # type: ignore[attr-defined]
        await pilot.pause()

    app = RunApp(session, {})
    app.write_log = lambda *a, **k: None  # type: ignore[method-assign]
    _drive(app, body)

    assert calls == ["pause"], f"pause button did not request a graceful pause — {calls}"
    assert session.read_status()["state"] == "running"


def test_restart_button_kills_and_restarts_from_checkpoint(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    import pytest

    pytest.importorskip("textual")

    from splinter.tui import RunApp

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    monkeypatch.setattr(RunApp, "_work", lambda self: None)
    session = _running_session(tmp_path, monkeypatch)

    calls: list[str] = []
    monkeypatch.setattr("splinter.procreg.request_stop", lambda: calls.append("pause"))
    monkeypatch.setattr("splinter.procreg.terminate_all", lambda: calls.append("terminate"))

    async def body(pilot: object) -> None:
        await pilot.press("s")  # type: ignore[attr-defined]
        await pilot.pause()
        await pilot.click("#stop-restart")  # type: ignore[attr-defined]
        await pilot.pause()

    app = RunApp(session, {})
    app.write_log = lambda *a, **k: None  # type: ignore[method-assign]
    _drive(app, body)

    assert calls == ["terminate"], f"restart button did not terminate — {calls}"
    status = session.read_status()
    assert status["state"] == "paused"
    assert status["reason"] == "user_kill_restart"
    assert app._restart_after_kill is True


def test_kill_button_terminates_and_exits(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    import pytest

    pytest.importorskip("textual")

    from splinter.tui import RunApp

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    monkeypatch.setattr(RunApp, "_work", lambda self: None)
    session = _running_session(tmp_path, monkeypatch)

    calls: list[str] = []
    monkeypatch.setattr("splinter.procreg.request_stop", lambda: calls.append("pause"))
    monkeypatch.setattr("splinter.procreg.terminate_all", lambda: calls.append("terminate"))

    async def body(pilot: object) -> None:
        await pilot.press("s")  # type: ignore[attr-defined]
        await pilot.pause()
        await pilot.click("#stop-kill")  # type: ignore[attr-defined]
        await pilot.pause()

    app = RunApp(session, {})
    app.write_log = lambda *a, **k: None  # type: ignore[method-assign]
    _drive(app, body)

    assert calls == ["terminate"], f"kill button did not terminate — {calls}"
    status = session.read_status()
    assert status["state"] == "paused"
    assert status["reason"] == "user_kill"


def test_cancel_button_leaves_the_run_alone(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    import pytest

    pytest.importorskip("textual")

    from splinter.tui import RunApp

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    monkeypatch.setattr(RunApp, "_work", lambda self: None)
    session = _running_session(tmp_path, monkeypatch)

    calls: list[str] = []
    monkeypatch.setattr("splinter.procreg.request_stop", lambda: calls.append("pause"))
    monkeypatch.setattr("splinter.procreg.terminate_all", lambda: calls.append("terminate"))

    async def body(pilot: object) -> None:
        await pilot.press("s")  # type: ignore[attr-defined]
        await pilot.pause()
        await pilot.click("#stop-cancel")  # type: ignore[attr-defined]
        await pilot.pause()

    app = RunApp(session, {})
    app.write_log = lambda *a, **k: None  # type: ignore[method-assign]
    _drive(app, body)

    assert calls == [], f"cancel must not touch the run — {calls}"
    assert session.read_status()["state"] == "running"


def test_q_while_running_opens_the_stop_bar(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    import pytest

    pytest.importorskip("textual")

    from splinter.tui import RunApp
    from splinter.tui_run_stop import StopBar

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    monkeypatch.setattr(RunApp, "_work", lambda self: None)
    session = _running_session(tmp_path, monkeypatch)

    exits: list[int] = []
    monkeypatch.setattr("splinter.procreg.terminate_all", lambda: None)

    async def body(pilot: object) -> None:
        await pilot.press("q")  # type: ignore[attr-defined]
        await pilot.pause()
        assert app.query_one("#stop-bar", StopBar).display

    app = RunApp(session, {})
    app.write_log = lambda *a, **k: None  # type: ignore[method-assign]
    monkeypatch.setattr(app, "exit", lambda code=0: exits.append(code))
    _drive(app, body, size=(160, 44))

    assert exits == [], "q must not exit while the run is live"


def test_q_when_not_running_exits(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    import pytest

    pytest.importorskip("textual")

    from splinter.tui import RunApp

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    monkeypatch.setattr(RunApp, "_work", lambda self: None)
    session = _idle_session(tmp_path, monkeypatch)

    exits: list[int] = []

    async def body(pilot: object) -> None:
        await pilot.press("q")  # type: ignore[attr-defined]
        await pilot.pause()

    app = RunApp(session, {})
    app.write_log = lambda *a, **k: None  # type: ignore[method-assign]
    monkeypatch.setattr(app, "exit", lambda code=0: exits.append(code))
    _drive(app, body)

    assert exits == [0], f"q must exit when the run is not live — {exits}"


def test_escape_cancels_the_stop_bar(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    import pytest

    pytest.importorskip("textual")

    from textual.containers import Horizontal

    from splinter.tui import RunApp
    from splinter.tui_run_stop import StopBar

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    monkeypatch.setattr(RunApp, "_work", lambda self: None)
    session = _running_session(tmp_path, monkeypatch)

    calls: list[str] = []
    monkeypatch.setattr("splinter.procreg.request_stop", lambda: calls.append("pause"))
    monkeypatch.setattr("splinter.procreg.terminate_all", lambda: calls.append("terminate"))

    async def body(pilot: object) -> None:
        await pilot.press("s")  # type: ignore[attr-defined]
        await pilot.pause()
        assert app.query_one("#stop-bar", StopBar).display
        await pilot.press("escape")  # type: ignore[attr-defined]
        await pilot.pause()
        assert not app.query_one("#stop-bar", StopBar).display
        assert app.query_one("#footer-bar", Horizontal).display

    app = RunApp(session, {})
    app.write_log = lambda *a, **k: None  # type: ignore[method-assign]
    _drive(app, body, size=(160, 44))

    assert calls == [], f"esc must not touch the run — {calls}"


def test_run_app_footer_and_bindings_show_the_new_keys(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    import pytest

    pytest.importorskip("textual")

    from splinter.tui import RunApp

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    _idle_session(tmp_path, monkeypatch)

    keys = {b.key: b for b in RunApp.BINDINGS}
    assert "s" in keys, "the stop menu needs a visible key"
    assert "q" in keys
    assert "ctrl+k" in keys
    assert "p" not in keys, "the old pause binding must be gone"
    assert "escape" not in keys, "escape must not kill the run behind the user's back"
    assert keys["s"].description == "Stop"
    assert keys["q"].description == "Quit"
    assert keys["ctrl+k"].description == "Cowabunga"


def test_log_lines_carry_a_dim_timestamp(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    import pytest

    pytest.importorskip("textual")

    from textual.widgets import RichLog

    from splinter.tui import RunApp

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    monkeypatch.setattr(RunApp, "_work", lambda self: None)
    session = _idle_session(tmp_path, monkeypatch)

    async def body(pilot: object) -> None:
        await pilot.pause()
        app.write_log("hello world", logging.WARNING)
        await app.run_action("set_view('logs')")
        await pilot.pause()
        log = app.query_one("#log", RichLog)
        lines = list(log.lines)
        assert lines, "nothing written to the log"
        plain = lines[-1].text
        assert "hello world" in plain, plain
        assert plain[:2].isdigit(), f"log line must start with HH:MM:SS — {plain!r}"
        assert plain[2:3] == ":" and plain[5:6] == ":", plain

    app = RunApp(session, {})
    _drive(app, body)
