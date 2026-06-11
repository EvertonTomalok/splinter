from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


def test_live_logger_routes_into_richlog(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    import pytest

    pytest.importorskip("textual")

    from splinter.memory.session import Session
    from splinter.tui import RunApp

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_us003_route")
    session.update_index("# us003\n")
    session.set_status("running", strategy="raphael", tasks=1, stage="run")

    captured: list[tuple[str, int]] = []

    def fake_pipeline(**kwargs: object) -> int:
        live = logging.getLogger("splinter.live")
        live.info("  🔧 tool a")
        live.info("  💬 text b")
        live.info("  ✏️ edit c")
        return 0

    monkeypatch.setattr("splinter.pipeline.run_pipeline", fake_pipeline)

    async def drive() -> None:
        app = RunApp(session, {})
        original_write_log = app.write_log

        def spy_write_log(msg: str, level: int = logging.INFO) -> None:
            captured.append((msg, level))
            original_write_log(msg, level)

        app.write_log = spy_write_log  # type: ignore[method-assign]

        async with app.run_test() as pilot:
            await pilot.pause()
            for _ in range(20):
                await pilot.pause(0.05)
                if app.workers and all(
                    w.state.name in ("SUCCESS", "ERROR") for w in app.workers
                ):
                    break
            await pilot.pause()
            await pilot.press("q")

    asyncio.run(drive())

    msgs = [m for m, _ in captured]
    assert any("🔧 tool a" in m for m in msgs), f"tool line missing — got: {msgs}"
    assert any("💬 text b" in m for m in msgs), f"text line missing — got: {msgs}"
    assert any("✏️ edit c" in m for m in msgs), f"edit line missing — got: {msgs}"

    tool_idx = next(i for i, m in enumerate(msgs) if "🔧 tool a" in m)
    text_idx = next(i for i, m in enumerate(msgs) if "💬 text b" in m)
    edit_idx = next(i for i, m in enumerate(msgs) if "✏️ edit c" in m)
    assert tool_idx < text_idx < edit_idx, "lines not in emission order"


def test_no_double_logging(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    import pytest

    pytest.importorskip("textual")

    from splinter.memory.session import Session
    from splinter.tui import RunApp

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_us003_dedup")
    session.update_index("# us003\n")
    session.set_status("running", strategy="raphael", tasks=1, stage="run")

    captured: list[str] = []

    def fake_pipeline(**kwargs: object) -> int:
        logging.getLogger("splinter.live").info("  🔧 unique_marker_xyz")
        return 0

    monkeypatch.setattr("splinter.pipeline.run_pipeline", fake_pipeline)

    async def drive() -> None:
        app = RunApp(session, {})
        original_write_log = app.write_log

        def spy_write_log(msg: str, level: int = logging.INFO) -> None:
            captured.append(msg)
            original_write_log(msg, level)

        app.write_log = spy_write_log  # type: ignore[method-assign]

        async with app.run_test() as pilot:
            await pilot.pause()
            for _ in range(20):
                await pilot.pause(0.05)
                if app.workers and all(
                    w.state.name in ("SUCCESS", "ERROR") for w in app.workers
                ):
                    break
            await pilot.pause()
            await pilot.press("q")

    asyncio.run(drive())

    count = sum(1 for m in captured if "unique_marker_xyz" in m)
    assert count == 1, f"expected 1 delivery, got {count} — double-logging guard failed"


def test_overview_renders_at_run_start(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    import pytest

    pytest.importorskip("textual")

    from splinter.analyze import render_overview
    from splinter.memory.session import Session
    from splinter.tui import RunApp

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_us003_overview")
    session.update_index("# us003 overview\n")
    session.set_status("running", strategy="raphael", tasks=1, stage="run")

    refresh_called: list[bool] = []
    orig_refresh = RunApp._refresh

    def patched_refresh(self: RunApp) -> None:
        refresh_called.append(True)
        orig_refresh(self)

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
                if app.workers and all(
                    w.state.name in ("SUCCESS", "ERROR") for w in app.workers
                ):
                    break
            await pilot.pause()
            await pilot.press("q")

    asyncio.run(drive())

    assert refresh_called, "_refresh never called during run"
    overview = render_overview(session, "running")
    assert overview.strip(), "render_overview returned empty for running session"


def test_propagation_restored_after_unmount(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    import pytest

    pytest.importorskip("textual")

    from splinter.memory.session import Session
    from splinter.tui import RunApp

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_us003_prop")
    session.update_index("# us003\n")
    session.set_status("running", strategy="raphael", tasks=1, stage="run")

    splog = logging.getLogger("splinter")
    original_propagate = splog.propagate

    def fake_pipeline(**kwargs: object) -> int:
        return 0

    monkeypatch.setattr("splinter.pipeline.run_pipeline", fake_pipeline)

    async def drive() -> None:
        app = RunApp(session, {})
        async with app.run_test() as pilot:
            await pilot.pause()
            for _ in range(20):
                await pilot.pause(0.05)
                if app.workers and all(
                    w.state.name in ("SUCCESS", "ERROR") for w in app.workers
                ):
                    break
            await pilot.pause()
            await pilot.press("q")

    asyncio.run(drive())

    assert splog.propagate == original_propagate, (
        f"propagate not restored: expected {original_propagate}, got {splog.propagate}"
    )
    assert logging.getLogger("splinter.live").level == logging.NOTSET
