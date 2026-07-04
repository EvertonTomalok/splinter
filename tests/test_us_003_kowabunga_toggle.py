from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


def _fake_pipeline(**kwargs: object) -> int:
    return 0


def test_ctrl_k_toggles_kowabunga_and_indicator(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    import pytest

    pytest.importorskip("textual")

    from textual.widgets import Static

    from splinter.enums import RunnerMode
    from splinter.memory.session import Session
    from splinter.tui import RunApp

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_us003_kowa_toggle")
    session.update_index("# us003 kowabunga\n")
    session.set_status("running", strategy="raphael", tasks=1, stage="run", pid=os.getpid())
    monkeypatch.setattr("splinter.pipeline.run_pipeline", _fake_pipeline)

    async def drive() -> None:
        app = RunApp(session, {})
        async with app.run_test() as pilot:
            await pilot.pause()

            assert app._kowabunga == RunnerMode.KOWABUNGA_OFF
            ind = app.query_one("#kowabunga-ind", Static)
            assert "KOWABUNGA: OFF" in str(ind.content)

            await pilot.press("ctrl+k")
            await pilot.pause()

            assert app._kowabunga == RunnerMode.KOWABUNGA_ON
            assert "KOWABUNGA: ON" in str(ind.content)

            await pilot.press("ctrl+k")
            await pilot.pause()

            assert app._kowabunga == RunnerMode.KOWABUNGA_OFF
            assert "KOWABUNGA: OFF" in str(ind.content)

            await pilot.press("q")

    asyncio.run(drive())


def test_kowabunga_toggle_persists_to_session(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    import pytest

    pytest.importorskip("textual")

    from splinter.enums import RunnerMode
    from splinter.memory.session import Session
    from splinter.tui import RunApp

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_us003_kowa_persist")
    session.update_index("# us003 kowabunga\n")
    session.set_status("running", strategy="raphael", tasks=1, stage="run", pid=os.getpid())
    monkeypatch.setattr("splinter.pipeline.run_pipeline", _fake_pipeline)

    async def drive() -> None:
        app = RunApp(session, {})
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("ctrl+k")
            await pilot.pause()
            await pilot.press("q")

    asyncio.run(drive())

    assert session.read_kowabunga() == RunnerMode.KOWABUNGA_ON


def test_kowabunga_indicator_reflects_persisted_state_on_mount(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    import pytest

    pytest.importorskip("textual")

    from textual.widgets import Static

    from splinter.enums import RunnerMode
    from splinter.memory.session import Session
    from splinter.tui import RunApp

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_us003_kowa_premount")
    session.update_index("# us003 kowabunga\n")
    session.set_status("running", strategy="raphael", tasks=1, stage="run", pid=os.getpid())
    session.set_kowabunga(RunnerMode.KOWABUNGA_ON)
    monkeypatch.setattr("splinter.pipeline.run_pipeline", _fake_pipeline)

    async def drive() -> None:
        app = RunApp(session, {})
        async with app.run_test() as pilot:
            await pilot.pause()

            assert app._kowabunga == RunnerMode.KOWABUNGA_ON
            ind = app.query_one("#kowabunga-ind", Static)
            assert "KOWABUNGA: ON" in str(ind.content)

            await pilot.press("q")

    asyncio.run(drive())


def test_paused_run_modal_cowabunga_actions_untouched() -> None:
    import pytest

    pytest.importorskip("textual")

    from splinter.tui import _AskUserModal

    modal = _AskUserModal(reason="needs input")
    assert hasattr(modal, "action_cowabunga")
    assert hasattr(modal, "action_jump_premium")
    bindings = {b[0]: b[1] for b in _AskUserModal.BINDINGS}
    assert bindings.get("c") == "action_cowabunga"
    assert bindings.get("p") == "jump_premium"
