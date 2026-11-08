from __future__ import annotations

import asyncio
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from textual.widgets import Static, TextArea

from splinter.memory.session import Session
from splinter.tui import PrdSessionApp
from splinter.tui_plan import PlanSessionApp
from splinter.tui_session_config import edit_session_config


@pytest.mark.parametrize("app_type", [PrdSessionApp, PlanSessionApp])
def test_configure_preserves_draft_and_resets_changed_model_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, app_type: type[PrdSessionApp | PlanSessionApp]
) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    monkeypatch.setattr(app_type, "_spawn", lambda *a, **kw: None)
    app = app_type(Session("configure"), {"no_ground": True})
    configure = AsyncMock(return_value=True)
    monkeypatch.setattr("splinter.tui_session_config.edit_session_config", configure)

    async def drive() -> None:
        async with app.run_test(size=(160, 44)) as pilot:
            app._set_busy(False)
            app._model_name = "old/model"
            app.claude_session = "old-session"
            editor = app.query_one(TextArea)
            editor.load_text("Keep this unsaved draft")
            crumb = str(app.query_one("#shell-crumb", Static).content)
            assert "configure" in crumb and "app.configure" in crumb
            await pilot.press("f2")
            configure.assert_awaited_once_with(app, busy=False)
            assert editor.text == "Keep this unsaved draft"
            assert app.claude_session == ""
            assert app._model_name != "old/model"

    asyncio.run(drive())


@pytest.mark.parametrize("saved", [True, None])
def test_configure_suspends_and_returns_save_result(
    saved: bool | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = Mock(run_async=AsyncMock(return_value=saved))
    constructor = Mock(return_value=child)
    monkeypatch.setattr("splinter.tui_configure.ConfigureApp", constructor)
    parent = Mock(suspend=Mock(return_value=nullcontext()))
    assert asyncio.run(edit_session_config(parent, busy=False)) == bool(saved)
    parent.suspend.assert_called_once()
    child.run_async.assert_awaited_once()


def test_configure_cannot_interrupt_active_model(monkeypatch: pytest.MonkeyPatch) -> None:
    constructor = Mock()
    monkeypatch.setattr("splinter.tui_configure.ConfigureApp", constructor)
    parent = Mock()
    assert not asyncio.run(edit_session_config(parent, busy=True))
    constructor.assert_not_called()
    parent.suspend.assert_not_called()
