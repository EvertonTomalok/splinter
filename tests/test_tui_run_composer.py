from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from splinter.memory.session import Session


def test_all_agents_message_changes_from_queued_to_delivered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")
    from splinter.tui import RunApp
    from splinter.tui_run_composer import Composer

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    monkeypatch.setattr(RunApp, "_work", lambda self: None)
    session = Session("ses_composer")
    session.set_status("running", strategy="agents", tasks=24, stage="run")
    app = RunApp(session, {})
    app.selected_story = 24

    async def drive() -> None:
        async with app.run_test(size=(120, 36)) as pilot:
            await pilot.pause()
            app.action_composer_target("all")
            composer = app.query_one(Composer)
            input_widget = composer.query_one("#user-cmd-input")
            input_widget.value = "do not edit migrations"
            await pilot.click("#user-cmd-input")
            await pilot.press("enter")
            assert "sending" in str(app.query_one("#composer-msgs").render())
            assert "do not edit migrations" in session.pop_live_commands(24)
            app._refresh()
            assert "✓ with the orchestrator" in str(app.query_one("#composer-msgs").render())

    asyncio.run(drive())
