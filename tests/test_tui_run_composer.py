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


def test_orchestrator_note_is_answered_and_acted_on_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")
    from splinter.orchestrator_chat import OrchestratorAnswer
    from splinter.tui import RunApp
    from splinter.tui_run_composer import Composer

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    monkeypatch.setattr(RunApp, "_work", lambda self: None)
    monkeypatch.setattr(
        "splinter.tui.ask_orchestrator",
        lambda *args: OrchestratorAnswer("on it", ((2, "orchestrator"),)),
    )
    session = Session("ses_composer_orch")
    session.set_status("paused", strategy="cascade", tasks=4, stage="run")
    app = RunApp(session, {})

    async def drive() -> None:
        async with app.run_test(size=(120, 36)) as pilot:
            await pilot.pause()
            app.action_composer_target("orch")
            composer = app.query_one(Composer)
            composer.query_one("#user-cmd-input").value = "run US-002 yourself"
            await pilot.click("#user-cmd-input")
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            app._refresh()
            assert "✓ with the orchestrator" in str(app.query_one("#composer-msgs").render())

    asyncio.run(drive())

    assert session.take_story_control(2) == "orchestrator"
