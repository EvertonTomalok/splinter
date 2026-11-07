"""P7: a PRD model error keeps the PRD screen open and offers Retry."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from textual.widgets import RichLog, TextArea

from splinter import prd_session
from splinter.memory.session import Session
from splinter.prd_session import Turn

SIZE = (160, 44)


def _convo(app: Any) -> str:
    return "\n".join(str(line.text) for line in app.query_one("#convo", RichLog).lines)


def _buttons(app: Any) -> list[str]:
    return [b.id or "" for b in app.query("#actions Button")]


async def _settle(app: Any, pilot: Any) -> None:
    for _ in range(2):
        await app.workers.wait_for_complete()
        await pilot.pause()


def test_prd_model_error_shows_retry_and_stays_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from splinter.tui import PrdSessionApp

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path / "home"))
    prd = tmp_path / "prd.md"
    prd.write_text("# Feature\n\n### US-001: Hello\nSay hello.\n")
    attempts: list[int] = []

    def flaky(*args: Any, **kw: Any) -> Turn:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("429 overloaded")
        return Turn(text="1. Which greeting?\n A) hi", session_id="sid-1", model="opus")

    monkeypatch.setattr(prd_session, "open_questions", flaky)
    app = PrdSessionApp(
        Session("ses_prd"),
        {"prd_path": str(prd), "strategy": None, "cowabunga": False, "no_ground": True},
    )

    async def drive() -> None:
        async with app.run_test(size=SIZE) as pilot:
            await _settle(app, pilot)
            await pilot.pause(0.1)
            assert app.is_running
            assert "429 overloaded" in _convo(app)
            assert "retry" in _buttons(app)
            assert not app.query_one("#entry", TextArea).disabled
            await pilot.click("#retry")
            await _settle(app, pilot)
            assert len(attempts) == 2
            assert "Which greeting?" in _convo(app)
            assert "retry" not in _buttons(app)
            assert app.phase == "chat"

    asyncio.run(drive())
