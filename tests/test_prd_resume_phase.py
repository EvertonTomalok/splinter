from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from splinter import tui, tui_run_setup
from splinter.memory.session import Session
from splinter.tui import PrdSessionApp


def test_run_that_never_started_resumes_at_run_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    monkeypatch.setattr(PrdSessionApp, "_spawn", lambda *args, **kw: None)
    session = Session("crashed-before-run")
    session.write("prd.md", "---\nstrategy: cascade\n---\n\n### US-001: One\nDo it.\n")
    session.set_status(
        "refining", source="prd", phase="run", claude_session="sid", strategy="cascade"
    )
    app = PrdSessionApp(session, {"resume": True, "prd_path": str(session.dir / "prd.md")})

    async def drive() -> None:
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            assert app.phase == "review"
            assert app.strategy == "cascade"
            app.query_one("#run-setup")

    asyncio.run(drive())
    assert session.read_status()["phase"] == "review"


def test_resumed_prd_that_never_ran_still_asks_about_parallel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    monkeypatch.setattr(tui_run_setup, "worktree_supported", lambda: True)
    session = Session("never-ran")
    prd = "### US-001: One\nDo it.\n\n### US-002: Two\nDo it.\n"
    session.write("prd.md", prd)
    session.set_status("refining", source="prd", phase="review", strategy="cascade")
    captured: dict[str, object] = {}

    class _App:
        def __init__(self, _session: Session, run_kwargs: dict[str, object]) -> None:
            captured.update(run_kwargs)

        def run(self) -> None:
            return None

    monkeypatch.setattr(tui, "PrdSessionApp", _App)
    tui._resume_prd(session, session.read_status())

    assert captured["parallel"] is None
    assert tui_run_setup.should_ask_parallel(prd, "cascade", None)
