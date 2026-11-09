from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

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
