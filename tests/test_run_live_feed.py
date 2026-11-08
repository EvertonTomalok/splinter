"""The live feed of the activity view, the action sentence, ``/ find`` and the rail fits.

No model runs: the pipeline worker is patched out and the streamed lines are written
to the log the way the providers write them.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from rich.console import Console
from rich.text import Text

from splinter.run_view import SpendRow
from splinter.tui_run_live import LiveAction, live_action, live_phrase, live_rows
from splinter.tui_run_palette import Ink
from splinter.tui_run_rail_parts import chip, split_markup
from splinter.tui_run_text import cell_len


def test_live_action_reads_tool_calls_and_thoughts() -> None:
    tool = live_action("🔧 Edit /x/.splinter/worktrees/s1/src/a.py", "14:55:18", 24, 3, 1.0)
    assert tool == LiveAction("14:55:18", "edit", "src/a.py", 24, 3, 1.0)
    thought = live_action("💬 round the wait up", "14:55:09", 24, 3, 1.0)
    assert thought is not None and thought.kind == "think"
    assert live_action("gate PASS", "14:55:20", 24, 3, 1.0) is None


def test_live_rows_light_the_newest_and_fold_the_rest() -> None:
    actions = [LiveAction("14:55:00", "read", f"f{n}.py", 1, 1, 0.0) for n in range(15)]
    actions.append(LiveAction("14:55:30", "edit", "src/a.py", 1, 1, 10.0))
    text = live_rows(actions, 100, phase=False, now=10.2)
    lines = text.plain.splitlines()
    assert "… 4 earlier actions" in lines[0]
    assert lines[-1].rstrip().endswith("src/a.py ▌")
    assert any("bold" in str(span.style) for span in text.spans)
    still = live_rows(actions, 100, phase=True, now=20.0)
    assert "▌" not in still.plain.splitlines()[-1]


def test_live_phrase_says_what_the_runner_does() -> None:
    edit = LiveAction("t", "edit", "src/api/middleware.py", 1, 1, 0.0)
    assert live_phrase("sonnet@high", edit) == "sonnet@high is editing src/api/middleware.py"
    think = LiveAction("t", "think", "Round the wait up", 1, 1, 0.0)
    assert live_phrase("sonnet@high", think) == "sonnet@high · round the wait up"


def test_split_line_fits_the_rail() -> None:
    spend = (
        SpendRow("run", 4.634, 0.0, ""),
        SpendRow("PRD + planner", 0.2041, 0.0, ""),
        SpendRow("jev ×6", 0.0052, 0.0, ""),
    )
    wide = Text.from_markup(split_markup(spend, 80)).plain
    assert wide == "run $4.6340 · PRD+planner $0.2041 · jev $0.0052 ×6"
    narrow = Text.from_markup(split_markup(spend, 40)).plain
    assert len(narrow) <= 40 and narrow.startswith("run $4.6340")


def test_chip_is_outlined_and_one_row() -> None:
    on = chip("all agents", True)
    assert Text.from_markup(on).plain == "▕ all agents ▏"
    assert cell_len(chip("x", False)) == 5


def _text(content: object) -> str:
    console = Console(width=240, color_system=None)
    with console.capture() as capture:
        console.print(content)
    return capture.get()


@pytest.fixture
def run_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    from splinter.tui import RunApp
    from tests.fixtures_run_view import FIXTURE_LADDER, build_session

    session = build_session(tmp_path, monkeypatch, jev=True)
    session.set_status("running", pid=os.getpid())
    monkeypatch.setattr("splinter.models.roster.load_ladder", lambda: FIXTURE_LADDER)
    monkeypatch.setattr(RunApp, "_run_pipeline_worker", lambda self, **kw: None)
    return RunApp(session, {"prd_path": "p.md"})


def test_streamed_actions_land_under_the_live_iteration(run_app) -> None:  # type: ignore[no-untyped-def]
    async def drive() -> None:
        async with run_app.run_test(size=(150, 46)) as pilot:
            await pilot.pause()
            run_app.write_log("🔧 Read src/api/middleware.py")
            run_app.write_log("🔧 Edit src/api/headers.py")
            await pilot.pause()
            body = _text(run_app.query_one("#view-static", Ink).content)
            assert "read  src/api/middleware.py" in body
            assert "edit  src/api/headers.py" in body
            sentence = _text(run_app.query_one("#sentence", Ink).content)
            assert "is editing src/api/headers.py" in sentence

    asyncio.run(drive())


def test_slash_finds_a_story_and_enter_opens_it(run_app) -> None:  # type: ignore[no-untyped-def]
    async def drive() -> None:
        async with run_app.run_test(size=(150, 46)) as pilot:
            await pilot.pause()
            await pilot.press("slash")
            await pilot.press("U", "S", "-", "0", "3", "1")
            await pilot.pause()
            overview = _text(run_app.query_one("#overview", Ink).content)
            assert "matches" in overview and "US-031" in overview
            await pilot.press("enter")
            await pilot.pause()
            assert run_app._story_number(run_app._last_view) == 31

    asyncio.run(drive())
