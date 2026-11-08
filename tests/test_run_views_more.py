"""Tests for the `spend`, `tier` and `diff` views of the run screen."""

from __future__ import annotations

import asyncio
import io
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pytest


def _session(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> Any:
    from tests.fixtures_run_view import build_session

    return build_session(tmp_path, monkeypatch)


def _view(session: Any) -> Any:
    from splinter.run_view import build_run_view
    from tests.fixtures_run_view import FIXTURE_LADDER

    return build_run_view(session, "running", datetime.now(timezone.utc), FIXTURE_LADDER)


def _context(session: Any, view: Any, **kwargs: Any) -> Any:
    from splinter.run_view_files import FileCache
    from splinter.tui_run_views import ViewContext

    return ViewContext(
        view=view,
        story=None,
        opened=frozenset(),
        files=FileCache(),
        session=session,
        multi=True,
        **kwargs,
    )


def _render(renderable: Any) -> str:
    from rich.console import Console

    console = Console(width=200, record=True, file=io.StringIO())
    console.print(renderable)
    return console.export_text()


def _step_values(text: str) -> list[float]:
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if "by step" in line)
    values: list[float] = []
    for line in lines[start + 1 :]:
        if "total" in line or "─" in line:
            break
        match = re.search(r"\$([0-9]+\.[0-9]+)", line)
        if match:
            values.append(float(match.group(1)))
    return values


def test_spend_rows_sum_to_the_total(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
    from splinter.tui_run_views_more import MoreState, set_more_state, spend_view

    session = _session(tmp_path, monkeypatch)
    view = _view(session)
    set_more_state(MoreState(spend_by="cost"))
    text = str(spend_view(_context(session, view)))
    values = _step_values(text)
    assert values, text
    assert round(sum(values), 4) == round(view.total_cost, 4)
    assert "PRD + planner" in text
    for row in view.spend:
        assert row.label in text


def test_spend_toggle_shows_by_time_in_bold(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from splinter.tui_run_views_more import MoreState, set_more_state, spend_view

    session = _session(tmp_path, monkeypatch)
    view = _view(session)
    set_more_state(MoreState(spend_by="time"))
    text = str(spend_view(_context(session, view)))
    assert "[bold]by time[/bold]" in text
    assert "[@click=app.set_spend_by('cost')]" in text


def test_tier_view_lists_the_pool_and_the_escalated_stories(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from splinter.tui_run_views_more import MoreState, set_more_state, tier_view
    from tests.fixtures_run_view import FIXTURE_LADDER

    session = _session(tmp_path, monkeypatch)
    view = _view(session)
    set_more_state(MoreState(tier="T1"))
    text = str(tier_view(_context(session, view)))
    tier = next(t for t in FIXTURE_LADDER.tiers if t.level == 1)
    for model in tier.models:
        assert model in text
    assert "closed" in text
    assert "US-024" in text
    assert "US-009" in text


def test_diff_renders_the_stat_and_the_patch(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from splinter.tui_run_views_more import MoreState, diff_view, set_more_state
    from splinter.vcs.diff import StoryDiff

    session = _session(tmp_path, monkeypatch)
    view = _view(session)
    set_more_state(
        MoreState(
            diff=StoryDiff(
                stat=" src/app.py | 2 +-",
                patch="diff --git a/src/app.py b/src/app.py\n+added line\n",
            )
        )
    )
    text = _render(diff_view(_context(session, view)))
    assert "src/app.py" in text
    assert "added line" in text


def test_app_spend_story_keeps_the_spend_view(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    import pytest

    pytest.importorskip("textual")

    from splinter.tui import RunApp

    session = _session(tmp_path, monkeypatch)
    monkeypatch.setattr("splinter.pipeline.run_pipeline", lambda **kwargs: 0)

    async def drive() -> None:
        app = RunApp(session, {})
        async with app.run_test() as pilot:
            await pilot.pause()
            await app.run_action("open_spend('cost')")
            await pilot.pause()
            assert app.view == "spend"
            await app.run_action("spend_story(9)")
            await pilot.pause()
            assert app.selected_story == 9
            assert app.view == "spend"
            await pilot.press("q")

    asyncio.run(drive())


def test_diff_worker_uses_the_fake_runner_and_refreshes_on_demand(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    import pytest

    pytest.importorskip("textual")

    from splinter.tui import RunApp

    session = _session(tmp_path, monkeypatch)
    monkeypatch.setattr("splinter.pipeline.run_pipeline", lambda **kwargs: 0)

    calls: list[list[str]] = []

    def fake_runner(argv: list[str], cwd: Path) -> str:
        calls.append(list(argv))
        if "--stat" in argv:
            return " src/app.py | 2 +-\n"
        return "diff --git a/src/app.py b/src/app.py\n+added line\n"

    async def drive() -> None:
        app = RunApp(session, {})
        app._diff_runner = fake_runner
        async with app.run_test() as pilot:
            await pilot.pause()
            await app.run_action("open_diff")
            for _ in range(20):
                await pilot.pause(0.05)
                if app._diff is not None:
                    break
            assert app._diff is not None
            assert app._diff.error == ""
            assert len([c for c in calls if "--stat" in c]) == 1
            await app.run_action("refresh_diff")
            for _ in range(20):
                await pilot.pause(0.05)
                if len([c for c in calls if "--stat" in c]) > 1:
                    break
            assert len([c for c in calls if "--stat" in c]) == 2
            for _ in range(10):
                app._refresh()
            assert len([c for c in calls if "--stat" in c]) == 2
            await pilot.press("q")

    asyncio.run(drive())


def test_diff_tab_loads_and_a_new_story_reloads_it(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from splinter.tui import RunApp

    session = _session(tmp_path, monkeypatch)
    monkeypatch.setattr("splinter.pipeline.run_pipeline", lambda **kwargs: 0)
    calls: list[list[str]] = []

    def fake_runner(argv: list[str], cwd: Path) -> str:
        calls.append(list(argv))
        return ""

    def stats() -> int:
        return len([c for c in calls if "--stat" in c])

    async def drive() -> None:
        app = RunApp(session, {})
        app._diff_runner = fake_runner
        async with app.run_test() as pilot:
            await pilot.pause()
            await app.run_action("set_view('diff')")
            for _ in range(20):
                await pilot.pause(0.05)
                if app._diff is not None:
                    break
            assert stats() == 1 and app._diff is not None
            app.select_story(3)
            for _ in range(20):
                await pilot.pause(0.05)
                if stats() == 2 and app._diff is not None:
                    break
            assert stats() == 2
            await pilot.press("q")

    asyncio.run(drive())
