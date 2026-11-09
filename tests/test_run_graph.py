from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from textual.widgets import Static

from splinter.run_graph import StoryBrief, build_graph, plan_excerpt, prd_stories, run_stories
from splinter.run_view import IterationRow, StoryRow, build_run_view
from splinter.tui_run_graph import detail_markup
from splinter.tui_run_graph_map import map_markup
from splinter.tui_run_palette import AMBER, LIME
from splinter.tui_run_rail_parts import groups_markup
from tests.fixtures_run_view import FIXTURE_LADDER, build_session

if TYPE_CHECKING:
    from splinter.memory.session import Session
    from splinter.tui import RunApp

_PRD = """# PRD

### US-001: Base verde
Do it.

### US-002: Segunda
Depends on: US-001

### US-003: Terceira
Depends on: US-002

### US-004: Quarta
Depends on: US-001
"""


def _row(number: int, state: str) -> StoryRow:
    return StoryRow(number, f"T{number}", "", state, "T1", "m", 2, 0.5, 1.0, 0)  # type: ignore[arg-type]


def test_stories_follow_the_order_the_run_uses_and_group_in_waves() -> None:
    stories = prd_stories(_PRD, True)
    assert [(s.number, s.id, s.wave) for s in stories] == [
        (1, "US-001", 1),
        (2, "US-002", 2),
        (3, "US-004", 2),
        (4, "US-003", 3),
    ]
    assert stories[1].title == "US-002: Segunda"
    assert stories[3].deps == ("US-002",)


def test_only_strategies_that_sort_use_dependency_order() -> None:
    assert [s.id for s in run_stories(_PRD, "cascade")][2] == "US-004"
    assert [s.id for s in run_stories(_PRD, "direct")][2] == "US-003"


def test_graph_says_what_is_done_running_next_and_later() -> None:
    rows = (_row(1, "pass"), _row(2, "live"), _row(3, "queued"), _row(4, "queued"))
    graph = build_graph(prd_stories(_PRD, True), rows)
    assert [n.state for n in graph.nodes] == ["done", "now", "ready", "blocked"]
    assert [[n.ident for n in wave] for wave in graph.waves] == [
        ["US-001"],
        ["US-002", "US-004"],
        ["US-003"],
    ]
    assert graph.nodes[0].dependents == ("US-002", "US-004")
    assert graph.nodes[3].title == "Terceira"


def test_failed_story_needs_the_user_and_holds_its_dependents() -> None:
    rows = (_row(1, "fail"), _row(2, "queued"), _row(3, "queued"), _row(4, "queued"))
    graph = build_graph(prd_stories(_PRD, True), rows)
    assert [n.state for n in graph.nodes[:2]] == ["needs", "blocked"]


def test_graph_without_a_prd_still_lists_every_story() -> None:
    graph = build_graph((), (_row(1, "live"), _row(2, "queued")))
    assert [n.number for n in graph.nodes] == [1, 2]
    assert len(graph.waves) == 1


def _session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Session:
    session = build_session(tmp_path, monkeypatch, stories=4, live=1, failed=0, asked=0)
    session.write("prd.md", _PRD)
    session.write("loop.md", "# Task 1/4: US-001: Base verde\n\n## Iteration 1\n")
    session.set_status("running", strategy="cascade", tasks=4, task_index=0)
    return session


def test_queued_stories_show_their_real_id_and_title(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _session(tmp_path, monkeypatch)
    view = build_run_view(session, "RUNNING", datetime.now(timezone.utc), FIXTURE_LADDER)
    assert view.stories[2].title == "US-004: Quarta"
    markup = groups_markup(view, None, 60, False, False)
    assert "US-004" in markup and "T3" not in markup
    assert "app.story_menu(3)" in markup


def _app(session: Session, monkeypatch: pytest.MonkeyPatch) -> RunApp:
    from splinter.tui import RunApp

    monkeypatch.setattr(RunApp, "_work", lambda self: None)
    return RunApp(session, {})


def test_clicking_a_story_offers_actions_and_plan_opens_the_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from splinter.tui_run_story_menu import StoryActions

    app = _app(_session(tmp_path, monkeypatch), monkeypatch)

    async def drive() -> None:
        async with app.run_test(size=(140, 44)) as pilot:
            await pilot.pause()
            await app.run_action("story_menu(3)")
            await pilot.pause()
            assert isinstance(app.screen, StoryActions)
            await pilot.press("p")
            await pilot.pause()
            assert not isinstance(app.screen, StoryActions)
            assert app.selected_story == 3 and app.view == "plan"

    asyncio.run(drive())


def test_map_routes_lines_around_nodes_and_lights_the_selected_edges() -> None:
    rows = (_row(1, "pass"), _row(2, "live"), _row(3, "queued"), _row(4, "queued"))
    graph = build_graph(prd_stories(_PRD, True), rows)
    markup = map_markup(graph, 2)
    assert "screen.pick(4)" in markup and "wave 3" in markup
    plain = re.sub(r"\[[^\]]*\]", "", markup)
    assert "─" in plain and "┐" in plain or "┬" in plain
    assert f"[{LIME}]" in markup and f"[{AMBER}]" in markup


def test_side_pane_shows_plan_and_iterations_at_once() -> None:
    rows = (_row(1, "live"), _row(2, "queued"), _row(3, "queued"), _row(4, "queued"))
    graph = build_graph(prd_stories(_PRD, True), rows)
    iteration = IterationRow(1, "T0", "m", "low", "RETRY", 0.5, 125.0, None, "", "")
    plan = plan_excerpt("# Plan\n\n1. Read the module\n\n2. Add the test\n")
    assert plan == ("1. Read the module", "2. Add the test")
    text = detail_markup(graph.nodes[0], StoryBrief(plan, (iteration,)), graph)
    assert "1. Read the module" in text and "RETRY" in text and "2m05s" in text
    assert "screen.act('logs')" in text and "US-002" in text


def test_graph_screen_navigates_opens_actions_and_goes_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from splinter.tui_run_graph import RunGraphScreen
    from splinter.tui_run_story_menu import StoryActions

    session = _session(tmp_path, monkeypatch)
    session.write("knowledge/plan-3.md", "# Plan\n\nFirst step of the plan\n")
    app = _app(session, monkeypatch)

    async def drive() -> None:
        async with app.run_test(size=(160, 44)) as pilot:
            await pilot.pause()
            await pilot.press("G")
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, RunGraphScreen)
            assert screen.selected == 1
            await pilot.press("right", "down")
            assert screen.selected == 3
            detail = str(screen.query_one("#graph-detail", Static).content)
            assert "First step of the plan" in detail and "US-001" in detail
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, StoryActions)
            await pilot.press("escape")
            await pilot.pause()
            assert app.screen is screen and screen.selected == 3
            await app.run_action("screen.pick(4)")
            await pilot.pause()
            assert screen.selected == 4
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, RunGraphScreen)
            await pilot.press("G")
            await pilot.pause()
            await pilot.press("right", "l")
            await pilot.pause()
            await pilot.pause()
            assert not isinstance(app.screen, RunGraphScreen)
            assert app.view == "logs" and app.selected_story == 2

    asyncio.run(drive())
