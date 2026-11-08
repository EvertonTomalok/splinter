"""Motion, scale and toast checks for the C2 run screen."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

import pytest

from splinter.run_view import IterationRow, RunView, Stage, StoryRow, StoryState
from splinter.tui_run_motion import ToastTracker
from tests.fixtures_run_view import FIXTURE_LADDER, build_session

if TYPE_CHECKING:
    from pathlib import Path

    from pytest import MonkeyPatch
    from textual.pilot import Pilot

    from splinter.memory.session import Session
    from splinter.tui import RunApp

#: The refresh of sixty stories must stay under this, on average.
REFRESH_BOUND_S = 0.05

#: Ticks the scale test measures.
TICKS = 20

#: The screen size the scale test runs at.
SIZE = (120, 36)


def _app(session: Session, monkeypatch: MonkeyPatch) -> RunApp:
    from splinter.tui import RunApp

    monkeypatch.setattr("splinter.models.roster.load_ladder", lambda: FIXTURE_LADDER)
    monkeypatch.setattr(RunApp, "_work", lambda self: None)
    return RunApp(session, {"prd_path": "payments-v2-prd.md"})


def _drive(
    app: RunApp, body: Callable[[Pilot], Awaitable[None]], size: tuple[int, int] = SIZE
) -> None:
    async def run() -> None:
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            await body(pilot)

    asyncio.run(run())


def test_motion_off_creates_no_timer(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    pytest.importorskip("textual")
    app = _app(build_session(tmp_path, monkeypatch), monkeypatch)
    app.animation_level = "none"

    async def body(_: Pilot) -> None:
        assert app._motion._timer is None

    _drive(app, body)


def test_motion_on_creates_one_timer(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    pytest.importorskip("textual")
    app = _app(build_session(tmp_path, monkeypatch), monkeypatch)

    async def body(_: Pilot) -> None:
        assert app._motion._timer is not None

    _drive(app, body)


def test_sixty_stories_refresh_under_bound(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    pytest.importorskip("textual")
    from splinter.tui_run_focus import FocusColumn
    from splinter.tui_run_rail import RunRail

    session = build_session(
        tmp_path, monkeypatch, stories=60, live=24, failed=9, asked=17, iters_each=5
    )
    app = _app(session, monkeypatch)
    app.animation_level = "none"

    async def body(_: Pilot) -> None:
        focus = app.query_one("#focus", FocusColumn)
        rail = app.query_one("#rail", RunRail)
        app._refresh()
        focus_renders = focus.renders
        rail_renders = rail.renders
        started = time.perf_counter()
        for _ in range(TICKS):
            app._refresh()
        average = (time.perf_counter() - started) / TICKS
        assert average < REFRESH_BOUND_S, f"refresh averaged {average:.4f}s"
        assert focus.renders == focus_renders
        assert rail.renders == rail_renders

    _drive(app, body)


def test_gate_pass_toast_fires_once(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    pytest.importorskip("textual")
    from splinter.tui import RunApp

    session = build_session(tmp_path, monkeypatch)
    session.set_status("running", stage="gate")
    app = _app(session, monkeypatch)
    messages: list[str] = []
    monkeypatch.setattr(
        RunApp, "notify", lambda self, message, **kwargs: messages.append(message)
    )

    async def body(_: Pilot) -> None:
        session.set_status("running", stage="eval")
        app._refresh()
        session.set_status("running", stage="eval")
        app._refresh()

    _drive(app, body)
    assert [item for item in messages if item.startswith("Gate passed")] == [
        "Gate passed · T24"
    ]


def test_toast_tracker_covers_asking_escalation_and_completion() -> None:
    tracker = ToastTracker()
    live = _story(24, "live")
    asked = _story(17, "asked")
    assert tracker.events(None, _view([live])) == []
    assert tracker.events(_view([live]), _view([live, asked])) == ["T17 is asking you"]
    escalated = _view(
        [live],
        iterations={24: (_iteration(1, "T0"), _iteration(2, "T1", escalated_to="1"))},
    )
    grown = _view(
        [live],
        iterations={
            24: (_iteration(1, "T0"), _iteration(2, "T1", escalated_to="1"), _iteration(3, "T1"))
        },
    )
    assert tracker.events(escalated, grown) == ["Escalated T24 to T1"]
    done = _view([_story(1, "pass")], stage="done", cost=12.412)
    assert tracker.events(_view([_story(1, "pass")]), done) == [
        "Run complete · 1/1 · $12.4120"
    ]
    assert tracker.events(done, done) == []


def _story(number: int, state: StoryState) -> StoryRow:
    return StoryRow(
        number=number,
        id=f"T{number}",
        title=f"Story {number}",
        state=state,
        tier="T0",
        model="flash",
        iterations=1,
        cost=0.1,
        elapsed_s=1.0,
        wave=0,
    )


def _iteration(n: int, tier: str, escalated_to: str | None = None) -> IterationRow:
    return IterationRow(
        n=n,
        tier=tier,
        model="flash",
        effort="low",
        verdict="RETRY",
        cost=0.01,
        duration_s=1.0,
        escalated_to=escalated_to,
        run_file="runs/iter-1.md",
        review_note="",
    )


def _view(
    stories: list[StoryRow],
    stage: Stage = "run",
    cost: float = 0.0,
    iterations: dict[int, tuple[IterationRow, ...]] | None = None,
) -> RunView:
    counts: dict[StoryState, int] = {}
    for row in stories:
        counts[row.state] = counts.get(row.state, 0) + 1
    live = next((row.number for row in stories if row.state == "live"), None)
    return RunView(
        stories=tuple(stories),
        iterations_by_story=iterations or {},
        live_story=live,
        stage=stage,
        spend=(),
        total_cost=cost,
        elapsed_s=0.0,
        ladder=(),
        needs_you=(),
        now=(),
        next_up=(),
        counts=counts,
    )
