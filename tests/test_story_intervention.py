from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from splinter import procreg
from splinter.agents.runner import RunResult, Task
from splinter.memory.knowledge import KnowledgeStore
from splinter.memory.session import Session
from splinter.obs.trace import Trace
from splinter.strategies import direct, intervention
from splinter.strategies.direct import DirectStrategy
from splinter.strategies.intervention import ORCHESTRATOR, REROUTE, TIER_UP, apply_intervention
from tests.fixtures_run_view import FIXTURE_LADDER, build_session


class _Proc:
    pid = 4242
    returncode = 0

    def __init__(self, during: Any = None) -> None:
        self.during = during

    def communicate(self, timeout: int | None = None) -> tuple[str, str]:
        if self.during:
            self.during()
        return "out", ""


@pytest.fixture
def killed(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    found: list[int] = []
    monkeypatch.setattr(procreg, "_kill", lambda proc: found.append(proc.pid))
    return found


def test_killing_a_story_stops_only_its_own_engine(
    monkeypatch: pytest.MonkeyPatch, killed: list[int]
) -> None:
    monkeypatch.setattr(
        procreg.subprocess, "Popen", lambda *_a, **_k: _Proc(lambda: procreg.kill_story(7))
    )
    with procreg.story_scope(7):
        procreg.run_subprocess(["engine"], timeout=5)
    assert killed == [4242]
    assert procreg.story_interrupted(7) and not procreg.story_interrupted(8)
    procreg.clear_story_interrupt(7)


def test_another_story_is_not_touched(
    monkeypatch: pytest.MonkeyPatch, killed: list[int]
) -> None:
    monkeypatch.setattr(
        procreg.subprocess, "Popen", lambda *_a, **_k: _Proc(lambda: procreg.kill_story(8))
    )
    with procreg.story_scope(7):
        procreg.run_subprocess(["engine"], timeout=5)
    assert killed == []
    procreg.clear_story_interrupt(8)


def test_an_interrupted_story_starts_no_new_engine(
    monkeypatch: pytest.MonkeyPatch, killed: list[int]
) -> None:
    procreg.kill_story(9)
    with procreg.story_scope(9), pytest.raises(procreg.StoryInterrupted):
        procreg.run_subprocess(["engine"], timeout=5)
    procreg.clear_story_interrupt(9)


def test_a_story_control_is_taken_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ctl")
    session.queue_story_control(3, TIER_UP)
    assert session.take_story_control(4) is None
    assert session.take_story_control(3) == TIER_UP
    assert session.take_story_control(3) is None


def test_orchestrator_jumps_to_the_premium_tier_with_a_note() -> None:
    done = apply_intervention(ORCHESTRATOR, 1, FIXTURE_LADDER, premium_tier=4, max_tier=5)
    assert done.tier == 4 and "orchestrator" in done.note


def test_orchestrator_never_lowers_the_tier() -> None:
    assert apply_intervention(ORCHESTRATOR, 5, FIXTURE_LADDER, premium_tier=4, max_tier=5).tier == 5


def test_raise_tier_stops_at_the_top() -> None:
    assert apply_intervention(TIER_UP, 1, FIXTURE_LADDER, premium_tier=4, max_tier=5).tier == 2
    assert apply_intervention(TIER_UP, 5, FIXTURE_LADDER, premium_tier=4, max_tier=5).tier == 5


def test_route_trips_the_current_model_and_keeps_the_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    failed: list[str] = []
    monkeypatch.setattr(intervention, "report_model_failure", failed.append)
    monkeypatch.setattr(intervention, "resolve_model", lambda tier, ladder: ("sonnet", "high"))
    done = apply_intervention(REROUTE, 1, FIXTURE_LADDER, premium_tier=4, max_tier=5)
    assert done.tier == 1 and failed == ["sonnet"]


def test_route_with_no_sibling_goes_up_a_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    ladder = SimpleNamespace(tiers=[SimpleNamespace(level=1, models=["only"])])
    done = apply_intervention(REROUTE, 1, ladder, premium_tier=4, max_tier=5)  # type: ignore[arg-type]
    assert done.tier == 2


class _Chain:
    def __init__(self, seen: list[int], session: Session) -> None:
        self.seen, self.session = seen, session

    def handle(self, ctx: Any) -> None:
        self.seen.append(ctx.tier)
        if len(self.seen) == 1:
            self.session.queue_story_control(1, TIER_UP)
            raise procreg.StoryInterrupted(1)
        ctx.run_result = MagicMock(spec=RunResult, text="ok", model="m", cost=0.0)
        ctx.verdict = MagicMock(decision=direct.Decision.PASS, passed=True)


def test_intervention_restarts_the_story_higher_without_burning_an_iteration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    session = Session("loop")
    session.write("knowledge/plan-1.md", "# Plan\n\n1. Do it\n")
    seen: list[int] = []
    monkeypatch.setattr(direct, "build_chain", lambda *_a, **_k: _Chain(seen, session))
    monkeypatch.setattr(direct, "_story_done", lambda *_a, **_k: True, raising=False)
    action = SimpleNamespace(stop=True, ask_user=False, next_tier=1)
    monkeypatch.setattr(direct.Evaluator, "next_action", lambda *_a, **_k: action)
    task = Task(description="US-001: One", acceptance="ok", id="US-001")
    DirectStrategy()._run_task_loop(
        task,
        session,
        FIXTURE_LADDER,
        Trace(session=session),
        KnowledgeStore(session),
        effort=None,
        budget=None,
        max_iterations=1,
        localization="",
        start_tier_override=0,
    )
    assert seen == [0, 1]
    assert "Intervention" in session.read("loop.md")


def _run_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Session:
    session = build_session(tmp_path, monkeypatch, stories=3, live=1, failed=0, asked=0)
    session.write("loop.md", "# Task 1/3: US-001: One\n\n## Iteration 1\n")
    session.write_run_live(1, {"provider": "claude", "model": "opus", "task_no": 1})
    session.write_run_live(2, {"provider": "claude", "model": "opus", "task_no": 2})
    session.set_status("running", parallel=True, tasks=3, task_index=0)
    return session


def _app(session: Session, monkeypatch: pytest.MonkeyPatch) -> Any:
    from splinter.tui import RunApp

    monkeypatch.setattr(RunApp, "_work", lambda self: None)
    monkeypatch.setattr(RunApp, "_run_state_now", lambda self: "RUNNING", raising=False)
    return RunApp(session, {})


def test_menu_confirms_then_stops_only_that_story(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from splinter.tui_run_story_menu import ConfirmIntervention

    session = _run_session(tmp_path, monkeypatch)
    app = _app(session, monkeypatch)
    stopped: list[int] = []
    monkeypatch.setattr(procreg, "kill_story", lambda n: stopped.append(n) or 1)
    restarts: list[bool] = []
    monkeypatch.setattr(type(app), "_do_pause_kill", lambda self: restarts.append(True))

    async def drive() -> None:
        async with app.run_test(size=(140, 44)) as pilot:
            await pilot.pause()
            view = app._last_view
            monkeypatch.setattr(
                app, "_story_row", lambda n: next(s for s in view.stories if s.number == n)
            )
            app._live_now = lambda n: True
            await app.run_action("story_do(1,'orchestrator')")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmIntervention)
            await pilot.press("escape")
            await pilot.pause()
            assert stopped == [] and session.take_story_control(1) is None
            await app.run_action("story_do(1,'orchestrator')")
            await pilot.pause()
            await pilot.press("y")
            await pilot.pause()
            assert stopped == [1] and restarts == []
            assert session.take_story_control(1) == ORCHESTRATOR

    asyncio.run(drive())


def test_retry_of_a_running_story_never_restarts_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _run_session(tmp_path, monkeypatch)
    app = _app(session, monkeypatch)
    restarts: list[bool] = []
    monkeypatch.setattr(type(app), "_do_pause_kill", lambda self: restarts.append(True))

    async def drive() -> None:
        async with app.run_test(size=(140, 44)) as pilot:
            await pilot.pause()
            app._live_now = lambda n: True
            app._retry_story(1, "use another approach")
            await pilot.pause()

    asyncio.run(drive())
    assert restarts == []
    assert "use another approach" in session.pop_live_commands(task_no=1)


@pytest.mark.parametrize("stopper", ["_do_pause_kill", "_do_kill_exit"])
def test_killing_the_run_also_stops_new_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stopper: str
) -> None:
    from splinter.tui import RunApp

    session = _run_session(tmp_path, monkeypatch)
    app = RunApp(session, {})
    monkeypatch.setattr(procreg, "terminate_all", lambda: None)
    monkeypatch.setattr(app, "write_log", lambda *_a, **_k: None)
    monkeypatch.setattr(app, "exit", lambda *_a, **_k: None)
    try:
        getattr(app, stopper)()
        assert procreg.stop_requested()
    finally:
        procreg.clear_stop()

