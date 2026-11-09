from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from splinter.agents.localizer import CodeAnchor
from splinter.agents.runner import Task
from splinter.memory.session import Session
from splinter.models.roster import Ladder, load_ladder
from splinter.pipeline import _localize_and_filter
from splinter.run_view import build_run_view
from splinter.strategies.cascade import CascadeStrategy
from splinter.strategies.direct import DirectStrategy
from splinter.strategies.intervention import ORCHESTRATOR, apply_intervention
from tests.fixtures_run_view import FIXTURE_LADDER, build_session


def _ladder(planner: str) -> Any:
    tiers = [
        SimpleNamespace(level=3, models=["minimax", "kimi"]),
        SimpleNamespace(level=4, models=["opus", "codex"]),
        SimpleNamespace(level=5, models=["fable"]),
    ]
    return SimpleNamespace(planner_model=planner, tiers=tiers)


def _handover(tier: int, planner: str) -> int:
    return apply_intervention(ORCHESTRATOR, tier, _ladder(planner), premium_tier=3, max_tier=5).tier


def test_the_orchestrator_finishes_on_the_tier_of_the_planner_model() -> None:
    assert _handover(1, "opus") == 4


def test_the_orchestrator_never_lowers_a_higher_tier() -> None:
    assert _handover(5, "opus") == 5


def test_a_planner_outside_the_ladder_falls_back_to_the_premium_tier() -> None:
    assert _handover(1, "sonnet") == 3


def _chain() -> list[Task]:
    first = Task(description="b", acceptance="ok", id="US-002", deps=["US-001"], target_files=["b"])
    second = Task(description="a", acceptance="ok", id="US-001", target_files=["a"])
    return [first, second]


def test_cascade_runs_dependencies_first() -> None:
    assert [task.id for task in CascadeStrategy().run_order(_chain())] == ["US-001", "US-002"]


def test_direct_keeps_the_given_order() -> None:
    assert [task.id for task in DirectStrategy().run_order(_chain())] == ["US-002", "US-001"]


def _no_filter(task: Task, ladder: Ladder, *, session: Session) -> str:
    return "ctx"


def test_localization_files_follow_the_run_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("order")
    tasks = _chain()
    anchors = [
        CodeAnchor(file="a", symbol="s", reason="r", confidence=0.9),
        CodeAnchor(file="b", symbol="s", reason="r", confidence=0.9),
    ]

    def fake_localize(prd: str, sess: Session, ladder: Ladder, **_: Any) -> list[CodeAnchor]:
        sess.write("knowledge/localization.md", "loc")
        return anchors

    monkeypatch.setattr("splinter.pipeline.localize", fake_localize)
    monkeypatch.setattr("splinter.pipeline.filter_task_context", _no_filter)

    _localize_and_filter(
        session,
        load_ladder(),
        "prd",
        tasks,
        False,
        False,
        1,
        2,
        run_order=[tasks[1], tasks[0]],
    )

    assert "### a" in session.read("knowledge/localization-1.md")
    assert "### b" in session.read("knowledge/localization-2.md")


def _rows(session: Session) -> dict[int, Any]:
    view = build_run_view(session, "RUNNING", datetime.now(timezone.utc), FIXTURE_LADDER)
    return {row.number: row for row in view.stories}


def _session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Session:
    return build_session(tmp_path, monkeypatch, stories=4, live=2, failed=0, asked=0)


def test_a_live_story_shows_the_tier_it_runs_on_now(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _session(tmp_path, monkeypatch)
    session.set_story_tier(2, 4)

    row = _rows(session)[2]

    assert row.state == "live" and row.tier == "T4"


def test_a_finished_story_keeps_the_tier_of_its_last_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _session(tmp_path, monkeypatch)
    before = _rows(session)[1].tier
    session.set_story_tier(1, 4)

    assert _rows(session)[1].tier == before
