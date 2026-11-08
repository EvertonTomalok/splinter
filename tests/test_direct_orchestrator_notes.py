from __future__ import annotations

from pathlib import Path

import pytest

from splinter.agents.runner import Task
from splinter.memory.knowledge import KnowledgeStore
from splinter.memory.session import Session
from splinter.obs.trace import Trace
from splinter.strategies.direct import DirectStrategy


class _Ladder:
    planner_model = "sonnet"
    planner_effort = "high"

    def effort_mapping(self, _effort: str) -> None:
        return None


def test_force_replan_passes_orchestrator_notes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_direct_note")
    session.queue_orchestrator_note("keep the public API stable")
    received: list[str] = []

    def fake_make_plan(*_args: object, **kwargs: object) -> str:
        received.append(str(kwargs["orchestrator_notes"]))
        return "implement it"

    monkeypatch.setattr("splinter.strategies.direct._make_plan", fake_make_plan)
    DirectStrategy()._run_task_loop(
        Task(description="change endpoint", acceptance="works"),
        session,
        _Ladder(),  # type: ignore[arg-type]
        Trace(session),
        KnowledgeStore(session),
        effort=None,
        budget=None,
        max_iterations=0,
        localization="",
        force_replan=True,
    )

    assert received == ["## Orchestrator notes\n- keep the public API stable"]


def test_given_plan_skips_the_planner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_direct_given_plan")

    def no_planner(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("the planner must not run when a plan is given")

    monkeypatch.setattr("splinter.strategies.direct._make_plan", no_planner)
    DirectStrategy()._run_task_loop(
        Task(description="define the proto", acceptance="stubs exist"),
        session,
        _Ladder(),  # type: ignore[arg-type]
        Trace(session),
        KnowledgeStore(session),
        task_index=1,
        effort=None,
        budget=None,
        max_iterations=0,
        localization="",
        plan_text="# Brief us-002-att1\nwrite report.proto and run make gen",
    )

    assert "run make gen" in session.read("knowledge/plan-2.md")
