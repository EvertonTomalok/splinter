"""Task routing (straight to the run), plan-mode resume routing, prune, and the hand-off."""

from __future__ import annotations

import io
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
import typer

from splinter import cli
from splinter.memory.session import Session


class _TTY(io.StringIO):
    def isatty(self) -> bool:
        return True


@pytest.fixture
def tty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("SPLINTER_NO_TUI", raising=False)


def _fake_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    # Set inside the test body: pytest's capture swaps sys.stdout back in after fixtures.
    monkeypatch.setattr(sys, "stdin", _TTY())
    monkeypatch.setattr(sys, "stdout", _TTY())


def _record(target: list[dict[str, Any]], rc: int) -> Any:
    def _f(run_kwargs: dict[str, Any], **kw: Any) -> int:
        target.append({**run_kwargs, **kw})
        return rc

    return _f


@pytest.mark.parametrize("cowabunga", [False, True])
def test_task_on_tty_goes_straight_to_the_run(
    tty: None, monkeypatch: pytest.MonkeyPatch, cowabunga: bool
) -> None:
    from splinter import tui, tui_plan

    planned: list[dict[str, Any]] = []
    ran: list[dict[str, Any]] = []
    monkeypatch.setattr(tui_plan, "run_plan_interactive", _record(planned, 7))
    monkeypatch.setattr(tui, "run_with_tui", _record(ran, 9))
    _fake_tty(monkeypatch)
    with pytest.raises(typer.Exit) as exc:
        cli.run(task="t.yaml", cowabunga=cowabunga)
    assert exc.value.exit_code == 9
    assert ran[0]["task_path"] == "t.yaml" and not planned


def test_resume_routes_task_refining_to_plan_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from splinter import tui, tui_plan

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_task")
    session.write("task.yaml", "description: x\n")
    session.set_status("refining", source="task", phase="chat")
    picked: list[str] = []
    monkeypatch.setattr(tui_plan, "_resume_plan", lambda s, st: picked.append(s.id) or 0)
    monkeypatch.setattr(tui, "_resume_prd", lambda s, st: picked.append("prd") or 0)
    assert tui.resume_session("ses_task") == 0
    assert picked == ["ses_task"]


def _aged(session: Session, seconds: float) -> None:
    old = datetime.now(timezone.utc).timestamp() - seconds
    os.utime(session.dir, (old, old))


def test_prune_keeps_plan_sessions_with_task_or_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from splinter.memory.session import list_sessions
    from splinter.prd_session import prd_session_is_resumable, prune_dead_prd_sessions

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    with_task = Session("ses_with_task")
    with_task.write("task.yaml", "description: x\n")
    with_task.set_status("refining", source="task", phase="chat")
    with_plan = Session("ses_with_plan")
    with_plan.write("knowledge/plan.md", "# Plan\n\n1. x\n")
    with_plan.set_status("refining", source="task", phase="review")
    empty = Session("ses_empty_task")
    empty.set_status("refining", source="task", phase="clarify")
    for s in (with_task, with_plan, empty):
        _aged(s, 300)

    assert prd_session_is_resumable(with_task) and prd_session_is_resumable(with_plan)
    assert prune_dead_prd_sessions(min_age_seconds=60.0) == ["ses_empty_task"]
    assert set(list_sessions()) == {"ses_with_task", "ses_with_plan"}


def test_run_plan_interactive_hands_off_to_run_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from splinter import tui, tui_plan

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))

    def fake_run(self: Any) -> int:
        self.session.write("task.yaml", "description: x\n")
        self.session.set_status("refining", source="task", phase="run", strategy="raphael")
        return 0

    ran: list[dict[str, Any]] = []
    monkeypatch.setattr(tui_plan.PlanSessionApp, "run", fake_run)
    monkeypatch.setattr(tui, "run_with_tui", _record(ran, 3))
    rc = tui_plan.run_plan_interactive({"task_path": "orig.yaml", "strategy": None, "effort": "x"})
    assert rc == 3
    call = ran[0]
    session = call["session"]
    assert call["task_path"] == str(session.dir / "task.yaml")
    assert call["prd_path"] is None and call["strategy"] == "raphael" and call["effort"] == "x"


def test_run_plan_interactive_abort_does_not_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from splinter import tui, tui_plan

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))

    def fake_run(self: Any) -> None:
        self.session.write("task.yaml", "description: x\n")
        return None

    ran: list[dict[str, Any]] = []
    monkeypatch.setattr(tui_plan.PlanSessionApp, "run", fake_run)
    monkeypatch.setattr(tui, "run_with_tui", _record(ran, 3))
    assert tui_plan.run_plan_interactive({"task_path": "orig.yaml"}) == 0
    assert not ran
    assert "resume later" in capsys.readouterr().out
