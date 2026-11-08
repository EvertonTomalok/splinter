"""PLAN views before the first story: run plan, files and filters per story."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from rich.console import Console

from splinter.memory.session import Session
from splinter.run_view import build_run_view
from splinter.run_view_files import FileCache
from splinter.tui_run_views import ViewContext, relevant_paths, view_registry

PRD = "# Demo\n\n### US-001: Charge API\n\n### US-002: Refunds\n"
LOC = "# Localization — Task 1\n\n### api/charge.go — \nfile: api/charge.go\nreason: charges\n"
FILTER = "Charges go through one adapter.\n\n**`api/charge.go`**: holds the Charge handler.\n"


@pytest.fixture
def session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Session:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    s = Session("ses_plan_views")
    s.write("prd.md", PRD)
    s.write("knowledge/localization-1.md", LOC)
    s.write("knowledge/filter-1.md", FILTER)
    s.write("knowledge/filter-2.md", "Nothing about refunds yet.")
    return s


def _text(session: Session, name: str, story: int | None = None) -> str:
    view = build_run_view(session, "running", datetime.now(timezone.utc))
    ctx = ViewContext(
        view=view, story=story, opened=frozenset(), files=FileCache(), session=session, multi=False
    )
    console = Console(width=120, color_system=None)
    with console.capture() as cap:
        console.print(view_registry()[name](ctx))
    return cap.get()


def test_files_lists_every_story_before_the_first_one_runs(session: Session) -> None:
    out = _text(session, "files")
    assert "T1 US-001 Charge API  1 file" in out
    assert "api/charge.go" in out
    assert "api/charge.go —" not in out


def test_filters_summarise_every_story_and_open_one(session: Session) -> None:
    out = _text(session, "filter")
    assert "T1 US-001 Charge API" in out
    assert "Charges go through one adapter." in out
    assert "T2 US-002 Refunds" in out
    detail = _text(session, "filter", story=1)
    assert "knowledge/filter-1.md" in detail
    assert "api/charge.go" in detail


def test_plan_shows_refine_progress_then_waves(session: Session) -> None:
    partial = {"total": 2, "done": [{"task": "US-001", "briefs": [{"id": "a", "goal": "adapter"}]}]}
    session.write("agents-run/briefs-partial.json", json.dumps(partial))
    out = _text(session, "plan")
    assert "writing the plan · 1/2 stories split into briefs" in out
    assert "adapter" in out
    briefs = [
        {"id": "a", "goal": "adapter", "files": ["api/charge.go"], "deps": [], "task_id": "US-001"},
        {"id": "b", "goal": "refund", "files": [], "deps": ["a"], "task_id": "US-002"},
    ]
    session.write("agents-run/briefs.json", json.dumps(briefs))
    out = _text(session, "plan")
    assert "wave 1" in out and "wave 2" in out
    assert "after a" in out


def test_overview_repaints_when_a_note_lands(session: Session) -> None:
    view = build_run_view(session, "running", datetime.now(timezone.utc))
    ctx = ViewContext(
        view=view, story=None, opened=frozenset(), files=FileCache(), session=session, multi=False
    )
    paths = relevant_paths(ctx, "filter")
    assert session.dir / "knowledge" / "filter-2.md" in paths
    assert session.dir / "agents-run" / "briefs-partial.json" in paths
