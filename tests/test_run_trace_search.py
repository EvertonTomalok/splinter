from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest
from textual.widgets import Input, TextArea, Tree

from splinter.memory.session import Session
from splinter.run_trace import TraceNode, build_trace, search_trace
from splinter.run_view import build_run_view
from splinter.tui import PrdSessionApp, RunApp
from splinter.tui_run_trace import TraceScreen
from tests.fixtures_run_view import FIXTURE_LADDER, build_session

_PRD = "# PRD\n\n### US-001: One\nDo it.\n\n### US-002: Two\nDepends on: US-001\n"


def _session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Session:
    session = build_session(tmp_path, monkeypatch, stories=2, live=1, failed=0, asked=0)
    session.write("prd.md", _PRD + "\nNeedle in the final PRD.\n")
    session.write("instructions.md", "Build the panel.\n")
    session.write("convo.md", "[cyan]Generating PRD[/]\x001. Which size?\x00[cyan]> 1C[/]")
    session.write("prd_versions.md", "- 000 · generate · 2 stories\n")
    session.write("prd-versions/000.md", "First draft with a needle.\n")
    session.write("knowledge/localization.md", "file: handlers.go\n")
    session.write("knowledge/filter-1.md", "handlers.go matters.\n")
    session.write("knowledge/plan-2.md", "# Plan\n\n1. Whole plan for the queued story\n")
    session.write("loop.md", "# Task 1/2: US-001: One\n\n## Iteration 1\n")
    session.set_status("running", strategy="cascade", tasks=2, parallel=True, pid=os.getpid())
    with (session.dir / "events.jsonl").open("a") as handle:
        event = {"type": "log", "ts": "2026-11-09T01:02:03Z", "payload": {"message": "log needle"}}
        handle.write(json.dumps(event) + "\n")
    return session


def _root(session: Session) -> TraceNode:
    view = build_run_view(session, "RUNNING", datetime.now(timezone.utc), FIXTURE_LADDER)
    return build_trace(session, view)


def _labels(node: TraceNode) -> list[str]:
    return [node.label, *(label for child in node.children for label in _labels(child))]


def test_trace_starts_at_the_prd_and_covers_context_and_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = "\n".join(_labels(_root(_session(tmp_path, monkeypatch))))
    for label in ("instructions", "conversation", "PRD v000", "final PRD", "run setup"):
        assert label in text
    for label in ("localize", "filter", "full log"):
        assert label in text


def test_conversation_is_readable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _root(_session(tmp_path, monkeypatch))
    start = next(node for node in root.children if "start" in node.label)
    convo = next(node for node in start.children if "conversation" in node.label)
    assert convo.detail == "Generating PRD\n\n1. Which size?\n\n> 1C"


def test_queued_story_opens_its_whole_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(_session(tmp_path, monkeypatch))
    queued = next(node for node in root.children if "queued" in node.label)
    plan = next(node for node in queued.children[0].children if "plan" in node.label)
    assert "Whole plan for the queued story" in plan.detail


def test_search_finds_every_place_once_with_its_position() -> None:
    leaf = TraceNode("a", "one\ntwo NEEDLE and needle")
    copy = TraceNode("b", "one\ntwo NEEDLE and needle")
    hidden = TraceNode("c", "needle", searchable=False)
    root = TraceNode("r", children=[leaf, copy, hidden])
    hits = search_trace(root, "needle")
    assert [(hit.node.label, hit.line, hit.column) for hit in hits] == [("a", 1, 4), ("a", 1, 15)]
    assert search_trace(root, "n") == []


def _app(session: Session, monkeypatch: pytest.MonkeyPatch) -> RunApp:
    monkeypatch.setattr("splinter.models.roster.load_ladder", lambda: FIXTURE_LADDER)
    monkeypatch.setattr(RunApp, "_run_pipeline_worker", lambda self, **kw: None)
    monkeypatch.setattr(RunApp, "_write_run_complete", lambda self: None)
    return RunApp(session, {"prd_path": "p.md"})


def test_ctrl_f_finds_across_the_run_and_walks_the_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _app(_session(tmp_path, monkeypatch), monkeypatch)

    async def drive() -> None:
        async with app.run_test(size=(160, 44)) as pilot:
            await pilot.pause()
            await pilot.press("L")
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, TraceScreen)
            await pilot.press("ctrl+f")
            await pilot.pause()
            field = screen.query_one("#trace-find", Input)
            assert field.has_focus
            await pilot.press(*"needle")
            await pilot.pause()
            area = screen.query_one("#trace-detail", TextArea)
            assert "needle" in area.selected_text.lower()
            first = area.text
            await pilot.press("enter")
            await pilot.pause()
            assert area.text != first and "needle" in area.selected_text.lower()
            await pilot.press("escape")
            await pilot.pause()
            assert app.screen is screen and field.value == ""
            assert screen.query_one("#trace-tree", Tree).has_focus
            await pilot.press("escape")
            await pilot.pause()
            assert app.screen is not screen

    asyncio.run(drive())


def test_generating_a_prd_keeps_what_the_user_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    monkeypatch.setattr(PrdSessionApp, "_spawn", lambda *args, **kw: None)
    session = Session("typed")
    app = PrdSessionApp(session, {"no_ground": True})

    async def drive() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            app.query_one("#instructions", TextArea).text = "Build a calculator"
            app._on_generate()
            await pilot.pause()

    asyncio.run(drive())
    assert session.read("instructions.md").strip() == "Build a calculator"
