"""The run trace: tree built from the session files, the full-screen trace, the log pane."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from textual.widgets import RichLog, TextArea, Tree

from splinter.run_trace import TraceNode, build_trace
from splinter.tui import RunApp
from splinter.tui_run_trace import TraceScreen
from tests.fixtures_run_view import FIXTURE_LADDER, build_session


def _app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RunApp:
    session = build_session(tmp_path, monkeypatch, jev=True)
    session.set_status("running", pid=os.getpid())
    monkeypatch.setattr("splinter.models.roster.load_ladder", lambda: FIXTURE_LADDER)
    monkeypatch.setattr(RunApp, "_run_pipeline_worker", lambda self, **kw: None)
    monkeypatch.setattr(RunApp, "_write_run_complete", lambda self: None)
    return RunApp(session, {"prd_path": "p.md"})


def _find(node: TraceNode, needle: str) -> TraceNode:
    if needle in node.label:
        return node
    for child in node.children:
        try:
            return _find(child, needle)
        except LookupError:
            continue
    raise LookupError(needle)


def test_trace_tree_tells_the_failed_story(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _app(tmp_path, monkeypatch)

    async def drive() -> None:
        async with app.run_test(size=(160, 44)) as pilot:
            await pilot.pause()
            root = build_trace(app.session, app._last_view)  # type: ignore[arg-type]
            story = _find(root, "US-009")
            assert story.expanded and "FAIL" in story.label
            last = _find(story, "iteration 5")
            labels = [child.label for child in last.children]
            assert any("review" in label for label in labels)
            assert any("actions" in label for label in labels)
            review = next(child for child in last.children if "review" in child.label)
            assert "call cancel_intent" in review.detail

    asyncio.run(drive())


def test_logs_is_the_first_run_topic_and_l_opens_the_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _app(tmp_path, monkeypatch)

    async def drive() -> None:
        async with app.run_test(size=(160, 44)) as pilot:
            await pilot.pause()
            app.write_log("🔧 Edit src/a.py")
            await app.run_action("select_story(9)")
            await app.run_action("set_view('plan')")
            await pilot.pause()
            assert not app.query_one("#log-pane").display
            await app.run_action("stage_view('run')")
            await pilot.pause()
            assert app.view == "logs"
            assert app.query_one("#log-pane").display
            assert not app.query_one("#view-body").display
            await pilot.press("L")
            await pilot.pause()
            assert isinstance(app.screen, TraceScreen)
            trace_log = app.screen.query_one("#trace-log", RichLog)
            assert any("src/a.py" in line.text for line in trace_log.lines)
            app.write_log("💬 still streaming")
            await pilot.pause()
            assert any("still streaming" in line.text for line in trace_log.lines)
            tree = app.screen.query_one("#trace-tree", Tree)
            groups = list(tree.root.children)
            story = next(
                node for group in groups for node in group.children if "US-009" in str(node.label)
            )
            tree.move_cursor(story)
            await pilot.pause()
            assert "US-009" in app.screen.query_one("#trace-detail", TextArea).text
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, TraceScreen)

    asyncio.run(drive())
