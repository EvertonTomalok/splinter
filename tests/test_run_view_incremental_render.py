from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from textual.widgets import Static

from splinter.obs.trace import RunEntry, Trace
from tests.fixtures_run_view import FIXTURE_LADDER, build_session


def test_idle_and_unrelated_story_do_not_repaint_selected_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from splinter.tui import RunApp

    session = build_session(tmp_path, monkeypatch)
    monkeypatch.setattr(RunApp, "_work", lambda self: None)
    monkeypatch.setattr("splinter.run_view.load_ladder", lambda: FIXTURE_LADDER)
    monkeypatch.setattr("splinter.models.roster.load_ladder", lambda: FIXTURE_LADDER)
    app = RunApp(session, {})
    app.animation_level = "none"

    async def drive() -> None:
        async with app.run_test(size=(120, 36)) as pilot:
            await app.run_action("select_story(9)")
            await app.run_action("set_view('eval')")
            await pilot.pause()
            updates: list[str] = []
            update = Static.update

            def track(widget: Static, content: object = "") -> None:
                if widget.id == "view-static":
                    updates.append(widget.id)
                update(widget, content)

            monkeypatch.setattr(Static, "update", track)
            app._refresh()
            app._refresh()
            Trace(session).add_entry(RunEntry("sonnet", 1, 3, {}, 0.25, 2.0, 23))
            app._refresh()
            assert updates == []
            session.write(
                "knowledge/task8-review-iter-5.md", "- decision: FAIL\n- reason: Changed\n"
            )
            app._refresh()
            assert updates == ["view-static"]
            updates.clear()
            await app.run_action("set_view('act')")
            await pilot.pause()
            updates.clear()
            session.append("loop.md", "\n# Task 59/59: Another story\n## Iteration 1\n")
            app._refresh()
            assert updates == []
            session.append("runs/task-9-iter-1.md", "\nChanged collapsed output\n")
            app._refresh()
            assert updates == []
            session.append("runs/task-9-iter-5.md", "\nChanged expanded output\n")
            app._refresh()
            assert updates == ["view-static"]

    asyncio.run(drive())


def test_view_key_reuses_parsed_story_until_history_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from splinter.run_view import RunViewBuilder
    from splinter.run_view_files import FileCache, _story_body
    from splinter.tui_run_view_key import view_content_key
    from splinter.tui_run_views import ViewContext

    session = build_session(tmp_path, monkeypatch)
    view = RunViewBuilder().build(session, "RUNNING", datetime.now(timezone.utc), FIXTURE_LADDER)
    ctx = ViewContext(view, 9, frozenset(), FileCache(), session, True, width=100)
    with patch("splinter.run_view_files._story_body", wraps=_story_body) as parse:
        first = view_content_key(ctx, "act")
        assert view_content_key(ctx, "act") == first
        assert view_content_key(replace(ctx, width=80), "act") != first
        assert parse.call_count == 1
        loop = session.read("loop.md")
        session.write(
            "loop.md", loop.replace("US-009: Story 9\n", "US-009: Story 9\n- gate: PASS\n")
        )
        assert view_content_key(ctx, "act") != first
        assert parse.call_count == 2
