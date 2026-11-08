from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from splinter.obs.events import Event, append_event
from splinter.obs.trace import RunEntry, Trace
from splinter.run_view import RunViewBuilder
from tests.fixtures_run_view import FIXTURE_LADDER, build_session

NOW = datetime(2026, 11, 8, 1, tzinfo=timezone.utc)


def test_idle_tick_only_updates_clock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = build_session(tmp_path, monkeypatch)
    builder = RunViewBuilder()
    before = builder.build(session, "RUNNING", NOW, FIXTURE_LADDER)
    after = builder.build(session, "RUNNING", NOW + timedelta(seconds=1), FIXTURE_LADDER)

    assert after.elapsed_s == before.elapsed_s + 1
    assert after.stories is before.stories
    assert after.iterations_by_story is before.iterations_by_story
    assert after.spend is before.spend


def test_completed_tick_does_not_recalculate_elapsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from splinter.run_view import elapsed

    session = build_session(tmp_path, monkeypatch)
    builder = RunViewBuilder()
    before = builder.build(session, "COMPLETED", NOW, FIXTURE_LADDER)
    with patch("splinter.run_view.elapsed", wraps=elapsed) as calculate:
        after = builder.build(session, "COMPLETED", NOW + timedelta(seconds=1), FIXTURE_LADDER)
    assert after is before
    assert calculate.call_count == 0


def test_appended_log_decodes_only_new_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = build_session(tmp_path, monkeypatch)
    builder = RunViewBuilder()
    before = builder.build(session, "RUNNING", NOW, FIXTURE_LADDER)
    append_event(session, Event("log", NOW.isoformat(), {"message": "new activity"}))
    with patch("json.loads", wraps=json.loads) as decode:
        after = builder.build(session, "RUNNING", NOW, FIXTURE_LADDER)

    assert after == before
    assert decode.call_count == 1
    assert after.stories is before.stories


def test_new_run_reuses_other_story_projections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = build_session(tmp_path, monkeypatch)
    builder = RunViewBuilder()
    before = builder.build(session, "RUNNING", NOW, FIXTURE_LADDER)
    Trace(session).add_entry(RunEntry("sonnet", 1, 3, {}, 0.125, 2.0, 23))
    after = builder.build(session, "RUNNING", NOW, FIXTURE_LADDER)

    assert after.total_cost == pytest.approx(before.total_cost + 0.125)
    assert after.stories[23].cost == pytest.approx(before.stories[23].cost + 0.125)
    assert after.stories[0] is before.stories[0]
    assert after.iterations_by_story[9] is before.iterations_by_story[9]
    assert after.iterations_by_story[24][-1].cost == pytest.approx(
        before.iterations_by_story[24][-1].cost + 0.125
    )


def test_parallel_review_edit_invalidates_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = build_session(tmp_path, monkeypatch, stories=2, live=2, failed=0, asked=0)
    session.write("loop.md", "# Task 1 [parallel]: First\n# Task 2 [parallel]: Second\n")
    session.write("knowledge/task0-review-iter-1.md", "- decision: RETRY\n(T0)\n")
    builder = RunViewBuilder()
    before = builder.build(session, "RUNNING", NOW, FIXTURE_LADDER)
    session.write("knowledge/task0-review-iter-1.md", "- decision: PASS\n(T0)\n")
    after = builder.build(session, "RUNNING", NOW, FIXTURE_LADDER)

    assert before.iterations_by_story[1][-1].verdict == "RETRY"
    assert after.iterations_by_story[1][-1].verdict == "PASS"
    assert after.stories[0].state == "pass"


def test_builder_can_switch_sessions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = build_session(tmp_path / "first", monkeypatch, stories=2, failed=0, asked=0)
    builder = RunViewBuilder()
    before = builder.build(first, "RUNNING", NOW, FIXTURE_LADDER)
    second = build_session(tmp_path / "second", monkeypatch, stories=3, failed=0, asked=0)
    after = builder.build(second, "RUNNING", NOW, FIXTURE_LADDER)

    assert len(before.stories) == 2
    assert len(after.stories) == 3


def test_ladder_cached_until_config_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = build_session(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    builder = RunViewBuilder()
    with patch("splinter.run_view.load_ladder", return_value=FIXTURE_LADDER) as load:
        builder.build(session, "RUNNING", NOW)
        builder.build(session, "RUNNING", NOW)
        assert load.call_count == 1
        config = tmp_path / ".splinter" / "config.yaml"
        config.parent.mkdir(exist_ok=True)
        config.write_text("models: {}\n")
        builder.build(session, "RUNNING", NOW)
        assert load.call_count == 2


def test_parallel_review_change_does_not_reread_other_reviews(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = build_session(tmp_path, monkeypatch, stories=2, failed=0, asked=0)
    session.write("loop.md", "# Task 1 [parallel]: First\n# Task 2 [parallel]: Second\n")
    session.write("knowledge/task0-review-iter-1.md", "- decision: RETRY\n(T0)\n")
    session.write("knowledge/task1-review-iter-1.md", "- decision: RETRY\n(T0)\n")
    builder = RunViewBuilder()
    before = builder.build(session, "RUNNING", NOW, FIXTURE_LADDER)
    session.write("knowledge/task0-review-iter-1.md", "- decision: PASS\n(T0)\n")
    read_text = Path.read_text
    reads: list[str] = []

    def read(path: Path, *args: object, **kwargs: object) -> str:
        reads.append(path.name)
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    after = builder.build(session, "RUNNING", NOW, FIXTURE_LADDER)
    assert "task0-review-iter-1.md" in reads
    assert "task1-review-iter-1.md" not in reads
    assert after.stories[1] is before.stories[1]


def test_replaced_review_invalidates_cache_with_preserved_mtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = build_session(tmp_path, monkeypatch, stories=1, failed=0, asked=0)
    session.write("loop.md", "# Task 1 [parallel]: First\n")
    note = session.dir / "knowledge/task0-review-iter-1.md"
    session.write("knowledge/task0-review-iter-1.md", "- decision: FAIL\n(T0)\n")
    builder = RunViewBuilder()
    before = builder.build(session, "RUNNING", NOW, FIXTURE_LADDER)
    stat = note.stat()
    replacement = note.with_suffix(".new")
    replacement.write_text("- decision: PASS\n(T0)\n")
    os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    replacement.replace(note)
    after = builder.build(session, "RUNNING", NOW, FIXTURE_LADDER)
    assert before.stories[0].state == "fail"
    assert after.stories[0].state == "pass"
