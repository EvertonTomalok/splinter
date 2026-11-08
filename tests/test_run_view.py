from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from tests.fixtures_run_view import FIXTURE_LADDER, build_session, build_single_session

if TYPE_CHECKING:
    import pytest


def test_run_view_groups_states_and_iterations(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from splinter.run_view import build_run_view

    session = build_session(tmp_path, monkeypatch)
    view = build_run_view(
        session, "RUNNING", datetime(2026, 11, 8, 1, tzinfo=timezone.utc), FIXTURE_LADDER
    )

    assert len(view.stories) == 58
    assert view.needs_you == (9, 17)
    assert view.now == (24,)
    assert view.next_up == (25, 26, 27, 28, 29)
    assert view.counts == {"pass": 21, "fail": 1, "paused": 0, "live": 1, "queued": 34, "asked": 1}
    assert len(view.iterations_by_story[9]) == 5
    assert view.iterations_by_story[9][1].escalated_to == "T1"
    assert view.iterations_by_story[9][3].escalated_to == "T2"
    assert view.iterations_by_story[24][-1].verdict == ""


def test_run_view_single_session(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
    from splinter.run_view import build_run_view

    session = build_single_session(tmp_path, monkeypatch)
    view = build_run_view(
        session, "COMPLETED", datetime(2026, 11, 8, tzinfo=timezone.utc), FIXTURE_LADDER
    )

    assert len(view.stories) == 1
    assert view.stories[0].id == "T1"


def test_spend_rows_include_jev_only_when_used(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from splinter.run_view import build_run_view

    session = build_session(tmp_path, monkeypatch)
    plain = build_run_view(session, "RUNNING", datetime.now(timezone.utc), FIXTURE_LADDER)
    jev_session = build_session(tmp_path, monkeypatch, jev=True)
    with_jev = build_run_view(jev_session, "RUNNING", datetime.now(timezone.utc), FIXTURE_LADDER)

    assert round(sum(row.cost for row in plain.spend), 4) == round(plain.total_cost, 4)
    assert all(not row.label.startswith("jev ·") for row in plain.spend)
    assert any(row.label == "jev · 6 calls" for row in with_jev.spend)


def test_truncated_loop_and_agents_status_are_safe(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from splinter.run_view import build_run_view

    session = build_session(tmp_path, monkeypatch)
    session.write("loop.md", "# Task 1/3: Story\n\n## Iteration 1\n- model: flash (tier 0)\n")
    truncated = build_run_view(session, "RUNNING", datetime.now(timezone.utc), FIXTURE_LADDER)
    session.set_status("running", strategy="agents", tasks=3, stage="run", task_index=0)
    agents = build_run_view(session, "RUNNING", datetime.now(timezone.utc), FIXTURE_LADDER)

    assert truncated.iterations_by_story[1][-1].verdict == ""
    assert len(agents.stories) == 3


def test_run_view_has_no_textual_dependency() -> None:
    import inspect

    import splinter.run_view as run_view

    assert "textual" not in inspect.getsource(run_view).lower()
