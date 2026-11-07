from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

    from splinter.memory.session import Session


def _session(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch", sid: str = "ses_ov") -> "Session":
    from splinter.memory.session import Session

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session(sid)
    session.update_index("# overview\n")
    return session


def _status(session: "Session", **fields: object) -> None:
    data: dict[str, object] = {
        "state": "running",
        "strategy": "cascade",
        "tasks": 3,
        "max_iterations": 5,
        "task_index": 1,
        "stage": "run",
        "started_at": "2026-07-05T01:49:58+00:00",
        "updated": "2026-07-05T02:00:00+00:00",
    }
    data.update(fields)
    session.write("status.json", json.dumps(data, indent=2))


def _run_events(
    session: "Session",
    entries: list[tuple[str, int, int, int, str, str]],
) -> None:
    lines = []
    for model, tier, iteration, task, role, ts in entries:
        lines.append(
            json.dumps(
                {
                    "type": "run",
                    "ts": ts,
                    "payload": {
                        "model": model,
                        "tier": tier,
                        "iteration": iteration,
                        "tokens": {"input": 10, "output": 20},
                        "cost": 0.01,
                        "latency_s": 1.0,
                        "task": task,
                        "role": role,
                        "cost_indeterminate": False,
                    },
                }
            )
        )
    session.write("events.jsonl", "\n".join(lines) + "\n")


def _loop(session: "Session") -> None:
    session.write(
        "loop.md",
        "# Task 1/3: US-001: Alpha feature\n"
        "\n"
        "## Iteration 1\n"
        "- model: sonnet (tier 1)\n"
        "- verdict: PASS — all good\n"
        "- corrections: none\n"
        "\n"
        "# Task 2/3: US-002: Beta feature\n"
        "\n"
        "## Iteration 1\n"
        "- model: haiku (tier 0)\n"
        "- verdict: RETRY — the gate failed on ruff\n"
        "- corrections: 1) fix the import order\n"
        "\n"
        "# Task 3/3: US-003: Gamma feature\n",
    )


def _running_session(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> "Session":
    session = _session(tmp_path, monkeypatch)
    monkeypatch.setattr("splinter.run_overview._tier_effort", lambda level, model: "high")
    _status(session)
    _loop(session)
    _run_events(
        session,
        [
            ("sonnet", 1, 1, 1, "run", "2026-07-05T01:50:00+00:00"),
            ("sonnet", 1, 1, 2, "run", "2026-07-05T01:55:00+00:00"),
        ],
    )
    session.write(
        "knowledge/localization.md",
        "# Localization\n\nfile: a.py\nsymbol: A\n\n"
        "file: b.py\nsymbol: B\n\nfile: a.py\nsymbol: A2\n",
    )
    session.write("knowledge/plan.md", "# Plan\n\n1. first\n2. second\n")
    session.write("prd_phases.md", "- generate · 3 stories\n- clarify\n- finalize · 3 stories\n")
    return session


def _plain(line: str) -> str:
    from rich.text import Text

    return Text.from_markup(line).plain


def test_header_line_shows_state_strategy_and_cost(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from splinter.run_overview import render_run_overview

    session = _running_session(tmp_path, monkeypatch)
    out = render_run_overview(session, "RUNNING")
    header = out.splitlines()[0]
    assert "RUNNING" in header
    assert "cascade" in header
    assert "$0.0200" in header
    assert header.count("·") >= 2, f"header should join parts with separators: {header}"


def test_header_elapsed_uses_trace_span_when_finished(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from splinter.run_overview import render_run_overview

    session = _running_session(tmp_path, monkeypatch)
    out = render_run_overview(session, "COMPLETED")
    header = out.splitlines()[0]
    assert "COMPLETED" in header
    assert "5m" in header or "300s" in header, f"elapsed missing from header: {header}"


def test_now_card_reports_task_stage_model_and_iteration(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from splinter.run_overview import render_run_overview

    session = _running_session(tmp_path, monkeypatch)
    out = render_run_overview(session, "RUNNING")
    assert "[bold]NOW[/]" in out
    assert "Task 2/3" in out
    assert "US-002" in out
    assert "sonnet@high" in out
    assert "iter 1/5" in out
    assert "last: RETRY" in out


def test_now_card_hidden_when_not_running(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from splinter.run_overview import render_run_overview

    session = _running_session(tmp_path, monkeypatch)
    out = render_run_overview(session, "COMPLETED")
    assert "[bold]NOW[/]" not in out


def test_now_card_awaiting_states_say_waiting_for_you(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from splinter.run_overview import render_run_overview

    session = _running_session(tmp_path, monkeypatch)
    _status(session, state="awaiting_user")
    out = render_run_overview(session, "AWAITING_USER")
    assert "waiting for you" in out
    assert "answer in the dialog" in out


def test_output_never_prints_unknown_placeholders(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from splinter.run_overview import render_run_overview

    session = _session(tmp_path, monkeypatch)
    _status(session, max_iterations="", stage="")
    session.write(
        "loop.md",
        "# Task 1/1: Only task\n\n## Iteration 1\n- model: sonnet\n",
    )
    _run_events(session, [("sonnet", 1, 1, 0, "run", "2026-07-05T01:50:00+00:00")])
    out = render_run_overview(session, "RUNNING")
    assert "T?" not in out
    assert "last: ?" not in out
    assert "iter ?" not in out
    assert "?" not in out, f"unknown marker leaked into the overview: {out}"


def test_localize_step_is_a_single_file_count_line(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from splinter.run_overview import render_run_overview

    session = _running_session(tmp_path, monkeypatch)
    out = render_run_overview(session, "RUNNING")
    assert "localize" in out
    assert "2 files" in out
    assert "hot" not in out
    assert "anchor" not in out


def test_tasks_table_has_one_aligned_row_per_task(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    import re

    from splinter.run_overview import render_run_overview

    session = _running_session(tmp_path, monkeypatch)
    out = render_run_overview(session, "RUNNING")
    assert "[bold]TASKS[/]" in out
    rows = [ln for ln in out.splitlines() if re.match(r"^\s*\d+\s+[○▶✓✗⏸]", _plain(ln))]
    assert len(rows) == 3, f"expected one row per task, got: {rows}"
    widths = {len(_plain(r)) for r in rows}
    assert len(widths) == 1, f"task rows not aligned: {[(r, len(_plain(r))) for r in rows]}"
    assert "US-001" in rows[0]
    assert "PASS" in rows[0]
    assert "US-003" in rows[2]


def test_only_the_current_task_shows_as_running(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    import re

    from splinter.run_overview import render_run_overview

    session = _session(tmp_path, monkeypatch)
    _status(session, task_index=1)
    session.write(
        "loop.md",
        "# Task 1/3: US-001: Alpha\n\n## Iteration 1\n- model: haiku (tier 0)\n"
        "- verdict: RETRY — first try failed\n\n"
        "# Task 2/3: US-002: Beta\n\n## Iteration 1\n- model: haiku (tier 0)\n"
        "- verdict: RETRY — second try failed\n\n"
        "# Task 3/3: US-003: Gamma\n",
    )
    out = render_run_overview(session, "RUNNING")
    rows = [ln for ln in out.splitlines() if re.match(r"^\s*\d+\s+[○▶✓✗⏸]", _plain(ln))]
    assert len(rows) == 3, rows
    assert "✗" in _plain(rows[0]), f"task 1 is not the current task: {_plain(rows[0])}"
    assert "▶" in _plain(rows[1]), f"task 2 is the current task: {_plain(rows[1])}"
    assert "○" in _plain(rows[2]), f"task 3 has not started: {_plain(rows[2])}"


def test_tasks_section_hidden_for_single_task(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from splinter.run_overview import render_run_overview

    session = _session(tmp_path, monkeypatch)
    _status(session, tasks=1, task_index=0)
    session.write("loop.md", "# Task 1/1: Only task\n\n## Iteration 1\n- verdict: PASS\n")
    out = render_run_overview(session, "RUNNING")
    assert "[bold]TASKS[/]" not in out


def test_waves_section_reads_agents_run_files(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from splinter.run_overview import render_run_overview

    session = _session(tmp_path, monkeypatch)
    _status(session, strategy="agents", tasks=3, task_index=0)
    session.write(
        "agents-run/README.md",
        "# Plan — demo\n\n## Waves & Briefs\n\n- Wave 1: B-01, B-02\n- Wave 2: B-03\n",
    )
    session.write(
        "agents-run/STATE.md",
        "# Run state\n\n## Merged\n\n- merged B-01 (mcode) into splinter/demo-int\n",
    )
    session.write("agents-run/checkpoint.json", json.dumps({"completed": ["B-01"]}))
    session.append(
        "events.md",
        "[AGENTS] brief B-02 engine=agy infra failure — STALL: idle watchdog",
    )
    session.append("events.md", "[AGENTS] brief B-02 engine=mcode fix round 1")
    session.append("events.md", "[AGENTS] brief B-03 assumed by orchestrator")

    out = render_run_overview(session, "RUNNING")
    assert "[bold]WAVES[/]" in out
    assert "Wave 1" in out
    assert "Wave 2" in out
    assert "B-01" in out
    assert "mcode" in out
    assert "merged" in out
    assert "fix 1/2" in out
    assert "↺ from agy" in out
    assert "assumed" in out


def test_waves_section_hidden_for_other_strategies(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from splinter.run_overview import render_run_overview

    session = _running_session(tmp_path, monkeypatch)
    out = render_run_overview(session, "RUNNING")
    assert "[bold]WAVES[/]" not in out


def test_sections_hidden_when_data_is_missing(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from splinter.run_overview import render_run_overview

    session = _session(tmp_path, monkeypatch)
    _status(session, tasks=1, task_index=0)
    out = render_run_overview(session, "RUNNING")
    assert "[bold]TASKS[/]" not in out
    assert "[bold]WAVES[/]" not in out


def test_prd_trajectory_is_a_single_dim_line_at_the_bottom(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from splinter.run_overview import render_run_overview

    session = _running_session(tmp_path, monkeypatch)
    out = render_run_overview(session, "RUNNING")
    last = out.splitlines()[-1]
    assert "generate" in last
    assert "→" in last
    assert "finalize" in last
    assert "[dim]" in last


def test_prd_line_absent_without_phases(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
    from splinter.run_overview import render_run_overview

    session = _session(tmp_path, monkeypatch)
    _status(session, tasks=1, task_index=0)
    out = render_run_overview(session, "RUNNING")
    assert "→" not in out


def test_header_total_includes_pre_run_and_jev_cost(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from splinter.run_overview import render_run_overview

    session = _running_session(tmp_path, monkeypatch)
    session.log_llm_usage("sonnet", {"input": 100, "output": 50}, 0.03)
    session.log_llm_usage("jev:typesafe/jev-1.14", {"input": 10, "output": 1}, 0.0015)
    session.log_llm_usage("jev:typesafe/jev-1.14", {"input": 10, "output": 1}, 0.0015)
    lines = render_run_overview(session, "RUNNING").splitlines()
    header, breakdown = _plain(lines[0]), _plain(lines[1])
    assert "$0.0530" in header, header
    assert "run $0.0200" in breakdown, breakdown
    assert "pre-run $0.0300" in breakdown, breakdown
    assert "jev $0.0030 ×2" in breakdown, breakdown


def test_header_without_pre_run_usage_shows_only_run_cost(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from splinter.run_overview import render_run_overview

    session = _running_session(tmp_path, monkeypatch)
    out = _plain(render_run_overview(session, "RUNNING"))
    assert "$0.0200" in out.splitlines()[0]
    assert "pre-run" not in out and "jev $" not in out, out


def test_last_verdict_has_a_single_dash(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from splinter.run_overview import render_run_overview

    session = _running_session(tmp_path, monkeypatch)
    out = _plain(render_run_overview(session, "RUNNING"))
    last = next(line for line in out.splitlines() if "last:" in line and "RETRY" in line)
    assert "— —" not in last and "RETRY — the gate failed" in last, last


def test_run_completion_total_includes_pre_run_cost(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from splinter.analyze import format_run_completion

    session = _running_session(tmp_path, monkeypatch)
    session.log_llm_usage("jev:typesafe/jev-1.14", {"input": 10, "output": 1}, 0.005)
    session.log_llm_usage("sonnet", {"input": 10, "output": 1}, 0.025)
    assert "$0.0500" in format_run_completion(session)
