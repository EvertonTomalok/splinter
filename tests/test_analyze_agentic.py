from __future__ import annotations

from pathlib import Path

import pytest

from splinter.memory.session import Session, new_session_id
from splinter.obs.agentic import render_agentic


@pytest.fixture
def session_with_agentic_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Session:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    return Session(new_session_id())


def test_render_agentic_groups_by_task_orders_by_stage(
    session_with_agentic_trace: Session,
) -> None:
    session = session_with_agentic_trace
    agentic_dir = session.dir / "agentic"
    agentic_dir.mkdir(parents=True, exist_ok=True)

    eval_line = (
        '{"stage":"eval","task_index":1,"iteration":1,'
        '"prompt":"eval prompt","response":"eval response","model":"opus","variant":"fast"}\n'
    )
    run_line = (
        '{"stage":"run","task_index":1,"iteration":1,'
        '"prompt":"run prompt","response":"run response","model":"sonnet","variant":""}\n'
    )
    (agentic_dir / "task-1.jsonl").write_text(eval_line + run_line)

    plan_line = (
        '{"stage":"plan","task_index":2,"iteration":1,'
        '"prompt":"plan prompt","response":"plan response","model":"opus","variant":""}\n'
    )
    (agentic_dir / "task-2.jsonl").write_text(plan_line)

    result = render_agentic(session)

    lines = result.split("\n")
    task1_idx = next(i for i, line in enumerate(lines) if "Task 1" in line)
    task2_idx = next(i for i, line in enumerate(lines) if "Task 2" in line)
    assert task1_idx < task2_idx

    task1_section = "\n".join(lines[task1_idx:task2_idx])
    run_idx = next(
        i for i, line in enumerate(task1_section.split("\n")) if line.startswith("run ·")
    )
    eval_idx = next(
        i for i, line in enumerate(task1_section.split("\n")) if line.startswith("eval ·")
    )
    assert run_idx < eval_idx

    eval_line = next(line for line in lines if line.startswith("eval ·"))
    assert "fast" in eval_line, "variant should appear in header"


def test_render_agentic_empty_prompt_response_no_payload(
    session_with_agentic_trace: Session,
) -> None:
    session = session_with_agentic_trace
    agentic_dir = session.dir / "agentic"
    agentic_dir.mkdir(parents=True, exist_ok=True)

    run_line = (
        '{"stage":"run","task_index":1,"iteration":1,'
        '"prompt":"run prompt","response":"run response","model":"sonnet","variant":""}\n'
    )
    gate_line = (
        '{"stage":"gate","task_index":1,"iteration":2,'
        '"prompt":"","response":"","model":"","variant":""}\n'
    )
    (agentic_dir / "task-1.jsonl").write_text(run_line + gate_line)

    result = render_agentic(session)

    lines = result.split("\n")
    gate_line = next(line for line in lines if line.startswith("gate ·"))

    gate_idx = lines.index(gate_line)

    for i in range(gate_idx + 1, len(lines)):
        assert not lines[i].startswith("prompt:")
        assert not lines[i].startswith("response:")


def test_render_agentic_missing_trace_files_returns_empty_state(
    session_with_agentic_trace: Session,
) -> None:
    session = session_with_agentic_trace

    result = render_agentic(session)

    assert result == "(no agentic trace)"


def test_render_agentic_empty_jsonl_returns_empty_state(
    session_with_agentic_trace: Session,
) -> None:
    session = session_with_agentic_trace
    agentic_dir = session.dir / "agentic"
    agentic_dir.mkdir(parents=True, exist_ok=True)

    (agentic_dir / "task-1.jsonl").write_text("")

    result = render_agentic(session)

    assert result == "(no agentic trace)"
