"""Agents progress: attempt log, the run screen's reading of it, codegen and breaker waits."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import splinter.agents.codegen as codegen_mod
from splinter.agents.codegen import run_codegen
from splinter.agents.progress import (
    ATTEMPTS,
    CHECKPOINT,
    STORIES,
    AttemptLog,
    brief_stories,
    stories_of,
    write_done,
    write_stories,
)
from splinter.agents.refiner import Brief
from splinter.agents.runner import Task
from splinter.configure import CodegenStep, _codegen_steps
from splinter.engines.health import CircuitBreaker
from splinter.memory.session import Session
from splinter.run_view_agents import agents_groups, agents_rows


def _session(tmp_path: Path) -> Session:
    session = Session.__new__(Session)
    session.id = "test"
    session.dir = tmp_path
    return session


def _tasks() -> list[Task]:
    return [
        Task(id="US-001", description="US-001: Como owner, quero fórmulas", acceptance="ok"),
        Task(id="US-002", description="screens", acceptance="ok"),
        Task(id="US-003", description="export", acceptance="ok"),
    ]


def _texts(tmp_path: Path) -> tuple[str, str, str]:
    def read(name: str) -> str:
        path = tmp_path / name
        return path.read_text() if path.exists() else ""

    return read(STORIES), read(ATTEMPTS), read(CHECKPOINT)


def test_stories_drop_a_repeated_id_and_briefs_keep_the_previous_story() -> None:
    stories = stories_of(_tasks())
    assert [(s.number, s.id, s.title) for s in stories] == [
        (1, "US-001", "Como owner, quero fórmulas"), (2, "US-002", "screens"),
        (3, "US-003", "export"),
    ]
    briefs = [
        Brief(id="r1a", goal="g", task_id="US-001"), Brief(id="r1b", goal="g"),
        Brief(id="b2", goal="g", task_id="us-002"),
    ]
    assert brief_stories(briefs, stories) == {"r1a": 1, "r1b": 1, "b2": 2}


def test_screen_reads_every_story_live_attempts_and_story_completion(tmp_path: Path) -> None:
    session = _session(tmp_path)
    write_stories(session, stories_of(_tasks()), {"r1a": 1, "r1b": 1, "b2": 2})
    log = AttemptLog(session)
    first = log.start(1, "r1a", "mcode", 1)
    log.end(1, first, "PASS")
    write_done(session, {"r1a"})
    live = log.start(2, "b2", "kimi", 3)

    groups = agents_groups(*_texts(tmp_path)[:2])
    assert [(n, title) for n, title, _ in groups] == [
        (1, "US-001: Como owner, quero fórmulas"), (2, "US-002: screens"),
        (3, "US-003: export"),
    ]
    assert groups[1][2] == [(live, "T3", "?")]
    rows = {row.number: row for row in agents_rows(*_texts(tmp_path))}
    assert rows[1].status == "pending"
    assert rows[2].status == "running"
    assert rows[3].status == "pending" and rows[3].iters == 0

    second = log.start(1, "r1b", "agy", 0)
    log.end(1, second, "PASS")
    write_done(session, {"r1a", "r1b"})
    rows = {row.number: row for row in agents_rows(*_texts(tmp_path))}
    assert rows[1].status == "pass" and rows[1].verdict == "PASS" and rows[1].iters == 2


def test_resume_closes_attempts_left_open_and_keeps_numbering(tmp_path: Path) -> None:
    session = _session(tmp_path)
    write_stories(session, stories_of(_tasks()), {"b2": 2})
    AttemptLog(session).start(2, "b2", "kimi", 3)
    resumed = AttemptLog(session)
    resumed.close_interrupted()
    assert resumed.start(2, "b2", "mcode", 3) == 2
    groups = agents_groups(*_texts(tmp_path)[:2])
    assert [verdict for _, _, verdict in groups[1][2]] == ["RETRY", "?"]


def test_merge_conflict_fails_the_story(tmp_path: Path) -> None:
    session = _session(tmp_path)
    write_stories(session, stories_of(_tasks()), {"b2": 2})
    log = AttemptLog(session)
    log.end(2, log.start(2, "b2", "kimi", 3), "FAIL", "merge conflict")
    rows = {row.number: row for row in agents_rows(*_texts(tmp_path))}
    assert rows[2].status == "failed"


def test_codegen_runs_only_steps_whose_paths_changed(tmp_path: Path) -> None:
    steps = (
        CodegenStep(cmd="make gen", paths=("api/",)),
        CodegenStep(cmd="make web-gen", paths=("web/",)),
    )
    runs: list[Any] = []

    def fake_run(cmd: Any, **kwargs: Any) -> MagicMock:
        runs.append(cmd)
        if isinstance(cmd, list):
            return MagicMock(stdout="?? api/proto/hub/agenda/v1/report.proto\n", returncode=0)
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch.object(codegen_mod.subprocess, "run", side_effect=fake_run):
        result = run_codegen(tmp_path, steps)
    assert result.ok and result.ran == ("make gen",)
    assert runs[1:] == ["make gen"]


def test_codegen_failure_carries_the_output(tmp_path: Path) -> None:
    def fake_run(cmd: Any, **kwargs: Any) -> MagicMock:
        if isinstance(cmd, list):
            return MagicMock(stdout=" M api/x.proto\n", returncode=0)
        return MagicMock(returncode=2, stdout="", stderr="x.proto:3: syntax error")

    with patch.object(codegen_mod.subprocess, "run", side_effect=fake_run):
        result = run_codegen(tmp_path, (CodegenStep(cmd="make gen"),))
    assert not result.ok and "syntax error" in result.failure


def test_codegen_config_accepts_strings_and_mappings() -> None:
    steps = _codegen_steps(["make gen", {"cmd": "buf generate", "paths": ["api/"], "timeout": 90}])
    assert steps == (
        CodegenStep(cmd="make gen"),
        CodegenStep(cmd="buf generate", paths=("api/",), timeout=90),
    )
    assert _codegen_steps("make gen") == ()


def test_open_breaker_reports_the_rest_of_its_cooldown() -> None:
    breaker = CircuitBreaker(failure_threshold=1, cooldown_s=300)
    assert breaker.wait_s(now=0) == 0
    breaker.record_failure(now=100)
    assert breaker.wait_s(now=160) == 240
    assert breaker.wait_s(now=400) == 0
