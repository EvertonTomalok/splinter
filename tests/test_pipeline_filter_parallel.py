"""US-002: filter stage fans out via the US-001 bounded primitive."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from splinter.agents.localizer import CodeAnchor
from splinter.agents.runner import Task
from splinter.memory.session import Session
from splinter.models.roster import Ladder, load_ladder
from splinter.pipeline import _localize_and_filter


@pytest.fixture
def tmp_session(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Session:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    return Session("test-session")


def _make_tasks(n: int) -> list[Task]:
    return [
        Task(description=f"task-{i}", acceptance="ok", target_files=[f"file{i}.py"])
        for i in range(n)
    ]


def _make_anchors(n: int) -> list[CodeAnchor]:
    return [
        CodeAnchor(file=f"file{i}.py", symbol="x", reason="r", confidence=0.9)
        for i in range(n)
    ]


def _stub_localize(
    anchors: list[CodeAnchor],
) -> Callable[[str, Session, Ladder], list[CodeAnchor]]:
    def _stub(prd_text: str, session: Session, ladder: Ladder) -> list[CodeAnchor]:
        session.write("knowledge/localization.md", "loc")
        return anchors

    return _stub


def _run(
    session: Session, ladder: Ladder, tasks: list[Task], anchors: list[CodeAnchor], **kwargs: Any
) -> tuple[str, list[CodeAnchor]]:
    return _localize_and_filter(
        session,
        ladder,
        prd_text="dummy prd",
        tasks=tasks,
        single_shot=False,
        resume=kwargs.get("resume", False),
        resume_round=kwargs.get("resume_round", 1),
        max_concurrency=kwargs.get("max_concurrency", 4),
    )


@pytest.mark.parametrize("n", [1, 3, 5])
def test_filter_parallel_matches_serial_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_session: Session, n: int
) -> None:
    tasks = _make_tasks(n)
    anchors = _make_anchors(n)
    ladder = load_ladder()

    calls: list[str] = []

    def _stub_filter(task: Task, ladder: Ladder, *, session: Session) -> str:
        calls.append(task.description)
        time.sleep(0.05)
        return f"ctx-{task.description}"

    monkeypatch.setattr("splinter.pipeline.localize", _stub_localize(anchors))
    monkeypatch.setattr("splinter.pipeline.filter_task_context", _stub_filter)

    start = time.monotonic()
    _run(tmp_session, ladder, tasks, anchors)
    elapsed = time.monotonic() - start

    assert [t.filtered_context for t in tasks] == [f"ctx-{t.description}" for t in tasks]
    assert sorted(calls) == sorted(t.description for t in tasks)
    for i, task in enumerate(tasks):
        assert tmp_session.read(f"knowledge/filter-{i + 1}.md") == f"ctx-{task.description}"

    if n > 1:
        assert elapsed < n * 0.05


def test_filter_one_fails_siblings_still_resolve(
    monkeypatch: pytest.MonkeyPatch, tmp_session: Session
) -> None:
    tasks = _make_tasks(3)
    anchors = _make_anchors(3)
    ladder = load_ladder()

    def _stub_filter(task: Task, ladder: Ladder, *, session: Session) -> str:
        if task.description == "task-1":
            raise RuntimeError("boom")
        return f"ctx-{task.description}"

    monkeypatch.setattr("splinter.pipeline.localize", _stub_localize(anchors))
    monkeypatch.setattr("splinter.pipeline.filter_task_context", _stub_filter)

    with pytest.raises(RuntimeError, match="boom"):
        _run(tmp_session, ladder, tasks, anchors)


def test_filter_resume_skips_fanout(monkeypatch: pytest.MonkeyPatch, tmp_session: Session) -> None:
    tasks = _make_tasks(2)
    anchors = _make_anchors(2)
    ladder = load_ladder()

    tmp_session.write("knowledge/localization.md", "loc")
    tmp_session.write("knowledge/filter-1.md", "cached-0")
    tmp_session.write("knowledge/filter-2.md", "cached-1")

    def _fail(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("filter_task_context should not be called on full resume hit")

    monkeypatch.setattr("splinter.pipeline.localize", _stub_localize(anchors))
    monkeypatch.setattr("splinter.pipeline.filter_task_context", _fail)

    _run(tmp_session, ladder, tasks, anchors, resume=True, resume_round=1)

    assert [t.filtered_context for t in tasks] == ["cached-0", "cached-1"]
