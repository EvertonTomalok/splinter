from __future__ import annotations

from pathlib import Path

import pytest

from splinter.task_plan import is_plan_index, load_plan_index

_README = """# Plan

Orchestrator notes here.

| Task | What | Model | Depends on |
|---|---|---|---|
| [T01](T01-a.md) | first | Sonnet medium | — |
| [T02](T02-b.md) | second | Opus xhigh | T01 |
| M1 | merge wave 1 | — | T01 T02 |
| [T03](T03-c.md) | third | Opus high | T01–T02, M1 |
| [T04](T04-d.md) | docs | Haiku low | all |
"""


def _plan(tmp_path: Path, readme: str = _README) -> Path:
    folder = tmp_path / "tasks" / "plan"
    folder.mkdir(parents=True)
    (folder / "README.md").write_text(readme)
    for name in ("T01-a", "T02-b", "T03-c", "T04-d"):
        body = f"# {name}\n\nDo {name}.\n\n## Acceptance\n- {name} ok\n"
        (folder / f"{name}.md").write_text(body)
    return folder


def test_readme_rows_become_tasks_with_deps_and_effort(tmp_path: Path) -> None:
    folder = _plan(tmp_path)
    index = load_plan_index(folder / "README.md")
    assert [t.id for t in index.tasks] == ["T01", "T02", "T03", "T04"]
    assert [t.deps for t in index.tasks] == [None, ["T01"], ["T01", "T02"], ["T01", "T02", "T03"]]
    assert [t.effort for t in index.tasks] == ["normal", "critical", "hard", "trivial"]
    assert index.text.startswith("# Plan")


def test_task_text_comes_from_its_file(tmp_path: Path) -> None:
    folder = _plan(tmp_path)
    first = load_plan_index(folder).tasks[0]
    assert first.description.startswith("T01: Task file: ")
    assert "Do T01-a." in first.description
    assert first.acceptance == "- T01-a ok"


def test_folder_and_readme_are_plan_indexes_but_a_task_file_is_not(tmp_path: Path) -> None:
    folder = _plan(tmp_path)
    assert is_plan_index(folder)
    assert is_plan_index(folder / "README.md")
    assert not is_plan_index(folder / "T01-a.md")
    assert not is_plan_index(tmp_path)


def test_readme_without_task_links_is_not_a_plan(tmp_path: Path) -> None:
    folder = _plan(tmp_path, "# Notes\n\n| a | b |\n|---|---|\n| x | y |\n")
    assert not is_plan_index(folder)
    with pytest.raises(ValueError, match="no task table"):
        load_plan_index(folder)


def test_missing_task_file_is_reported(tmp_path: Path) -> None:
    folder = _plan(tmp_path)
    (folder / "T03-c.md").unlink()
    with pytest.raises(ValueError, match="T03 links a missing file"):
        load_plan_index(folder)


def test_table_without_model_column_defaults_to_normal(tmp_path: Path) -> None:
    readme = "| Task | What | Depends on |\n|---|---|---|\n| [T01](T01-a.md) | a | — |\n"
    folder = _plan(tmp_path, readme)
    (task,) = load_plan_index(folder).tasks
    assert task.effort == "normal" and task.deps is None


def test_pipeline_loads_every_task_and_keeps_the_readme(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from splinter import pipeline
    from splinter.memory.session import Session

    folder = _plan(tmp_path)
    monkeypatch.chdir(tmp_path)

    def _stop(*args: object, **kwargs: object) -> object:
        raise RuntimeError("stop before running")

    monkeypatch.setattr(pipeline, "get_strategy", _stop)
    session = Session("plan-index-test")
    session.write("knowledge/plan-index.md", "# Task plan: 4 tasks (stale editor bundle)\n")
    with pytest.raises(RuntimeError, match="stop before running"):
        pipeline.run_pipeline(task_path=str(folder / "README.md"), session=session)
    assert session.read("knowledge/plan-index.md") == _README


def test_seed_task_plans_writes_task_text_and_keeps_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from splinter.memory.session import Session
    from splinter.task_plan import seed_task_plans

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path / "home"))
    folder = _plan(tmp_path)
    index = load_plan_index(folder)
    session = Session("seed-test")
    session.write("knowledge/plan-2.md", "# Plan\n\nkept\n")
    session.write("knowledge/plan-T03.md", "# Plan\n\nby id\n")
    assert seed_task_plans(session, index.tasks, index.path) == 2
    first = session.read("knowledge/plan-1.md")
    assert first.startswith("# Plan\n\nT01: Task file: ") and "Do T01-a." in first
    assert f"`{index.path}`" in first
    assert session.read("knowledge/plan-2.md") == "# Plan\n\nkept\n"
    assert session.read("knowledge/plan-3.md") == ""
    assert "Do T04-d." in session.read("knowledge/plan-4.md")


def test_readme_run_skips_localize_and_planner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from splinter import pipeline
    from splinter.memory.session import Session

    folder = _plan(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path / "home"))

    def _no_localize(*args: object, **kwargs: object) -> object:
        raise AssertionError("task mode must not localize")

    def _stop(*args: object, **kwargs: object) -> object:
        raise RuntimeError("stop at the gate")

    monkeypatch.setattr(pipeline, "_localize_and_filter", _no_localize)
    monkeypatch.setattr(pipeline, "_resolve_gate", _stop)
    session = Session("readme-direct")
    with pytest.raises(RuntimeError, match="stop at the gate"):
        pipeline.run_pipeline(task_path=str(folder / "README.md"), session=session)
    assert [session.read(f"knowledge/plan-{n}.md") != "" for n in (1, 2, 3, 4)] == [True] * 4
    assert "Do T03-c." in session.read("knowledge/plan-3.md")
