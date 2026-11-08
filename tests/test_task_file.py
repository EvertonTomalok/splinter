from __future__ import annotations

from pathlib import Path

import pytest

from splinter.pipeline import _load_task_from_yaml
from splinter.task_file import read_task_mapping, split_markdown_task

_TASK_MD = """# T02 — Run screen shell

Build the shell.

## Rules

- No new dependency.

## Acceptance

- [ ] Header shows the story count.
- [ ] `q` quits.

## Tests

- Pilot test for the header.
"""


def test_split_markdown_task_takes_only_the_acceptance_section() -> None:
    description, acceptance = split_markdown_task(_TASK_MD)
    assert acceptance == "- [ ] Header shows the story count.\n- [ ] `q` quits."
    assert "## Rules" in description and "## Tests" in description
    assert "Header shows the story count" not in description


def test_split_markdown_task_without_acceptance_is_all_description() -> None:
    assert split_markdown_task("# T09\n\nJust do it.\n") == ("# T09\n\nJust do it.", "")


def test_split_markdown_task_accepts_acceptance_criteria_heading() -> None:
    _, acceptance = split_markdown_task("# T\n\nbody\n\n## Acceptance Criteria\n- ok\n")
    assert acceptance == "- ok"


def test_markdown_task_names_its_file_and_plan_index(tmp_path: Path) -> None:
    plan = tmp_path / "tasks" / "c2"
    plan.mkdir(parents=True)
    (plan / "README.md").write_text("# plan\n")
    task = plan / "T02-run-shell.md"
    task.write_text(_TASK_MD)
    data = read_task_mapping(task)
    assert data["description"].startswith(f"Task file: `{task}`\nPlan index")
    assert f"`{plan / 'README.md'}`" in data["description"]
    assert data["acceptance"].startswith("- [ ] Header")


def test_markdown_task_without_index_names_only_itself(tmp_path: Path) -> None:
    task = tmp_path / "T01.md"
    task.write_text("# T01\n\nbody\n")
    data = read_task_mapping(task)
    assert data["description"] == f"Task file: `{task}`\n\n# T01\n\nbody"
    assert data["acceptance"] == ""


def test_empty_markdown_task_is_rejected(tmp_path: Path) -> None:
    task = tmp_path / "T01.md"
    task.write_text("## Acceptance\n- ok\n")
    with pytest.raises(ValueError, match="empty"):
        read_task_mapping(task)


def test_yaml_task_must_be_a_mapping(tmp_path: Path) -> None:
    task = tmp_path / "task.yaml"
    task.write_text("- a\n- b\n")
    with pytest.raises(ValueError, match="mapping"):
        read_task_mapping(task)


def test_pipeline_loads_a_markdown_task(tmp_path: Path) -> None:
    task = tmp_path / "T02-run-shell.md"
    task.write_text(_TASK_MD)
    loaded = _load_task_from_yaml(str(task))
    assert loaded.acceptance.startswith("- [ ] Header")
    assert "Build the shell." in loaded.description
    assert loaded.effort == "normal"
