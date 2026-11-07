"""Plan mode: pure planner-conversation logic for ``splinter run --task``."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from splinter import plan_session
from splinter.agents.runner import Task
from splinter.memory.session import Session


class _Ladder:
    planner_model = "sonnet"
    planner_effort = "high"
    planner_timeout = 1234
    prd_model = "opus"

    def step_pool(self, step: str) -> list[str]:
        return [f"{step}-a", f"{step}-b"]


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []

    def fake_run(prompt: str, model: str, **kw: Any) -> tuple[object, str]:
        captured.append({"prompt": prompt, "model": model, **kw})
        result = SimpleNamespace(text="reply", tokens={"input": 3, "output": 4}, cost=0.5)
        return result, "sid-1"

    monkeypatch.setattr(plan_session, "load_ladder", lambda: _Ladder())
    monkeypatch.setattr(plan_session, "run_provider_session", fake_run)
    return captured


def test_open_plan_questions_uses_planner_step(calls: list[dict[str, Any]]) -> None:
    turn = plan_session.open_plan_questions("## Task\nAdd a flag", localization="a.py:1")
    call = calls[0]
    assert call["model"] == "sonnet"
    assert call["variant"] == "high"
    assert call["timeout"] == 1234
    assert call["role"] == "plan"
    assert call["pool"] == ["planner-a", "planner-b"]
    assert call["output_format"] == "json"
    assert call["session"] is None
    assert turn.text == "reply" and turn.session_id == "sid-1" and turn.model == "sonnet"
    assert turn.tokens == {"input": 3, "output": 4} and turn.cost == 0.5


def test_open_plan_questions_prompt_rules(calls: list[dict[str, Any]]) -> None:
    plan_session.open_plan_questions("## Task\nAdd a flag", localization="a.py:1 anchor")
    prompt = calls[0]["prompt"]
    assert "Add a flag" in prompt
    assert "a.py:1 anchor" in prompt
    assert "3-8" in prompt
    assert "(A/B/C/D)" in prompt
    assert "never answer the questions yourself" in prompt
    assert "no preamble" in prompt
    for topic in ("approach", "files", "data shapes", "edge cases", "tests", "out of scope"):
        assert topic in prompt


def test_refine_plan_prompt(calls: list[dict[str, Any]]) -> None:
    plan_session.refine_plan("1A, 2C", resume="sid-0", current_task="## Task\nEdited")
    call = calls[0]
    assert call["session"] == "sid-0"
    assert "1A, 2C" in call["prompt"]
    assert "Edited" in call["prompt"]
    assert "## Decisions" in call["prompt"]
    assert "## Open Questions" in call["prompt"]
    assert "None — ready to plan." in call["prompt"]
    assert "RESUMING" not in call["prompt"]


def test_refine_plan_reseeds_when_conversation_lost(calls: list[dict[str, Any]]) -> None:
    plan_session.refine_plan("1A", resume="", current_task="## Task\nKeep me")
    prompt = calls[0]["prompt"]
    assert "RESUMING" in prompt
    assert calls[0]["session"] is None


def test_write_plan_includes_contract_and_standards(
    calls: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(plan_session, "load_standards", lambda: "- Type hints everywhere")
    plan_session.write_plan(resume="sid-0", current_task="## Task\nX", autodecide=False)
    prompt = calls[0]["prompt"]
    assert "- Type hints everywhere" in prompt
    assert "numbered" in prompt.lower()
    assert "files and functions" in prompt
    assert "another LLM" in prompt
    assert "Output ONLY the plan markdown" in prompt
    assert "sensible defaults" not in prompt


def test_write_plan_autodecide(
    calls: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(plan_session, "load_standards", lambda: "")
    plan_session.write_plan(resume="", current_task="## Task\nX", autodecide=True)
    prompt = calls[0]["prompt"]
    assert "sensible defaults" in prompt
    assert "RESUMING" in prompt


def test_revise_plan_prompt(calls: list[dict[str, Any]]) -> None:
    plan_session.revise_plan("drop step 3", resume="", current_plan="1. a\n2. b")
    prompt = calls[0]["prompt"]
    assert "drop step 3" in prompt
    assert "1. a\n2. b" in prompt
    assert "RESUMING" in prompt
    assert "COMPLETE" in prompt


def test_ask_logs_usage_on_session(calls: list[dict[str, Any]]) -> None:
    logged: list[tuple[str, dict[str, int], float]] = []
    fake = SimpleNamespace(log_llm_usage=lambda m, t, c: logged.append((m, t, c)))
    plan_session.open_plan_questions("t", session=fake)
    assert logged == [("sonnet", {"input": 3, "output": 4}, 0.5)]


def test_task_text_and_parse_roundtrip() -> None:
    task = Task(
        description="Add --dry-run flag\nto the CLI", acceptance="- prints plan\n- no writes"
    )
    text = plan_session.task_text(task)
    assert "## Task" in text and "## Acceptance Criteria" in text
    assert plan_session.parse_task_text(text) == (task.description, task.acceptance)


def test_parse_task_text_without_headers_is_all_description() -> None:
    assert plan_session.parse_task_text("just do it") == ("just do it", "")


def _session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Session:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    return Session("ses_plan")


def test_save_plan_writes_both_files_without_doubling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _session(tmp_path, monkeypatch)
    plan_session.save_plan(session, "# Plan\n\n1. Edit cli.py\n")
    expected = "# Plan\n\n1. Edit cli.py\n"
    assert session.read("knowledge/plan.md") == expected
    assert session.read("knowledge/plan-1.md") == expected


def test_save_plan_adds_header(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = _session(tmp_path, monkeypatch)
    plan_session.save_plan(session, "1. Do it")
    assert session.read("knowledge/plan.md") == "# Plan\n\n1. Do it\n"


def test_save_task_roundtrip_keeps_extra_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from splinter.pipeline import _load_task_from_yaml

    session = _session(tmp_path, monkeypatch)
    base = {"description": "old", "acceptance": "old", "effort": "hard", "eval_skill": "rev"}
    path = plan_session.save_task(session, "new desc", "- ok", base)
    assert path == session.dir / "task.yaml"
    data = yaml.safe_load(path.read_text())
    assert data == {"description": "new desc", "acceptance": "- ok", "effort": "hard",
                    "eval_skill": "rev"}
    task = _load_task_from_yaml(str(path))
    assert task.description == "new desc" and task.effort == "hard"
