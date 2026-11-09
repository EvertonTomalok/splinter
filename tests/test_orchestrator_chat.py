from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from splinter.orchestrator_chat import ask_orchestrator

_STORIES = (
    SimpleNamespace(number=2, title="US-002: tokens", state="fail", tier="T1"),
    SimpleNamespace(number=4, title="US-020: goals", state="live", tier="T1"),
)
_LADDER: Any = SimpleNamespace(planner_model="opus", planner_effort="high")


def _reply(monkeypatch: pytest.MonkeyPatch, text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake(prompt: str, model: str, **kwargs: Any) -> str:
        calls.append({"prompt": prompt, "model": model, **kwargs})
        return text

    monkeypatch.setattr("splinter.orchestrator_chat.run_text", fake)
    return calls


def test_the_planner_model_reads_the_note_and_the_stories(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _reply(monkeypatch, '{"reply": "ok", "actions": []}')

    ask_orchestrator("run them yourself", _STORIES, _LADDER)

    assert calls[0]["model"] == "opus" and calls[0]["role"] == "orchestrator"
    assert "run them yourself" in calls[0]["prompt"] and "US-020: goals" in calls[0]["prompt"]


def test_the_answer_carries_its_reply_and_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    text = '{"reply": "on it", "actions": [{"story": 2, "action": "orchestrator"}]}'
    _reply(monkeypatch, f"sure\n{text}")

    answer = ask_orchestrator("run 2", _STORIES, _LADDER)

    assert answer.reply == "on it" and answer.actions == ((2, "orchestrator"),)


def test_unknown_stories_and_actions_are_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    items = '[{"story": 99, "action": "tier_up"}, {"story": 2, "action": "delete"}, "x"]'
    _reply(monkeypatch, f'{{"reply": "no", "actions": {items}}}')

    assert ask_orchestrator("x", _STORIES, _LADDER).actions == ()


def test_plain_text_is_the_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    _reply(monkeypatch, "no json here")

    answer = ask_orchestrator("x", _STORIES, _LADDER)

    assert answer.reply == "no json here" and answer.actions == ()


def test_a_crash_is_reported_as_the_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> str:
        raise RuntimeError("provider down")

    monkeypatch.setattr("splinter.orchestrator_chat.run_text", boom)

    answer = ask_orchestrator("x", _STORIES, _LADDER)

    assert "provider down" in answer.reply and answer.actions == ()
