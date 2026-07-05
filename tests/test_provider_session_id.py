"""US-002: per-task provider session UUID — generated once, reused across turns."""

from __future__ import annotations

import json
import uuid
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from splinter.providers import claude_cli as claude_cli_module
from splinter.providers import codex as codex_module
from splinter.providers import cursor as cursor_module
from splinter.providers import dispatch as dispatch_module
from splinter.providers import opencode as opencode_module
from splinter.providers.base import ModelProvider, ProviderResponse
from splinter.providers.claude_cli import ClaudeProvider
from splinter.providers.codex import CodexProvider
from splinter.providers.cursor import CursorProvider
from splinter.providers.dispatch import run_provider_session
from splinter.providers.opencode import OpencodeProvider

# ── helpers ───────────────────────────────────────────────────────────────────


def _fake_proc(stdout: str, returncode: int = 0) -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")


def _cmd_value_after(cmd: list[str], flag: str) -> str | None:
    if flag not in cmd:
        return None
    idx = cmd.index(flag)
    return cmd[idx + 1] if idx + 1 < len(cmd) else None


def _stream_json_result(*, session_id: str, text: str = "ok") -> str:
    return json.dumps(
        {
            "type": "result",
            "result": text,
            "session_id": session_id,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )


def _patch_claude_subprocess(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Patch claude_cli.run_subprocess to echo back --session-id/--resume as session_id."""
    captured: list[list[str]] = []

    def fake_subprocess(
        cmd: list[str], timeout: int = 0, cwd: object = None, on_line: object = None
    ) -> object:
        captured.append(cmd)
        sid = _cmd_value_after(cmd, "--session-id") or _cmd_value_after(cmd, "--resume") or ""
        return _fake_proc(_stream_json_result(session_id=sid))

    monkeypatch.setattr(claude_cli_module, "run_subprocess", fake_subprocess)
    return captured


# ── first turn: create-with-id ────────────────────────────────────────────────


def test_first_turn_creates_with_session_id(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_claude_subprocess(monkeypatch)
    resp, sid = run_provider_session("do the thing", "sonnet", session=None, timeout=5)

    cmd = captured[0]
    assert "--session-id" in cmd
    assert "--resume" not in cmd
    created = _cmd_value_after(cmd, "--session-id")
    assert created is not None
    uuid.UUID(created)  # raises ValueError if malformed
    assert sid == created
    assert resp.session_id == created


# ── second turn: resume ───────────────────────────────────────────────────────


def test_second_turn_resumes_with_uuid(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_claude_subprocess(monkeypatch)
    existing = str(uuid.uuid4())
    resp, sid = run_provider_session("continue", "sonnet", session=existing, timeout=5)

    cmd = captured[0]
    assert _cmd_value_after(cmd, "--resume") == existing
    assert "--session-id" not in cmd
    assert sid == existing
    assert resp.session_id == existing


# ── same uuid reused across iterations (run_task) ────────────────────────────


class _RecordingProvider(ModelProvider):
    name = "claude"
    supports_session_create = True

    def __init__(self) -> None:
        self.calls: list[dict[str, str | None]] = []

    def run(
        self,
        prompt: str,
        model: str,
        *,
        variant: str | None = None,
        output_format: str = "json",
        session: str | None = None,
        session_id: str | None = None,
        timeout: int | None = None,
        agent: str = "build",
        cwd: str | None = None,
    ) -> ProviderResponse:
        self.calls.append({"session": session, "session_id": session_id})
        sid = session_id or session
        return ProviderResponse(
            text="ok", tokens={"input": 1, "output": 1}, cost=0.01, raw={}, session_id=sid
        )


def test_same_uuid_reused_across_iterations(monkeypatch: pytest.MonkeyPatch) -> None:
    from splinter.agents.runner import Task, run_task

    fake_provider = _RecordingProvider()
    monkeypatch.setattr(dispatch_module, "get_provider", lambda _name: fake_provider)
    monkeypatch.setattr(
        "splinter.agents.runner.resolve_model", lambda t, _lad: ("sonnet", "claude")
    )
    monkeypatch.setattr("splinter.agents.runner.resolve_variant", lambda *a, **kw: "low")
    monkeypatch.setattr("splinter.agents.runner.record_exchange", lambda *a, **kw: None)

    ladder = Mock()
    ladder.tier_timeout.return_value = 600
    task = Task(description="test", acceptance="works", suggested_tier=0)

    result1 = run_task(task, "plan", 0, ladder, iteration=1, task_index=0)
    result2 = run_task(
        task,
        "plan",
        0,
        ladder,
        opencode_session=result1.opencode_session,
        iteration=2,
        task_index=0,
    )

    assert fake_provider.calls[0]["session"] is None
    created = fake_provider.calls[0]["session_id"]
    assert created is not None
    uuid.UUID(created)
    assert result1.opencode_session == created

    assert fake_provider.calls[1]["session"] == created
    assert fake_provider.calls[1]["session_id"] is None
    assert result2.opencode_session == created


# ── session_id absent: single-shot flow unchanged ────────────────────────────


def test_session_id_none_unchanged_single_shot(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_claude_subprocess(monkeypatch)
    ClaudeProvider().run("hello", "sonnet", session=None, session_id=None, timeout=5)

    cmd = captured[0]
    assert "--session-id" not in cmd
    assert "--resume" not in cmd


# ── non-claude providers accept-and-ignore session_id ────────────────────────


def test_non_claude_provider_ignores_session_id(monkeypatch: pytest.MonkeyPatch) -> None:
    oc_captured: list[list[str]] = []

    def fake_oc_subprocess(
        cmd: list[str], timeout: int = 0, cwd: object = None, on_line: object = None
    ) -> object:
        oc_captured.append(cmd)
        return _fake_proc(json.dumps({"text": "ok", "session_id": "oc-sess"}))

    monkeypatch.setattr(opencode_module, "run_subprocess", fake_oc_subprocess)
    oc_resp = OpencodeProvider().run(
        "hi", "opencode-go/minimax-m3", session=None, session_id="ignored", timeout=5
    )
    assert oc_resp.session_id == "oc-sess"
    assert "-s" not in oc_captured[0]

    codex_captured: list[list[str]] = []

    def fake_codex_subprocess(
        cmd: list[str], timeout: int = 0, cwd: object = None, on_line: object = None
    ) -> object:
        codex_captured.append(cmd)
        return _fake_proc(
            '{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}\n'
        )

    monkeypatch.setattr(codex_module, "run_subprocess", fake_codex_subprocess)
    codex_resp = CodexProvider().run(
        "hi", "codex/gpt-5-codex", session=None, session_id="ignored", timeout=5
    )
    assert codex_resp is not None
    assert "resume" not in codex_captured[0]

    cursor_captured: list[list[str]] = []

    def fake_cursor_subprocess(
        cmd: list[str], timeout: int = 0, cwd: object = None, on_line: object = None
    ) -> object:
        cursor_captured.append(cmd)
        return _fake_proc(
            json.dumps({"type": "result", "result": "ok", "session_id": "cur-sess", "usage": {}})
        )

    monkeypatch.setattr(cursor_module, "run_subprocess", fake_cursor_subprocess)
    cursor_resp = CursorProvider().run(
        "hi", "cursor/auto", session=None, session_id="ignored", timeout=5
    )
    assert cursor_resp.session_id == "cur-sess"
    assert "--resume" not in cursor_captured[0]
