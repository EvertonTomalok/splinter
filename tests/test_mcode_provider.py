"""Tests for the mcode CLI provider adapter."""

from __future__ import annotations

from typing import Any

import pytest

from splinter import procreg
from splinter.providers import mcode as mcode_module
from splinter.providers.base import ProviderGapError, ProviderResponse
from splinter.providers.mcode import McodeProvider, _strip_prefix

_SAMPLE_NDJSON = (
    '{"type":"system","session_id":"sess-123"}\n'
    '{"type":"assistant","message":{"content":[{"type":"text","text":"hello world"}]}}\n'
    '{"type":"result","usage":{"input_tokens":120,"output_tokens":34},"cost":0.0123}\n'
)


def _install_fake(
    monkeypatch: pytest.MonkeyPatch,
    stdout: str = _SAMPLE_NDJSON,
    returncode: int = 0,
    stderr: str = "",
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_subprocess(
        cmd: list[str],
        *,
        timeout: int,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        idle_timeout: float | None = None,
        on_line: Any = None,
    ) -> procreg.CompletedProcess:
        captured["cmd"] = cmd
        captured["timeout"] = timeout
        captured["cwd"] = cwd
        captured["env"] = env
        captured["idle_timeout"] = idle_timeout
        captured["on_line"] = on_line
        if on_line is not None:
            for line in stdout.splitlines():
                on_line(line)
        return procreg.CompletedProcess(returncode=returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(mcode_module, "run_subprocess", fake_subprocess)
    return captured


# ── argv construction ────────────────────────────────────────────────────────


def test_strip_prefix_removes_mcode_prefix() -> None:
    assert _strip_prefix("mcode/minimax-m3") == "minimax-m3"


def test_strip_prefix_noop_without_prefix() -> None:
    assert _strip_prefix("minimax-m3") == "minimax-m3"
    assert _strip_prefix("codex/gpt-5-codex") == "codex/gpt-5-codex"


def test_run_cmd_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fake(monkeypatch)
    mcode_module.run("do the task", "mcode/minimax-m3", timeout=60)

    assert captured["cmd"] == [
        "mcode",
        "exec",
        "--cwd",
        ".",
        "--model",
        "minimax-m3",
        "--permission",
        "smart",
        "--timeout",
        "30m",
        "--output-format",
        "stream-json",
        "do the task",
    ]
    assert captured["timeout"] == 60
    assert "mcode/minimax-m3" not in captured["cmd"]


def test_run_continue_session_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fake(monkeypatch)
    mcode_module.run("continue this", "minimax-m3", session="sess-9", timeout=30)

    cmd = captured["cmd"]
    idx = cmd.index("--continue")
    assert cmd[idx + 1] == "sess-9"
    assert cmd[-1] == "continue this"


def test_run_file_pairs_order(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fake(monkeypatch)
    mcode_module.run(
        "go", "minimax-m3", timeout=30, files=("brief.md", "contract.md", "extra.md")
    )

    cmd = captured["cmd"]
    pairs = [cmd[i + 1] for i, flag in enumerate(cmd) if flag == "--file"]
    assert pairs == ["brief.md", "contract.md", "extra.md"]
    file_idx = cmd.index("--file")
    assert cmd[file_idx : file_idx + 6] == [
        "--file",
        "brief.md",
        "--file",
        "contract.md",
        "--file",
        "extra.md",
    ]
    assert cmd[-1] == "go"


def test_run_forwards_cwd_env_idle_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fake(monkeypatch)
    env = {"PATH": "/shims:/usr/bin"}
    mcode_module.run(
        "hi",
        "minimax-m3",
        timeout=45,
        cwd="/tmp/worktree",
        env=env,
        idle_timeout=300.0,
    )

    assert captured["cwd"] == "/tmp/worktree"
    assert captured["env"] == env
    assert captured["idle_timeout"] == 300.0
    cmd = captured["cmd"]
    idx = cmd.index("--cwd")
    assert cmd[idx + 1] == "/tmp/worktree"
    assert captured["on_line"] is mcode_module._stream_event


def test_run_default_timeout_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fake(monkeypatch)
    monkeypatch.setattr("splinter.configure.configured_timeout", lambda: 90)
    mcode_module.run("hi", "minimax-m3")
    assert captured["timeout"] == 90


def test_run_stream_callback_tolerates_garbage(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = "not json\n{broken\n" + _SAMPLE_NDJSON
    _install_fake(monkeypatch, stdout=stdout)
    result = mcode_module.run("hi", "minimax-m3", timeout=30)
    assert result.text == "hello world"


# ── NDJSON parsing ───────────────────────────────────────────────────────────


def test_parse_text_flat_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = '{"type":"text","text":"flat hello"}\n'
    _install_fake(monkeypatch, stdout=stdout)
    result = mcode_module.run("hi", "minimax-m3", timeout=30)
    assert result.text == "flat hello"


def test_parse_text_nested_and_flat_shapes_combined(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = (
        '{"type":"assistant","message":{"content":[{"type":"text","text":"nested"},'
        '{"type":"text","text":"blocks"}]}}\n'
        '{"type":"text","text":"flat"}\n'
    )
    _install_fake(monkeypatch, stdout=stdout)
    result = mcode_module.run("hi", "minimax-m3", timeout=30)
    assert result.text == "nested\nblocks\nflat"


def test_parse_tokens_alt_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = '{"type":"result","usage":{"input":7,"output":5}}\n'
    _install_fake(monkeypatch, stdout=stdout)
    result = mcode_module.run("hi", "minimax-m3", timeout=30)
    assert result.tokens == {"input": 7, "output": 5}


def test_parse_tokens_nested_message_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = (
        '{"type":"assistant","message":{"usage":{"input_tokens":9,"output_tokens":8},'
        '"content":[{"type":"text","text":"nested usage"}]}}\n'
    )
    _install_fake(monkeypatch, stdout=stdout)
    result = mcode_module.run("hi", "minimax-m3", timeout=30)
    assert result.tokens == {"input": 9, "output": 8}
    assert result.text == "nested usage"


def test_parse_tokens_and_cost_from_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(monkeypatch)
    result = mcode_module.run("hi", "minimax-m3", timeout=30)
    assert result.tokens == {"input": 120, "output": 34}
    assert result.cost == pytest.approx(0.0123)
    assert result.cost_indeterminate is False
    assert result.session_id == "sess-123"


def test_parse_cost_from_total_cost_usd(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = '{"type":"result","total_cost_usd":0.75}\n'
    _install_fake(monkeypatch, stdout=stdout)
    result = mcode_module.run("hi", "minimax-m3", timeout=30)
    assert result.cost == pytest.approx(0.75)
    assert result.cost_indeterminate is False


def test_parse_cost_from_usage_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = '{"type":"result","usage":{"input_tokens":1,"output_tokens":1,"cost":0.5}}\n'
    _install_fake(monkeypatch, stdout=stdout)
    result = mcode_module.run("hi", "minimax-m3", timeout=30)
    assert result.cost == pytest.approx(0.5)
    assert result.cost_indeterminate is False


def test_parse_cost_indeterminate_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = '{"type":"result","usage":{"input_tokens":3,"output_tokens":2}}\n'
    _install_fake(monkeypatch, stdout=stdout)
    result = mcode_module.run("hi", "minimax-m3", timeout=30)
    assert result.cost == 0.0
    assert result.cost_indeterminate is True


# ── failure paths ────────────────────────────────────────────────────────────


def test_run_nonzero_exit_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(monkeypatch, returncode=1, stderr="auth error")
    with pytest.raises(RuntimeError, match="mcode exited 1: auth error"):
        mcode_module.run("hi", "minimax-m3", timeout=30)


def test_run_error_event_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = '{"type":"error","message":"model stream ended"}\n'
    _install_fake(monkeypatch, stdout=stdout)
    with pytest.raises(RuntimeError, match="model stream ended"):
        mcode_module.run("hi", "minimax-m3", timeout=30)


def test_run_result_error_event_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = '{"type":"result_error","error":"bad payload"}\n'
    _install_fake(monkeypatch, stdout=stdout)
    with pytest.raises(RuntimeError, match="bad payload"):
        mcode_module.run("hi", "minimax-m3", timeout=30)


def test_run_is_error_flag_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = '{"type":"result","is_error":true,"text":"failed hard"}\n'
    _install_fake(monkeypatch, stdout=stdout)
    with pytest.raises(RuntimeError, match="failed hard"):
        mcode_module.run("hi", "minimax-m3", timeout=30)


# ── McodeProvider ────────────────────────────────────────────────────────────


def test_provider_returns_provider_response(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(monkeypatch)
    resp = McodeProvider().run("hello", "mcode/minimax-m3", timeout=30)

    assert isinstance(resp, ProviderResponse)
    assert resp.text == "hello world"
    assert resp.session_id == "sess-123"
    assert resp.tokens == {"input": 120, "output": 34}
    assert resp.cost == pytest.approx(0.0123)
    assert resp.cost_indeterminate is False


def test_provider_gap_on_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(monkeypatch, returncode=1, stderr="429 rate limit exceeded")
    with pytest.raises(ProviderGapError) as exc_info:
        McodeProvider().run("hi", "mcode/minimax-m3", timeout=30)
    assert exc_info.value.kind == "rate_limit"
    assert exc_info.value.provider == "mcode"
    assert exc_info.value.model == "mcode/minimax-m3"


def test_provider_gap_on_error_event_balance(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = '{"type":"error","message":"insufficient balance, please top up"}\n'
    _install_fake(monkeypatch, stdout=stdout)
    with pytest.raises(ProviderGapError) as exc_info:
        McodeProvider().run("hi", "mcode/minimax-m3", timeout=30)
    assert exc_info.value.kind == "insufficient_balance"


def test_provider_plain_error_not_a_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(monkeypatch, returncode=1, stderr="connection refused")
    with pytest.raises(RuntimeError, match="connection refused") as exc_info:
        McodeProvider().run("hi", "mcode/minimax-m3", timeout=30)
    assert not isinstance(exc_info.value, ProviderGapError)


def test_provider_passes_session_as_continue(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fake(monkeypatch)
    McodeProvider().run("continue", "minimax-m3", session="sess-9", timeout=30)
    cmd = captured["cmd"]
    assert cmd[cmd.index("--continue") + 1] == "sess-9"
