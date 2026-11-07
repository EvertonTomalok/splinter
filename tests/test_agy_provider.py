"""Tests for the agy CLI provider adapter."""

from __future__ import annotations

from typing import Any

import pytest

from splinter import procreg
from splinter.providers import agy as agy_module
from splinter.providers.agy import AgyProvider, _final_model, _strip_prefix
from splinter.providers.base import ProviderGapError, ProviderResponse

_SAMPLE_NDJSON = (
    '{"type":"assistant","message":{"content":[{"type":"text","text":"agy says hi"}]}}\n'
    '{"type":"result","usage":{"input_tokens":10,"output_tokens":4},"total_cost_usd":0.002}\n'
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

    monkeypatch.setattr(agy_module, "run_subprocess", fake_subprocess)
    return captured


# ── model effort suffix ──────────────────────────────────────────────────────


def test_strip_prefix_removes_agy_prefix() -> None:
    assert _strip_prefix("agy/gemini-3.8-flash") == "gemini-3.8-flash"
    assert _strip_prefix("gemini-3.8-flash") == "gemini-3.8-flash"


@pytest.mark.parametrize(
    "model,variant,expected",
    [
        ("gemini-3.8-flash", None, "gemini-3.8-flash-low"),
        ("gemini-3.8-flash", "low", "gemini-3.8-flash-low"),
        ("gemini-3.8-flash", "medium", "gemini-3.8-flash-medium"),
        ("gemini-3.8-flash", "high", "gemini-3.8-flash-high"),
        ("gemini-3.8-flash", "max", "gemini-3.8-flash-low"),
        ("gemini-3.8-flash", "auto", "gemini-3.8-flash-low"),
        ("agy/gemini-3.8-flash", "high", "gemini-3.8-flash-high"),
        ("gemini-3.8-flash-high", None, "gemini-3.8-flash-low"),
        ("gemini-3.8-flash-low", "high", "gemini-3.8-flash-high"),
        ("gemini-3.8-flash-medium", "medium", "gemini-3.8-flash-medium"),
        ("", None, "gemini-3.8-flash-low"),
        ("agy/", "high", "gemini-3.8-flash-high"),
    ],
)
def test_final_model_suffix(model: str, variant: str | None, expected: str) -> None:
    assert _final_model(model, variant) == expected


def test_run_model_flag_default_low(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fake(monkeypatch)
    agy_module.run("edit files", "gemini-3.8-flash", timeout=60)

    cmd = captured["cmd"]
    assert cmd == [
        "agy",
        "-p",
        "edit files",
        "--model",
        "gemini-3.8-flash-low",
        "--mode",
        "accept-edits",
        "--sandbox",
        "--output-format",
        "stream-json",
    ]
    assert captured["timeout"] == 60


def test_run_model_flag_high_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fake(monkeypatch)
    agy_module.run("edit files", "agy/gemini-3.8-flash", variant="high", timeout=30)

    cmd = captured["cmd"]
    idx = cmd.index("--model")
    assert cmd[idx + 1] == "gemini-3.8-flash-high"


def test_run_model_flag_replaces_existing_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fake(monkeypatch)
    agy_module.run("edit files", "gemini-3.8-flash-high", variant="low", timeout=30)

    cmd = captured["cmd"]
    idx = cmd.index("--model")
    assert cmd[idx + 1] == "gemini-3.8-flash-low"


# ── argv construction ────────────────────────────────────────────────────────


def test_run_add_dirs_flags_order(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fake(monkeypatch)
    agy_module.run("go", "gemini-3.8-flash", timeout=30, add_dirs=("/wt/a", "/wt/b"))

    cmd = captured["cmd"]
    tail = cmd[cmd.index("--output-format") :]
    assert tail == [
        "--output-format",
        "stream-json",
        "--add-dir",
        "/wt/a",
        "--add-dir",
        "/wt/b",
    ]


def test_run_forwards_cwd_env_idle_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fake(monkeypatch)
    env = {"PATH": "/shims:/usr/bin"}
    agy_module.run(
        "hi",
        "gemini-3.8-flash",
        timeout=45,
        cwd="/tmp/worktree",
        env=env,
        idle_timeout=120.0,
    )

    assert captured["cwd"] == "/tmp/worktree"
    assert captured["env"] == env
    assert captured["idle_timeout"] == 120.0
    assert captured["on_line"] is agy_module._stream_event


def test_run_default_timeout_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fake(monkeypatch)
    monkeypatch.setattr("splinter.configure.configured_timeout", lambda: 90)
    agy_module.run("hi", "gemini-3.8-flash")
    assert captured["timeout"] == 90


def test_run_session_not_in_argv_but_echoed(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fake(monkeypatch)
    result = agy_module.run("continue", "gemini-3.8-flash", timeout=30, session="sess-7")

    cmd = captured["cmd"]
    assert "sess-7" not in cmd
    assert "--continue" not in cmd
    assert result.session_id == "sess-7"


# ── NDJSON parsing ───────────────────────────────────────────────────────────


def test_parse_sample_text_tokens_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(monkeypatch)
    result = agy_module.run("hi", "gemini-3.8-flash", timeout=30)

    assert result.text == "agy says hi"
    assert result.tokens == {"input": 10, "output": 4}
    assert result.cost == pytest.approx(0.002)
    assert result.cost_indeterminate is False


def test_parse_text_flat_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = '{"type":"text","text":"flat hello"}\n'
    _install_fake(monkeypatch, stdout=stdout)
    result = agy_module.run("hi", "gemini-3.8-flash", timeout=30)
    assert result.text == "flat hello"


def test_parse_cost_from_usage_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = '{"type":"result","usage":{"input_tokens":2,"output_tokens":1,"cost":0.5}}\n'
    _install_fake(monkeypatch, stdout=stdout)
    result = agy_module.run("hi", "gemini-3.8-flash", timeout=30)
    assert result.cost == pytest.approx(0.5)
    assert result.cost_indeterminate is False


def test_parse_cost_indeterminate_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = '{"type":"result","usage":{"input_tokens":3,"output_tokens":2}}\n'
    _install_fake(monkeypatch, stdout=stdout)
    result = agy_module.run("hi", "gemini-3.8-flash", timeout=30)
    assert result.cost == 0.0
    assert result.cost_indeterminate is True


def test_parse_stream_callback_tolerates_garbage(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = "not json\n{broken\n" + _SAMPLE_NDJSON
    _install_fake(monkeypatch, stdout=stdout)
    result = agy_module.run("hi", "gemini-3.8-flash", timeout=30)
    assert result.text == "agy says hi"


# ── failure paths ────────────────────────────────────────────────────────────


def test_run_nonzero_exit_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(monkeypatch, returncode=1, stderr="sandbox failure")
    with pytest.raises(RuntimeError, match="agy exited 1: sandbox failure"):
        agy_module.run("hi", "gemini-3.8-flash", timeout=30)


def test_run_error_event_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = '{"type":"error","message":"quota exceeded for today"}\n'
    _install_fake(monkeypatch, stdout=stdout)
    with pytest.raises(RuntimeError, match="quota exceeded for today"):
        agy_module.run("hi", "gemini-3.8-flash", timeout=30)


def test_run_is_error_flag_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = '{"type":"result","is_error":true,"text":"edit rejected"}\n'
    _install_fake(monkeypatch, stdout=stdout)
    with pytest.raises(RuntimeError, match="edit rejected"):
        agy_module.run("hi", "gemini-3.8-flash", timeout=30)


# ── AgyProvider ──────────────────────────────────────────────────────────────


def test_provider_returns_provider_response(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(monkeypatch)
    resp = AgyProvider().run("hello", "agy/gemini-3.8-flash", variant="high", timeout=30)

    assert isinstance(resp, ProviderResponse)
    assert resp.text == "agy says hi"
    assert resp.tokens == {"input": 10, "output": 4}
    assert resp.cost == pytest.approx(0.002)
    assert resp.cost_indeterminate is False


def test_provider_maps_variant_to_model_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fake(monkeypatch)
    AgyProvider().run("hello", "gemini-3.8-flash", variant="medium", timeout=30)
    cmd = captured["cmd"]
    idx = cmd.index("--model")
    assert cmd[idx + 1] == "gemini-3.8-flash-medium"


def test_provider_gap_on_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(monkeypatch, returncode=1, stderr="429 rate limit exceeded")
    with pytest.raises(ProviderGapError) as exc_info:
        AgyProvider().run("hi", "agy/gemini-3.8-flash", timeout=30)
    assert exc_info.value.kind == "rate_limit"
    assert exc_info.value.provider == "agy"
    assert exc_info.value.model == "agy/gemini-3.8-flash"


def test_provider_gap_on_insufficient_balance(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = '{"type":"error","message":"insufficient balance"}\n'
    _install_fake(monkeypatch, stdout=stdout)
    with pytest.raises(ProviderGapError) as exc_info:
        AgyProvider().run("hi", "agy/gemini-3.8-flash", timeout=30)
    assert exc_info.value.kind == "insufficient_balance"


def test_provider_plain_error_not_a_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake(monkeypatch, returncode=1, stderr="connection refused")
    with pytest.raises(RuntimeError, match="connection refused") as exc_info:
        AgyProvider().run("hi", "agy/gemini-3.8-flash", timeout=30)
    assert not isinstance(exc_info.value, ProviderGapError)
