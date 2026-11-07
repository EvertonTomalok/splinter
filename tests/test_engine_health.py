"""Tests for circuit breakers, engine probes and the probe report."""

from __future__ import annotations

from typing import Any

import pytest

from splinter.engines import health
from splinter.engines.catalog import PROBE_PROMPT, Engine, engine_by_key
from splinter.engines.health import (
    BreakerState,
    CircuitBreaker,
    EngineHealth,
    EngineProbe,
    binary_for,
    is_reviewer,
    probe,
    probe_report,
    reviewer_keys,
)
from splinter.procreg import CompletedProcess


def _engine(key: str) -> Engine:
    engine = engine_by_key(key)
    assert engine is not None
    return engine


class _FakeRunner:
    def __init__(self, returncode: int, output: str) -> None:
        self._returncode = returncode
        self._output = output
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        cmd: list[str],
        *,
        timeout: int,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        idle_timeout: float | None = None,
    ) -> CompletedProcess:
        self.calls.append({"cmd": cmd, "cwd": cwd, "env": env, "timeout": timeout})
        return CompletedProcess(self._returncode, self._output, "")


def _fake_which(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(health.shutil, "which", lambda name: f"/opt/fake/{name}")


# ── CircuitBreaker ───────────────────────────────────────────────────────────


def test_breaker_opens_after_threshold() -> None:
    breaker = CircuitBreaker(failure_threshold=2, cooldown_s=300.0)
    assert breaker.state(now=0.0) is BreakerState.CLOSED
    assert breaker.allow(now=0.0) is True
    breaker.record_failure(now=1.0)
    assert breaker.state(now=1.5) is BreakerState.CLOSED
    assert breaker.allow(now=1.5) is True
    breaker.record_failure(now=2.0)
    assert breaker.state(now=3.0) is BreakerState.OPEN
    assert breaker.allow(now=3.0) is False
    assert breaker.allow(now=250.0) is False


def test_breaker_half_open_allows_one_trial_after_cooldown() -> None:
    breaker = CircuitBreaker(failure_threshold=2, cooldown_s=300.0)
    breaker.record_failure(now=1.0)
    breaker.record_failure(now=2.0)
    assert breaker.state(now=301.9) is BreakerState.OPEN
    assert breaker.allow(now=301.9) is False
    assert breaker.state(now=302.0) is BreakerState.HALF_OPEN
    assert breaker.allow(now=302.0) is True
    breaker.record_failure(now=302.0)
    assert breaker.state(now=302.5) is BreakerState.OPEN
    assert breaker.allow(now=302.5) is False


def test_breaker_success_resets() -> None:
    breaker = CircuitBreaker(failure_threshold=2, cooldown_s=300.0)
    breaker.record_failure(now=1.0)
    breaker.record_failure(now=2.0)
    assert breaker.state(now=3.0) is BreakerState.OPEN
    breaker.record_success()
    assert breaker.state(now=3.0) is BreakerState.CLOSED
    assert breaker.allow(now=3.0) is True


def test_breaker_single_failure_threshold() -> None:
    breaker = CircuitBreaker(failure_threshold=1, cooldown_s=10.0)
    breaker.record_failure(now=0.0)
    assert breaker.state(now=1.0) is BreakerState.OPEN
    assert breaker.state(now=10.0) is BreakerState.HALF_OPEN


# ── EngineHealth ─────────────────────────────────────────────────────────────


def test_engine_health_snapshot() -> None:
    registry = EngineHealth()
    assert registry.snapshot() == {}
    assert registry.allow("mcode") is True
    registry.trip("mcode")
    registry.trip("mcode")
    registry.trip("agy")
    snapshot = registry.snapshot()
    assert snapshot["mcode"] is BreakerState.OPEN
    assert snapshot["agy"] is BreakerState.CLOSED
    registry.reset("mcode")
    assert registry.snapshot()["mcode"] is BreakerState.CLOSED


# ── binary_for / reviewer helpers ────────────────────────────────────────────


def test_binary_for_mapping() -> None:
    assert binary_for(_engine("mcode")) == "mcode"
    assert binary_for(_engine("mcode-flash")) == "mcode"
    assert binary_for(_engine("agy")) == "agy"
    assert binary_for(_engine("sonnet")) == "claude-credit"
    assert binary_for(_engine("claude-plan")) == "claude"
    assert binary_for(_engine("codex")) == "codex"
    assert binary_for(_engine("glm")) == "opencode"
    assert binary_for(_engine("opencode-minimax")) == "opencode"


def test_reviewer_helpers() -> None:
    assert reviewer_keys() == ("reviewer-opus", "reviewer-fable", "reviewer-sonnet")
    assert is_reviewer("reviewer-opus") is True
    assert is_reviewer("mcode") is False


# ── probe ────────────────────────────────────────────────────────────────────


def test_probe_absent_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(health, "binary_for", lambda engine: "splinter-no-such-bin")
    result = probe(_engine("mcode"))
    assert result.up is False
    assert result.binary == ""
    assert result.detail == "absent"


def test_probe_up_on_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_which(monkeypatch)
    runner = _FakeRunner(0, "ok\n")
    monkeypatch.setattr(health, "run_subprocess", runner)
    result = probe(_engine("mcode"))
    assert result.up is True
    assert result.binary == "/opt/fake/mcode"
    assert result.detail == ""
    assert runner.calls[0]["timeout"] == 120


def test_probe_down_without_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_which(monkeypatch)
    monkeypatch.setattr(health, "run_subprocess", _FakeRunner(0, "token expired\n"))
    result = probe(_engine("sonnet"))
    assert result.up is False
    assert result.detail.startswith("token expired")


def test_probe_down_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_which(monkeypatch)
    monkeypatch.setattr(health, "run_subprocess", _FakeRunner(1, ""))
    result = probe(_engine("glm"))
    assert result.up is False
    assert result.detail == "exit status 1"


def test_probe_down_on_error_with_output(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_which(monkeypatch)
    runner = _FakeRunner(2, "quota exhausted\nsecond line\n")
    monkeypatch.setattr(health, "run_subprocess", runner)
    result = probe(_engine("codex"))
    assert result.up is False
    assert result.detail == "quota exhausted exit status 2"


def test_probe_reviewer_is_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_which(monkeypatch)
    monkeypatch.setattr(health, "run_subprocess", _FakeRunner(0, "ok"))
    result = probe(_engine("reviewer-opus"))
    assert result.up is False
    assert result.detail == "unsupported probe provider"


def test_mcode_probe_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_which(monkeypatch)
    runner = _FakeRunner(0, "ok")
    monkeypatch.setattr(health, "run_subprocess", runner)
    assert probe(_engine("mcode")).up is True
    cmd, cwd, env = runner.calls[0]["cmd"], runner.calls[0]["cwd"], runner.calls[0]["env"]
    assert cmd == [
        "mcode",
        "exec",
        "--cwd",
        cwd,
        "--model",
        "minimax/MiniMax-M3",
        "--permission",
        "smart",
        "--timeout",
        "90s",
        "--output-format",
        "text",
        PROBE_PROMPT,
    ]
    assert cwd is not None and cwd != ""
    assert env is None


def test_mcode_flash_probe_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_which(monkeypatch)
    runner = _FakeRunner(0, "ok")
    monkeypatch.setattr(health, "run_subprocess", runner)
    assert probe(_engine("mcode-flash")).up is True
    cmd = runner.calls[0]["cmd"]
    assert cmd[0] == "mcode"
    assert cmd[cmd.index("--model") + 1] == "minimax/MiniMax-M3.1-Flash"


def test_kimi_probe_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_which(monkeypatch)
    runner = _FakeRunner(0, "ok")
    monkeypatch.setattr(health, "run_subprocess", runner)
    assert probe(_engine("kimi")).up is True
    cmd = runner.calls[0]["cmd"]
    assert cmd == [
        "kimi", "-p", PROBE_PROMPT, "--model", _engine("kimi").model,
        "--output-format", "text",
    ]
    assert "--auto" not in cmd


def test_kimi_flash_probe_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_which(monkeypatch)
    runner = _FakeRunner(0, "ok")
    monkeypatch.setattr(health, "run_subprocess", runner)
    assert probe(_engine("kimi-flash")).up is True
    assert runner.calls[0]["cmd"][0] == "kimi"


def test_agy_probe_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_which(monkeypatch)
    runner = _FakeRunner(0, "ok")
    monkeypatch.setattr(health, "run_subprocess", runner)
    assert probe(_engine("agy")).up is True
    cmd, env = runner.calls[0]["cmd"], runner.calls[0]["env"]
    assert cmd == [
        "agy",
        "-p",
        PROBE_PROMPT,
        "--model",
        "gemini-3.8-flash-low",
        "--mode",
        "accept-edits",
        "--print-timeout",
        "90s",
    ]
    assert env is None


def test_sonnet_probe_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_which(monkeypatch)
    runner = _FakeRunner(0, "ok")
    monkeypatch.setattr(health, "run_subprocess", runner)
    assert probe(_engine("sonnet")).up is True
    cmd, env = runner.calls[0]["cmd"], runner.calls[0]["env"]
    assert cmd == [
        "claude-credit",
        "--via",
        "auto",
        PROBE_PROMPT,
        "--model",
        "claude-sonnet-5-5",
        "--effort",
        "low",
    ]
    assert env is None


def test_claude_plan_probe_argv_strips_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "secret-token")
    _fake_which(monkeypatch)
    runner = _FakeRunner(0, "ok")
    monkeypatch.setattr(health, "run_subprocess", runner)
    assert probe(_engine("claude-plan")).up is True
    cmd, env = runner.calls[0]["cmd"], runner.calls[0]["env"]
    assert cmd == [
        "claude",
        "-p",
        PROBE_PROMPT,
        "--model",
        "claude-sonnet-5-5",
        "--effort",
        "low",
    ]
    assert env is not None
    assert "ANTHROPIC_API_KEY" not in env
    assert "ANTHROPIC_AUTH_TOKEN" not in env
    assert env.get("PATH")


def test_codex_probe_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_which(monkeypatch)
    runner = _FakeRunner(0, "ok")
    monkeypatch.setattr(health, "run_subprocess", runner)
    assert probe(_engine("codex")).up is True
    cmd, cwd, env = runner.calls[0]["cmd"], runner.calls[0]["cwd"], runner.calls[0]["env"]
    assert cmd[:-1] == [
        "codex",
        "exec",
        "-m",
        _engine("codex").model.removeprefix("codex/"),
        "--skip-git-repo-check",
        "-s",
        "read-only",
        "-c",
        'model_reasoning_effort="low"',
        "-C",
        cwd,
    ]
    assert cmd[-1] == PROBE_PROMPT
    assert env is None


def test_opencode_probe_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_which(monkeypatch)
    runner = _FakeRunner(0, "ok")
    monkeypatch.setattr(health, "run_subprocess", runner)
    assert probe(_engine("glm")).up is True
    glm_cmd = runner.calls[0]["cmd"]
    assert glm_cmd == [
        "opencode",
        "run",
        "--standalone",
        "-m",
        "zai-coding-plan/glm-5.3-flash",
        PROBE_PROMPT,
    ]
    assert probe(_engine("opencode-minimax")).up is True
    assert runner.calls[1]["cmd"] == [
        "opencode",
        "run",
        "--standalone",
        "-m",
        "minimax-coding-plan/MiniMax-M3",
        PROBE_PROMPT,
    ]


# ── probe_report ─────────────────────────────────────────────────────────────


def test_probe_report_header_and_states() -> None:
    probes = [
        EngineProbe(key="mcode", up=True, binary="mcode"),
        EngineProbe(key="sonnet", up=False, binary="claude-credit", detail="token expired"),
        EngineProbe(key="codex", up=False, binary="codex"),
    ]
    report = probe_report(probes)
    assert "Reviewer availability is inferred from the Sonnet ping" in report
    assert "reviewer models are not pinged." in report
    assert "| Engine | Launch | State |" in report
    assert "| up |" in report
    assert "| down |" in report
    assert "| absent |" in report


def test_probe_report_infers_reviewers_from_sonnet() -> None:
    probes = [EngineProbe(key="sonnet", up=False, binary="claude-credit")]
    report = probe_report(probes)
    assert report.count("| down |") == 4
    for name in ("Opus", "Fable", "Sonnet high"):
        assert f"Final reviewer: {name}" in report


def test_probe_report_reviewers_inherit_up_sonnet() -> None:
    probes = [EngineProbe(key="sonnet", up=True, binary="claude-credit")]
    report = probe_report(probes)
    assert report.count("| up |") == 4


def test_probe_report_codex_model_label() -> None:
    probes = [EngineProbe(key="codex", up=True, binary="codex")]
    assert "GPT gpt-10 (codex)" in probe_report(probes, codex_model="gpt-10")
    assert "GPT default (codex)" in probe_report(probes)


def test_probe_report_lists_all_catalog_rows() -> None:
    report = probe_report([])
    assert report.count("\n") == 17
    assert report.count("| absent |") == 13
