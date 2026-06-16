"""Tests for the cursor CLI provider adapter."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from splinter.providers import cursor as cursor_module
from splinter.providers.cursor import CursorProvider
from splinter.providers.cursor import list_models as _real_list_models


def _fake_proc(stdout: str, returncode: int = 0) -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")


def _capture(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    captured: dict[str, object] = {}

    def fake_subprocess(
        cmd: list[str], timeout: int = 0, cwd: str = ".", on_line: object = None
    ) -> object:
        captured["cmd"] = list(cmd)
        captured["on_line"] = on_line
        return _fake_proc("ok")

    monkeypatch.setattr(cursor_module, "run_subprocess", fake_subprocess)
    return captured


def test_run_cmd_basic(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture(monkeypatch)
    cursor_module.run("hello", timeout=60)
    cmd = captured["cmd"]
    assert cmd[0] == "agent"
    assert "-p" in cmd
    assert "--trust" in cmd
    assert "--output-format" in cmd
    assert cmd[cmd.index("--output-format") + 1] == "stream-json"
    assert "--" in cmd
    assert cmd[-1] == "hello"


def test_run_passes_model_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """cursor/ prefix stripped; --model passed with bare id."""
    captured = _capture(monkeypatch)
    cursor_module.run("do it", model="cursor/gpt-5.3-codex")
    cmd = captured["cmd"]
    assert "--model" in cmd
    idx = cmd.index("--model")
    assert cmd[idx + 1] == "gpt-5.3-codex"


def test_run_omits_model_flag_for_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    """cursor/auto should not add --model (Cursor picks its own default)."""
    captured = _capture(monkeypatch)
    cursor_module.run("do it", model="cursor/auto")
    assert "--model" not in captured["cmd"]


def test_run_omits_model_flag_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture(monkeypatch)
    cursor_module.run("do it", model=None)
    assert "--model" not in captured["cmd"]


def test_run_passes_resume_session(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture(monkeypatch)
    cursor_module.run("do it", session="sid-123")
    cmd = captured["cmd"]
    assert "--resume" in cmd
    assert cmd[cmd.index("--resume") + 1] == "sid-123"


def test_provider_run_passes_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """CursorProvider.run() forwards model to the module-level run()."""
    captured = _capture(monkeypatch)
    provider = CursorProvider()
    provider.run("prompt", "cursor/claude-opus-4-8-high")
    cmd = captured["cmd"]
    assert "--model" in cmd
    assert cmd[cmd.index("--model") + 1] == "claude-opus-4-8-high"


def test_provider_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    """provider_for('cursor/X') resolves to the CursorProvider."""
    from splinter.models.roster import provider_for
    from splinter.providers.dispatch import get_provider

    assert provider_for("cursor/gpt-5.3-codex") == "cursor"
    p = get_provider("cursor")
    assert isinstance(p, CursorProvider)


def test_run_raises_on_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_subprocess(
        cmd: list[str], timeout: int = 0, cwd: str = ".", on_line: object = None
    ) -> object:
        return _fake_proc("", returncode=1)

    monkeypatch.setattr(cursor_module, "run_subprocess", fake_subprocess)
    with pytest.raises(RuntimeError, match="agent exited 1"):
        cursor_module.run("fail")


def test_run_streams_lines_to_live_logger(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    captured: dict[str, object] = {}

    def fake_subprocess(
        cmd: list[str], timeout: int = 0, cwd: str = ".", on_line: object = None
    ) -> object:
        captured["on_line"] = on_line
        if callable(on_line):
            on_line("live cursor line")
        return _fake_proc("ok")

    monkeypatch.setattr(cursor_module, "run_subprocess", fake_subprocess)

    with caplog.at_level(logging.INFO, logger="splinter.live"):
        cursor_module.run("do it")

    assert captured["on_line"] is cursor_module._stream_cursor_line
    assert any("live cursor line" in rec.message for rec in caplog.records)


def test_stream_cursor_tool_call_logs_summary(caplog: pytest.LogCaptureFixture) -> None:
    line = (
        '{"type":"tool_call","subtype":"started","tool_call":'
        '{"shellToolCall":{"args":{"description":"Run tests","command":"pytest -q"}}}}'
    )
    with caplog.at_level(logging.INFO, logger="splinter.live"):
        cursor_module._stream_cursor_line(line)
    assert any("🔧 shell [started] Run tests" in rec.message for rec in caplog.records)


def test_run_parses_stream_json_tokens_and_session(monkeypatch: pytest.MonkeyPatch) -> None:
    usage = (
        '"usage":{"inputTokens":1000,"outputTokens":200,'
        '"cacheReadTokens":50,"cacheWriteTokens":10}'
    )
    stdout = (
        '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Draft"}]}}\n'
        f'{{"type":"result","result":"Final answer","session_id":"sid-abc",{usage}}}\n'
    )

    def fake_subprocess(
        cmd: list[str], timeout: int = 0, cwd: str = ".", on_line: object = None
    ) -> object:
        return _fake_proc(stdout)

    monkeypatch.setattr(cursor_module, "run_subprocess", fake_subprocess)
    result = cursor_module.run("do it", model="cursor/composer-2.5")

    assert result.text == "Final answer"
    assert result.session_id == "sid-abc"
    assert result.tokens["input"] == 1000
    assert result.tokens["output"] == 200
    assert result.cost > 0


def test_list_models_parses_output(monkeypatch: pytest.MonkeyPatch) -> None:
    sample = (
        "Available models\n\n"
        "auto - Auto\n"
        "gpt-5.3-codex - Codex 5.3\n"
        "claude-opus-4-8-high - Opus 4.8 1M\n"
    )

    def fake_run(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(stdout=sample, returncode=0, stderr="")

    monkeypatch.setattr(cursor_module.subprocess, "run", fake_run)
    models = _real_list_models()
    assert "cursor/auto" in models
    assert "cursor/gpt-5.3-codex" in models
    assert "cursor/claude-opus-4-8-high" in models
    assert all(m.startswith("cursor/") for m in models)


def test_fetch_pricing_inline_and_family(monkeypatch: pytest.MonkeyPatch) -> None:
    sample = (
        "auto - Auto · $3 in / $15 out\n"
        "claude-opus-4-8-thinking-high - Opus 4.8 1M Thinking\n"
        "gpt-5.3-codex-xhigh - Codex 5.3 Extra High\n"
    )

    def fake_run(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(stdout=sample, returncode=0, stderr="")

    monkeypatch.setattr(cursor_module.subprocess, "run", fake_run)
    monkeypatch.setattr("splinter.models.public_pricing.fetch_public_pricing", lambda: {})
    prices = cursor_module.fetch_pricing()
    # Inline rate parsed from the description.
    assert prices["cursor/auto"].input == pytest.approx(3.0)
    # Variants priced from the family table (no inline rate in the listing).
    assert prices["cursor/claude-opus-4-8-thinking-high"].input == pytest.approx(5.0)
    assert prices["cursor/gpt-5.3-codex-xhigh"].output == pytest.approx(40.0)


def test_fetch_pricing_cli_failure_still_prices_auto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(stdout="", returncode=1, stderr="401 Unauthorized")

    monkeypatch.setattr(cursor_module.subprocess, "run", fake_run)
    monkeypatch.setattr("splinter.models.public_pricing.fetch_public_pricing", lambda: {})
    prices = cursor_module.fetch_pricing()
    assert len(prices) > 0
    assert "cursor/auto" in prices


def test_cursor_provider_supports_pricing() -> None:
    provider = CursorProvider()
    assert provider.supports_pricing is True
    assert hasattr(provider, "fetch_pricing")
