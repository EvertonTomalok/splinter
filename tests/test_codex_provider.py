"""Tests for the codex CLI provider adapter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from splinter.providers import codex as codex_module
from splinter.providers.codex import (
    CodexProvider,
    CodexResult,
    _calc_cost,
    _normalize_effort,
    _parse_jsonl,
    _strip_prefix,
)

# ── _parse_jsonl ─────────────────────────────────────────────────────────────

_SAMPLE_JSONL = (
    '{"type":"thread.started","thread_id":"019eb885-0bf2-7be2-b265-81dc3637472b"}\n'
    '{"type":"turn.started"}\n'
    '{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"hello world"}}\n'
    '{"type":"turn.completed","usage":{"input_tokens":100,"cached_input_tokens":50,'
    '"output_tokens":20,"reasoning_output_tokens":5}}\n'
)


def test_parse_jsonl_text_extraction() -> None:
    result = _parse_jsonl(_SAMPLE_JSONL)
    assert result["text"] == "hello world"


def test_parse_jsonl_session_id() -> None:
    result = _parse_jsonl(_SAMPLE_JSONL)
    assert result["session_id"] == "019eb885-0bf2-7be2-b265-81dc3637472b"


def test_parse_jsonl_tokens() -> None:
    result = _parse_jsonl(_SAMPLE_JSONL)
    tokens = result["tokens"]
    assert tokens["input"] == 100
    assert tokens["output"] == 20
    assert tokens["cached_input"] == 50
    assert tokens["reasoning"] == 5


def test_parse_jsonl_multi_message() -> None:
    jsonl = (
        '{"type":"thread.started","thread_id":"abc-123"}\n'
        '{"type":"item.completed","item":{"id":"0","type":"agent_message","text":"line one"}}\n'
        '{"type":"item.completed","item":{"id":"1","type":"agent_message","text":"line two"}}\n'
        '{"type":"turn.completed","usage":{"input_tokens":5,"cached_input_tokens":0,'
        '"output_tokens":3,"reasoning_output_tokens":0}}\n'
    )
    result = _parse_jsonl(jsonl)
    assert result["text"] == "line one\nline two"


def test_parse_jsonl_ignores_non_agent_message_items() -> None:
    jsonl = (
        '{"type":"thread.started","thread_id":"xyz"}\n'
        '{"type":"item.completed","item":{"id":"0","type":"tool_call","text":"ignored"}}\n'
        '{"type":"item.completed","item":{"id":"1","type":"agent_message","text":"kept"}}\n'
        '{"type":"turn.completed","usage":{"input_tokens":1,"cached_input_tokens":0,'
        '"output_tokens":1,"reasoning_output_tokens":0}}\n'
    )
    result = _parse_jsonl(jsonl)
    assert result["text"] == "kept"


def test_parse_jsonl_empty_stdout() -> None:
    result = _parse_jsonl("")
    assert result["text"] == ""
    assert result["session_id"] is None
    assert result["tokens"] == {}


def test_parse_jsonl_invalid_lines_skipped() -> None:
    jsonl = "not json\n{broken\n" + _SAMPLE_JSONL
    result = _parse_jsonl(jsonl)
    assert result["text"] == "hello world"


# ── _calc_cost ───────────────────────────────────────────────────────────────


def test_calc_cost_known_model() -> None:
    tokens = {"input": 1_000_000, "output": 1_000_000}
    cost, indeterminate = _calc_cost("gpt-5-codex", tokens)
    assert cost == pytest.approx(50.0)
    assert indeterminate is False


def test_calc_cost_unknown_model_returns_zero() -> None:
    tokens = {"input": 999999, "output": 999999}
    cost, indeterminate = _calc_cost("unknown-model", tokens)
    assert cost == 0.0
    assert indeterminate is True


def test_calc_cost_zero_tokens() -> None:
    cost, indeterminate = _calc_cost("gpt-5-codex", {"input": 0, "output": 0})
    assert cost == 0.0
    assert indeterminate is False


# ── _normalize_effort ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("low", "low"),
        ("medium", "medium"),
        ("high", "high"),
        ("minimal", "low"),
        ("auto", None),
        ("xhigh", "xhigh"),
        ("max", "xhigh"),
        ("bogus", None),
        (None, None),
    ],
)
def test_normalize_effort(raw: str | None, expected: str | None) -> None:
    assert _normalize_effort(raw) == expected


# ── _strip_prefix ────────────────────────────────────────────────────────────


def test_strip_prefix_removes_codex_prefix() -> None:
    assert _strip_prefix("codex/gpt-5-codex") == "gpt-5-codex"


def test_strip_prefix_noop_without_prefix() -> None:
    assert _strip_prefix("gpt-5-codex") == "gpt-5-codex"


def test_strip_prefix_noop_other_prefix() -> None:
    assert _strip_prefix("opencode/some-model") == "opencode/some-model"


# ── run() subprocess integration ─────────────────────────────────────────────


def _fake_proc(stdout: str, returncode: int = 0) -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")


def test_run_normal_cmd_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_subprocess(cmd: list[str], timeout: int = 0, on_line: object = None) -> object:
        captured["cmd"] = cmd
        return _fake_proc(_SAMPLE_JSONL)

    monkeypatch.setattr(codex_module, "run_subprocess", fake_subprocess)
    codex_module.run("hello", "codex/gpt-5-codex", timeout=60)

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[0] == "codex"
    assert cmd[1] == "exec"
    assert "--json" in cmd
    assert "--skip-git-repo-check" in cmd
    assert "--dangerously-bypass-approvals-and-sandbox" in cmd
    assert "-m" in cmd
    idx = cmd.index("-m")
    assert cmd[idx + 1] == "gpt-5-codex"
    assert cmd[-1] == "hello"


def test_run_strips_codex_prefix_from_model_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_subprocess(cmd: list[str], timeout: int = 0, on_line: object = None) -> object:
        captured["cmd"] = cmd
        return _fake_proc(_SAMPLE_JSONL)

    monkeypatch.setattr(codex_module, "run_subprocess", fake_subprocess)
    codex_module.run("hi", "codex/gpt-5-codex", timeout=30)

    cmd = captured["cmd"]
    idx = cmd.index("-m")
    assert cmd[idx + 1] == "gpt-5-codex"
    assert "codex/gpt-5-codex" not in cmd


def test_run_effort_flag_added(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_subprocess(cmd: list[str], timeout: int = 0, on_line: object = None) -> object:
        captured["cmd"] = cmd
        return _fake_proc(_SAMPLE_JSONL)

    monkeypatch.setattr(codex_module, "run_subprocess", fake_subprocess)
    codex_module.run("hi", "codex/gpt-5-codex", effort="high", timeout=30)

    cmd = captured["cmd"]
    assert "-c" in cmd
    idx = cmd.index("-c")
    assert cmd[idx + 1] == "model_reasoning_effort=high"


def test_run_effort_alias_mapped(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_subprocess(cmd: list[str], timeout: int = 0, on_line: object = None) -> object:
        captured["cmd"] = cmd
        return _fake_proc(_SAMPLE_JSONL)

    monkeypatch.setattr(codex_module, "run_subprocess", fake_subprocess)
    codex_module.run("hi", "codex/gpt-5-codex", effort="minimal", timeout=30)

    cmd = captured["cmd"]
    idx = cmd.index("-c")
    assert cmd[idx + 1] == "model_reasoning_effort=low"


def test_run_auto_effort_omitted(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_subprocess(cmd: list[str], timeout: int = 0, on_line: object = None) -> object:
        captured["cmd"] = cmd
        return _fake_proc(_SAMPLE_JSONL)

    monkeypatch.setattr(codex_module, "run_subprocess", fake_subprocess)
    codex_module.run("hi", "codex/gpt-5-codex", effort="auto", timeout=30)

    cmd = captured["cmd"]
    assert "-c" not in cmd


def test_run_resume_uses_subcommand(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    session_id = "019eb885-0bf2-7be2-b265-81dc3637472b"

    def fake_subprocess(cmd: list[str], timeout: int = 0, on_line: object = None) -> object:
        captured["cmd"] = cmd
        return _fake_proc(_SAMPLE_JSONL)

    monkeypatch.setattr(codex_module, "run_subprocess", fake_subprocess)
    codex_module.run("continue", "codex/gpt-5-codex", resume=session_id, timeout=30)

    cmd = captured["cmd"]
    assert cmd[0] == "codex"
    assert cmd[1] == "exec"
    assert cmd[2] == "resume"
    assert session_id in cmd
    assert cmd[-1] == "continue"
    assert cmd[cmd.index(session_id) - 1] != "continue"


def test_run_returns_session_id(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_subprocess(cmd: list[str], timeout: int = 0, on_line: object = None) -> object:
        return _fake_proc(_SAMPLE_JSONL)

    monkeypatch.setattr(codex_module, "run_subprocess", fake_subprocess)
    result = codex_module.run("hi", "codex/gpt-5-codex", timeout=30)

    assert result.session_id == "019eb885-0bf2-7be2-b265-81dc3637472b"


def test_run_nonzero_exit_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_subprocess(cmd: list[str], timeout: int = 0, on_line: object = None) -> object:
        return SimpleNamespace(returncode=1, stdout="", stderr="auth error")

    monkeypatch.setattr(codex_module, "run_subprocess", fake_subprocess)
    with pytest.raises(RuntimeError, match="codex exited 1"):
        codex_module.run("hi", "codex/gpt-5-codex", timeout=30)


def test_run_calls_on_line_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_subprocess(cmd: list[str], timeout: int = 0, on_line: object = None) -> object:
        if on_line is not None:
            for line in _SAMPLE_JSONL.splitlines():
                on_line(line)
        return _fake_proc(_SAMPLE_JSONL)

    monkeypatch.setattr(codex_module, "run_subprocess", fake_subprocess)
    codex_module.run("hi", "codex/gpt-5-codex", timeout=30)


# ── provider routing ─────────────────────────────────────────────────────────


def test_provider_for_codex_prefix() -> None:
    from splinter.models.roster import provider_for

    assert provider_for("codex/gpt-5-codex") == "codex"


def test_provider_for_opencode_unchanged() -> None:
    from splinter.models.roster import provider_for

    assert provider_for("opencode-go/minimax-m3") == "opencode"
    assert provider_for("opencode/deepseek-v4-flash-free") == "opencode"


def test_provider_for_claude_unchanged() -> None:
    from splinter.models.roster import provider_for

    assert provider_for("sonnet") == "claude"
    assert provider_for("opus") == "claude"


def test_get_provider_returns_codex_provider() -> None:
    from splinter.providers.registry import get_provider

    provider = get_provider("codex")
    assert isinstance(provider, CodexProvider)
    assert provider.name == "codex"


def test_available_providers_includes_codex() -> None:
    from splinter.providers.registry import available_providers

    assert "codex" in available_providers()


# ── CodexProvider.run ────────────────────────────────────────────────────────


def test_codex_provider_returns_provider_response(monkeypatch: pytest.MonkeyPatch) -> None:
    from splinter.providers.base import ProviderResponse

    def fake_subprocess(cmd: list[str], timeout: int = 0, on_line: object = None) -> object:
        return _fake_proc(_SAMPLE_JSONL)

    monkeypatch.setattr(codex_module, "run_subprocess", fake_subprocess)
    provider = CodexProvider()
    resp = provider.run("hello", "codex/gpt-5-codex", timeout=30)

    assert isinstance(resp, ProviderResponse)
    assert resp.text == "hello world"
    assert resp.session_id == "019eb885-0bf2-7be2-b265-81dc3637472b"
    assert resp.tokens["input"] == 100
    assert resp.tokens["output"] == 20


def test_codex_provider_gap_on_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    from splinter.providers.base import ProviderGapError

    def fake_subprocess(cmd: list[str], timeout: int = 0, on_line: object = None) -> object:
        return SimpleNamespace(returncode=1, stdout="", stderr="429 rate limit exceeded")

    monkeypatch.setattr(codex_module, "run_subprocess", fake_subprocess)
    provider = CodexProvider()
    with pytest.raises(ProviderGapError) as exc_info:
        provider.run("hi", "codex/gpt-5-codex", timeout=30)
    assert exc_info.value.kind == "rate_limit"
    assert exc_info.value.provider == "codex"


def test_codex_provider_gap_on_text_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from splinter.providers.base import ProviderGapError

    gap_jsonl = (
        '{"type":"thread.started","thread_id":"abc"}\n'
        '{"type":"error","message":"insufficient balance, please top up"}\n'
        '{"type":"turn.completed","usage":{"input_tokens":1,"cached_input_tokens":0,'
        '"output_tokens":5,"reasoning_output_tokens":0}}\n'
    )

    def fake_subprocess(cmd: list[str], timeout: int = 0, on_line: object = None) -> object:
        return _fake_proc(gap_jsonl)

    monkeypatch.setattr(codex_module, "run_subprocess", fake_subprocess)
    provider = CodexProvider()
    with pytest.raises(ProviderGapError) as exc_info:
        provider.run("hi", "codex/gpt-5-codex", timeout=30)
    assert exc_info.value.kind == "insufficient_balance"


def test_codex_provider_does_not_treat_normal_text_as_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    jsonl = (
        '{"type":"thread.started","thread_id":"abc"}\n'
        '{"type":"item.completed","item":{"id":"0","type":"agent_message",'
        '"text":"Add classifyVendorError coverage for rate limit and timeout"}}\n'
        '{"type":"turn.completed","usage":{"input_tokens":1,"cached_input_tokens":0,'
        '"output_tokens":5,"reasoning_output_tokens":0}}\n'
    )

    def fake_subprocess(cmd: list[str], timeout: int = 0, on_line: object = None) -> object:
        return _fake_proc(jsonl)

    monkeypatch.setattr(codex_module, "run_subprocess", fake_subprocess)
    provider = CodexProvider()
    resp = provider.run("hi", "codex/gpt-5-codex", timeout=30)
    assert "rate limit" in resp.text.lower()


def test_codex_provider_passes_session_as_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    session_id = "019eb885-0bf2-7be2-b265-81dc3637472b"

    def fake_subprocess(cmd: list[str], timeout: int = 0, on_line: object = None) -> object:
        captured["cmd"] = cmd
        return _fake_proc(_SAMPLE_JSONL)

    monkeypatch.setattr(codex_module, "run_subprocess", fake_subprocess)
    provider = CodexProvider()
    provider.run("continue", "codex/gpt-5-codex", session=session_id, timeout=30)

    cmd = captured["cmd"]
    assert "resume" in cmd
    assert session_id in cmd


# ── dispatch routing ──────────────────────────────────────────────────────────


def test_dispatch_run_text_routes_codex(monkeypatch: pytest.MonkeyPatch) -> None:
    from splinter.providers import dispatch

    def fake_run(
        prompt: str,
        model: str,
        *,
        effort: str | None = None,
        resume: str | None = None,
        timeout: int | None = None,
    ) -> CodexResult:
        return CodexResult(
            text="dispatched",
            tokens={"input": 10, "output": 5},
            cost=0.0,
            raw={},
            session_id=None,
        )

    monkeypatch.setattr(codex_module, "run", fake_run)
    text = dispatch.run_text("hello", "codex/gpt-5-codex", timeout=30)
    assert text == "dispatched"


def test_dispatch_run_text_session_routes_codex(monkeypatch: pytest.MonkeyPatch) -> None:
    from splinter.providers import dispatch

    def fake_run(
        prompt: str,
        model: str,
        *,
        effort: str | None = None,
        resume: str | None = None,
        timeout: int | None = None,
    ) -> CodexResult:
        return CodexResult(
            text="session-text",
            tokens={"input": 1, "output": 1},
            cost=0.0,
            raw={},
            session_id="new-sid",
        )

    monkeypatch.setattr(codex_module, "run", fake_run)
    text, sid = dispatch.run_text_session("hello", "codex/gpt-5-codex", timeout=30)
    assert text == "session-text"
    assert sid == "new-sid"


def test_dispatch_run_provider_session_routes_codex(monkeypatch: pytest.MonkeyPatch) -> None:
    from splinter.providers import dispatch
    from splinter.providers.base import ProviderResponse

    def fake_run(
        prompt: str,
        model: str,
        *,
        effort: str | None = None,
        resume: str | None = None,
        timeout: int | None = None,
    ) -> CodexResult:
        return CodexResult(
            text="full-resp",
            tokens={"input": 2, "output": 2},
            cost=0.05,
            raw={"_session_id": "sid-xyz"},
            session_id="sid-xyz",
        )

    monkeypatch.setattr(codex_module, "run", fake_run)
    resp, sid = dispatch.run_provider_session("hello", "codex/gpt-5-codex", timeout=30)
    assert isinstance(resp, ProviderResponse)
    assert resp.text == "full-resp"
    assert resp.cost == pytest.approx(0.05)
    assert sid == "sid-xyz"
