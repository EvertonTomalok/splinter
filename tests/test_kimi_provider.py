"""Tests for the kimi (kimi-code) CLI provider adapter."""

from __future__ import annotations

from typing import Any

import pytest

import splinter.providers.kimi as kimi
from splinter.procreg import CompletedProcess
from splinter.providers.base import ProviderGapError, ProviderResponse


def _fake_proc(stdout: str = "", returncode: int = 0, stderr: str = "") -> CompletedProcess:
    return CompletedProcess(returncode=returncode, stdout=stdout, stderr=stderr)


class TestArgv:
    def test_basic_prompt_mode_without_auto(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        def fake(cmd: list[str], **kw: Any) -> CompletedProcess:
            captured["cmd"] = cmd
            captured["kw"] = kw
            return _fake_proc('{"type":"text","text":"ok"}')

        monkeypatch.setattr(kimi, "run_subprocess", fake)
        kimi.run("do it", "kimi/kimi-code/k3.1-pro", timeout=60)
        cmd = captured["cmd"]
        assert cmd[0] == "kimi"
        assert cmd[1:4] == ["-p", "do it", "--output-format", "stream-json"][:3]
        assert "--auto" not in cmd
        assert cmd[-2:] == ["-m", "kimi-code/k3.1-pro"]

    def test_prefix_and_default_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        def fake(cmd: list[str], **kw: Any) -> CompletedProcess:
            captured["cmd"] = cmd
            return _fake_proc('{"type":"text","text":"ok"}')

        monkeypatch.setattr(kimi, "run_subprocess", fake)
        kimi.run("p", "kimi/kimi-code/k3", timeout=60)
        assert captured["cmd"][-2:] == ["-m", "kimi-code/k3"]

        kimi.run("p", "", timeout=60)
        assert "-m" not in captured["cmd"]

    def test_session_resume_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        def fake(cmd: list[str], **kw: Any) -> CompletedProcess:
            captured["cmd"] = cmd
            return _fake_proc('{"type":"text","text":"ok"}')

        monkeypatch.setattr(kimi, "run_subprocess", fake)
        kimi.run("p", "", session="s-1", timeout=60)
        assert captured["cmd"][-2:] == ["-S", "s-1"]

    def test_add_dirs_and_passthrough(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        def fake(cmd: list[str], **kw: Any) -> CompletedProcess:
            captured.update(kw=kw, cmd=cmd)
            return _fake_proc('{"type":"text","text":"ok"}')

        monkeypatch.setattr(kimi, "run_subprocess", fake)
        kimi.run(
            "p",
            "",
            timeout=60,
            cwd="/wt",
            env={"PATH": "/x"},
            idle_timeout=30.0,
            add_dirs=("/a", "/b"),
        )
        cmd = captured["cmd"]
        assert cmd[-4:] == ["--add-dir", "/a", "--add-dir", "/b"]
        assert captured["kw"]["cwd"] == "/wt"
        assert captured["kw"]["env"] == {"PATH": "/x"}
        assert captured["kw"]["idle_timeout"] == 30.0


class TestParse:
    def test_nested_and_flat_text(self) -> None:
        stdout = "\n".join(
            [
                '{"session_id":"s-9","message":{"content":[{"type":"text","text":"part one "}]}}',
                '{"type":"text","text":"part two"}',
            ]
        )
        result = kimi._parse_output(stdout)
        assert result.text.split() == ["part", "one", "part", "two"]
        assert result.session_id == "s-9"

    def test_usage_tokens(self) -> None:
        stdout = (
            '{"type":"result","usage":{"input_tokens":11,"output_tokens":7,"cost":0.002}}'
        )
        result = kimi._parse_output(stdout)
        assert result.tokens == {"input": 11, "output": 7}
        assert result.cost == 0.002
        assert result.cost_indeterminate is False

    def test_no_usage_indeterminate(self) -> None:
        result = kimi._parse_output('{"type":"text","text":"ok"}')
        assert result.cost == 0.0
        assert result.cost_indeterminate is True

    def test_error_event_raises(self) -> None:
        with pytest.raises(RuntimeError):
            kimi._parse_output('{"type":"error","message":"boom"}')

    def test_nonzero_exit_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            kimi,
            "run_subprocess",
            lambda cmd, **kw: _fake_proc(returncode=1, stderr="auth failed"),
        )
        with pytest.raises(RuntimeError, match="kimi exited 1"):
            kimi.run("p", "", timeout=60)


class TestProvider:
    def test_run_maps_to_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            kimi,
            "run_subprocess",
            lambda cmd, **kw: _fake_proc(
                '{"session_id":"s-2","type":"text","text":"hello",'
                '"usage":{"input_tokens":3,"output_tokens":4,"cost":0.001}}'
            ),
        )
        resp = kimi.KimiProvider().run("hi", "kimi/kimi-code/k3.1-pro", timeout=60)
        assert isinstance(resp, ProviderResponse)
        assert resp.text == "hello"
        assert resp.session_id == "s-2"
        assert resp.tokens == {"input": 3, "output": 4}
        assert resp.cost == 0.001

    def test_gap_wrapped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            kimi,
            "run_subprocess",
            lambda cmd, **kw: _fake_proc(returncode=1, stderr="429 rate limit exceeded"),
        )
        with pytest.raises(ProviderGapError):
            kimi.KimiProvider().run("hi", "kimi/kimi-code/k3.1-pro", timeout=60)
