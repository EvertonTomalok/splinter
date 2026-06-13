"""Tests for the cursor CLI provider adapter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from splinter.providers import cursor as cursor_module


def _fake_proc(stdout: str, returncode: int = 0) -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")


def test_run_cmd_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_subprocess(cmd: list[str], timeout: int = 0, cwd: str = ".") -> object:
        captured["cmd"] = cmd
        return _fake_proc("ok")

    monkeypatch.setattr(cursor_module, "run_subprocess", fake_subprocess)
    cursor_module.run("hello", timeout=60)

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[0] == "agent"
    assert cmd[1] == "-p"
    assert "--" in cmd
    assert cmd[-1] == "hello"
