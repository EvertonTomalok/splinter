from __future__ import annotations

from typing import Any

import pytest

from splinter import procreg


class _Proc:
    pid = 4242
    returncode = 0

    def communicate(self, timeout: int | None = None) -> tuple[str, str]:
        return "", ""

    def poll(self) -> int:
        return 0


def _launch(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    def fake(cmd: list[str], **popen: Any) -> _Proc:
        seen.update(popen)
        return _Proc()

    monkeypatch.setattr(procreg.subprocess, "Popen", fake)
    procreg.run_subprocess(["engine"], timeout=5, **kwargs)
    return seen


def test_an_engine_never_inherits_the_pwd_of_the_main_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PWD", "/repo")

    seen = _launch(monkeypatch, cwd="/repo/.splinter/worktrees/US-002")

    assert seen["env"]["PWD"] == "/repo/.splinter/worktrees/US-002"


def test_a_given_environment_is_kept_and_gets_the_pwd(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _launch(monkeypatch, cwd="/wt", env={"A": "1", "PWD": "/repo"})

    assert seen["env"] == {"A": "1", "PWD": "/wt"}


def test_no_cwd_leaves_the_environment_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _launch(monkeypatch)["env"] is None
