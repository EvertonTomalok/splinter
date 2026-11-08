"""Unit tests for the procreg idle watchdog and child-env composition."""

from __future__ import annotations

import io
import subprocess
import threading
import time
from collections.abc import Iterator
from typing import Any, cast
from unittest.mock import patch

import pytest

import splinter.procreg as procreg
from splinter.procreg import IdleTimeoutError, build_child_env


class _FakeProc:
    returncode = 0

    def __init__(self, *, poll_result: int | None = 0) -> None:
        self._poll_result = poll_result
        self.stdout = io.StringIO("")
        self.stderr = io.StringIO("")

    def poll(self) -> int | None:
        return self._poll_result

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        return "", ""

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def terminate(self) -> None:
        return None


class TestBuildChildEnv:
    def test_no_args_inherits(self) -> None:
        assert build_child_env() is None

    def test_explicit_env_wins(self) -> None:
        env = build_child_env({"PATH": "/x", "SECRET": "1"}, ("SECRET",))
        assert env == {"PATH": "/x"}

    def test_unset_filters_environ(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SPLINTER_TEST_KEEP", "yes")
        monkeypatch.setenv("SPLINTER_TEST_DROP", "no")
        env = build_child_env(unset=("SPLINTER_TEST_DROP",))
        assert env is not None
        assert "SPLINTER_TEST_DROP" not in env
        assert env["SPLINTER_TEST_KEEP"] == "yes"


class TestRunSubprocessEnvPassthrough:
    @pytest.mark.parametrize("binary", ["mcode", "agy", "claude", "codex", "opencode"])
    def test_guard_resolves_launcher_before_shims(self, binary: str) -> None:
        from unittest.mock import patch

        env = {"PATH": "/shims:/original", "SPLINTER_LAUNCH_PATH": "/original"}
        with (
            patch("shutil.which", return_value=f"/original/{binary}") as which,
            patch.object(procreg.subprocess, "Popen", return_value=_FakeProc()) as popen,
        ):
            procreg.run_subprocess([binary, "prompt"], timeout=5, env=env)
        which.assert_called_once_with(binary, path="/original")
        assert popen.call_args.args[0] == [f"/original/{binary}", "prompt"]
        assert popen.call_args.kwargs["env"] == {"PATH": "/shims:/original"}

    def test_env_forwarded_to_popen(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        def fake_popen(cmd: list[str], **kwargs: Any) -> _FakeProc:
            captured.update(kwargs)
            return _FakeProc()

        monkeypatch.setattr(procreg.subprocess, "Popen", fake_popen)
        procreg.run_subprocess(["true"], timeout=5, env={"PATH": "/shims:/bin"})
        assert captured["env"] == {"PATH": "/shims:/bin"}

    def test_env_none_means_inherit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        def fake_popen(cmd: list[str], **kwargs: Any) -> _FakeProc:
            captured.update(kwargs)
            return _FakeProc()

        monkeypatch.setattr(procreg.subprocess, "Popen", fake_popen)
        procreg.run_subprocess(["true"], timeout=5)
        assert captured["env"] is None


class TestStreamIdleWatchdog:
    def test_idle_timeout_raises(self) -> None:
        proc = _FakeProc(poll_result=None)
        with pytest.raises(IdleTimeoutError) as excinfo:
            procreg._stream(
                proc,
                ["mcode", "exec"],
                timeout=30,
                on_line=lambda line: None,
                idle_timeout=0.001,
            )
        assert excinfo.value.idle_seconds == 0.001
        assert excinfo.value.cmd == ["mcode", "exec"]

    def test_no_idle_timeout_only_deadline(self) -> None:
        proc = _FakeProc(poll_result=0)
        out, err = procreg._stream(
            proc, ["true"], timeout=5, on_line=lambda line: None, idle_timeout=None
        )
        assert out == "" and err == ""


class TestIdleTimeoutErrorAttrs:
    def test_message_and_attrs(self) -> None:
        err = IdleTimeoutError(480.0, ["agy", "-p"])
        assert err.idle_seconds == 480.0
        assert err.cmd == ["agy", "-p"]
        assert "no output for 480s" in str(err)


class _OpenPipe:
    """A pipe with no output yet: iterating blocks until it closes."""

    def __init__(self) -> None:
        self.closed = threading.Event()

    def __iter__(self) -> Iterator[str]:
        self.closed.wait(5)
        return iter(())


class _LiveProc:
    returncode = 0

    def __init__(self) -> None:
        self.stdout = _OpenPipe()
        self.stderr = _OpenPipe()
        self.polls = 0
        self.exited = False

    def poll(self) -> int | None:
        self.polls += 1
        return 0 if self.exited else None

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def close(self) -> None:
        self.exited = True
        self.stdout.closed.set()
        self.stderr.closed.set()


class TestStreamWakeups:
    def test_idle_watchdog_blocks_instead_of_polling(self) -> None:
        proc = _LiveProc()
        try:
            with (
                patch.object(procreg.time, "sleep", side_effect=AssertionError("polled")),
                pytest.raises(IdleTimeoutError),
            ):
                procreg._stream(
                    cast(Any, proc), ["x"], timeout=30, on_line=lambda _: None, idle_timeout=0.05
                )
        finally:
            proc.close()
        assert proc.polls <= 4

    def test_deadline_blocks_instead_of_polling(self) -> None:
        proc = _LiveProc()
        try:
            with (
                patch.object(procreg.time, "sleep", side_effect=AssertionError("polled")),
                patch.object(procreg, "_WATCHDOG_TICK_S", 0.02),
                pytest.raises(subprocess.TimeoutExpired),
            ):
                procreg._stream(cast(Any, proc), ["x"], timeout=0, on_line=lambda _: None)
        finally:
            proc.close()

    def test_exit_wakes_the_watchdog_before_the_tick(self) -> None:
        proc = _LiveProc()
        timer = threading.Timer(0.05, proc.close)
        timer.start()
        started = time.monotonic()
        with patch.object(procreg, "_WATCHDOG_TICK_S", 3.0):
            procreg._stream(cast(Any, proc), ["x"], timeout=30, on_line=lambda _: None)
        timer.join()
        assert time.monotonic() - started < 1.0
        assert proc.polls <= 3

    def test_child_that_closed_its_pipes_is_waited_with_a_tick(self) -> None:
        proc = _FakeProc(poll_result=None)
        results: list[subprocess.TimeoutExpired | int] = [
            subprocess.TimeoutExpired(["x"], 1),
            subprocess.TimeoutExpired(["x"], 1),
            subprocess.TimeoutExpired(["x"], 1),
            0,
        ]
        waits: list[float | None] = []

        def fake_wait(timeout: float | None = None) -> int:
            waits.append(timeout)
            result = results.pop(0)
            if isinstance(result, subprocess.TimeoutExpired):
                raise result
            proc._poll_result = 0
            return result

        with patch.object(proc, "wait", side_effect=fake_wait):
            out, err = procreg._stream(
                cast(Any, proc), ["x"], timeout=30, on_line=lambda _: None
            )
        assert (out, err) == ("", "")
        assert len(waits) == 4
        assert all(step is not None and 0 < step <= 1.0 for step in waits)
