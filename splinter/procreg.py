"""Registry of running child processes so the UI can kill them on shutdown.

Provider CLIs (claude, opencode) are long-running. If the TUI quits or the run
errors while one is blocked in ``communicate()``, the worker thread would hang on
it and leave the terminal wedged. Running each child in its own process group and
tracking it here lets :func:`terminate_all` kill the whole tree, unblocking the
worker so the process can exit cleanly.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

_lock = threading.Lock()
_active: set[subprocess.Popen[str]] = set()
_stop_event = threading.Event()
_interrupt_event = threading.Event()


class DirectiveInterrupt(RuntimeError):
    """Raised by :func:`run_subprocess` when the in-flight provider process is
    killed to apply a live user directive. The pipeline loop catches it, merges
    the queued directive into corrections, and restarts the same session — so a
    single provider process keeps following the conversation."""


def request_stop() -> None:
    """Signal the pipeline to stop gracefully after the current iteration."""
    _stop_event.set()


def stop_requested() -> bool:
    """Return True if a graceful stop has been requested."""
    return _stop_event.is_set()


def clear_stop() -> None:
    """Clear the stop flag (called on resume so a fresh run isn't pre-stopped)."""
    _stop_event.clear()


def request_interrupt() -> bool:
    """Kill the in-flight provider process so the loop restarts it with a pending
    user directive applied. Returns False (a no-op) when nothing is running — the
    directive then just lands on the next iteration. Provider-agnostic: the kill
    is a SIGTERM to the process group, same path as a user kill."""
    with _lock:
        if not _active:
            return False
    _interrupt_event.set()
    terminate_all()
    return True


def interrupt_requested() -> bool:
    return _interrupt_event.is_set()


def clear_interrupt() -> None:
    _interrupt_event.clear()


@dataclass(frozen=True)
class CompletedProcess:
    returncode: int
    stdout: str
    stderr: str


def _kill(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.terminate()
        except OSError:
            pass


def run_subprocess(
    cmd: list[str],
    *,
    timeout: int,
    cwd: str | None = None,
    on_line: Callable[[str], None] | None = None,
) -> CompletedProcess:
    """Run ``cmd`` capturing output, in its own process group, killable on shutdown.

    If ``on_line`` is given, stdout/stderr are read line-by-line and each line is
    handed to the callback as it arrives (live streaming) — used to surface the
    model's tool-calls/text in the run pane while it works.
    """
    proc: subprocess.Popen[str] = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        start_new_session=True,  # own process group so killpg reaches subagents
    )
    with _lock:
        _active.add(proc)
    try:
        if on_line is None:
            stdout, stderr = proc.communicate(timeout=timeout)
        else:
            stdout, stderr = _stream(proc, cmd, timeout, on_line)
    except subprocess.TimeoutExpired:
        _kill(proc)
        if on_line is None:
            proc.communicate()
        else:
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        raise
    finally:
        with _lock:
            _active.discard(proc)
    # A pending interrupt is one-shot: consume it on the next process to finish.
    # If this process was signal-killed (negative returncode) it was the target —
    # surface a DirectiveInterrupt so the loop restarts with the directive. A
    # clean exit during the race window just clears the flag and returns normally.
    if interrupt_requested():
        clear_interrupt()
        if proc.returncode is not None and proc.returncode < 0:
            raise DirectiveInterrupt()
    return CompletedProcess(returncode=proc.returncode, stdout=stdout, stderr=stderr)


def _stream(
    proc: subprocess.Popen[str], cmd: list[str], timeout: int, on_line: Callable[[str], None]
) -> tuple[str, str]:
    """Read stdout/stderr line-by-line, invoke ``on_line`` per line, enforce timeout."""
    deadline = time.monotonic() + timeout
    out: list[str] = []
    err: list[str] = []
    assert proc.stdout is not None
    assert proc.stderr is not None

    def _drain(stream: Any, sink: list[str]) -> None:
        for line in stream:
            sink.append(line)
            try:
                on_line(line.rstrip("\n"))
            except Exception:  # noqa: BLE001 — a logging hiccup must not kill the run
                pass

    out_thread = threading.Thread(target=_drain, args=(proc.stdout, out), daemon=True)
    err_thread = threading.Thread(target=_drain, args=(proc.stderr, err), daemon=True)
    out_thread.start()
    err_thread.start()

    while proc.poll() is None:
        if time.monotonic() > deadline:
            raise subprocess.TimeoutExpired(cmd, timeout)
        time.sleep(0.05)

    out_thread.join()
    err_thread.join()
    return "".join(out), "".join(err)


def terminate_all() -> None:
    """Kill every tracked child process group (best effort)."""
    with _lock:
        procs = list(_active)
    for proc in procs:
        _kill(proc)
