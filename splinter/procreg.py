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

_lock = threading.Lock()
_active: set[subprocess.Popen[str]] = set()


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
    cmd: list[str], *, timeout: int, cwd: str | None = None
) -> CompletedProcess:
    """Run ``cmd`` capturing output, in its own process group, killable on shutdown."""
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
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill(proc)
        proc.communicate()
        raise
    finally:
        with _lock:
            _active.discard(proc)
    return CompletedProcess(returncode=proc.returncode, stdout=stdout, stderr=stderr)


def terminate_all() -> None:
    """Kill every tracked child process group (best effort)."""
    with _lock:
        procs = list(_active)
    for proc in procs:
        _kill(proc)
