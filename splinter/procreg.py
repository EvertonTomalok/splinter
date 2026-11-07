"""Registry of running child processes so the UI can kill them on shutdown.

Provider CLIs (claude, opencode) are long-running. If the TUI quits or the run
errors while one is blocked in ``communicate()``, the worker thread would hang on
it and leave the terminal wedged. Running each child in its own process group and
tracking it here lets :func:`terminate_all` kill the whole tree, unblocking the
worker so the process can exit cleanly.
"""

from __future__ import annotations

import contextvars
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


def request_stop() -> None:
    """Signal the pipeline to stop gracefully after the current iteration."""
    _stop_event.set()


def stop_requested() -> bool:
    """Return True if a graceful stop has been requested."""
    return _stop_event.is_set()


def clear_stop() -> None:
    """Clear the stop flag (called on resume so a fresh run isn't pre-stopped)."""
    _stop_event.clear()


@dataclass(frozen=True)
class CompletedProcess:
    returncode: int
    stdout: str
    stderr: str


class IdleTimeoutError(RuntimeError):
    """Raised when a streaming child produces no output for ``idle_timeout``."""

    def __init__(self, idle_seconds: float, cmd: list[str]) -> None:
        self.idle_seconds = idle_seconds
        self.cmd = cmd
        super().__init__(f"no output for {idle_seconds:.0f}s: {' '.join(cmd[:3])}…")


def build_child_env(
    env: dict[str, str] | None = None,
    unset: tuple[str, ...] | list[str] = (),
) -> dict[str, str] | None:
    """Compose the child environment: explicit ``env`` wins; else os.environ minus ``unset``.

    Returns ``None`` (inherit the parent environment) unless an explicit env is
    given or keys must be dropped — e.g. subscription-only claude launches strip
    ``ANTHROPIC_API_KEY``/``ANTHROPIC_AUTH_TOKEN``.
    """
    if env is not None:
        return {k: v for k, v in env.items() if k not in set(unset)}
    if not unset:
        return None
    drop = set(unset)
    return {k: v for k, v in os.environ.items() if k not in drop}


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
    env: dict[str, str] | None = None,
    idle_timeout: float | None = None,
) -> CompletedProcess:
    """Run ``cmd`` capturing output, in its own process group, killable on shutdown.

    If ``on_line`` is given, stdout/stderr are read line-by-line and each line is
    handed to the callback as it arrives (live streaming) — used to surface the
    model's tool-calls/text in the run pane while it works. ``env`` replaces the
    child environment (PATH shims for guarded engine runs). ``idle_timeout``
    (seconds, streaming mode only) kills the group and raises
    :class:`IdleTimeoutError` when no line arrives on either pipe in time — the
    workhorse STALL watchdog.
    """
    if env is not None and "SPLINTER_LAUNCH_PATH" in env:
        import shutil

        executable = shutil.which(cmd[0], path=env["SPLINTER_LAUNCH_PATH"])
        if executable is None:
            raise FileNotFoundError(cmd[0])
        cmd = [executable, *cmd[1:]]
        env = {key: value for key, value in env.items() if key != "SPLINTER_LAUNCH_PATH"}
    proc: subprocess.Popen[str] = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        env=env,
        start_new_session=True,  # own process group so killpg reaches subagents
    )
    with _lock:
        _active.add(proc)
    try:
        if on_line is None:
            stdout, stderr = proc.communicate(timeout=timeout)
        else:
            stdout, stderr = _stream(
                proc, cmd, timeout, on_line, idle_timeout=idle_timeout
            )
    except (subprocess.TimeoutExpired, IdleTimeoutError):
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
    return CompletedProcess(returncode=proc.returncode, stdout=stdout, stderr=stderr)


def _stream(
    proc: subprocess.Popen[str],
    cmd: list[str],
    timeout: int,
    on_line: Callable[[str], None],
    *,
    idle_timeout: float | None = None,
) -> tuple[str, str]:
    """Read stdout/stderr line-by-line, invoke ``on_line`` per line, enforce timeout."""
    deadline = time.monotonic() + timeout
    out: list[str] = []
    err: list[str] = []
    last_activity = [time.monotonic()]
    assert proc.stdout is not None
    assert proc.stderr is not None

    def _drain(stream: Any, sink: list[str], ctx: contextvars.Context) -> None:
        for line in stream:
            sink.append(line)
            last_activity[0] = time.monotonic()
            try:
                # Run the callback inside the caller's context: threads start with
                # an empty Context, so agentic_scope's ContextVar (task/iteration
                # attribution for record_action) would otherwise be unset here.
                ctx.run(on_line, line.rstrip("\n"))
            except Exception:  # noqa: BLE001 — a logging hiccup must not kill the run
                pass

    # One copy per drain thread — a single Context cannot be entered concurrently.
    out_thread = threading.Thread(
        target=_drain, args=(proc.stdout, out, contextvars.copy_context()), daemon=True
    )
    err_thread = threading.Thread(
        target=_drain, args=(proc.stderr, err, contextvars.copy_context()), daemon=True
    )
    out_thread.start()
    err_thread.start()

    while proc.poll() is None:
        now = time.monotonic()
        if now > deadline:
            raise subprocess.TimeoutExpired(cmd, timeout)
        if idle_timeout is not None and now - last_activity[0] > idle_timeout:
            raise IdleTimeoutError(idle_timeout, cmd)
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
