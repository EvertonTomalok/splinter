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
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

_lock = threading.Lock()
_active: set[subprocess.Popen[str]] = set()
_stop_event = threading.Event()
_WATCHDOG_TICK_S = 1.0
_story: contextvars.ContextVar[int | None] = contextvars.ContextVar("procreg_story", default=None)
_by_story: dict[int, set[subprocess.Popen[str]]] = {}
_interrupted: set[int] = set()
_log_story: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "procreg_log_story", default=None
)


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


class StoryInterrupted(RuntimeError):
    """A person stopped this story's engine to change how the story continues."""

    def __init__(self, story: int) -> None:
        self.story = story
        super().__init__(f"story {story} was interrupted")


@contextmanager
def story_scope(story: int) -> Iterator[None]:
    """Every child started inside belongs to ``story``, so it can be stopped alone."""
    token = _story.set(story)
    try:
        yield
    finally:
        _story.reset(token)


def bind_log_story(story: int) -> None:
    """Mark what this worker logs from here on as belonging to ``story``."""
    _log_story.set(story)


@contextmanager
def log_story_scope(story: int) -> Iterator[None]:
    """Mark what is logged inside as belonging to ``story``."""
    token = _log_story.set(story)
    try:
        yield
    finally:
        _log_story.reset(token)


def current_log_story() -> int | None:
    return _log_story.get()


def kill_story(story: int) -> int:
    """Stop the engines of one story and refuse new ones until the story takes note."""
    with _lock:
        _interrupted.add(story)
        procs = list(_by_story.get(story, ()))
    for proc in procs:
        _kill(proc)
    return len(procs)


def story_interrupted(story: int) -> bool:
    with _lock:
        return story in _interrupted


def clear_story_interrupt(story: int) -> None:
    with _lock:
        _interrupted.discard(story)


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


def _child_env(env: dict[str, str] | None, cwd: object) -> dict[str, str] | None:
    if cwd is None:
        return env
    return {**(os.environ if env is None else env), "PWD": str(cwd)}


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
    if _shutting_down.is_set():
        raise ShuttingDownError(f"splinter is exiting: not starting {cmd[0]}")
    owner = _story.get()
    if owner is not None and story_interrupted(owner):
        raise StoryInterrupted(owner)
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
        env=_child_env(env, cwd),
        start_new_session=True,  # own process group so killpg reaches subagents
    )
    with _lock:
        _active.add(proc)
        if owner is not None:
            _by_story.setdefault(owner, set()).add(proc)
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
            if owner is not None:
                _by_story.get(owner, set()).discard(proc)
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
        step = _watchdog_step(cmd, timeout, deadline, idle_timeout, last_activity[0])
        _block_for_exit(proc, (out_thread, err_thread), step)

    out_thread.join()
    err_thread.join()
    return "".join(out), "".join(err)


def _watchdog_step(
    cmd: list[str],
    timeout: int,
    deadline: float,
    idle_timeout: float | None,
    last_activity: float,
) -> float:
    """Seconds the watchdog may block for; raises once a limit has passed."""
    now = time.monotonic()
    if now > deadline:
        raise subprocess.TimeoutExpired(cmd, timeout)
    step = min(_WATCHDOG_TICK_S, deadline - now)
    if idle_timeout is None:
        return step
    idle = now - last_activity
    if idle > idle_timeout:
        raise IdleTimeoutError(idle_timeout, cmd)
    return min(step, idle_timeout - idle)


def _block_for_exit(
    proc: subprocess.Popen[str], drains: tuple[threading.Thread, ...], step: float
) -> None:
    """Sleep up to ``step``; a pipe closing wakes it at once.

    ``Popen.wait(timeout=...)`` polls every 50 ms at best, so a live pipe is
    waited on through its drain thread. Only a child that closed both pipes
    and kept running falls back to it.
    """
    for drain in drains:
        if drain.is_alive():
            drain.join(step)
            return
    try:
        proc.wait(timeout=step)
    except subprocess.TimeoutExpired:
        pass


_shutting_down = threading.Event()


class ShuttingDownError(RuntimeError):
    """Raised by :func:`run_subprocess` once :func:`shutdown` began."""


def _descendants(root: int) -> list[int]:
    """Every live descendant pid of ``root`` (by parent pid, any process group)."""
    try:
        out = subprocess.run(
            ["ps", "-A", "-o", "pid=,ppid="], capture_output=True, text=True, timeout=5
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    children: dict[int, list[int]] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            children.setdefault(int(parts[1]), []).append(int(parts[0]))
    found: list[int] = []
    stack = [root]
    while stack:
        for child in children.get(stack.pop(), []):
            found.append(child)
            stack.append(child)
    return found


def _signal_all(pids: list[int], sig: int) -> None:
    for pid in pids:
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def shutdown(grace: float = 2.0) -> None:
    """Stop for good: refuse new children, then kill every child and grandchild.

    Tracked provider groups get SIGTERM first; then every descendant of this
    process (gate commands, git, anything spawned outside the registry) gets
    SIGTERM, and SIGKILL after ``grace`` seconds if still alive.
    """
    _shutting_down.set()
    _stop_event.set()
    terminate_all()
    pids = _descendants(os.getpid())
    _signal_all(pids, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while pids and time.monotonic() < deadline:
        time.sleep(0.1)
        pids = [pid for pid in pids if pid in set(_descendants(os.getpid()))]
    _signal_all(pids, signal.SIGKILL)


def terminate_all() -> None:
    """Kill every tracked child process group (best effort)."""
    with _lock:
        procs = list(_active)
    for proc in procs:
        _kill(proc)
