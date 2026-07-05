from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from splinter.enums import DEFAULT_RUNNER_MODE, RunnerMode

log = logging.getLogger("splinter.session")

#: Serialises the read-modify-write of ``worktrees.json`` across parallel worker
#: threads. Module-level so it holds even for sessions built via ``__new__``
#: (which bypasses ``__init__``), and shared by every Session in-process.
_WORKTREE_LOCK = threading.Lock()

#: Serialises the read-modify-write of ``status.json``. The kowabunga read path
#: clears its error flag from many parallel worker threads + the cascade dispatch
#: thread; without this lock those clearing-writes race concurrent ``set_status``
#: calls (e.g. ``state="completed"``) and clobber/revert fields.
_STATUS_LOCK = threading.Lock()

#: Per-session-id locks guarding ``Session.append``'s critical section (rotate +
#: fan-out write). Keyed by session id rather than a single module-global lock so
#: independent worktree sessions don't serialise against each other; keyed by id
#: rather than instance because ``Session(sid)`` objects are recreated freely
#: (each worker thread builds its own) and must still share one lock per session.
_APPEND_LOCKS: dict[str, threading.Lock] = {}
_APPEND_LOCKS_GUARD = threading.Lock()


def _append_lock(session_id: str) -> threading.Lock:
    with _APPEND_LOCKS_GUARD:
        return _APPEND_LOCKS.setdefault(session_id, threading.Lock())


def _base_dir() -> Path:
    """Where session memory lives.

    ``SPLINTER_HOME`` overrides; otherwise a ``.splinter`` dir under the current
    working directory — sessions live where ``splinter`` was invoked.
    """
    env = os.environ.get("SPLINTER_HOME")
    if env:
        return Path(env)
    return Path.cwd() / ".splinter"


def _sessions_dir() -> Path:
    return _base_dir() / "sessions"


def new_session_id() -> str:
    now = datetime.now(timezone.utc)
    return f"ses_{now.strftime('%Y%m%d-%H%M%S')}"


def session_dir(session_id: str) -> Path:
    d = _sessions_dir() / session_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "knowledge").mkdir(exist_ok=True)
    return d


def list_sessions() -> list[str]:
    """All session ids, newest first by mtime."""
    sd = _sessions_dir()
    if not sd.exists():
        return []
    entries = [e for e in sd.iterdir() if e.is_dir()]
    entries.sort(key=lambda e: e.stat().st_mtime, reverse=True)
    return [e.name for e in entries]


def latest_session_id() -> str | None:
    sessions = list_sessions()
    return sessions[0] if sessions else None


def delete_session(session_id: str) -> None:
    """Remove a session directory and everything in it."""
    import shutil

    d = _sessions_dir() / session_id
    if d.exists():
        shutil.rmtree(d)


def resolve_session(session_id: str | None = None) -> str:
    if session_id:
        return session_id
    sid = latest_session_id()
    if sid is None:
        sid = new_session_id()
    return sid


NEXT_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "next_planner_model",
        "next_planner_effort",
        "next_runner_model",
        "next_runner_effort",
        "next_eval_model",
        "next_eval_effort",
        "next_skip_planner",
        "next_skip_eval",
        "next_skip_final_eval",
    }
)

_EVENTS_TAG_RE = re.compile(r"^\[([^\]]+)\]\s*(.*)$")


class Session:
    def __init__(self, session_id: str | None = None) -> None:
        self.id = resolve_session(session_id)
        self.dir = _sessions_dir() / self.id

    def _ensure_dir(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "knowledge").mkdir(exist_ok=True)

    def index_path(self) -> Path:
        return self.dir / "index.md"

    def read_index(self) -> str:
        p = self.index_path()
        if p.exists():
            return p.read_text()
        return ""

    def write(self, filename: str, content: str) -> Path:
        self._ensure_dir()
        p = self.dir / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return p

    def append(self, filename: str, content: str) -> Path:
        # US-001 thread-safety: the whole append is serialised per session so
        # concurrent cascade workers never interleave. US-004: an ``events.md``
        # append is routed into the canonical ``events.jsonl`` — the eager 4-file
        # fan-out (rotate/tail/compact) is gone; those views render on demand.
        with _append_lock(self.id):
            self._ensure_dir()
            p = self.dir / filename
            p.parent.mkdir(parents=True, exist_ok=True)
            payload = content if content.endswith("\n") else f"{content}\n"
            if filename == "events.md":
                self._append_events_jsonl(payload)
                return p
            with open(p, "a") as f:
                f.write(payload)
        return p

    def _append_events_jsonl(self, payload: str) -> None:
        """Route an ``events.md`` append into the canonical ``events.jsonl`` as one
        ``log``-typed record per non-empty line — the sole write path for the
        chronological log (see :mod:`splinter.obs.events`). ``events.md`` itself is
        never written; :meth:`render_events_md` renders it back on demand.

        Caller runs under ``_append_lock(self.id)`` (held by :meth:`append`)."""
        from splinter.obs import events

        ts = datetime.now(timezone.utc).isoformat()
        for raw_line in payload.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            m = _EVENTS_TAG_RE.match(line)
            if m:
                stage = m.group(1).strip().lower().replace(" ", "_")
                message = m.group(2).strip() or line
            else:
                stage = "event"
                message = line
            events.append_event(
                self,
                events.Event(
                    type="log",
                    ts=ts,
                    payload={"stage": stage, "message": message, "raw": line},
                ),
            )

    def render_events_md(self) -> str:
        """Full chronological events log, rendered on demand from ``events.jsonl``."""
        from splinter.obs import events

        lines = [
            str(ev.payload.get("raw") or ev.payload.get("message", ""))
            for ev in events.load_log_events(self)
        ]
        return "\n".join(lines) + ("\n" if lines else "")

    def render_events_tail(self, max_bytes: int) -> str:
        """Tail slice of :meth:`render_events_md`, trimmed to a whole-line boundary."""
        data = self.render_events_md().encode("utf-8")
        if len(data) <= max_bytes:
            return data.decode("utf-8")
        tail = data[-max_bytes:]
        nl = tail.find(b"\n")
        trimmed = tail[nl + 1 :] if nl >= 0 else tail
        return trimmed.decode("utf-8", errors="replace")

    def update_index(self, summary: str) -> None:
        self.write("index.md", summary)

    def has(self, what: str) -> bool:
        p = self.dir / what
        if p.exists() and p.stat().st_size > 0:
            return True
        idx = self.read_index()
        return what in idx

    def read(self, filename: str) -> str:
        p = self.dir / filename
        if p.exists():
            return p.read_text()
        return ""

    def knowledge_dir(self) -> Path:
        self._ensure_dir()
        return self.dir / "knowledge"

    def round_dir(self, n: int) -> Path:
        rd = self.dir / f"eval-fix-{n}"
        rd.mkdir(parents=True, exist_ok=True)
        return rd

    def read_next_config(self) -> dict[str, str]:
        data = self.read_status()
        return {k: v for k in NEXT_CONFIG_KEYS if (v := str(data.get(k, ""))) and v}

    def clear_next_config(self) -> None:
        data = self.read_status()
        state = str(data.get("state", "running"))
        self.set_status(state, **{k: "" for k in NEXT_CONFIG_KEYS})

    def status_path(self) -> Path:
        return self.dir / "status.json"

    def set_status(self, state: str, **fields: Any) -> None:
        """Persist run state (running/completed/failed) plus arbitrary fields."""
        self._ensure_dir()
        with _STATUS_LOCK:
            data = self.read_status()
            if "started_at" not in data:
                data["started_at"] = datetime.now(timezone.utc).isoformat()
            data["state"] = state
            data["updated"] = datetime.now(timezone.utc).isoformat()
            data.update(fields)
            self.status_path().write_text(json.dumps(data, indent=2))

    def read_status(self) -> dict[str, Any]:
        p = self.status_path()
        if p.exists():
            try:
                loaded: dict[str, Any] = json.loads(p.read_text())
                return loaded
            except json.JSONDecodeError:
                return {}
        return {}

    def read_kowabunga(self) -> RunnerMode:
        """Session-scoped kowabunga toggle; absent or bad value -> OFF, never raises."""
        raw = self.read_status().get("kowabunga")
        try:
            return RunnerMode(raw) if raw is not None else DEFAULT_RUNNER_MODE
        except ValueError:
            return DEFAULT_RUNNER_MODE

    def set_kowabunga(self, mode: RunnerMode) -> None:
        state = str(self.read_status().get("state", "running"))
        self.set_status(state, kowabunga=str(RunnerMode(mode)))

    def read_cowabunga(self) -> bool:
        """Is kowabunga ON? Re-read from session state on every scheduling decision.

        Fault-tolerant by contract: any failure reading the persisted state falls
        back to OFF and records ``kowabunga_read_error`` so the runner screen can
        surface a warning. The error is logged, never silently swallowed. A clean
        read clears a previously-set error flag.
        """
        try:
            data = self.read_status()
        except Exception as exc:
            log.warning("kowabunga: state read failed, defaulting OFF: %s", exc)
            self._flag_kowabunga_read_error()
            return False
        raw = data.get("kowabunga")
        try:
            mode = RunnerMode(raw) if raw is not None else DEFAULT_RUNNER_MODE
        except ValueError:
            mode = DEFAULT_RUNNER_MODE
        if data.get("kowabunga_read_error"):
            self.set_status(str(data.get("state", "running")), kowabunga_read_error=False)
        return mode == RunnerMode.KOWABUNGA_ON

    def _flag_kowabunga_read_error(self) -> None:
        """Persist the read-error flag directly, bypassing ``read_status`` (which just
        failed), so the warning survives to the runner screen."""
        self._ensure_dir()
        p = self.status_path()
        with _STATUS_LOCK:
            data: dict[str, Any] = {}
            try:
                if p.exists():
                    loaded = json.loads(p.read_text())
                    if isinstance(loaded, dict):
                        data = loaded
            except Exception:
                data = {}
            data["state"] = data.get("state", "running")
            data["kowabunga_read_error"] = True
            p.write_text(json.dumps(data, indent=2))

    def log_llm_usage(self, model: str, tokens: dict[str, int], cost: float) -> None:
        """Accumulate LLM usage outside the main run trace (PRD, planner, etc.)."""
        p = self.dir / "pre_run_usage.json"
        try:
            data: dict[str, Any] = json.loads(p.read_text()) if p.exists() else {}
        except json.JSONDecodeError:
            data = {}
        data["input"] = int(data.get("input", 0)) + tokens.get("input", 0)
        data["output"] = int(data.get("output", 0)) + tokens.get("output", 0)
        data["cost"] = float(data.get("cost", 0.0)) + cost
        # Per-model breakdown
        models: dict[str, Any] = data.get("models", {})
        m = models.get(model, {})
        m["input"] = int(m.get("input", 0)) + tokens.get("input", 0)
        m["output"] = int(m.get("output", 0)) + tokens.get("output", 0)
        m["cost"] = float(m.get("cost", 0.0)) + cost
        models[model] = m
        data["models"] = models
        self._ensure_dir()
        p.write_text(json.dumps(data))

    def read_pre_run_usage(self) -> dict[str, Any]:
        p = self.dir / "pre_run_usage.json"
        if p.exists():
            try:
                loaded: dict[str, Any] = json.loads(p.read_text())
                return loaded
            except json.JSONDecodeError:
                pass
        return {}

    def _directive_path(self, task_no: int | None) -> Path:
        """Queue file for a directive. ``task_no`` (1-based) scopes it to a single
        parallel task; ``None`` is the shared queue any task loop may drain."""
        if task_no is None:
            return self.dir / "pending_directive.txt"
        return self.dir / f"pending_directive.task-{task_no}.txt"

    def queue_live_command(self, text: str, task_no: int | None = None) -> None:
        """Append a live user directive to the pending queue (TUI → pipeline).

        ``task_no`` (1-based) targets a single running parallel task; when omitted
        the directive lands in the shared queue that the next task loop to poll
        will pick up. The directive is *never* applied by killing the in-flight
        provider — it is merged into corrections at the top of the next iteration,
        so the model decides when to act on it.
        """
        self._ensure_dir()
        p = self._directive_path(task_no)
        with open(p, "a") as f:
            f.write(text.strip())
            f.write("\n---\n")

    def pop_live_commands(self, task_no: int | None = None) -> str:
        """Read and clear pending live directives for ``task_no`` plus the shared
        queue. Returns empty string if none.

        A parallel task loop passes its own 1-based ``task_no`` so it drains only
        its scoped queue (no cross-task races) and the shared queue. A single-task
        loop passing ``None`` drains just the shared queue.
        """
        paths = [self.dir / "pending_directive.txt"]
        if task_no is not None:
            paths.append(self._directive_path(task_no))
        parts: list[str] = []
        for p in paths:
            if not p.exists():
                continue
            try:
                content = p.read_text().strip()
                p.unlink()
                if content:
                    parts.append(content)
            except Exception:
                continue
        return "\n---\n".join(parts)

    def set_worktree(self, task_id: str, path: str, branch: str) -> None:
        """Persist worktree path+branch for task_id so resume can reattach.

        Locked read-modify-write: parallel tasks register their worktrees
        concurrently and an unguarded write would lose entries (last-write-wins),
        orphaning the dropped task's worktree + branch on resume.
        """
        self._ensure_dir()
        p = self.dir / "worktrees.json"
        with _WORKTREE_LOCK:
            try:
                data: dict[str, Any] = json.loads(p.read_text()) if p.exists() else {}
            except json.JSONDecodeError:
                data = {}
            data[task_id] = {"path": path, "branch": branch}
            p.write_text(json.dumps(data, indent=2))

    def read_worktrees(self) -> dict[str, dict[str, str]]:
        """Return {task_id: {path, branch}} map persisted by set_worktree."""
        p = self.dir / "worktrees.json"
        if not p.exists():
            return {}
        try:
            loaded: dict[str, Any] = json.loads(p.read_text())
            return {k: dict(v) for k, v in loaded.items() if isinstance(v, dict)}
        except json.JSONDecodeError:
            return {}

    def is_empty(self) -> bool:
        """No real work captured — only a status stamp / blank scaffolding.

        Used to garbage-collect sessions that get a ``refining`` status the
        moment a PRD UI mounts but are abandoned before anything is written.
        """
        if not self.dir.exists():
            return True
        for name in ("prd.md", "loop.md", "eval.md"):
            if self.read(name).strip():
                return False
        events_path = self.dir / "events.jsonl"
        if events_path.exists() and events_path.stat().st_size > 0:
            return False
        kdir = self.dir / "knowledge"
        if kdir.exists() and any(p.stat().st_size > 0 for p in kdir.glob("*.md")):
            return False
        return True
