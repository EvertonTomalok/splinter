from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
        self._ensure_dir()
        p = self.dir / filename
        with open(p, "a") as f:
            f.write(content)
            if not content.endswith("\n"):
                f.write("\n")
        return p

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

    def is_empty(self) -> bool:
        """No real work captured — only a status stamp / blank scaffolding.

        Used to garbage-collect sessions that get a ``refining`` status the
        moment a PRD UI mounts but are abandoned before anything is written.
        """
        if not self.dir.exists():
            return True
        for name in ("prd.md", "loop.md", "trace.md", "eval.md"):
            if self.read(name).strip():
                return False
        kdir = self.dir / "knowledge"
        if kdir.exists() and any(p.stat().st_size > 0 for p in kdir.glob("*.md")):
            return False
        return True
