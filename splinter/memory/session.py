from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _base_dir() -> Path:
    """Where session memory lives.

    ``SPLINTER_HOME`` overrides; otherwise a ``splinter`` dir under the system
    temp folder, so runs don't litter the working tree.
    """
    env = os.environ.get("SPLINTER_HOME")
    if env:
        return Path(env)
    return Path(tempfile.gettempdir()) / "splinter"


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


class Session:
    def __init__(self, session_id: str | None = None) -> None:
        self.id = resolve_session(session_id)
        self.dir = session_dir(self.id)

    def index_path(self) -> Path:
        return self.dir / "index.md"

    def read_index(self) -> str:
        p = self.index_path()
        if p.exists():
            return p.read_text()
        return ""

    def write(self, filename: str, content: str) -> Path:
        p = self.dir / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return p

    def append(self, filename: str, content: str) -> Path:
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
        d = self.dir / "knowledge"
        d.mkdir(exist_ok=True)
        return d

    def status_path(self) -> Path:
        return self.dir / "status.json"

    def set_status(self, state: str, **fields: Any) -> None:
        """Persist run state (running/completed/failed) plus arbitrary fields."""
        data = self.read_status()
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
