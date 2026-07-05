"""Canonical event log: one JSONL file per session, source of truth for both the
run trace (cost/tokens/latency per model call) and the chronological log
(human-readable stage/message lines). ``trace.md``, ``events.md`` and every
``analyze`` view are on-demand renderers over this file — nothing else writes
them eagerly.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from splinter.memory.session import Session

if TYPE_CHECKING:
    from splinter.obs.trace import RunEntry

log = logging.getLogger("splinter.events")

EVENTS_FILENAME = "events.jsonl"

#: Guards the append so parallel cascade workers (and the log handler) never
#: interleave partial JSON lines in the same session file.
_LOCK = threading.Lock()


@dataclass(frozen=True)
class Event:
    type: Literal["run", "log"]
    ts: str
    payload: dict[str, Any] = field(default_factory=dict)


def events_path(session: Session) -> Path:
    return session.dir / EVENTS_FILENAME


def append_event(session: Session, event: Event) -> None:
    """Append one JSON line to ``{session.dir}/events.jsonl``. Thread-safe.

    Raises on I/O failure instead of swallowing it — a dropped event silently
    corrupts cost/trajectory accounting downstream, so callers must see it.
    """
    session._ensure_dir()
    path = events_path(session)
    line = json.dumps({"type": event.type, "ts": event.ts, "payload": event.payload}) + "\n"
    try:
        with _LOCK, open(path, "a") as f:
            f.write(line)
    except OSError:
        log.error("events: failed to append to %s", path, exc_info=True)
        raise


def _read_records(session: Session) -> list[dict[str, Any]]:
    path = events_path(session)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    malformed = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                malformed += 1
    if malformed:
        log.warning("events: skipped %d malformed line(s) in %s", malformed, path)
    return records


def load_run_entries(session: Session) -> list["RunEntry"]:
    """Deserialize every ``type: run`` record into a :class:`RunEntry`, in file order.

    Malformed records are skipped (counted + logged), never silently dropped
    without a trace.
    """
    from splinter.obs.trace import RunEntry

    path = events_path(session)
    entries: list[RunEntry] = []
    skipped = 0
    for rec in _read_records(session):
        if rec.get("type") != "run":
            continue
        payload = rec.get("payload") or {}
        try:
            entries.append(
                RunEntry(
                    model=str(payload["model"]),
                    tier=int(payload["tier"]),
                    iteration=int(payload["iteration"]),
                    tokens={str(k): int(v) for k, v in (payload.get("tokens") or {}).items()},
                    cost=float(payload["cost"]),
                    latency_s=float(payload.get("latency_s", 0.0)),
                    task=int(payload.get("task", 0)),
                    role=str(payload.get("role", "run")),
                    cost_indeterminate=bool(payload.get("cost_indeterminate", False)),
                    ts=str(rec.get("ts", "")),
                )
            )
        except (KeyError, TypeError, ValueError):
            skipped += 1
    if skipped:
        log.warning("events: skipped %d malformed run record(s) in %s", skipped, path)
    return entries


def load_log_events(session: Session) -> list[Event]:
    """All ``type: log`` records, in file order."""
    out: list[Event] = []
    for rec in _read_records(session):
        if rec.get("type") != "log":
            continue
        out.append(
            Event(type="log", ts=str(rec.get("ts", "")), payload=dict(rec.get("payload") or {}))
        )
    return out
