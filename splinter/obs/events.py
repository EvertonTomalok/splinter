"""Canonical event log: one JSONL file per session, source of truth for both the
run trace (cost/tokens/latency per model call) and the chronological log
(human-readable stage/message lines). ``trace.md``, ``events.md`` and every
``analyze`` view are on-demand renderers over this file — nothing else writes
them eagerly.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from splinter.memory.session import Session
from splinter.obs.errors import report_obs_error

if TYPE_CHECKING:
    from splinter.obs.trace import RunEntry

log = logging.getLogger("splinter.events")

EVENTS_FILENAME = "events.jsonl"

#: Baseline schema stamped onto every migrated legacy record (US-008).
SCHEMA_VERSION = 1

#: Guards the append so parallel cascade workers (and the log handler) never
#: interleave partial JSON lines in the same session file.
_LOCK = threading.Lock()

#: Legacy ``events.md`` line prefix: ``[HH:MM:SS] message`` or ``[STAGE] message``.
_EVENTS_TAG_RE = re.compile(r"^\[([^\]]+)\]\s*(.*)$")

#: Legacy ``trace.md`` run line (the retired ``Trace.from_markdown`` format).
_LEGACY_RUN_RE = re.compile(
    r"- (?:task (\d+) )?iter (\d+): (.+?) \((?:T(\d+)|eval)\) "
    r"tokens=\{([^}]*)\} cost=\$([\d.]+)( \[!cost\])? ([\d.]+)s(?: @ (\S+))?"
)


@dataclass(frozen=True)
class Event:
    type: Literal[
        "run",
        "log",
        "live_message_fired",
        "live_message_queued",
        "abnormal_termination",
    ]
    ts: str
    payload: dict[str, Any] = field(default_factory=dict)


def events_path(session: Session) -> Path:
    return session.dir / EVENTS_FILENAME


@dataclass
class LiveMessageFiredEvent:
    """A live user directive was fired directly into a running provider session."""

    task_no: int
    session_id: str
    summary: str
    schema_version: int = SCHEMA_VERSION

    def to_event(self) -> Event:
        return Event(
            type="live_message_fired",
            ts=datetime.now(timezone.utc).isoformat(),
            payload={
                "task_no": self.task_no,
                "session_id": self.session_id,
                "summary": self.summary,
                "schema_version": self.schema_version,
            },
        )

    def emit(self, session: Session) -> None:
        append_event(session, self.to_event())


@dataclass
class LiveMessageQueuedEvent:
    """A live user directive was queued for the next iteration (no live session active)."""

    task_no: int
    summary: str
    schema_version: int = SCHEMA_VERSION

    def to_event(self) -> Event:
        return Event(
            type="live_message_queued",
            ts=datetime.now(timezone.utc).isoformat(),
            payload={
                "task_no": self.task_no,
                "summary": self.summary,
                "schema_version": self.schema_version,
            },
        )

    def emit(self, session: Session) -> None:
        append_event(session, self.to_event())


@dataclass
class AbnormalTerminationEvent:
    """A task run ended unexpectedly (crash, timeout, unhandled error)."""

    task_no: int
    reason: str
    schema_version: int = SCHEMA_VERSION

    def to_event(self) -> Event:
        return Event(
            type="abnormal_termination",
            ts=datetime.now(timezone.utc).isoformat(),
            payload={
                "task_no": self.task_no,
                "reason": self.reason,
                "schema_version": self.schema_version,
            },
        )

    def emit(self, session: Session) -> None:
        append_event(session, self.to_event())


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
    migrate_legacy(session)  # one-shot: build events.jsonl from legacy files if missing
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


# --- US-008: one-shot legacy migration ------------------------------------


def _legacy_run_records(session: Session) -> list[dict[str, Any]]:
    """Parse legacy ``trace.md`` run lines into canonical ``run`` records so cost
    and per-model views survive migration. Malformed lines route to errors.jsonl."""
    trace_md = session.dir / "trace.md"
    if not trace_md.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        text = trace_md.read_text()
    except OSError as exc:
        report_obs_error(session, "events.migrate_legacy", "read", exc, detail="trace.md")
        return []
    for m in _LEGACY_RUN_RE.finditer(text):
        try:
            tokens: dict[str, int] = {
                tm.group(1): int(tm.group(2))
                for tm in re.finditer(r"'(\w+)':\s*(\d+)", m.group(5))
            }
            is_eval = m.group(4) is None
            records.append(
                {
                    "type": "run",
                    "ts": m.group(9) or "",
                    "payload": {
                        "model": m.group(3),
                        "tier": 0 if is_eval else int(m.group(4)),
                        "iteration": int(m.group(2)),
                        "tokens": tokens,
                        "cost": float(m.group(6)),
                        "latency_s": float(m.group(8)),
                        "task": int(m.group(1)) if m.group(1) else 0,
                        "role": "eval" if is_eval else "run",
                        "cost_indeterminate": m.group(7) is not None,
                        "schema_version": SCHEMA_VERSION,
                    },
                }
            )
        except (ValueError, TypeError) as exc:
            report_obs_error(
                session, "events.migrate_legacy", "decode", exc, detail="trace.md run line"
            )
    return records


def _legacy_log_records(session: Session) -> list[dict[str, Any]]:
    """Rebuild ``log`` records from legacy ``events.md`` (preferred: preserves the
    verbatim ``raw`` line, so the rendered log is byte-identical) falling back to
    ``events.compact.jsonl``. Malformed compact lines route to errors.jsonl."""
    events_md = session.dir / "events.md"
    if events_md.exists():
        records: list[dict[str, Any]] = []
        try:
            lines = events_md.read_text().splitlines()
        except OSError as exc:
            report_obs_error(session, "events.migrate_legacy", "read", exc, detail="events.md")
            return []
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            tag = _EVENTS_TAG_RE.match(line)
            stage = tag.group(1).strip().lower().replace(" ", "_") if tag else "event"
            message = (tag.group(2).strip() or line) if tag else line
            records.append(
                {
                    "type": "log",
                    "ts": "",
                    "payload": {
                        "stage": stage,
                        "message": message,
                        "raw": line,
                        "task": 0,
                        "schema_version": SCHEMA_VERSION,
                    },
                }
            )
        return records

    compact = session.dir / "events.compact.jsonl"
    if not compact.exists():
        return []
    records = []
    for raw in compact.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            report_obs_error(
                session, "events.migrate_legacy", "decode", exc, detail="events.compact.jsonl"
            )
            continue
        message = str(data.get("message", ""))
        records.append(
            {
                "type": "log",
                "ts": str(data.get("ts", "")),
                "payload": {
                    "stage": str(data.get("stage", "event")),
                    "message": message,
                    "raw": message,
                    "task": 0,
                    "schema_version": SCHEMA_VERSION,
                },
            }
        )
    return records


def migrate_legacy(session: Session) -> bool:
    """One-shot: if ``events.jsonl`` is absent, rebuild it from the legacy
    ``trace.md`` (run entries) + ``events.md``/``events.compact.jsonl`` (log).

    Idempotent: once ``events.jsonl`` exists this is a no-op, so it is safe to
    call on every load. Malformed legacy input never aborts — bad lines are
    routed to ``errors.jsonl`` and skipped. Returns ``True`` iff it migrated.
    """
    path = events_path(session)
    if path.exists():
        return False
    if not session.dir.exists():
        return False

    records = _legacy_run_records(session) + _legacy_log_records(session)
    if not records:
        return False

    try:
        session._ensure_dir()
        with _LOCK, open(path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
    except OSError as exc:
        report_obs_error(session, "events.migrate_legacy", "write", exc)
        return False
    log.info("events: migrated %d legacy record(s) into %s", len(records), path)
    return True
