"""Persistent, once-per-consumer notes for the run orchestrator."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from splinter.memory.session import Session

REQUIRED: dict[str, frozenset[str]] = {
    "direct": frozenset({"replan", "final_eval"}),
    "agents": frozenset({"refine", "acceptance", "final_eval"}),
}

_LOCK = threading.Lock()
_NOTE_RE = re.compile(
    r"^### note ([\w-]+) · ([^\n]+)\nreads: ?([^\n]*)\n(.*?)(?=^### note |\Z)", re.M | re.S
)


@dataclass(frozen=True)
class OrchestratorNote:
    note_id: str
    timestamp: str
    reads: frozenset[str]
    text: str


def required_consumers(strategy: str) -> frozenset[str]:
    return REQUIRED["agents"] if strategy in {"agents", "splinter"} else REQUIRED["direct"]


_PATH = "knowledge/orchestrator-notes.md"


def _sections(raw: str) -> tuple[str, str]:
    pending = ""
    applied = ""
    if "## Pending" in raw:
        pending = raw.split("## Pending", 1)[1].split("## Applied", 1)[0].strip()
    if "## Applied" in raw:
        applied = raw.split("## Applied", 1)[1].strip()
    return pending, applied


def _parse(raw: str) -> list[OrchestratorNote]:
    notes: list[OrchestratorNote] = []
    for match in _NOTE_RE.finditer(raw.strip()):
        reads = frozenset(part.strip() for part in match.group(3).split(",") if part.strip())
        text = match.group(4).strip()
        notes.append(OrchestratorNote(match.group(1), match.group(2), reads, text))
    return notes


def _render_note(note: OrchestratorNote, *, ended: bool = False) -> str:
    suffix = " (run ended)" if ended else ""
    reads = ", ".join(sorted(note.reads))
    return (
        f"### note {note.note_id} · {note.timestamp}{suffix}\nreads: {reads}\n{note.text}"
    ).strip()


def _write(session: Session, pending: list[OrchestratorNote], applied: list[str]) -> None:
    pending_text = "\n\n".join(_render_note(note) for note in pending)
    applied_text = "\n\n".join(applied)
    session.write(
        _PATH,
        f"# Orchestrator notes\n\n## Pending\n{pending_text + chr(10) if pending_text else ''}"
        f"\n## Applied\n{applied_text}\n",
    )


def queue_note(session: Session, text: str, now: datetime) -> str:
    note_id = uuid4().hex[:12]
    with _LOCK:
        pending_text, applied_text = _sections(session.read(_PATH))
        pending = _parse(pending_text)
        applied = [_render_note(note) for note in _parse(applied_text)]
        pending.append(OrchestratorNote(note_id, now.isoformat(), frozenset(), text.strip()))
        _write(session, pending, applied)
    return note_id


def notes_for(session: Session, consumer: str, strategy: str) -> str:
    with _LOCK:
        pending_text, applied_text = _sections(session.read(_PATH))
        pending = _parse(pending_text)
        applied = [_render_note(note) for note in _parse(applied_text)]
        remaining: list[OrchestratorNote] = []
        delivered: list[OrchestratorNote] = []
        needed = required_consumers(strategy)
        for note in pending:
            if consumer in note.reads:
                remaining.append(note)
                continue
            updated = OrchestratorNote(
                note.note_id,
                note.timestamp,
                note.reads | {consumer},
                note.text,
            )
            delivered.append(updated)
            if needed <= updated.reads:
                applied.append(_render_note(updated))
                session.write_receipt(updated.note_id)
            else:
                remaining.append(updated)
        _write(session, remaining, applied)
    if not delivered:
        return ""
    lines = "\n".join(f"- {note.text}" for note in delivered)
    return f"## Orchestrator notes\n{lines}"


def close_pending(session: Session) -> None:
    with _LOCK:
        pending_text, applied_text = _sections(session.read(_PATH))
        pending = _parse(pending_text)
        applied = [_render_note(note) for note in _parse(applied_text)]
        for note in pending:
            applied.append(_render_note(note, ended=True))
            session.write_receipt(note.note_id)
        _write(session, [], applied)
