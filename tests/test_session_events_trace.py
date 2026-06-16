from __future__ import annotations

import json
from pathlib import Path

from splinter.memory.session import Session


def test_events_append_creates_tail_and_compact(tmp_path: Path, monkeypatch: "object") -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_events_files")

    session.append("events.md", "[RUN] first line")
    session.append("events.md", "second line")

    tail_md = session.dir / "events.tail.md"
    compact = session.dir / "events.compact.jsonl"
    compact_tail = session.dir / "events.compact.tail.jsonl"

    assert tail_md.exists()
    assert compact.exists()
    assert compact_tail.exists()

    rows = [json.loads(line) for line in compact.read_text().splitlines() if line.strip()]
    assert rows[0]["stage"] == "run"
    assert rows[0]["message"] == "first line"
    assert rows[1]["stage"] == "event"
    assert rows[1]["message"] == "second line"


def test_events_rotation_keeps_recent_file(tmp_path: Path, monkeypatch: "object") -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    monkeypatch.setenv("SPLINTER_EVENTS_MAX_BYTES", "80")
    monkeypatch.setenv("SPLINTER_EVENTS_ROTATIONS", "2")
    session = Session("ses_events_rotate")

    session.append("events.md", "a" * 40)
    session.append("events.md", "b" * 40)
    session.append("events.md", "c" * 40)

    assert (session.dir / "events.1.md").exists()
    assert "c" * 40 in session.read("events.md")


def test_events_tail_files_trimmed(tmp_path: Path, monkeypatch: "object") -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    monkeypatch.setenv("SPLINTER_EVENTS_TAIL_MAX_BYTES", "64")
    monkeypatch.setenv("SPLINTER_EVENTS_COMPACT_TAIL_MAX_BYTES", "256")
    session = Session("ses_events_trim")

    for i in range(30):
        session.append("events.md", f"[RUN] line-{i:03d}-" + ("z" * 20))

    tail_md = session.dir / "events.tail.md"
    compact_tail = session.dir / "events.compact.tail.jsonl"

    assert tail_md.exists()
    assert compact_tail.exists()
    assert tail_md.stat().st_size <= 64
    assert compact_tail.stat().st_size <= 256
    assert "line-029" in tail_md.read_text()

    rows = [json.loads(line) for line in compact_tail.read_text().splitlines() if line.strip()]
    assert rows[-1]["message"].startswith("line-029")
