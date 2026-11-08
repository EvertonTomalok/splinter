from __future__ import annotations

import json
import tracemalloc
from collections.abc import Iterator
from pathlib import Path
from typing import Any, BinaryIO

import pytest

from splinter.memory.session import Session
from splinter.obs.event_reader import RunEntryReader
from splinter.obs.trace import RunEntry, Trace


@pytest.fixture
def session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Session:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("incremental")
    Trace(session).add_entry(RunEntry("flash", 0, 1, {}, 0.01, 1.0))
    return session


def test_partial_utf8_record_waits_for_newline(session: Session) -> None:
    reader = RunEntryReader()
    assert len(reader.read(session)) == 1
    path = session.dir / "events.jsonl"
    record = json.loads(path.read_text())
    record["payload"]["model"] = "módelo"
    data = (json.dumps(record, ensure_ascii=False) + "\n").encode()
    cut = data.index("ó".encode()) + 1
    with path.open("ab") as stream:
        stream.write(data[:cut])
    assert len(reader.read(session)) == 1
    with path.open("ab") as stream:
        stream.write(data[cut:-1])
    assert len(reader.read(session)) == 1
    with path.open("ab") as stream:
        stream.write(b"\n")
    assert [entry.model for entry in reader.read(session)] == ["flash", "módelo"]
    assert reader.read(session) == Trace.from_jsonl(session).entries


@pytest.mark.parametrize("mode", ["truncate", "replace", "rewrite", "regrow"])
def test_changed_log_resets_without_duplicate_cost(session: Session, mode: str) -> None:
    reader = RunEntryReader()
    reader.read(session)
    path = session.dir / "events.jsonl"
    record = json.loads(path.read_text())
    record["payload"]["model"] = "other"
    record["payload"]["cost"] = 0.25
    data = json.dumps(record) + "\n"
    if mode == "replace":
        replacement = path.with_suffix(".new")
        replacement.write_text(data)
        replacement.replace(path)
    elif mode == "truncate":
        path.write_text("")
        assert reader.read(session) == []
        path.write_text(data)
    else:
        path.write_text(data * (2 if mode == "regrow" else 1))
    assert reader.read(session) == Trace.from_jsonl(session).entries
    assert reader.read(session)[0].cost == 0.25


def test_deleted_log_and_new_session_reset_reader(session: Session) -> None:
    reader = RunEntryReader()
    reader.read(session)
    (session.dir / "events.jsonl").unlink()
    assert reader.read(session) == []
    other = Session("other")
    Trace(other).add_entry(RunEntry("sonnet", 1, 1, {}, 0.25, 2.0))
    assert reader.read(other) == Trace.from_jsonl(other).entries


def test_malformed_complete_lines_do_not_block_later_entries(session: Session) -> None:
    reader = RunEntryReader()
    reader.read(session)
    path = session.dir / "events.jsonl"
    with path.open("ab") as stream:
        stream.write(b"not-json\n[]\n\xff\n")
    Trace(session).add_entry(RunEntry("sonnet", 1, 1, {}, 0.25, 2.0))
    assert [entry.model for entry in reader.read(session)] == ["flash", "sonnet"]
    revision = reader.revision
    reader.read(session)
    assert reader.revision == revision


def test_reader_migrates_legacy_trace(session: Session) -> None:
    (session.dir / "events.jsonl").unlink()
    session.write(
        "trace.md",
        "- iter 1: flash (T0) tokens={'input': 1} cost=$0.0100 1.0s\n",
    )
    assert RunEntryReader().read(session) == Trace.from_jsonl(session).entries


def test_log_only_history_has_bounded_peak_memory(session: Session) -> None:
    line = json.dumps({"type": "log", "payload": {"message": "x" * 160}}) + "\n"
    session.write("events.jsonl", line * 10_000)
    reader = RunEntryReader()
    tracemalloc.start()
    try:
        assert reader.read(session) == []
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1_000_000, f"log-only history retained {peak:,} bytes"


def test_invalid_run_payload_does_not_lose_following_entries(session: Session) -> None:
    path = session.dir / "events.jsonl"
    malformed = json.loads(path.read_text())
    malformed["payload"]["tokens"] = [1, 2]
    with path.open("a") as stream:
        stream.write(json.dumps(malformed) + "\n")
    Trace(session).add_entry(RunEntry("sonnet", 1, 1, {}, 0.25, 2.0))
    reader = RunEntryReader()
    assert [entry.model for entry in reader.read(session)] == ["flash", "sonnet"]
    assert reader.read(session) == Trace.from_jsonl(session).entries


def test_read_failure_does_not_advance_past_uncommitted_entries(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    reader = RunEntryReader()
    records = reader._records

    def interrupted(stream: BinaryIO, size: int) -> Iterator[dict[str, Any]]:
        yield next(records(stream, size))
        raise OSError("interrupted read")

    with monkeypatch.context() as scoped:
        scoped.setattr(reader, "_records", interrupted)
        with pytest.raises(OSError, match="interrupted read"):
            reader.read(session)
    assert reader.read(session) == Trace.from_jsonl(session).entries
    assert len(reader.entries) == 1
