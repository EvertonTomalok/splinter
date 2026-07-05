from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

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


_LINE_RE = re.compile(r"^t\d+-\d+$")


@pytest.mark.parametrize("n_threads,m_appends", [(2, 50), (8, 100), (16, 25)])
def test_append_is_thread_safe(
    tmp_path: Path, monkeypatch: "object", n_threads: int, m_appends: int
) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session(f"ses_concurrent_{n_threads}_{m_appends}")
    barrier = threading.Barrier(n_threads)

    def worker(tid: int) -> None:
        barrier.wait()
        for i in range(m_appends):
            session.append("loop.md", f"t{tid}-{i}")

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        list(pool.map(worker, range(n_threads)))

    lines = session.read("loop.md").splitlines()
    assert len(lines) == n_threads * m_appends
    for line in lines:
        assert _LINE_RE.match(line), f"torn/interleaved line: {line!r}"

    expected = {f"t{tid}-{i}" for tid in range(n_threads) for i in range(m_appends)}
    assert set(lines) == expected


def test_events_rotation_is_atomic_under_concurrency(tmp_path: Path, monkeypatch: "object") -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    monkeypatch.setenv("SPLINTER_EVENTS_MAX_BYTES", "200")
    monkeypatch.setenv("SPLINTER_EVENTS_ROTATIONS", "1000")
    session = Session("ses_concurrent_rotation")
    n_threads, m_appends = 8, 30
    barrier = threading.Barrier(n_threads)

    def worker(tid: int) -> None:
        barrier.wait()
        for i in range(m_appends):
            session.append("events.md", f"t{tid}-{i}")

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        list(pool.map(worker, range(n_threads)))

    rotation_re = re.compile(r"^events(\.\d+)?\.md$")
    all_lines: list[str] = []
    for path in sorted(session.dir.iterdir()):
        if rotation_re.match(path.name):
            all_lines.extend(line for line in path.read_text().splitlines() if line.strip())

    assert len(all_lines) == n_threads * m_appends
    for line in all_lines:
        assert _LINE_RE.match(line), f"torn/interleaved line across rotation: {line!r}"

    expected = {f"t{tid}-{i}" for tid in range(n_threads) for i in range(m_appends)}
    assert set(all_lines) == expected
