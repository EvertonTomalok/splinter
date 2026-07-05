from __future__ import annotations

import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from splinter.memory.session import Session


def test_events_append_routes_to_jsonl_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """events.md appends land in the canonical events.jsonl; the old 4-file
    fan-out (events.md, events.tail.md, events.compact*.jsonl) is gone."""
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_events_files")

    session.append("events.md", "[RUN] first line")
    session.append("events.md", "second line")

    assert (session.dir / "events.jsonl").exists()
    assert not (session.dir / "events.md").exists()
    assert not (session.dir / "events.tail.md").exists()
    assert not (session.dir / "events.compact.jsonl").exists()
    assert not (session.dir / "events.compact.tail.jsonl").exists()

    rendered = session.render_events_md()
    assert "first line" in rendered
    assert "second line" in rendered


def test_render_events_md_preserves_line_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_events_order")

    session.append("events.md", "[RUN] alpha")
    session.append("events.md", "[EVAL] beta")
    session.append("events.md", "gamma")

    rendered = session.render_events_md()
    assert rendered.index("alpha") < rendered.index("beta") < rendered.index("gamma")


def test_render_events_tail_trims_to_max_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_events_trim")

    for i in range(30):
        session.append("events.md", f"[RUN] line-{i:03d}-" + ("z" * 20))

    tail = session.render_events_tail(256)

    assert len(tail.encode("utf-8")) <= 256
    assert "line-029" in tail
    assert "line-000" not in tail


_LINE_RE = re.compile(r"^t\d+-\d+$")


@pytest.mark.parametrize("n_threads,m_appends", [(2, 50), (8, 100), (16, 25)])
def test_append_is_thread_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n_threads: int, m_appends: int
) -> None:
    """US-001: concurrent appends never tear or interleave (loop.md path)."""
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


def test_render_events_tail_returns_full_text_under_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_events_small")

    session.append("events.md", "one line only")

    full = session.render_events_md()
    tail = session.render_events_tail(64 * 1024)
    assert tail == full
