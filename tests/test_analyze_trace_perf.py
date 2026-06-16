from __future__ import annotations

from pathlib import Path

from splinter.memory.session import Session
from splinter.tui import TRACE_PAGE_BYTES, _trace_md, _trace_render


def test_trace_md_tails_large_events_file(tmp_path: Path, monkeypatch: "object") -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_trace_tail")
    session.write(
        "trace.md",
        "# Trace\n- total runs: 3\n- total cost: $0.0123\n"
        "- total tokens: {'input': 900, 'output': 400}\n",
    )
    lines = [f"line-{i:04d}" for i in range(5000)]
    session.write("events.md", "\n".join(lines))

    out = _trace_md(session)

    assert "offset" in out
    assert "line-4999" in out
    assert "line-0000" not in out
    assert len(out) < 400_000


def test_trace_md_keeps_full_small_events_file(tmp_path: Path, monkeypatch: "object") -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_trace_small")
    session.write("trace.md", "# Trace\n- total runs: 1\n- total cost: $0.0001\n")
    session.write("events.md", "step-1\nstep-2\nstep-3\n")

    out = _trace_md(session)

    assert "offset 0 bytes" in out
    assert "step-1" in out
    assert "step-3" in out


def test_trace_md_collapses_repeated_lines(tmp_path: Path, monkeypatch: "object") -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_trace_collapse")
    session.append("events.md", "[RUN] same event")
    session.append("events.md", "[RUN] same event")
    session.append("events.md", "[RUN] same event")
    session.append("events.md", "[EVAL] another event")

    out = _trace_md(session)

    assert "x3" in out
    assert "another event" in out


def test_trace_md_expand_selected_row(tmp_path: Path, monkeypatch: "object") -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_trace_expand")
    session.append("events.md", "[RUN] very long event payload for expansion")

    out = _trace_md(session, selected_row=1, expanded_row=1)

    assert "```" in out
    assert "very long event payload for expansion" in out


def test_trace_render_supports_offset_paging(tmp_path: Path, monkeypatch: "object") -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_trace_page")
    for i in range(5000):
        session.append("events.md", f"[RUN] page-line-{i:04d}")

    newest = _trace_render(session, offset_from_end=0)
    older = _trace_render(session, offset_from_end=TRACE_PAGE_BYTES)

    assert newest.has_older
    assert older.has_newer
