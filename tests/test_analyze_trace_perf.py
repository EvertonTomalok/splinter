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


def test_trace_refresh_appends_only_new_events(
    tmp_path: Path, monkeypatch: "object"
) -> None:
    import asyncio

    from textual.widgets import Markdown

    from splinter.tui import AnalyzeApp

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))  # type: ignore[attr-defined]
    session = Session("ses_trace_refresh")
    for i in range(5):
        session.append("events.md", f"[RUN] event-{i}")

    appended: list[str] = []
    updated: list[str] = []
    real_append = Markdown.append
    real_update = Markdown.update
    monkeypatch.setattr(  # type: ignore[attr-defined]
        Markdown, "append", lambda self, md: (appended.append(md), real_append(self, md))[1]
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        Markdown, "update", lambda self, md: (updated.append(md), real_update(self, md))[1]
    )

    async def drive() -> None:
        app = AnalyzeApp(session)
        async with app.run_test() as pilot:
            await pilot.pause()
            app._render_trace(force=True)  # full render anchors append tracking
            await pilot.pause()
            appended.clear()
            updated.clear()
            session.append("events.md", "[RUN] event-NEW")
            app._render_trace(force=False)  # refresh — should append, not reload
            await pilot.pause()
            assert any("event-NEW" in m for m in appended)
            assert not updated  # no full re-render
            await pilot.press("q")

    asyncio.run(drive())
