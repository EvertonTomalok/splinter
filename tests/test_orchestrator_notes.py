from __future__ import annotations

from pathlib import Path

import pytest

from splinter.memory.orchestrator_notes import close_pending, notes_for
from splinter.memory.session import Session


@pytest.fixture
def session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Session:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    return Session("ses_notes")


def test_note_reaches_direct_consumers_once(session: Session) -> None:
    note_id = session.queue_orchestrator_note("preserve Retry-After")

    assert "## Pending" in session.read("knowledge/orchestrator-notes.md")
    assert "preserve Retry-After" in notes_for(session, "replan", "raphael")
    assert "preserve Retry-After" in notes_for(session, "final_eval", "raphael")
    assert "## Applied" in session.read("knowledge/orchestrator-notes.md")
    assert notes_for(session, "final_eval", "raphael") == ""
    assert session.receipt_exists(note_id)


def test_close_pending_marks_note_as_run_ended(session: Session) -> None:
    note_id = session.queue_orchestrator_note("defer story 31")

    close_pending(session)

    content = session.read("knowledge/orchestrator-notes.md")
    assert "## Pending\n\n## Applied" in content
    assert "(run ended)" in content
    assert session.receipt_exists(note_id)
