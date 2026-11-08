from __future__ import annotations

from pathlib import Path

import pytest

from splinter.memory.session import Session


@pytest.fixture
def session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Session:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    return Session("ses_broadcast")


def test_broadcast_reaches_each_task_once(session: Session) -> None:
    directive_ids = session.broadcast_live_command("keep migrations untouched", [1, 3])

    assert session.directive_pending(1)
    assert session.directive_pending(3)
    assert session.pop_live_commands(1) == "keep migrations untouched\n---"
    assert session.pop_live_commands(3) == "keep migrations untouched\n---"
    assert not session.directive_pending(1)
    assert not session.directive_pending(3)
    assert all(session.receipt_exists(directive_id) for directive_id in directive_ids)


def test_legacy_directive_file_still_pops(session: Session) -> None:
    session.write("pending_directive.task-3.txt", "legacy instruction\n---\n")

    assert session.pop_live_commands(3) == "legacy instruction\n---"
