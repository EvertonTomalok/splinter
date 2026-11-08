"""A resumed PRD session in the questions phase shows its questions again.

No model is called: the app starts with ``resume=True`` and only reads the
session files.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from splinter.memory.session import Session, new_session_id
from splinter.tui import PrdSessionApp
from splinter.tui_questions import QuestionForm

PRD = "---\nfeature: demo\n---\n\n# Demo\n\n### US-001: Demo story\n"
QUESTIONS = (
    "1. Gate G0: qual aprovação vale para iniciar US-002?\n\n"
    "   A. Aprovar todas as fórmulas.\n"
    "   B. Aprovar só o piloto.\n\n"
    "2. A2: qual população entra na taxa de confirmação?\n\n"
    "   A. Todos os agendamentos.\n"
    "   B. Só os confirmáveis.\n"
)


def _session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Session:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path / "home"))
    session = Session(new_session_id())
    session.write("prd.md", PRD)
    session.set_status("refining", source="prd", phase="chat", claude_session="sid", strategy="?")
    return session


def _resume(session: Session) -> PrdSessionApp:
    kwargs: dict[str, Any] = {"prd_path": str(session.dir / "prd.md"), "resume": True}
    return PrdSessionApp(session, kwargs)


async def _questions(app: PrdSessionApp) -> list[str]:
    async with app.run_test(size=(160, 44)) as pilot:
        for _ in range(3):
            await app.workers.wait_for_complete()
            await pilot.pause()
        assert app.has_class("-asking")
        return [q.text for q in app.query_one("#question-form", QuestionForm).questions]


def test_old_session_recovers_questions_from_the_conversation_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _session(tmp_path, monkeypatch)
    log = ["[dim]PRD generated.[/]", QUESTIONS, "[cyan]> fulfilled[/]", "[red]ERROR: boom[/]"]
    session.write("convo.md", "\x00".join(log))
    texts = asyncio.run(_questions(_resume(session)))
    assert texts == [
        "Gate G0: qual aprovação vale para iniciar US-002?",
        "A2: qual população entra na taxa de confirmação?",
    ]


def test_questions_file_wins_over_the_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = _session(tmp_path, monkeypatch)
    session.write("convo.md", QUESTIONS)
    session.write(
        "questions.md", "1. Which provider?\nLabel: provider\nA) Stripe — live\nB) Adyen\n"
    )
    assert asyncio.run(_questions(_resume(session))) == ["Which provider?"]
