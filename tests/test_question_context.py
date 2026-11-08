"""Question context: the PRD excerpt, the explain prompt and the ``?`` pane.

The cheap model is never called: ``run_provider_session`` and the app's
``explain`` seam are replaced in every test.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from textual.widgets import Static

from splinter import prd_session, question_context
from splinter.memory.session import Session, new_session_id
from splinter.prd_session import Turn
from splinter.question_context import (
    Explanation,
    clean_explanation,
    explain,
    explain_prompt,
    prd_excerpt,
)
from splinter.tui import PrdSessionApp
from splinter.tui_questions import parse_questions

PRD = """---
feature: relatorios
---

# INS-322 — Relatórios core

## 1. Visão geral

Ama Hub possui onze relatórios financeiros.

## 4. Gates

- G0: aprovação das fórmulas antes de iniciar US-002.
- G1: revisão de performance depois do piloto.

### US-002: Relatório de agenda e produção

Mostra a produção clínica por profissional.

## 9. Fora de escopo

Exportação para Excel.
"""

QUESTIONS = (
    "1. [aprovação do gate] Gate G0: qual aprovação vale para iniciar US-002?\n"
    "Why: a seção 4 trava US-002 até G0 ser aprovado.\n"
    "A) Aprovar todas as fórmulas — piloto e pós-piloto juntos\n"
    "B) Só fórmulas do piloto — o resto ganha novo gate\n"
)


def _question() -> Any:
    return parse_questions(QUESTIONS)[0]


def test_why_line_becomes_the_question_context() -> None:
    question = _question()
    assert question.label == "aprovação do gate"
    assert question.context == "a seção 4 trava US-002 até G0 ser aprovado."
    assert [o.letter for o in question.options] == ["A", "B"]


def test_excerpt_finds_the_sections_that_name_the_ids() -> None:
    sections = prd_excerpt(PRD, _question())
    headings = [section.heading for section in sections]
    assert headings[0] == "4. Gates"
    assert "G0: aprovação das fórmulas antes de iniciar US-002." in sections[0].lines[0]
    assert "9. Fora de escopo" not in headings


def test_prompt_asks_for_the_question_language_and_forbids_filler() -> None:
    prompt = explain_prompt(_question(), prd_excerpt(PRD, _question()), "INS-322")
    assert "same language the question is written in" in prompt
    assert "no praise of the question" in prompt
    assert "never invent it" in prompt
    assert "A) Aprovar todas as fórmulas" in prompt
    assert "[4. Gates]" in prompt


def test_clean_explanation_drops_markdown() -> None:
    assert clean_explanation("```\n## Resumo\n**A** muda tudo\n\n\n\nfim\n```") == (
        "Resumo\nA muda tudo\n\nfim"
    )


def test_explain_uses_the_tier_zero_model(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_session(prompt: str, model: str, **kw: Any) -> tuple[Any, str]:
        calls.append({"model": model, **kw})
        result = SimpleNamespace(text="Decide quem aprova.", tokens={"input": 9}, cost=0.0002)
        return result, "sid"

    monkeypatch.setattr(question_context, "run_provider_session", fake_session)
    monkeypatch.setattr(question_context, "explain_model", lambda: ("cheap/flash", ["cheap/flash"]))
    out = explain(_question(), [], "")
    assert out == Explanation(
        text="Decide quem aprova.",
        model="cheap/flash",
        cost=0.0002,
        tokens={"input": 9, "output": 0},
    )
    assert calls[0]["model"] == "cheap/flash"
    assert calls[0]["role"] == "explain"


def test_question_mark_opens_the_context_pane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(
        prd_session,
        "_ask",
        lambda prompt, *, resume, session=None: Turn(text=QUESTIONS, session_id="s", model="m"),
    )
    asked: list[str] = []

    def fake_explain(question: Any, sections: Any, title: str) -> Explanation:
        asked.append(question.text)
        return Explanation(text="Decide quem aprova.\nA libera tudo.", model="cheap", cost=0.0001)

    monkeypatch.setattr("splinter.tui.explain", fake_explain)
    monkeypatch.setattr("splinter.tui.explain_model", lambda: ("cheap", ["cheap"]))
    src = tmp_path / "ins-322-prd.md"
    src.write_text(PRD)
    kwargs: dict[str, Any] = {"strategy": "cascade", "prd_path": str(src), "no_ground": True}
    session = Session(new_session_id())
    app = PrdSessionApp(session, kwargs)

    async def drive() -> None:
        async with app.run_test(size=(160, 44)) as pilot:
            for _ in range(3):
                await app.workers.wait_for_complete()
                await pilot.pause()
            await pilot.press("question_mark")
            for _ in range(3):
                await app.workers.wait_for_complete()
                await pilot.pause()
            assert app.has_class("-context")
            body = str(app.query_one("#context-body", Static).render())
            assert "4. Gates" in body
            assert "Decide quem aprova." in body
            assert asked == ["Gate G0: qual aprovação vale para iniciar US-002?"]
            assert "cheap" in session.read_pre_run_usage()["models"]
            await pilot.press("escape")
            await pilot.pause()
            assert not app.has_class("-context")
            await pilot.press("question_mark")
            await pilot.pause()
            assert len(asked) == 1

    asyncio.run(drive())
