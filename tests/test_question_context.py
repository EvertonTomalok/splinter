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
    "\n"
    "2. Quem vê a lista de pacientes?\n"
    "A) Todos\n"
    "B) Só quem tem patients.view\n"
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
    assert "language the question is written in" in prompt
    assert "speak to the reader directly" in prompt
    assert "Never write: excerpt" in prompt
    assert "never invent it" in prompt
    assert "Do not run tools" in prompt
    assert "A) Aprovar todas as fórmulas" in prompt
    assert "Section 4. Gates:" in prompt
    assert "[4. Gates]" not in prompt


def test_clean_explanation_drops_markdown() -> None:
    assert clean_explanation("```\n## Resumo\n**A** muda tudo\n\n\n\nfim\n```") == (
        "Resumo\nA muda tudo\n\nfim"
    )


def test_explain_runs_low_effort_outside_the_repo(monkeypatch: pytest.MonkeyPatch) -> None:
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
    assert calls[0]["variant"] == "low"
    assert calls[0]["cwd"] != str(Path.cwd())


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


def test_explainer_is_haiku_then_tier_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    from splinter.models import roster

    tier = SimpleNamespace(models=["zai/glm-flash", "haiku"])
    monkeypatch.setattr(roster, "_project_config", lambda: {})
    monkeypatch.setattr(roster, "load_ladder", lambda: SimpleNamespace(tiers=[tier]))
    model, pool = question_context.explain_model()
    assert model == "haiku"
    assert pool == ["haiku@low", "zai/glm-flash@low"]


def test_models_explain_in_config_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    from splinter.models import roster

    config = {"models": {"explain": ["opencode/glm-flash", "sonnet@medium"]}}
    monkeypatch.setattr(roster, "_project_config", lambda: config)
    model, pool = question_context.explain_model()
    assert model == "opencode/glm-flash"
    assert pool == ["opencode/glm-flash@low", "sonnet@medium"]


def test_claude_explain_role_runs_text_only(monkeypatch: pytest.MonkeyPatch) -> None:
    from splinter.providers import claude_cli

    seen: list[list[str]] = []

    def fake_subprocess(cmd: list[str], **_: Any) -> Any:
        seen.append(cmd)
        line = '{"type":"result","result":"ok","usage":{}}'
        return SimpleNamespace(returncode=0, stdout=line, stderr="")

    monkeypatch.setattr(claude_cli, "run_subprocess", fake_subprocess)
    provider = claude_cli.ClaudeProvider()
    provider.run("hi", "haiku", variant="low", agent="explain", timeout=5)
    provider.run("hi", "haiku", variant="low", agent="prd", timeout=5)
    explain_cmd, prd_cmd = seen
    assert explain_cmd[explain_cmd.index("--setting-sources") + 1] == ""
    assert explain_cmd[explain_cmd.index("--tools") + 1] == ""
    assert "--strict-mcp-config" in explain_cmd
    assert "--setting-sources" not in prd_cmd


def test_context_pane_shows_the_loader_while_waiting() -> None:
    from splinter.tui_prd_outline import context_markup

    waiting = context_markup(_question(), [], model="haiku", waiting="⠹ 3s")
    assert "⠹ 3s" in waiting
    assert "haiku is reading the question" in waiting
    done = context_markup(_question(), [], explanation=Explanation(text="ok", model="haiku"))
    assert "reading the question" not in done


def _app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, asked: list[str]) -> PrdSessionApp:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(
        prd_session,
        "_ask",
        lambda prompt, *, resume, session=None: Turn(text=QUESTIONS, session_id="s", model="m"),
    )

    def fake_explain(question: Any, sections: Any, title: str) -> Explanation:
        asked.append(question.text)
        return Explanation(text="Explicado.", model="cheap")

    monkeypatch.setattr("splinter.tui.explain", fake_explain)
    monkeypatch.setattr("splinter.tui.explain_model", lambda: ("cheap", ["cheap"]))
    src = tmp_path / "ins-322-prd.md"
    src.write_text(PRD)
    kwargs: dict[str, Any] = {"strategy": "cascade", "prd_path": str(src), "no_ground": True}
    return PrdSessionApp(Session(new_session_id()), kwargs)


async def _settle(app: Any, pilot: Any) -> None:
    for _ in range(3):
        await app.workers.wait_for_complete()
        await pilot.pause()


def test_moving_on_with_the_pane_open_does_not_call_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[str] = []
    app = _app(tmp_path, monkeypatch, asked)

    async def drive() -> None:
        async with app.run_test(size=(160, 44)) as pilot:
            await _settle(app, pilot)
            await pilot.press("question_mark")
            await _settle(app, pilot)
            assert len(asked) == 1
            await pilot.press("right")
            await _settle(app, pilot)
            assert len(asked) == 1
            body = str(app.query_one("#context-body", Static).render())
            assert "? explains this one" in body
            await pilot.press("question_mark")
            await _settle(app, pilot)
            assert asked[-1] == "Quem vê a lista de pacientes?"
            assert app.has_class("-context")

    asyncio.run(drive())


def test_a_note_stays_with_its_question(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from textual.widgets import TextArea

    from splinter.tui_questions import QuestionForm

    asked: list[str] = []
    app = _app(tmp_path, monkeypatch, asked)
    spawned: list[dict[str, Any]] = []

    async def drive() -> None:
        async with app.run_test(size=(160, 44)) as pilot:
            await _settle(app, pilot)
            app._spawn = lambda fn, **kw: spawned.append(kw)  # type: ignore[method-assign]
            entry = app.query_one("#entry", TextArea)
            form = app.query_one("#question-form", QuestionForm)
            entry.text = "Não existe isso!"
            await pilot.pause()
            answered = str(app.query_one("#answered", Static).render())
            assert "1 Não existe isso!" in answered
            form.action_next_question()
            await pilot.pause()
            assert entry.text == ""
            assert "1 Não existe isso!" in str(app.query_one("#answered", Static).render())
            form.action_prev_question()
            await pilot.pause()
            assert entry.text == "Não existe isso!"
            form.action_next_question()
            await pilot.pause()
            entry.text = "só gestores"
            app.action_send()
            await pilot.pause()
            answers = [call["answers"] for call in spawned if "answers" in call]
            assert answers == ["Q1: Não existe isso!; Q2: só gestores"]

    asyncio.run(drive())
