"""Question form display modes: one at a time or all at once (T08).

Widget-level tests run a bare form inside a host app; the pilots drive the
real PRD and plan apps with every planner turn mocked at ``_ask`` so the
answers can be traced into the exact prompt that reaches the provider seam.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
import yaml
from textual.app import App, ComposeResult
from textual.widgets import Button, Static, TextArea

from splinter import plan_session, prd_session
from splinter.memory.session import Session, new_session_id
from splinter.prd_session import Turn
from splinter.tui import PrdSessionApp
from splinter.tui_plan import PlanSessionApp
from splinter.tui_questions import QuestionForm, parse_questions
from splinter.tui_questions_one import AnsweredList, AnswerRow, QuestionIndex
from splinter.tui_stepper import PhaseStepper
from splinter.ui_prefs import questions_mode

QUESTIONS_TEXT = (
    "1. Which provider?\n"
    "A) Stripe\n"
    "B) Adyen\n"
    "C) Both\n"
    "\n"
    "2. What triggers a refund?\n"
    "A) Webhook\n"
    "B) Admin\n"
    "\n"
    "3. How is currency rounded?\n"
    "A) Banker's\n"
    "B) Half up\n"
)
PLAN_QUESTIONS_TEXT = (
    "1. Where should the flag live?\nA) cli.py  B) config.py\n\n2. Default value?\nA) on  B) off"
)
PRD = "---\nfeature: demo\nstrategy: cascade\n---\n\n# Demo\n\n### US-001: Demo story\n"
SIZE = (120, 40)


class _FormApp(App[None]):
    def __init__(self, form: QuestionForm) -> None:
        super().__init__()
        self.form = form
        self.completed = 0

    def compose(self) -> ComposeResult:
        yield QuestionIndex(id="question-index")
        yield self.form

    def on_question_form_answers_changed(self, event: QuestionForm.AnswersChanged) -> None:
        self.query_one("#question-index", QuestionIndex).refresh_from(
            event.form.questions, event.answers
        )

    def on_question_form_completed(self, event: QuestionForm.Completed) -> None:
        self.completed += 1

    def action_jump_question(self, n: int | str) -> None:
        self.form.jump_to(int(n))


class _StepperApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.clicked: list[int] = []

    def compose(self) -> ComposeResult:
        yield PhaseStepper(id="stepper")

    def on_phase_stepper_step_clicked(self, event: PhaseStepper.StepClicked) -> None:
        self.clicked.append(event.index)


def _form_app(prefs_root: Path | None = None) -> _FormApp:
    questions = parse_questions(QUESTIONS_TEXT)
    return _FormApp(QuestionForm(questions, prefs_root=prefs_root))


def _counter(app: _FormApp) -> str:
    return str(app.query_one("#q-counter", Static).render())


async def _settle(app: Any, pilot: Any) -> None:
    for _ in range(2):
        await app.workers.wait_for_complete()
        await pilot.pause()


def test_one_mode_picks_advances_and_keeps_answer() -> None:
    app = _form_app()

    async def drive() -> None:
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause()
            assert app.form.mode == "one"
            assert _counter(app) == "1 / 3"
            await pilot.click("#ans-B")
            await pilot.pause()
            assert _counter(app) == "2 / 3"
            assert app.form.compile_answer("") == "1B"
            await pilot.press("left")
            await pilot.pause()
            assert _counter(app) == "1 / 3"
            assert "-picked" in app.query_one("#ans-B", AnswerRow).classes
            assert app.form.compile_answer("") == "1B"

    asyncio.run(drive())


def test_toggle_mode_keeps_answers_and_compile_equal() -> None:
    app = _form_app()

    async def drive() -> None:
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause()
            await pilot.click("#ans-B")
            await pilot.pause()
            one_mode = app.form.compile_answer("notes")
            await pilot.press("m")
            await pilot.pause()
            assert app.form.mode == "all"
            radios = list(app.query("#question-form RadioSet"))
            assert len(radios) == 3
            assert all(radio.display for radio in radios)
            assert app.query_one("#q-page").display is False
            picked = [b for b in app.query("#question-form RadioButton") if b.value]
            assert len(picked) == 1
            assert app.form.compile_answer("notes") == one_mode == "1B + notes"
            app.form.action_toggle_mode()
            await pilot.pause()
            assert app.form.mode == "one"
            assert app.form.compile_answer("notes") == "1B + notes"

    asyncio.run(drive())


def test_mode_persisted_to_ui_yaml(tmp_path: Path) -> None:
    root = tmp_path / ".splinter"
    assert questions_mode(root) == "one"
    app = _form_app(prefs_root=root)

    async def drive() -> None:
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause()
            app.form.action_toggle_mode()
            await pilot.pause()
            data = yaml.safe_load((root / "ui.yaml").read_text())
            assert data == {"questions_mode": "all"}
            assert questions_mode(root) == "all"
            form = QuestionForm(parse_questions(QUESTIONS_TEXT), prefs_root=root)
            assert form.mode == "all"

    asyncio.run(drive())


def test_question_index_row_click_jumps() -> None:
    app = _form_app()

    async def drive() -> None:
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause()
            index = app.query_one("#question-index", QuestionIndex)
            assert index.display is False
            await pilot.click("#ans-A")
            await pilot.pause()
            assert index.display is True
            await pilot.click("#question-index", offset=(2, 2))
            await pilot.pause()
            assert _counter(app) == "3 / 3"

    asyncio.run(drive())


def test_stepper_posts_step_clicked_only_for_reached_steps() -> None:
    app = _StepperApp()

    async def drive() -> None:
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause()
            stepper = app.query_one("#stepper", PhaseStepper)
            offset = stepper.content_offset.x
            spans = stepper._spans
            assert spans
            await pilot.click(stepper, offset=(spans[2][0] + offset, 0))
            await pilot.pause()
            assert app.clicked == []
            stepper.set_current(3)
            await pilot.pause()
            spans = stepper._spans
            await pilot.click(stepper, offset=(spans[1][0] + offset, 0))
            await pilot.pause()
            assert app.clicked == [1]
            await pilot.click(stepper, offset=(spans[3][0] + offset, 0))
            await pilot.pause()
            assert app.clicked == [1]

    asyncio.run(drive())


def test_prd_one_mode_answers_reach_send_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path / "home"))
    prompts: list[str] = []

    def fake_ask(prompt: str, *, resume: str | None, session: object = None) -> Turn:
        prompts.append(prompt)
        if "User answers:" in prompt:
            return Turn(
                text="## Working Draft\n\nsame\n\n## Open Questions\nNone — ready to finalize.",
                session_id="sid",
                model="sonnet",
            )
        return Turn(text=QUESTIONS_TEXT, session_id="sid", model="sonnet")

    monkeypatch.setattr(prd_session, "_ask", fake_ask)
    src = tmp_path / "demo-prd.md"
    src.write_text(PRD)
    run_kwargs: dict[str, Any] = {
        "strategy": None,
        "prd_path": str(src),
        "task_path": None,
        "effort": None,
        "budget": None,
        "max_iterations": 5,
        "cowabunga": False,
        "no_ground": True,
    }
    app = PrdSessionApp(Session(new_session_id()), run_kwargs)

    async def drive() -> None:
        async with app.run_test(size=SIZE) as pilot:
            await _settle(app, pilot)
            form = app.query_one("#question-form", QuestionForm)
            assert form.mode == "one"
            await pilot.click("#ans-A")
            await pilot.pause()
            await pilot.click("#ans-A")
            await pilot.pause()
            await pilot.click("#ans-B")
            await pilot.pause()
            assert "finalize" in str(app.query_one("#q-keys", Static).render())
            form.action_toggle_mode()
            assert form.compile_answer("notes") == "1A, 2A, 3B + notes"
            form.action_toggle_mode()
            assert form.mode == "one"
            app.query_one("#entry", TextArea).text = "notes"
            app.action_send()
            await _settle(app, pilot)
            sent = next(p for p in prompts if "User answers:" in p)
            assert "1A, 2A, 3B + Q3: notes" in sent

    asyncio.run(drive())


def test_plan_one_mode_answers_reach_send_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path / "home"))
    path = tmp_path / "hello.yaml"
    path.write_text(
        yaml.safe_dump({"description": "Add a hello command", "acceptance": "- prints hello"})
    )
    prompts: list[str] = []

    def fake_ask(prompt: str, *, resume: str | None, session: object = None) -> Turn:
        prompts.append(prompt)
        if "User answers:" in prompt:
            return Turn(text="## Decisions\n- cli.py", session_id="sid", model="sonnet")
        return Turn(text=PLAN_QUESTIONS_TEXT, session_id="sid", model="sonnet")

    monkeypatch.setattr(plan_session, "_ask", fake_ask)
    kwargs: dict[str, Any] = {
        "task_path": str(path),
        "strategy": None,
        "no_ground": True,
    }
    app = PlanSessionApp(Session("ses_qmodes"), kwargs)

    async def drive() -> None:
        async with app.run_test(size=SIZE) as pilot:
            await _settle(app, pilot)
            form = app.query_one("#question-form", QuestionForm)
            assert form.mode == "one"
            await pilot.click("#ans-A")
            await pilot.pause()
            await pilot.click("#ans-B")
            await pilot.pause()
            assert app.focused is app.query_one("#write-plan", Button)
            app.query_one("#entry", TextArea).text = "keep it small"
            app.action_send()
            await _settle(app, pilot)
            sent = next(p for p in prompts if "User answers:" in p)
            assert "1A, 2B + keep it small" in sent

    asyncio.run(drive())


def test_question_label_and_reason_are_parsed() -> None:
    text = "1. [primary provider] Which provider?\nA) Stripe — live traffic\nB) Adyen\n"
    question = parse_questions(text)[0]
    assert question.label == "primary provider"
    assert question.text == "Which provider?"
    assert question.options[0].text == "Stripe — live traffic"


def test_answered_list_shows_picks_and_dashes() -> None:
    questions = parse_questions("1. [p] Q?\nA) Stripe — live\nB) Adyen\n\n2. R?\nA) x\nB) y\n")

    class _App(App[None]):
        def compose(self) -> ComposeResult:
            yield AnsweredList(id="answered")

    app = _App()

    async def drive() -> None:
        async with app.run_test(size=SIZE) as pilot:
            answered = app.query_one(AnsweredList)
            answered.refresh_from(questions, {1: "A"})
            await pilot.pause()
            rendered = str(answered.render())
            assert "1 Stripe" in rendered
            assert "2 —" in rendered
            assert "live" not in rendered

    asyncio.run(drive())


def test_prd_next_on_last_question_finalizes_with_the_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path / "home"))
    prompts: list[str] = []

    def fake_ask(prompt: str, *, resume: str | None, session: object = None) -> Turn:
        prompts.append(prompt)
        if "output the COMPLETE final PRD" in prompt:
            return Turn(text=PRD, session_id="sid", model="sonnet")
        return Turn(text=QUESTIONS_TEXT, session_id="sid", model="sonnet")

    monkeypatch.setattr(prd_session, "_ask", fake_ask)
    src = tmp_path / "demo-prd.md"
    src.write_text(PRD)
    run_kwargs: dict[str, Any] = {
        "strategy": "cascade",
        "prd_path": str(src),
        "task_path": None,
        "cowabunga": False,
        "no_ground": True,
    }
    app = PrdSessionApp(Session(new_session_id()), run_kwargs)
    app._to_strategy_phase = lambda: None  # type: ignore[method-assign]

    async def drive() -> None:
        async with app.run_test(size=SIZE) as pilot:
            await _settle(app, pilot)
            assert app.has_class("-asking")
            outline = str(app.query_one("#outline", Static).render())
            assert "demo-prd.md · 1 story" in outline
            assert "US-001 Demo story" in outline
            await pilot.click("#ans-A")
            await pilot.pause()
            await pilot.press("right")
            await pilot.pause()
            assert "finalize" in str(app.query_one("#q-keys", Static).render())
            await pilot.press("right")
            await _settle(app, pilot)
            final = next(p for p in prompts if "output the COMPLETE final PRD" in p)
            assert "User answers:\n1A" in final

    asyncio.run(drive())


def test_label_line_names_the_question() -> None:
    text = (
        "1. What exact text must hello.py print?\n"
        "Label: output text\n"
        "Why: US-001 only requires stdout to contain hello.\n"
        "A) hello world — plain print\n"
        "B) Hello, World! — conventional\n"
    )
    question = parse_questions(text)[0]
    assert question.label == "output text"
    assert question.text == "What exact text must hello.py print?"
    assert question.context == "US-001 only requires stdout to contain hello."
    assert [o.letter for o in question.options] == ["A", "B"]
