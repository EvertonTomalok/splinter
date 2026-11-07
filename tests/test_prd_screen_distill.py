"""Pilot tests for the distilled PRD conversation screen (brief C2).

Every model call is mocked: ``PrdSessionApp._spawn`` is patched so no worker
ever reaches a provider, and the callbacks that normally follow a model turn
(``_after_questions`` and friends) are driven directly.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from textual.widgets import Button, RadioButton, RadioSet, TextArea

from splinter.memory.session import Session, new_session_id
from splinter.tui import PrdSessionApp
from splinter.tui_questions import QuestionForm
from splinter.tui_status import BusyStatus
from splinter.tui_stepper import PhaseStepper

QUESTIONS = (
    "1. Which database should we use?\n"
    "A) Postgres\n"
    "B) SQLite\n"
    "\n"
    "2. Should the API be public?\n"
    "A) Yes\n"
    "B) No\n"
    "\n"
    "3. Which rollout strategy?\n"
    "A) Blue-green\n"
    "B) Canary\n"
)

PRD = """---
feature: demo
strategy: cascade
---

# Demo

### US-001: Demo story
"""


@pytest.fixture
def session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Session:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    return Session(new_session_id())


@pytest.fixture
def run_kwargs() -> dict[str, Any]:
    return {
        "strategy": None,
        "prd_path": None,
        "task_path": None,
        "effort": None,
        "budget": None,
        "max_iterations": 5,
        "cowabunga": False,
    }


def _app(session: Session, run_kwargs: dict[str, Any]) -> PrdSessionApp:
    app = PrdSessionApp(session, run_kwargs)
    app._spawn = lambda fn, **kw: None  # type: ignore[method-assign]
    return app


def _prd_file(tmp_path: Path) -> Path:
    src = tmp_path / "my-feature-prd.md"
    src.write_text(PRD)
    return src


def test_generate_phase_hides_right_composer_and_send_button(
    session: Session, run_kwargs: dict[str, Any]
) -> None:
    async def drive() -> None:
        app = _app(session, run_kwargs)
        async with app.run_test(size=(160, 44)) as pilot:
            await pilot.pause()
            assert app.query_one("#entry").display is False
            assert list(app.query("#send")) == []
            assert len(list(app.query("#generate"))) == 1
            assert app.query_one("#generate", Button).label == "Generate PRD"
            assert app.query_one("#stepper", PhaseStepper).current_label == "Draft"

    asyncio.run(drive())


def test_stepper_follows_the_current_phase(
    session: Session, run_kwargs: dict[str, Any], tmp_path: Path
) -> None:
    async def drive() -> None:
        run_kwargs["prd_path"] = str(_prd_file(tmp_path))
        app = _app(session, run_kwargs)
        async with app.run_test(size=(160, 44)) as pilot:
            await pilot.pause()
            stepper = app.query_one("#stepper", PhaseStepper)
            assert stepper.current_label == "Questions"
            assert app.sub_title == "Questions"

            app._to_strategy_phase()
            await pilot.pause()
            assert stepper.current_label == "Run setup"
            assert app.query_one("#cowabunga", Button).tooltip == (
                "Pick the recommended strategy and run"
            )

            from splinter.tui_run_setup import RunSetupModal

            assert isinstance(app.screen, RunSetupModal)

    asyncio.run(drive())


def test_question_form_renders_and_send_submits_answers(
    session: Session, run_kwargs: dict[str, Any], tmp_path: Path
) -> None:
    spawned: list[dict[str, Any]] = []

    async def drive() -> None:
        run_kwargs["prd_path"] = str(_prd_file(tmp_path))
        app = _app(session, run_kwargs)
        app._spawn = lambda fn, **kw: spawned.append(kw)  # type: ignore[method-assign]
        async with app.run_test(size=(160, 44)) as pilot:
            await pilot.pause()
            app._after_questions(QUESTIONS, "sid")
            await pilot.pause()

            form = app.query_one("#question-form", QuestionForm)
            assert form.display is True
            assert len(form.questions) == 3
            assert len(list(app.query("#question-form RadioSet"))) == 3

            first = app.query_one("#q-1 RadioButton", RadioButton)
            third_set = app.query_one("#q-3", RadioSet)
            third = list(third_set.query(RadioButton))[1]
            first.value = True
            third.value = True
            await pilot.pause()

            app.query_one("#entry", TextArea).text = "notes"
            app.action_send()
            await pilot.pause()

            assert spawned, "the answer was not sent"
            answers = [call["answers"] for call in spawned if "answers" in call]
            assert answers == ["1A, 3B + notes"]

    asyncio.run(drive())


def test_no_repeated_instruction_lines_in_the_conversation(
    session: Session, run_kwargs: dict[str, Any], tmp_path: Path
) -> None:
    async def drive() -> None:
        run_kwargs["prd_path"] = str(_prd_file(tmp_path))
        app = _app(session, run_kwargs)
        async with app.run_test(size=(160, 44)) as pilot:
            await pilot.pause()
            app._after_questions(QUESTIONS, "sid")
            await pilot.pause()
            convo = "\n".join(app._convo_lines)
            assert "Answer (e.g." not in convo
            assert "Answer remaining questions" not in convo
            assert "Type 'accept'" not in convo
            assert "Continue: answer" not in convo

    asyncio.run(drive())


def test_left_pane_title_names_the_prd_file(
    session: Session, run_kwargs: dict[str, Any], tmp_path: Path
) -> None:
    src = tmp_path / "my-feature-prd.md"
    src.write_text(PRD)

    async def drive() -> None:
        run_kwargs["prd_path"] = str(src)
        app = _app(session, run_kwargs)
        async with app.run_test(size=(160, 44)) as pilot:
            await pilot.pause()
            editor = app.query_one("#draft-edit")
            assert editor.border_title == "PRD · my-feature-prd.md"
            assert editor.border_subtitle == ""

    asyncio.run(drive())


def test_draft_editor_is_flagged_as_edited_after_a_change(
    session: Session, run_kwargs: dict[str, Any], tmp_path: Path
) -> None:
    src = tmp_path / "my-feature-prd.md"
    src.write_text(PRD)

    async def drive() -> None:
        run_kwargs["prd_path"] = str(src)
        app = _app(session, run_kwargs)
        async with app.run_test(size=(160, 44)) as pilot:
            await pilot.pause()
            editor = app.query_one("#draft-edit", TextArea)
            editor.text = PRD + "\nmore"
            await pilot.pause()
            assert editor.border_subtitle == "edited"

    asyncio.run(drive())


def test_busy_line_shows_action_and_hides_when_idle(
    session: Session, run_kwargs: dict[str, Any]
) -> None:
    async def drive() -> None:
        app = _app(session, run_kwargs)
        async with app.run_test(size=(160, 44)) as pilot:
            await pilot.pause()
            status = app.query_one("#busy", BusyStatus)
            assert status.display is False

            app._set_busy(True, "drafting questions…")
            await pilot.pause()
            assert status.display is True
            assert status.action == "drafting questions"
            rendered = str(status.render())
            assert "drafting questions" in rendered
            assert "s" in rendered

            app._set_busy(False, "answers · fulfilled · cowabunga")
            await pilot.pause()
            assert status.display is False

    asyncio.run(drive())


def test_cowabunga_button_tooltip_says_what_it_does(
    session: Session, run_kwargs: dict[str, Any], tmp_path: Path
) -> None:
    async def drive() -> None:
        run_kwargs["prd_path"] = str(_prd_file(tmp_path))
        app = _app(session, run_kwargs)
        async with app.run_test(size=(160, 44)) as pilot:
            await pilot.pause()
            app._after_questions(QUESTIONS, "sid")
            await pilot.pause()
            button = app.query_one("#cowabunga", Button)
            assert button.tooltip == "Answer the open questions yourself and finalize"

    asyncio.run(drive())


def test_send_button_label_and_border_hint_are_not_duplicated(
    session: Session, run_kwargs: dict[str, Any], tmp_path: Path
) -> None:
    async def drive() -> None:
        run_kwargs["prd_path"] = str(_prd_file(tmp_path))
        app = _app(session, run_kwargs)
        async with app.run_test(size=(160, 44)) as pilot:
            await pilot.pause()
            app._after_questions(QUESTIONS, "sid")
            await pilot.pause()
            assert app.query_one("#send", Button).label == "Send"
            assert app.query_one("#entry").border_subtitle == "↵ newline"

    asyncio.run(drive())
