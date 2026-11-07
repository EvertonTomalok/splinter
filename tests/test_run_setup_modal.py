"""Pilot and unit tests for ``RunSetupModal`` and its ``PrdSessionApp`` wiring."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from textual.app import App, ComposeResult
from textual.pilot import Pilot
from textual.widgets import Checkbox, Input, OptionList, Static

from splinter.agents import gate
from splinter.memory.session import Session
from splinter.strategies.registry import registered_strategies
from splinter.tui import PrdSessionApp
from splinter.tui_run_setup import GateSettingsModal, RunSetup, RunSetupModal

_TWO_STORIES_PRD = (
    "---\nfeature: demo\n---\n\n"
    "### US-001: First task\n**Description:** a\n\n"
    "### US-002: Second task\n**Description:** b\n"
)

_SINGLE_STORY_PRD = (
    "---\nfeature: demo\n---\n\n"
    "### US-001: Only task\n**Description:** a\n"
)


class _HostApp(App[RunSetup | None]):
    def __init__(self, modal: RunSetupModal) -> None:
        super().__init__()
        self._modal = modal
        self.result: RunSetup | None = None
        self.dismissed = False

    def compose(self) -> ComposeResult:
        yield Static("host")

    def on_mount(self) -> None:
        def _on_done(val: RunSetup | None) -> None:
            self.result = val
            self.dismissed = True

        self.push_screen(self._modal, _on_done)


Drive = Callable[[_HostApp, Pilot[Any]], Awaitable[None]]


def _run_modal(modal: RunSetupModal, drive: Drive) -> _HostApp:
    app = _HostApp(modal)

    async def main() -> None:
        async with app.run_test(size=(160, 44)) as pilot:
            await pilot.pause()
            await drive(app, pilot)

    asyncio.run(main())
    return app


def test_all_five_strategies_visible_and_selectable_at_160x44(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    monkeypatch.setattr("splinter.vcs.worktree.worktree_supported", lambda: True)
    session = Session("ses_modal_5")
    modal = RunSetupModal(_TWO_STORIES_PRD, session)

    async def drive(app: _HostApp, pilot: Pilot[Any]) -> None:
        ol = app.screen.query_one("#run-setup-strategy", OptionList)
        assert ol.option_count == 5
        expected = [cls.name for cls in registered_strategies()]
        assert expected == ["adaptive", "agents", "cascade", "direct", "sprint"]
        for idx, cls in enumerate(registered_strategies()):
            opt = ol.get_option_at_index(idx)
            prompt = str(opt.prompt)
            alias = cls.aliases[0] if cls.aliases else cls.name
            assert opt.id == cls.name
            assert alias in prompt
            assert cls.name in prompt
            assert cls.summary in prompt
            ol.highlighted = idx
            await pilot.pause()
            assert ol.highlighted == idx

    _run_modal(modal, drive)


def test_recommended_strategy_highlighted_on_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    monkeypatch.setattr("splinter.vcs.worktree.worktree_supported", lambda: False)
    session = Session("ses_modal_rec")
    modal = RunSetupModal(_TWO_STORIES_PRD, session)

    async def drive(app: _HostApp, pilot: Pilot[Any]) -> None:
        ol = app.screen.query_one("#run-setup-strategy", OptionList)
        names = [cls.name for cls in registered_strategies()]
        cascade_idx = names.index("cascade")
        assert ol.highlighted == cascade_idx
        prompt = str(ol.get_option_at_index(cascade_idx).prompt)
        assert "★ recommended · 2 stories, dependency-ordered" in prompt

    _run_modal(modal, drive)


def test_parallel_checkbox_hidden_for_direct_and_shown_for_cascade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    monkeypatch.setattr("splinter.vcs.worktree.worktree_supported", lambda: True)
    session = Session("ses_modal_par")
    modal = RunSetupModal(_TWO_STORIES_PRD, session)

    async def drive(app: _HostApp, pilot: Pilot[Any]) -> None:
        ol = app.screen.query_one("#run-setup-strategy", OptionList)
        cb = app.screen.query_one("#run-setup-parallel", Checkbox)
        names = [cls.name for cls in registered_strategies()]

        assert ol.highlighted == names.index("cascade")
        assert cb.display is True

        ol.highlighted = names.index("direct")
        await pilot.pause()
        assert cb.display is False

        ol.highlighted = names.index("cascade")
        await pilot.pause()
        assert cb.display is True

    _run_modal(modal, drive)


def test_run_button_returns_run_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    monkeypatch.setattr("splinter.vcs.worktree.worktree_supported", lambda: True)
    session = Session("ses_modal_run")
    modal = RunSetupModal(_TWO_STORIES_PRD, session)

    async def drive(app: _HostApp, pilot: Pilot[Any]) -> None:
        cb = app.screen.query_one("#run-setup-parallel", Checkbox)
        cb.value = True
        await pilot.click("#run-setup-run")
        await pilot.pause()
        assert app.dismissed is True
        assert app.result == RunSetup(strategy="cascade", parallel=True)

    _run_modal(modal, drive)


def test_enter_on_strategy_list_returns_run_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    monkeypatch.setattr("splinter.vcs.worktree.worktree_supported", lambda: False)
    session = Session("ses_modal_enter")
    modal = RunSetupModal(_SINGLE_STORY_PRD, session)

    async def drive(app: _HostApp, pilot: Pilot[Any]) -> None:
        await pilot.press("enter")
        await pilot.pause()
        assert app.dismissed is True
        assert app.result == RunSetup(strategy="direct", parallel=False)

    _run_modal(modal, drive)


def test_back_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    monkeypatch.setattr("splinter.vcs.worktree.worktree_supported", lambda: False)
    session = Session("ses_modal_back")
    modal = RunSetupModal(_TWO_STORIES_PRD, session)

    async def drive(app: _HostApp, pilot: Pilot[Any]) -> None:
        await pilot.click("#run-setup-back")
        await pilot.pause()
        assert app.dismissed is True
        assert app.result is None

    _run_modal(modal, drive)


def test_cowabunga_returns_recommended_strategy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    monkeypatch.setattr("splinter.vcs.worktree.worktree_supported", lambda: True)
    session = Session("ses_modal_cowa")
    modal = RunSetupModal(_TWO_STORIES_PRD, session)

    async def drive(app: _HostApp, pilot: Pilot[Any]) -> None:
        ol = app.screen.query_one("#run-setup-strategy", OptionList)
        names = [cls.name for cls in registered_strategies()]
        ol.highlighted = names.index("sprint")
        await pilot.pause()
        await pilot.click("#run-setup-cowabunga")
        await pilot.pause()
        assert app.dismissed is True
        assert app.result == RunSetup(strategy="cascade", parallel=False)

    _run_modal(modal, drive)


def test_settings_modal_persists_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    monkeypatch.setattr("splinter.vcs.worktree.worktree_supported", lambda: False)
    session = Session("ses_modal_gate")
    modal = RunSetupModal(_TWO_STORIES_PRD, session)

    async def drive(app: _HostApp, pilot: Pilot[Any]) -> None:
        await pilot.click("#run-setup-settings")
        await pilot.pause()
        assert isinstance(app.screen, GateSettingsModal)
        inp = app.screen.query_one("#gate-input", Input)
        inp.value = "pytest -q; ruff check"
        await pilot.click("#gate-save")
        await pilot.pause()

        checks = gate.configured_gate_checks(str(tmp_path), session.dir)
        assert checks is not None
        assert [c["cmd"] for c in checks] == ["pytest -q", "ruff check"]
        summary = str(app.screen.query_one("#run-setup-summary", Static).render())
        assert "pytest -q; ruff check" in summary

    _run_modal(modal, drive)


def test_stories_truncated_at_12_with_plus_n_more(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    monkeypatch.setattr("splinter.vcs.worktree.worktree_supported", lambda: False)
    stories = "\n".join(f"### US-{i:03d}: Story {i}\n**Description:** x\n" for i in range(1, 16))
    prd = f"---\nfeature: big\n---\n\n{stories}"
    session = Session("ses_modal_15")
    modal = RunSetupModal(prd, session)

    async def drive(app: _HostApp, pilot: Pilot[Any]) -> None:
        stories_widget = app.screen.query_one("#run-setup-stories", Static)
        rendered = str(stories_widget.render())
        assert "US-001 · Story 1" in rendered
        assert "US-012 · Story 12" in rendered
        assert "US-013" not in rendered
        assert "+3 more" in rendered

    _run_modal(modal, drive)


def test_prd_session_app_review_actions_and_typed_strategy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_prd_wire")
    app = PrdSessionApp(session, {"strategy": None, "prd_path": None})
    app.final_prd = _TWO_STORIES_PRD
    app.phase = "review"
    app._busy = False

    with patch.object(app, "_read_draft", return_value=_TWO_STORIES_PRD):
        with patch.object(app, "_say"):
            with patch.object(app, "_set_busy"):
                with patch.object(app, "_set_preview"):
                    app._on_review("raphael")
                    assert app.strategy == "direct"
                    assert "strategy: direct" in app.final_prd


def test_to_strategy_phase_enters_review_and_opens_modal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_prd_to_strat")
    app = PrdSessionApp(session, {"strategy": None, "prd_path": None})
    app.final_prd = _TWO_STORIES_PRD
    pushed: list[object] = []

    with (
        patch.object(app, "_say"),
        patch.object(app, "_show_stories"),
        patch.object(app, "_show_final_eval_hint"),
        patch.object(app, "_render_actions") as mock_actions,
        patch.object(app, "_set_busy"),
        patch.object(app, "_read_draft", return_value=_TWO_STORIES_PRD),
        patch.object(app, "push_screen", side_effect=lambda s, cb=None: pushed.append(s)),
    ):
        app._to_strategy_phase()

    assert app.phase == "review"
    mock_actions.assert_called_once_with("review")
    assert len(pushed) == 1
    assert isinstance(pushed[0], RunSetupModal)


@pytest.mark.parametrize("saved_phase", ["strategy", "review"])
def test_resume_strategy_or_review_lands_in_review_phase(
    saved_phase: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session(f"ses_resume_{saved_phase}")
    session.write("prd.md", _TWO_STORIES_PRD)
    session.set_status("refining", source="prd", phase=saved_phase, strategy="cascade")
    app = PrdSessionApp(session, {"strategy": None, "prd_path": None, "resume": True})

    with (
        patch.object(app, "_mount_draft_editor"),
        patch.object(app, "_replay_convo"),
        patch.object(app, "_say"),
        patch.object(app, "_show_stories"),
        patch.object(app, "_show_final_eval_hint"),
        patch.object(app, "_render_actions") as mock_actions,
        patch.object(app, "_set_busy"),
    ):
        app._resume()

    assert app.phase == "review"
    mock_actions.assert_called_once_with("review")

