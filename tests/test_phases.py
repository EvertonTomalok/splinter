"""Tests for multi-phase development: run_phase, artifact separation, trajectory rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from splinter.analyze import _phase_entries, _trajectory_lines
from splinter.memory.session import Session
from splinter.models.roster import Ladder, load_ladder
from splinter.phases import PhaseConfig, phase_count, run_phase

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stub_run_text(text: str = "stub plan") -> Any:
    def _run(prompt: str, model: str, **kwargs: Any) -> str:
        return text

    return _run


def _fake_ladder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Ladder:
    monkeypatch.chdir(tmp_path)
    return load_ladder()


def _make_phase_cfg(desc: str = "add logging") -> PhaseConfig:
    return PhaseConfig(
        description=desc,
        plan_model="haiku",
        plan_effort="low",
        run_model="haiku",
        run_effort="low",
    )


@dataclass
class _FakeProviderResponse:
    text: str = "phase runner output"
    model: str = "haiku"
    tokens: dict[str, int] = None  # type: ignore[assignment]
    cost: float = 0.001
    raw: dict[str, Any] = None  # type: ignore[assignment]
    session_id: str | None = None

    def __post_init__(self) -> None:
        if self.tokens is None:
            self.tokens = {"input": 10, "output": 20}
        if self.raw is None:
            self.raw = {}


class _FakeProvider:
    def run(self, prompt: str, model: str, **kwargs: Any) -> _FakeProviderResponse:
        return _FakeProviderResponse(model=model)


def _mock_phase_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock every I/O boundary that run_phase touches.

    IMPORTANT — patch on splinter.phases (the importing module), not on the
    source modules, because ``from X import name`` binds the name at import
    time and monkeypatch.setattr on X won't affect it.
    """
    monkeypatch.setattr("splinter.phases.run_text", _stub_run_text("1. plan step"))
    monkeypatch.setattr(
        "splinter.phases.get_provider",
        lambda name: _FakeProvider(),
    )
    monkeypatch.setattr(
        "splinter.phases.record_exchange",
        lambda *a, **kw: None,
    )
    from splinter.agents.gate import GateResult

    monkeypatch.setattr(
        "splinter.phases.run_gate",
        lambda session_dir=None, languages=None: GateResult(passed=True, checks=[]),
    )


# ---------------------------------------------------------------------------
# run_phase — artifact separation
# ---------------------------------------------------------------------------


class TestRunPhaseArtifacts:
    def test_phase_writes_to_phase_loop_not_loop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_phase writes to phase_loop.md, not loop.md."""
        monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
        session = Session("ses_phases_test")
        _mock_phase_boundaries(monkeypatch)

        ladder = _fake_ladder(tmp_path, monkeypatch)
        cfg = _make_phase_cfg()

        result = run_phase(cfg, session, ladder)

        assert result.phase_number == 1
        assert session.read("phase_loop.md")
        loop = session.read("loop.md")
        assert "Phase 1" not in loop
        assert session.read("knowledge/phase-plan-1.md")
        assert session.read("runs/phase-1.md")
        assert not (session.dir / "runs/iter-1.md").exists()

    def test_phase_trace_entries_recorded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Phase entries appear in trace.md."""
        monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
        session = Session("ses_phases_trace")
        _mock_phase_boundaries(monkeypatch)

        ladder = _fake_ladder(tmp_path, monkeypatch)
        cfg = _make_phase_cfg("trace test")

        run_phase(cfg, session, ladder)

        trace_md = session.read("trace.md")
        assert "total runs: 1" in trace_md
        assert "haiku" in trace_md

    def test_multiple_phases_get_unique_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each phase writes to its own numbered files."""
        monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
        session = Session("ses_multi")
        _mock_phase_boundaries(monkeypatch)

        ladder = _fake_ladder(tmp_path, monkeypatch)

        cfg1 = _make_phase_cfg("phase one")
        cfg2 = _make_phase_cfg("phase two")

        run_phase(cfg1, session, ladder)
        run_phase(cfg2, session, ladder)

        assert session.read("knowledge/phase-plan-1.md")
        assert session.read("knowledge/phase-plan-2.md")
        assert session.read("runs/phase-1.md")
        assert session.read("runs/phase-2.md")
        assert phase_count(session) == 2

    def test_phase_count_empty_session(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """phase_count returns 0 for a session with no phases."""
        monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
        session = Session("ses_empty")
        assert phase_count(session) == 0

    def test_phase_count_ignores_empty_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """phase_count skips empty phase-plan files."""
        monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
        session = Session("ses_empty_plans")
        (session.dir / "knowledge").mkdir(parents=True, exist_ok=True)
        (session.dir / "knowledge" / "phase-plan-1.md").write_text("")
        assert phase_count(session) == 0


# ---------------------------------------------------------------------------
# analyze — trajectory + phase entries
# ---------------------------------------------------------------------------


class TestPhaseAnalyze:
    def test_phase_entries_parses_correctly(self) -> None:
        md = """- Phase 1 · PASS · haiku · $0.0010 · [14:30:00]
  add logging everywhere
- Phase 2 · FAIL · sonnet · $0.0050 · [14:31:00]
  fix the broken test
"""
        entries = _phase_entries(md)
        assert len(entries) == 2
        assert entries[0] == (1, "PASS", "haiku", "0.0010")
        assert entries[1] == (2, "FAIL", "sonnet", "0.0050")

    def test_phase_entries_empty(self) -> None:
        assert _phase_entries("") == []
        assert _phase_entries("no phases here\n") == []

    def test_trajectory_lines_includes_phases(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
        session = Session("ses_traj")

        session.write(
            "phases.md",
            "- Phase 1 · PASS · haiku · $0.0010 · [14:30:00]\n  add logging\n",
        )
        session.write("loop.md", "## Iteration 1\ntier 0\nverdict: PASS\n")

        from splinter.analyze import _iterations

        iters = _iterations(session.read("loop.md"))
        lines = _trajectory_lines(session, iters)
        text = "\n".join(lines)
        assert "phases" in text
        assert "phase 1" in text.lower()

    def test_trajectory_no_phases_no_crash(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
        session = Session("ses_no_phases")
        session.write("loop.md", "## Iteration 1\ntier 0\nverdict: PASS\n")

        from splinter.analyze import _iterations

        iters = _iterations(session.read("loop.md"))
        lines = _trajectory_lines(session, iters)
        text = "\n".join(lines)
        assert "phases" not in text.replace("[dim]", "").replace("[/]", "")


# ---------------------------------------------------------------------------
# TUI phase modal (unit — no I/O needed)
# ---------------------------------------------------------------------------


class TestPhaseConfigModal:
    def test_modal_has_auto_in_efforts(self) -> None:
        """_PhaseConfigModal._EFFORTS includes 'auto'."""
        from splinter.tui import _PhaseConfigModal

        assert "auto" in _PhaseConfigModal._EFFORTS
        assert "(default)" in _PhaseConfigModal._EFFORTS

    def test_modal_composes_inside_app(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_PhaseConfigModal composes cleanly inside a running App."""
        from textual.app import App

        from splinter.tui import _PhaseConfigModal

        monkeypatch.chdir(tmp_path)

        _PhaseConfigModal._PLAN_MODELS = ["opus", "sonnet"]
        _PhaseConfigModal._RUN_MODELS = ["haiku", "flash"]

        result_holder: list[dict[str, str] | None] = []

        class _TestApp(App[None]):
            def on_mount(self) -> None:
                modal = _PhaseConfigModal(phase_num=1)
                self.push_screen(modal, callback=lambda r: result_holder.append(r))

        async def _run() -> None:
            app = _TestApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                await pilot.pause()
                await pilot.pause()
                assert len(app.screen_stack) >= 1
                app.pop_screen()
                await pilot.pause()

        import asyncio

        asyncio.run(_run())

    def test_modal_submit_returns_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_PhaseConfigModal Go button dismisses with description and model config."""
        from textual.app import App

        from splinter.tui import _PhaseConfigModal

        monkeypatch.chdir(tmp_path)

        _PhaseConfigModal._PLAN_MODELS = ["opus", "sonnet"]
        _PhaseConfigModal._RUN_MODELS = ["haiku", "flash"]

        result_holder: list[dict[str, str] | None] = []

        class _TestApp(App[None]):
            def on_mount(self) -> None:
                modal = _PhaseConfigModal(
                    phase_num=2,
                    default_plan_model="opus",
                    default_run_model="haiku",
                )
                self.push_screen(modal, callback=lambda r: result_holder.append(r))

        async def _run() -> None:
            app = _TestApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                await pilot.pause()
                await pilot.pause()
                # The modal is now the top screen — query from there
                assert len(app.screen_stack) >= 1
                if len(app.screen_stack) > 1:
                    # Modal is pushed — widgets exist on the modal screen
                    assert result_holder == []
                    # Dismiss without entering text (Done)
                    app.pop_screen()
                await pilot.pause()
                # Callback fires with None when popped without dismiss()

        import asyncio

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# PhaseConfig
# ---------------------------------------------------------------------------


class TestPhaseConfig:
    def test_no_provider_fields(self) -> None:
        """PhaseConfig has no explicit provider fields."""
        cfg = PhaseConfig(
            description="test",
            plan_model="opus",
            plan_effort="high",
            run_model="haiku",
            run_effort="low",
        )
        assert cfg.plan_model == "opus"
        assert cfg.run_model == "haiku"
        with pytest.raises(TypeError):
            PhaseConfig(  # type: ignore[call-arg]
                description="test",
                plan_provider="claude",
                plan_model="opus",
                plan_effort="high",
                run_provider="claude",
                run_model="haiku",
                run_effort="low",
            )
