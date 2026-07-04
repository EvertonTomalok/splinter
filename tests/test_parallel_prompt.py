"""The parallel flag is tri-state: explicit on the CLI, or asked on the PRD
accept screen when omitted. These cover the decision (`_should_ask_parallel`)
and the non-interactive coercion (None -> sequential)."""

from __future__ import annotations

from pathlib import Path

import pytest

_TWO_STORIES = (
    "---\nstrategy: cascade\n---\n"
    "### US-001: First\n**Description:** do a\n"
    "### US-002: Second\n**Description:** do b\n"
)
_ONE_STORY = "---\nstrategy: cascade\n---\n### US-001: Only\n**Description:** do a\n"


def _app(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch", **run_kwargs):
    from splinter.memory.session import Session, new_session_id
    from splinter.tui import PrdSessionApp

    monkeypatch.chdir(tmp_path)
    session = Session(new_session_id())
    return PrdSessionApp(session, run_kwargs)


def _force_worktree(monkeypatch: "pytest.MonkeyPatch", supported: bool) -> None:
    from splinter.vcs import worktree

    monkeypatch.setattr(worktree, "worktree_supported", lambda: supported)


class TestShouldAskParallel:
    def test_asks_when_unspecified_multitask_two_stories(
        self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
    ) -> None:
        _force_worktree(monkeypatch, True)
        app = _app(tmp_path, monkeypatch, strategy="cascade")
        app.final_prd = _TWO_STORIES
        assert app._should_ask_parallel() is True

    def test_no_ask_when_explicit_true(
        self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
    ) -> None:
        _force_worktree(monkeypatch, True)
        app = _app(tmp_path, monkeypatch, strategy="cascade", parallel=True)
        app.final_prd = _TWO_STORIES
        assert app._should_ask_parallel() is False

    def test_no_ask_when_explicit_false(
        self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
    ) -> None:
        _force_worktree(monkeypatch, True)
        app = _app(tmp_path, monkeypatch, strategy="cascade", parallel=False)
        app.final_prd = _TWO_STORIES
        assert app._should_ask_parallel() is False

    def test_no_ask_for_direct_strategy(
        self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
    ) -> None:
        _force_worktree(monkeypatch, True)
        app = _app(tmp_path, monkeypatch, strategy="raphael")
        app.final_prd = _TWO_STORIES
        assert app._should_ask_parallel() is False

    def test_no_ask_with_single_story(
        self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
    ) -> None:
        _force_worktree(monkeypatch, True)
        app = _app(tmp_path, monkeypatch, strategy="cascade")
        app.final_prd = _ONE_STORY
        assert app._should_ask_parallel() is False

    def test_no_ask_without_worktree_support(
        self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
    ) -> None:
        _force_worktree(monkeypatch, False)
        app = _app(tmp_path, monkeypatch, strategy="cascade")
        app.final_prd = _TWO_STORIES
        assert app._should_ask_parallel() is False


class TestBeginRunWiring:
    def _prep(self, tmp_path, monkeypatch, worktree, **kw):
        _force_worktree(monkeypatch, worktree)
        app = _app(tmp_path, monkeypatch, strategy=kw.pop("strategy", "cascade"), **kw)
        app.final_prd = kw.pop("prd", _TWO_STORIES)
        pushed: dict[str, object] = {}
        app.push_screen = lambda screen, cb=None: pushed.update(screen=screen, cb=cb)
        app._finish_run = lambda: pushed.update(finished=True)
        return app, pushed

    def test_accept_pushes_modal_and_choice_true(
        self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
    ) -> None:
        from splinter.tui import _ParallelModal

        app, pushed = self._prep(tmp_path, monkeypatch, worktree=True)
        app._begin_run()
        assert isinstance(pushed["screen"], _ParallelModal)
        assert "finished" not in pushed  # deferred until the user answers
        pushed["cb"](True)  # user picks Parallel
        assert app.run_kwargs["parallel"] is True
        assert pushed.get("finished") is True

    def test_modal_escape_defaults_sequential(
        self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
    ) -> None:
        app, pushed = self._prep(tmp_path, monkeypatch, worktree=True)
        app._begin_run()
        pushed["cb"](None)  # Escape / dismissed
        assert app.run_kwargs["parallel"] is False
        assert pushed.get("finished") is True

    def test_cowabunga_skips_modal(
        self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
    ) -> None:
        app, pushed = self._prep(tmp_path, monkeypatch, worktree=True)
        app._begin_run(auto=True)
        assert "screen" not in pushed  # no prompt
        assert app.run_kwargs["parallel"] is False
        assert pushed.get("finished") is True


class TestCliCoercion:
    def test_unspecified_parallel_defaults_sequential_non_tty(
        self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
    ) -> None:
        """No --parallel on a non-interactive run -> run_pipeline gets False, not None."""
        from typer.testing import CliRunner

        import splinter.pipeline as pipeline
        from splinter.cli import app

        prd = tmp_path / "prd.md"
        prd.write_text(_TWO_STORIES)
        captured: dict[str, object] = {}

        def _fake_run_pipeline(**kwargs):
            captured.update(kwargs)
            return 0

        monkeypatch.setattr(pipeline, "run_pipeline", _fake_run_pipeline)
        monkeypatch.setenv("SPLINTER_NO_TUI", "1")

        result = CliRunner().invoke(app, ["run", "--prd", str(prd)])
        assert result.exit_code == 0
        assert captured["parallel"] is False

    def test_explicit_no_parallel_passes_false(
        self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
    ) -> None:
        from typer.testing import CliRunner

        import splinter.pipeline as pipeline
        from splinter.cli import app

        prd = tmp_path / "prd.md"
        prd.write_text(_TWO_STORIES)
        captured: dict[str, object] = {}
        monkeypatch.setattr(
            pipeline, "run_pipeline", lambda **kw: captured.update(kw) or 0
        )
        monkeypatch.setenv("SPLINTER_NO_TUI", "1")

        result = CliRunner().invoke(app, ["run", "--prd", str(prd), "--no-parallel"])
        assert result.exit_code == 0
        assert captured["parallel"] is False
