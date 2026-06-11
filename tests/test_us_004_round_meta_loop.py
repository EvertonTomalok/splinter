"""US-004: Pipeline round meta-loop — integration tests.

Covers:
 1. final-eval PASS  → status "completed", no AskUserPause, localize called once
 2. final-eval FAIL  → AskUserPause / status "awaiting_user", round_index=1,
                       next_effort = bump_effort(start)
 3. resume round>0   → _clear_round_caches, localize re-invoked, bumped effort
                       forwarded to strat.execute
 4. round-eval notes → knowledge/previous_rounds.md written; content lands in
                       the next plan's code_ctx via _plan_all_tasks
"""

from __future__ import annotations

from pathlib import Path

import pytest

from splinter.agents.final_eval import FinalEvalResult
from splinter.agents.runner import RunResult
from splinter.memory.knowledge import KnowledgeStore
from splinter.memory.session import Session
from splinter.models.roster import bump_effort
from splinter.pipeline import _clear_round_caches, _load_round_history, run_pipeline

# ── shared helpers ────────────────────────────────────────────────────────────


def _run_result() -> RunResult:
    return RunResult(text="done", model="m", tier=0, tokens={}, cost=0.0, raw={})


def _pass_fe(name: str = "gate") -> list[FinalEvalResult]:
    return [FinalEvalResult(name=name, passed=True, output="all checks passed")]


def _fail_fe(name: str = "gate") -> list[FinalEvalResult]:
    return [FinalEvalResult(name=name, passed=False, output="assertion failed: missing tests")]


def _config_with_fe() -> dict:
    return {
        "final_eval": [{"name": "gate", "kind": "command", "cmd": "pytest"}],
        "gate_checks": [],
        "defaults": {"timeout": 3600},
    }


def _fake_execute(self_, tasks, sess, ladder, **kwargs):
    return [_run_result()]


@pytest.fixture()
def session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Session:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    s = Session("ses_us004")
    s.update_index("# us-004 test\n")
    return s


@pytest.fixture()
def task_yaml(tmp_path: Path) -> str:
    p = tmp_path / "task.yaml"
    p.write_text(
        "description: Implement feature X\n"
        "acceptance: Feature X works correctly\n"
        "effort: normal\n"
    )
    return str(p)


def _base_patches(monkeypatch: pytest.MonkeyPatch, *, fe_results: list) -> None:
    """Patch all expensive I/O so run_pipeline runs without real LLMs."""
    monkeypatch.setattr("splinter.pipeline.localize", lambda *a, **kw: [])
    monkeypatch.setattr("splinter.pipeline._resolve_gate", lambda *a, **kw: None)
    from splinter.strategies.cascade import CascadeStrategy
    monkeypatch.setattr(CascadeStrategy, "execute", _fake_execute)
    monkeypatch.setattr("splinter.configure.load_config", lambda *a, **kw: _config_with_fe())
    fe_iter = iter(fe_results)
    monkeypatch.setattr(
        "splinter.agents.final_eval.run_all_final_evals",
        lambda *a, **kw: next(fe_iter),
    )


# ── 1. PASS ends pipeline cleanly ─────────────────────────────────────────────


class TestFinalEvalPass:
    def test_status_completed(
        self,
        session: Session,
        task_yaml: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _base_patches(monkeypatch, fe_results=[_pass_fe()])
        rc = run_pipeline(task_path=task_yaml, session=session)
        assert rc == 0
        assert session.read_status()["state"] == "completed"

    def test_localize_called_exactly_once(
        self,
        session: Session,
        task_yaml: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[int] = []
        monkeypatch.setattr(
            "splinter.pipeline.localize",
            lambda *a, **kw: calls.append(1) or [],
        )
        monkeypatch.setattr("splinter.pipeline._resolve_gate", lambda *a, **kw: None)
        from splinter.strategies.cascade import CascadeStrategy
        monkeypatch.setattr(CascadeStrategy, "execute", _fake_execute)
        monkeypatch.setattr("splinter.configure.load_config", lambda *a, **kw: _config_with_fe())
        monkeypatch.setattr(
            "splinter.agents.final_eval.run_all_final_evals",
            lambda *a, **kw: _pass_fe(),
        )

        run_pipeline(task_path=task_yaml, session=session)
        assert len(calls) == 1

    def test_no_ask_user_pause(
        self,
        session: Session,
        task_yaml: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _base_patches(monkeypatch, fe_results=[_pass_fe()])
        rc = run_pipeline(task_path=task_yaml, session=session)
        assert rc == 0

    def test_no_final_eval_configured_also_completes(
        self,
        session: Session,
        task_yaml: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When no final_eval is configured, pipeline completes without checking."""
        monkeypatch.setattr("splinter.pipeline.localize", lambda *a, **kw: [])
        monkeypatch.setattr("splinter.pipeline._resolve_gate", lambda *a, **kw: None)
        from splinter.strategies.cascade import CascadeStrategy
        monkeypatch.setattr(CascadeStrategy, "execute", _fake_execute)
        monkeypatch.setattr(
            "splinter.configure.load_config",
            lambda *a, **kw: {"gate_checks": [], "defaults": {"timeout": 3600}},
        )

        rc = run_pipeline(task_path=task_yaml, session=session)
        assert rc == 0
        assert session.read_status()["state"] == "completed"


# ── 2. RETRY / fail → AskUserPause ───────────────────────────────────────────


class TestFinalEvalRetry:
    def test_returns_code_3(
        self,
        session: Session,
        task_yaml: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _base_patches(monkeypatch, fe_results=[_fail_fe()])
        rc = run_pipeline(task_path=task_yaml, session=session)
        assert rc == 3

    def test_status_awaiting_user(
        self,
        session: Session,
        task_yaml: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _base_patches(monkeypatch, fe_results=[_fail_fe()])
        run_pipeline(task_path=task_yaml, session=session)
        assert session.read_status()["state"] == "awaiting_user"

    def test_status_round_index_1(
        self,
        session: Session,
        task_yaml: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _base_patches(monkeypatch, fe_results=[_fail_fe()])
        run_pipeline(task_path=task_yaml, session=session)
        assert session.read_status()["round_index"] == 1

    def test_status_next_effort_bumped(
        self,
        session: Session,
        task_yaml: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _base_patches(monkeypatch, fe_results=[_fail_fe()])
        run_pipeline(task_path=task_yaml, session=session)
        assert session.read_status()["next_effort"] == bump_effort("normal")

    def test_round_eval_note_written(
        self,
        session: Session,
        task_yaml: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _base_patches(monkeypatch, fe_results=[_fail_fe()])
        run_pipeline(task_path=task_yaml, session=session)

        ks = KnowledgeStore(session)
        assert "round-eval-0" in ks.list_notes()
        assert "assertion failed" in ks.read_note("round-eval-0")

    def test_corrections_in_status(
        self,
        session: Session,
        task_yaml: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _base_patches(monkeypatch, fe_results=[_fail_fe()])
        run_pipeline(task_path=task_yaml, session=session)
        status = session.read_status()
        assert "assertion failed" in status.get("ask_corrections", "")


# ── 3. Resume with round_index > 0 ───────────────────────────────────────────


class TestResumeNewRound:
    def _seed_resume(self, session: Session, effort: str = "normal") -> str:
        bumped = bump_effort(effort)
        session.set_status("awaiting_user", round_index=1, next_effort=bumped)
        return bumped

    def test_localize_reinvoked_on_resume(
        self,
        session: Session,
        task_yaml: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """force_rerun=True on resume with round_index>0 re-invokes localize."""
        self._seed_resume(session)
        session.write("knowledge/localization.md", "# old localization\n")

        calls: list[int] = []
        monkeypatch.setattr(
            "splinter.pipeline.localize",
            lambda *a, **kw: calls.append(1) or [],
        )
        monkeypatch.setattr("splinter.pipeline._resolve_gate", lambda *a, **kw: None)
        from splinter.strategies.cascade import CascadeStrategy
        monkeypatch.setattr(CascadeStrategy, "execute", _fake_execute)
        monkeypatch.setattr("splinter.configure.load_config", lambda *a, **kw: _config_with_fe())
        monkeypatch.setattr(
            "splinter.agents.final_eval.run_all_final_evals",
            lambda *a, **kw: _pass_fe(),
        )

        run_pipeline(task_path=task_yaml, session=session, resume=True)
        assert len(calls) == 1, "localize must re-run on round > 0"

    def test_plan_cache_cleared_on_resume(
        self,
        session: Session,
        task_yaml: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """_clear_round_caches removes plan/filter files before re-planning."""
        self._seed_resume(session)
        session.write("knowledge/plan.md", "# stale plan\n")
        session.write("knowledge/plan-1.md", "# stale plan-1\n")
        session.write("knowledge/filter-1.md", "# stale filter\n")

        _base_patches(monkeypatch, fe_results=[_pass_fe()])
        run_pipeline(task_path=task_yaml, session=session, resume=True)

        assert not (session.dir / "knowledge" / "plan.md").exists()
        assert not (session.dir / "knowledge" / "plan-1.md").exists()
        assert not (session.dir / "knowledge" / "filter-1.md").exists()

    def test_bumped_effort_forwarded_to_execute(
        self,
        session: Session,
        task_yaml: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """strat.execute receives bumped effort read from status on resume."""
        bumped = self._seed_resume(session, effort="normal")

        captured: list[dict] = []

        def _capturing_execute(self_, tasks, sess, ladder, **kwargs):
            captured.append(kwargs)
            return [_run_result()]

        monkeypatch.setattr("splinter.pipeline.localize", lambda *a, **kw: [])
        monkeypatch.setattr("splinter.pipeline._resolve_gate", lambda *a, **kw: None)
        from splinter.strategies.cascade import CascadeStrategy
        monkeypatch.setattr(CascadeStrategy, "execute", _capturing_execute)
        monkeypatch.setattr("splinter.configure.load_config", lambda *a, **kw: _config_with_fe())
        monkeypatch.setattr(
            "splinter.agents.final_eval.run_all_final_evals",
            lambda *a, **kw: _pass_fe(),
        )

        run_pipeline(task_path=task_yaml, session=session, resume=True)

        assert len(captured) == 1
        assert captured[0]["effort"] == bumped

    def test_localize_cache_not_reused_on_new_round(
        self,
        session: Session,
        task_yaml: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Even with resume=True and localization.md on disk, new round re-localizes."""
        self._seed_resume(session)
        session.write("knowledge/localization.md", "cached content")

        calls: list[int] = []
        monkeypatch.setattr(
            "splinter.pipeline.localize",
            lambda *a, **kw: calls.append(1) or [],
        )
        monkeypatch.setattr("splinter.pipeline._resolve_gate", lambda *a, **kw: None)
        from splinter.strategies.cascade import CascadeStrategy
        monkeypatch.setattr(CascadeStrategy, "execute", _fake_execute)
        monkeypatch.setattr("splinter.configure.load_config", lambda *a, **kw: _config_with_fe())
        monkeypatch.setattr(
            "splinter.agents.final_eval.run_all_final_evals",
            lambda *a, **kw: _pass_fe(),
        )

        run_pipeline(task_path=task_yaml, session=session, resume=True)
        assert calls, "localize must NOT use the cache when force_rerun=True"


# ── 4. Round eval history in plan context ─────────────────────────────────────


class TestRoundHistory:
    def test_previous_rounds_md_written_when_notes_exist(
        self,
        session: Session,
        task_yaml: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When round-eval-N notes exist, previous_rounds.md is written before execute."""
        session.set_status("awaiting_user", round_index=1, next_effort="hard")
        ks = KnowledgeStore(session)
        ks.write_note("round-eval-0", "missing feature X implementation")

        _base_patches(monkeypatch, fe_results=[_pass_fe()])

        run_pipeline(task_path=task_yaml, session=session, resume=True)

        prev = session.read("knowledge/previous_rounds.md")
        assert prev.strip(), "previous_rounds.md must be non-empty"
        assert "round-eval-0" in prev
        assert "missing feature X implementation" in prev

    def test_previous_rounds_md_empty_without_notes(
        self,
        session: Session,
        task_yaml: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No round-eval notes → previous_rounds.md not written."""
        _base_patches(monkeypatch, fe_results=[_pass_fe()])

        run_pipeline(task_path=task_yaml, session=session)

        assert not (session.dir / "knowledge" / "previous_rounds.md").exists()

    def test_plan_all_tasks_prepends_previous_rounds(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Unit test: _plan_all_tasks reads knowledge/previous_rounds.md and
        prepends its content to code_ctx passed to _make_plan."""
        monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)

        from splinter.agents.runner import Task
        from splinter.models.roster import load_ladder
        from splinter.strategies import direct as direct_mod
        from splinter.strategies.direct import DirectStrategy

        sess = Session("ses_plan_test")
        sess.update_index("# t\n")
        sess.write("knowledge/previous_rounds.md", "ROUND_SENTINEL_VALUE")

        task = Task(description="test task", acceptance="it works")
        ladder = load_ladder()

        captured_ctxs: list[str] = []

        def _spy_make_plan(tsk, ldr, code_ctx, session=None):
            captured_ctxs.append(code_ctx)
            return "spy plan"

        monkeypatch.setattr(direct_mod, "_make_plan", _spy_make_plan)

        strat = DirectStrategy()
        strat._plan_all_tasks([task], sess, ladder, localization="")

        assert captured_ctxs, "_make_plan must have been called"
        assert any("ROUND_SENTINEL_VALUE" in ctx for ctx in captured_ctxs), (
            "previous_rounds.md content must be prepended to code_ctx"
        )

    def test_run_task_loop_fallback_plan_prepends_previous_rounds(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Fallback plan path in _run_task_loop also reads previous_rounds.md."""
        monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)

        from splinter.agents.runner import Task
        from splinter.models.roster import load_ladder
        from splinter.strategies import direct as direct_mod
        from splinter.strategies.direct import DirectStrategy

        sess = Session("ses_fallback_test")
        sess.update_index("# t\n")
        sess.write("knowledge/previous_rounds.md", "FALLBACK_SENTINEL")

        task = Task(description="test fallback plan", acceptance="works")
        ladder = load_ladder()

        captured: list[str] = []

        def _spy_make_plan(tsk, ldr, code_ctx, session=None):
            captured.append(code_ctx)
            return "fallback spy plan"

        monkeypatch.setattr(direct_mod, "_make_plan", _spy_make_plan)

        strat = DirectStrategy()
        strat._plan_all_tasks([task], sess, ladder, localization="some context")

        assert any("FALLBACK_SENTINEL" in ctx for ctx in captured)


# ── helpers tests (_clear_round_caches, _load_round_history) ──────────────────


class TestHelpers:
    def test_clear_round_caches_removes_plan_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
        s = Session("ses_clear")
        s.write("knowledge/plan.md", "p")
        s.write("knowledge/plan-1.md", "p1")
        s.write("knowledge/plan-2.md", "p2")
        s.write("knowledge/filter-1.md", "f1")
        s.write("knowledge/localization-1.md", "l1")
        s.write("knowledge/localization.md", "main-loc")  # must be kept

        _clear_round_caches(s)

        assert not (s.dir / "knowledge" / "plan.md").exists()
        assert not (s.dir / "knowledge" / "plan-1.md").exists()
        assert not (s.dir / "knowledge" / "filter-1.md").exists()
        assert not (s.dir / "knowledge" / "localization-1.md").exists()
        assert (s.dir / "knowledge" / "localization.md").exists()

    def test_load_round_history_empty_when_no_notes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
        s = Session("ses_hist_empty")
        assert _load_round_history(s) == ""

    def test_load_round_history_concatenates_notes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
        s = Session("ses_hist")
        s.write("knowledge/round-eval-0.md", "first failure")
        s.write("knowledge/round-eval-1.md", "second failure")

        hist = _load_round_history(s)
        assert "round-eval-0" in hist
        assert "first failure" in hist
        assert "round-eval-1" in hist
        assert "second failure" in hist
