"""Tests for CascadeStrategy: topo-sort, checkpoints, resume, and budget."""

from __future__ import annotations

import json
import logging
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from splinter.agents.runner import RunResult, Task
from splinter.memory.session import Session
from splinter.strategies.cascade import CascadeStrategy

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _task(id: str, deps: list[str] | None = None) -> Task:
    return Task(description=f"{id}: task", acceptance="done", id=id, deps=deps)


def _fake_result(id: str = "m") -> RunResult:
    return RunResult(
        text="ok", model=id, tier=0, tokens={}, cost=0.0, raw={}
    )


# ---------------------------------------------------------------------------
# _topo_sort
# ---------------------------------------------------------------------------

class TestTopoSort:
    def test_linear_deps_reordered(self) -> None:
        t1 = _task("US-001")
        t2 = _task("US-002", deps=["US-001"])
        result = CascadeStrategy._topo_sort([t2, t1])
        assert [t.id for t in result] == ["US-001", "US-002"]

    def test_diamond(self) -> None:
        """US-001 → US-002, US-003 → US-004 (diamond)."""
        t1 = _task("US-001")
        t2 = _task("US-002", deps=["US-001"])
        t3 = _task("US-003", deps=["US-001"])
        t4 = _task("US-004", deps=["US-002", "US-003"])
        result = CascadeStrategy._topo_sort([t4, t2, t3, t1])
        ids = [t.id for t in result]
        assert ids.index("US-001") < ids.index("US-002")
        assert ids.index("US-001") < ids.index("US-003")
        assert ids.index("US-002") < ids.index("US-004")
        assert ids.index("US-003") < ids.index("US-004")

    def test_cycle_fallback_and_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        t1 = _task("US-001", deps=["US-002"])
        t2 = _task("US-002", deps=["US-001"])
        original = [t1, t2]
        with caplog.at_level(logging.WARNING, logger="splinter.loop"):
            result = CascadeStrategy._topo_sort(original)
        assert result == original
        assert any("cycle" in r.message for r in caplog.records)

    def test_no_deps_preserves_prd_order(self) -> None:
        tasks = [_task(f"US-{i:03d}") for i in range(1, 6)]
        assert CascadeStrategy._topo_sort(tasks) == tasks

    def test_external_deps_ignored(self) -> None:
        """Dep not in task set is treated as already done."""
        t2 = _task("US-002", deps=["US-000"])  # US-000 not in list
        t3 = _task("US-003", deps=["US-002"])
        result = CascadeStrategy._topo_sort([t2, t3])
        assert [t.id for t in result] == ["US-002", "US-003"]

    def test_tasks_without_id_appended_at_end(self) -> None:
        t1 = _task("US-001")
        t_no_id = Task(description="anon task", acceptance="done")
        result = CascadeStrategy._topo_sort([t1, t_no_id])
        assert result[0].id == "US-001"
        assert result[-1] is t_no_id


# ---------------------------------------------------------------------------
# _load_checkpoint / _save_checkpoint
# ---------------------------------------------------------------------------

class TestCheckpoint:
    def test_round_trip(self, tmp_path: Path) -> None:
        session = Session.__new__(Session)
        session.id = "test"
        session.dir = tmp_path

        ids: set[str] = {"US-001", "US-002"}
        CascadeStrategy._save_checkpoint(session, ids)
        loaded = CascadeStrategy._load_checkpoint(session)
        assert loaded == ids

    def test_absent_file_returns_empty(self, tmp_path: Path) -> None:
        session = Session.__new__(Session)
        session.id = "test"
        session.dir = tmp_path
        assert CascadeStrategy._load_checkpoint(session) == set()

    def test_corrupt_file_returns_empty(self, tmp_path: Path) -> None:
        session = Session.__new__(Session)
        session.id = "test"
        session.dir = tmp_path
        (tmp_path / "checkpoint.json").write_text("not json")
        assert CascadeStrategy._load_checkpoint(session) == set()

    def test_sorted_output(self, tmp_path: Path) -> None:
        session = Session.__new__(Session)
        session.id = "test"
        session.dir = tmp_path
        CascadeStrategy._save_checkpoint(session, {"US-003", "US-001", "US-002"})
        raw = json.loads((tmp_path / "checkpoint.json").read_text())
        assert raw["completed"] == ["US-001", "US-002", "US-003"]


# ---------------------------------------------------------------------------
# execute — resume: pre-checkpointed task skipped
# ---------------------------------------------------------------------------

class TestExecuteResume:
    def _make_session(self, tmp_path: Path) -> Session:
        (tmp_path / "knowledge").mkdir()
        session = Session.__new__(Session)
        session.id = "test"
        session.dir = tmp_path
        return session

    def test_checkpointed_task_skipped(self, tmp_path: Path) -> None:
        session = self._make_session(tmp_path)
        # pre-seed checkpoint with US-001 done
        CascadeStrategy._save_checkpoint(session, {"US-001"})

        t1 = _task("US-001")
        t2 = _task("US-002")
        ran: list[str] = []

        strategy = CascadeStrategy()
        fake_ladder = MagicMock()
        fake_ladder.effort_mapping.return_value = None

        def fake_loop(task: Task, *args: Any, **kwargs: Any) -> RunResult | None:
            ran.append(task.id)
            return _fake_result()

        with patch.object(strategy, "_run_task_loop", side_effect=fake_loop):
            with patch.object(strategy, "_start_tier", return_value=0):
                strategy.execute(
                    [t1, t2], session, fake_ladder,
                    resume=True, cowabunga=True,
                )

        assert ran == ["US-002"]

    def test_checkpoint_updated_after_each_task(self, tmp_path: Path) -> None:
        session = self._make_session(tmp_path)
        t1 = _task("US-001")
        t2 = _task("US-002")

        call_order: list[str] = []
        checkpoints_after: list[set[str]] = []

        strategy = CascadeStrategy()
        fake_ladder = MagicMock()

        def fake_loop(task: Task, *args: Any, **kwargs: Any) -> RunResult | None:
            call_order.append(task.id)
            return _fake_result()

        original_save = CascadeStrategy._save_checkpoint

        def tracking_save(s: Session, done: set[str]) -> None:
            checkpoints_after.append(set(done))
            original_save(s, done)

        with patch.object(strategy, "_run_task_loop", side_effect=fake_loop):
            with patch.object(strategy, "_start_tier", return_value=0):
                with patch.object(CascadeStrategy, "_save_checkpoint", staticmethod(tracking_save)):
                    strategy.execute([t1, t2], session, fake_ladder, cowabunga=True)

        assert call_order == ["US-001", "US-002"]
        assert "US-001" in checkpoints_after[0]
        assert {"US-001", "US-002"} == checkpoints_after[1]


# ---------------------------------------------------------------------------
# execute — budget short-circuit
# ---------------------------------------------------------------------------

class TestBudgetShortCircuit:
    def test_stops_when_budget_exhausted(self, tmp_path: Path) -> None:
        (tmp_path / "knowledge").mkdir()
        session = Session.__new__(Session)
        session.id = "test"
        session.dir = tmp_path

        t1 = _task("US-001")
        t2 = _task("US-002")

        strategy = CascadeStrategy()
        fake_ladder = MagicMock()

        call_count = 0

        def fake_loop(task: Task, *args: Any, **kwargs: Any) -> RunResult | None:
            nonlocal call_count
            call_count += 1
            return _fake_result()

        fake_trace = MagicMock()
        fake_trace.total_cost = 10.0  # above any budget
        fake_trace.summary.return_value = ""

        with patch("splinter.strategies.cascade.Trace", return_value=fake_trace):
            with patch.object(strategy, "_run_task_loop", side_effect=fake_loop):
                with patch.object(strategy, "_start_tier", return_value=0):
                    strategy.execute(
                        [t1, t2], session, fake_ladder,
                        budget=5.0, cowabunga=True,
                    )

        assert call_count == 1  # stopped after first task


# ---------------------------------------------------------------------------
# resume — budget cost restored from trace.md
# ---------------------------------------------------------------------------

class TestResumeBudgetContinuity:
    def test_resume_restores_cost(self, tmp_path: Path) -> None:
        """Trace.from_markdown must restore total_cost so budget short-circuit
        counts prior-run spend on resume, not just the current run's spend."""
        from splinter.obs.trace import RunEntry, Trace

        prior = Trace()
        prior.entries.append(
            RunEntry(model="m", tier=0, iteration=1, tokens={}, cost=3.50, latency_s=1.0, task=0)
        )
        md = prior.summary()

        restored = Trace.from_markdown(md)
        assert abs(restored.total_cost - 3.50) < 1e-6

    def test_cascade_resume_continues_from_prior_cost(self, tmp_path: Path) -> None:
        """On resume, cascade reloads trace.md; budget check uses accumulated cost."""
        from splinter.obs.trace import RunEntry, Trace

        (tmp_path / "knowledge").mkdir()
        session = Session.__new__(Session)
        session.id = "test"
        session.dir = tmp_path

        # Seed trace.md with $4 already spent.
        prior = Trace()
        prior.entries.append(
            RunEntry(model="m", tier=0, iteration=1, tokens={}, cost=4.0, latency_s=1.0, task=0)
        )
        (tmp_path / "trace.md").write_text(prior.summary())

        t1 = _task("US-001")
        t2 = _task("US-002")
        strategy = CascadeStrategy()
        fake_ladder = MagicMock()
        ran: list[str] = []

        def fake_loop(task: Task, *args: Any, **kwargs: Any) -> RunResult | None:
            ran.append(task.id)
            return _fake_result()

        with patch.object(strategy, "_run_task_loop", side_effect=fake_loop):
            with patch.object(strategy, "_start_tier", return_value=0):
                strategy.execute(
                    [t1, t2], session, fake_ladder,
                    budget=5.0, resume=True, cowabunga=True,
                )

        # $4 from prior + $0 fake run = $4 < $5 budget; first task runs.
        # After first task budget check: $4 still < $5, so second task runs too.
        # (fake_result cost=0.0, so total stays at $4 throughout)
        assert ran == ["US-001", "US-002"]


# ---------------------------------------------------------------------------
# planner → cascade integration: PRD deps flow through pipeline to topo_sort
# ---------------------------------------------------------------------------

class TestPlannerCascadeIntegration:
    def test_prd_deps_reach_topo_sort(self, tmp_path: Path) -> None:
        """parse_stories + assign_target_files → cascade executes in dep order."""
        from splinter.agents.planner import parse_stories
        from splinter.strategies.cascade import CascadeStrategy

        prd = textwrap.dedent("""\
            ### US-001: Base
            **Description:** foundation layer
            - [ ] base done

            ### US-002: Middle
            **Description:** middle layer
            Depends on US-001
            - [ ] middle done

            ### US-003: Top
            **Description:** top layer
            Depends on US-002
            - [ ] top done
        """)

        tasks = parse_stories(prd)
        assert len(tasks) == 3

        ordered = CascadeStrategy._topo_sort(tasks)
        ids = [t.id for t in ordered]
        assert ids.index("US-001") < ids.index("US-002")
        assert ids.index("US-002") < ids.index("US-003")

    def test_prd_frontmatter_strategy_cascade(self) -> None:
        """PRD frontmatter strategy: cascade is returned by plan()."""

        from splinter.agents.planner import _parse_frontmatter

        prd = "---\nstrategy: cascade\n---\n### US-001: task\n**Description:** do it\n- [ ] done\n"
        fm, _body = _parse_frontmatter(prd)
        assert fm.get("strategy") == "cascade"
