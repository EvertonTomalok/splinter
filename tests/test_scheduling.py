"""Tests for DagScheduler, topo_order, and BudgetPool (US-003)."""

from __future__ import annotations

from splinter.agents.runner import Task
from splinter.scheduling import (
    BudgetPool,
    DagScheduler,
    TaskState,
    default_max_concurrency,
    topo_order,
)


def _t(id: str, deps: list[str] | None = None) -> Task:
    return Task(description=f"{id} task", acceptance="done", id=id, deps=deps)


class TestDagSchedulerReady:
    def test_no_deps_all_ready(self) -> None:
        tasks = [_t("A"), _t("B"), _t("C")]
        sched = DagScheduler(tasks)
        ready_ids = {t.id for t in sched.ready()}
        assert ready_ids == {"A", "B", "C"}

    def test_linear_chain_only_head_ready(self) -> None:
        sched = DagScheduler([_t("A"), _t("B", deps=["A"]), _t("C", deps=["B"])])
        assert [t.id for t in sched.ready()] == ["A"]

    def test_dep_passed_unblocks_next(self) -> None:
        sched = DagScheduler([_t("A"), _t("B", deps=["A"])])
        sched.mark_running("A")
        assert sched.ready() == []
        sched.mark_passed("A")
        assert [t.id for t in sched.ready()] == ["B"]

    def test_fan_out_both_ready_after_root(self) -> None:
        sched = DagScheduler([_t("A"), _t("B", deps=["A"]), _t("C", deps=["A"])])
        sched.mark_passed("A")
        ready_ids = {t.id for t in sched.ready()}
        assert ready_ids == {"B", "C"}

    def test_fan_in_waits_for_all_deps(self) -> None:
        sched = DagScheduler([_t("A"), _t("B"), _t("C", deps=["A", "B"])])
        sched.mark_passed("A")
        assert sched.ready() == [t for t in sched.ready() if t.id != "C"]
        sched.mark_passed("B")
        ready_ids = {t.id for t in sched.ready()}
        assert "C" in ready_ids


class TestDagSchedulerMarkFailed:
    def test_failed_blocks_direct_dependent(self) -> None:
        sched = DagScheduler([_t("A"), _t("B", deps=["A"])])
        sched.mark_failed("A")
        assert sched.state("B") == TaskState.BLOCKED

    def test_failed_blocks_transitive_dependents(self) -> None:
        sched = DagScheduler([_t("A"), _t("B", deps=["A"]), _t("C", deps=["B"])])
        sched.mark_failed("A")
        assert sched.state("B") == TaskState.BLOCKED
        assert sched.state("C") == TaskState.BLOCKED

    def test_independent_task_unaffected_by_failure(self) -> None:
        sched = DagScheduler([_t("A"), _t("B", deps=["A"]), _t("D")])
        sched.mark_failed("A")
        assert sched.state("D") == TaskState.PENDING
        assert _t("D").id in {t.id for t in sched.ready()}

    def test_blocked_reason_recorded(self) -> None:
        sched = DagScheduler([_t("A"), _t("B", deps=["A"])])
        sched.mark_failed("A")
        reason = sched.blocked_reason("B")
        assert "A" in reason

    def test_is_done_after_all_terminal(self) -> None:
        sched = DagScheduler([_t("A"), _t("B")])
        sched.mark_passed("A")
        sched.mark_failed("B")
        assert sched.is_done() is True

    def test_is_done_false_while_pending(self) -> None:
        sched = DagScheduler([_t("A"), _t("B")])
        sched.mark_passed("A")
        assert sched.is_done() is False

    def test_has_running(self) -> None:
        sched = DagScheduler([_t("A")])
        assert sched.has_running() is False
        sched.mark_running("A")
        assert sched.has_running() is True
        sched.mark_passed("A")
        assert sched.has_running() is False


class TestTopoOrder:
    def test_linear_chain(self) -> None:
        tasks = [_t("A"), _t("B", deps=["A"]), _t("C", deps=["B"])]
        ordered = topo_order(tasks)
        assert ordered is not None
        ids = [t.id for t in ordered]
        assert ids.index("A") < ids.index("B") < ids.index("C")

    def test_cycle_returns_none(self) -> None:
        tasks = [_t("A", deps=["B"]), _t("B", deps=["A"])]
        assert topo_order(tasks) is None

    def test_no_deps_preserves_prd_order(self) -> None:
        tasks = [_t("A"), _t("B"), _t("C")]
        ordered = topo_order(tasks)
        assert ordered is not None
        assert [t.id for t in ordered] == ["A", "B", "C"]

    def test_diamond(self) -> None:
        t1, t2, t3, t4 = _t("A"), _t("B", deps=["A"]), _t("C", deps=["A"]), _t("D", deps=["B", "C"])
        ordered = topo_order([t4, t2, t3, t1])
        assert ordered is not None
        ids = [t.id for t in ordered]
        assert ids.index("A") < ids.index("B")
        assert ids.index("A") < ids.index("C")
        assert ids.index("B") < ids.index("D")
        assert ids.index("C") < ids.index("D")

    def test_tasks_without_id_appended(self) -> None:
        t_a = _t("A")
        t_anon = Task(description="anon", acceptance="done")
        ordered = topo_order([t_a, t_anon])
        assert ordered is not None
        assert ordered[0].id == "A"
        assert ordered[-1] is t_anon


class TestDefaultMaxConcurrency:
    def test_returns_at_least_one(self) -> None:
        result = default_max_concurrency()
        assert result >= 1

    def test_returns_int(self) -> None:
        assert isinstance(default_max_concurrency(), int)


class TestBudgetPool:
    def test_not_exhausted_without_budget(self) -> None:
        pool = BudgetPool(_budget=None)
        assert pool.exhausted() is False

    def test_exhausted_when_over_budget(self) -> None:
        pool = BudgetPool(_budget=5.0)
        pool.add_cost(5.01)
        assert pool.exhausted() is True

    def test_not_exhausted_when_under_budget(self) -> None:
        pool = BudgetPool(_budget=10.0)
        pool.add_cost(9.99)
        assert pool.exhausted() is False

    def test_base_cost_included_in_check(self) -> None:
        pool = BudgetPool(_budget=5.0)
        pool.add_cost(2.0)
        assert pool.exhausted(base_cost=3.5) is True
        assert pool.exhausted(base_cost=2.9) is False

    def test_total_extra_accumulates(self) -> None:
        pool = BudgetPool(_budget=10.0)
        pool.add_cost(1.0)
        pool.add_cost(2.5)
        assert abs(pool.total_extra - 3.5) < 1e-9
