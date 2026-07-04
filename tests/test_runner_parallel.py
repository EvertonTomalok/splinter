"""Tests for Task.parallelizable field and validate_deps (US-002)."""

from __future__ import annotations

import pytest

from splinter.agents.runner import Task, validate_deps


def _t(id: str, deps: list[str] | None = None, parallelizable: bool | None = None) -> Task:
    return Task(
        description=f"{id} task", acceptance="done", id=id, deps=deps, parallelizable=parallelizable
    )


class TestIsParallelizable:
    def test_no_deps_defaults_true(self) -> None:
        assert _t("A").is_parallelizable() is True

    def test_with_deps_defaults_false(self) -> None:
        assert _t("B", deps=["A"]).is_parallelizable() is False

    def test_explicit_true_overrides_deps(self) -> None:
        assert _t("C", deps=["A"], parallelizable=True).is_parallelizable() is True

    def test_explicit_false_overrides_no_deps(self) -> None:
        assert _t("D", parallelizable=False).is_parallelizable() is False

    def test_explicit_none_derives_from_empty_deps(self) -> None:
        t = Task(description="task", acceptance="done", parallelizable=None)
        assert t.is_parallelizable() is True

    def test_backward_compat_no_parallelizable_field(self) -> None:
        t = Task(description="task", acceptance="done")
        assert t.is_parallelizable() is True


class TestValidateDeps:
    def test_valid_linear_chain(self) -> None:
        tasks = [_t("A"), _t("B", deps=["A"]), _t("C", deps=["B"])]
        validate_deps(tasks)  # no exception

    def test_no_deps_ok(self) -> None:
        tasks = [_t("A"), _t("B")]
        validate_deps(tasks)

    def test_empty_list_ok(self) -> None:
        validate_deps([])

    def test_unknown_dep_raises(self) -> None:
        tasks = [_t("A", deps=["UNKNOWN"])]
        with pytest.raises(ValueError, match="unknown"):
            validate_deps(tasks)

    def test_unknown_dep_names_offender(self) -> None:
        tasks = [_t("A", deps=["X-999"])]
        with pytest.raises(ValueError, match="X-999"):
            validate_deps(tasks)

    def test_cycle_raises(self) -> None:
        tasks = [_t("A", deps=["B"]), _t("B", deps=["A"])]
        with pytest.raises(ValueError, match="cycle"):
            validate_deps(tasks)

    def test_self_dep_is_cycle(self) -> None:
        tasks = [_t("A", deps=["A"])]
        with pytest.raises(ValueError, match="cycle"):
            validate_deps(tasks)

    def test_tasks_without_id_skipped(self) -> None:
        t_anon = Task(description="anon", acceptance="done")
        tasks = [_t("A"), t_anon]
        validate_deps(tasks)  # no exception; anon tasks ignored

    def test_three_node_cycle(self) -> None:
        tasks = [_t("A", deps=["C"]), _t("B", deps=["A"]), _t("C", deps=["B"])]
        with pytest.raises(ValueError, match="cycle"):
            validate_deps(tasks)


# --- kowabunga scheduler ordering (US-004) ------------------------------------


def _te(id: str, effort: str) -> Task:
    return Task(description=f"{id} task", acceptance="done", id=id, effort=effort)


def _reorder(ready: list[Task], ladder: object, *, cowabunga: bool) -> list[Task]:
    """Mirror the exact reorder the parallel scheduler applies each dispatch pass."""
    from splinter.agents.evaluator import is_premium_task

    if cowabunga:
        return sorted(ready, key=lambda t: 0 if is_premium_task(t, ladder) else 1)
    return list(ready)


class TestKowabungaSchedulerOrdering:
    def test_is_premium_task_by_effort(self) -> None:
        from splinter.agents.evaluator import is_premium_task
        from splinter.models.roster import load_ladder

        ladder = load_ladder()
        assert is_premium_task(_te("A", "hard"), ladder) is True
        assert is_premium_task(_te("B", "critical"), ladder) is True
        assert is_premium_task(_te("C", "normal"), ladder) is False
        assert is_premium_task(_te("D", "trivial"), ladder) is False

    def test_on_reorders_premium_first(self) -> None:
        from splinter.models.roster import load_ladder

        ladder = load_ladder()
        ready = [_te("A", "normal"), _te("B", "hard"), _te("C", "normal"), _te("D", "critical")]
        out = [t.id for t in _reorder(ready, ladder, cowabunga=True)]
        # Premium tasks jump ahead; stable within groups (B before D, A before C).
        assert out == ["B", "D", "A", "C"]

    def test_off_leaves_order_unchanged(self) -> None:
        from splinter.models.roster import load_ladder

        ladder = load_ladder()
        ready = [_te("A", "normal"), _te("B", "hard"), _te("C", "normal"), _te("D", "critical")]
        out = [t.id for t in _reorder(ready, ladder, cowabunga=False)]
        assert out == ["A", "B", "C", "D"]
