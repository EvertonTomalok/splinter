"""Tests for SprintStrategy (Leonardo — flash-first)."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from splinter.models.roster import load_ladder
from splinter.strategies import sprint as sprint_module
from splinter.strategies.adaptive import AdaptiveStrategy
from splinter.strategies.registry import available_strategies, get_strategy
from splinter.strategies.sprint import (
    SprintStrategy,  # noqa: F401 (imported for registration side-effect)
)

# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_sprint_registered() -> None:
    assert isinstance(get_strategy("sprint"), SprintStrategy)


def test_michelangelo_alias_registered() -> None:
    assert isinstance(get_strategy("michelangelo"), SprintStrategy)


def test_sprint_in_available_strategies() -> None:
    names = available_strategies()
    assert "sprint" in names
    assert "michelangelo" in names


# ---------------------------------------------------------------------------
# Docstring validation
# ---------------------------------------------------------------------------


def test_sprint_docstring_no_us004_block() -> None:
    """Module docstring must not contain 'Blocked until US-004'."""
    doc = inspect.getdoc(sprint_module)
    assert doc is not None, "sprint module missing docstring"
    assert "Blocked until US-004" not in doc, (
        f"sprint docstring should not mention US-004 blocking: {doc}"
    )


def test_sprint_docstring_describes_flash_first() -> None:
    """Module docstring must describe flash-first behavior."""
    doc = inspect.getdoc(sprint_module)
    assert doc is not None, "sprint module missing docstring"
    # Expect both "flash" and "cheapest" to be present
    assert "flash" in doc.lower(), (
        f"sprint docstring should mention 'flash': {doc}"
    )
    assert "cheapest" in doc.lower(), (
        f"sprint docstring should mention 'cheapest': {doc}"
    )


# ---------------------------------------------------------------------------
# _route_tier: always returns floor (level 0)
# ---------------------------------------------------------------------------


def test_route_tier_always_floor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    ladder = load_ladder()
    strategy = SprintStrategy()
    floor = min(t.level for t in ladder.tiers)
    for effort in ("trivial", "normal", "hard", "critical"):
        assert strategy._route_tier(effort, ladder) == floor, f"effort={effort}"


def test_route_tier_returns_zero_for_all_efforts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    ladder = load_ladder()
    strategy = SprintStrategy()
    for effort in ("trivial", "normal", "hard", "critical"):
        tier = strategy._route_tier(effort, ladder)
        assert tier == 0, f"expected T0 for effort={effort}, got T{tier}"


def test_route_tier_ignores_budget_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sprint ignores budget/remaining_efforts and always returns the cheapest tier."""
    monkeypatch.chdir(tmp_path)
    ladder = load_ladder()
    strategy = SprintStrategy()
    floor = min(t.level for t in ladder.tiers)
    for effort in ("trivial", "normal", "hard", "critical"):
        assert strategy._route_tier(effort, ladder, 0.001, [effort]) == floor
        assert strategy._route_tier(effort, ladder, 100.0, [effort]) == floor
        assert strategy._route_tier(effort, ladder, None, None) == floor


# ---------------------------------------------------------------------------
# Class flags and logging
# ---------------------------------------------------------------------------


def test_sprint_log_routing_detail_false() -> None:
    """SprintStrategy sets _log_routing_detail to False."""
    strategy = SprintStrategy()
    assert strategy._log_routing_detail is False


def test_sprint_no_execute_override() -> None:
    """SprintStrategy does not override execute() — uses adaptive's."""
    assert "execute" not in SprintStrategy.__dict__


def test_sprint_short_log_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """When _log_routing_detail=False, log omits budget details."""
    from unittest.mock import MagicMock

    from splinter.agents.runner import RunResult, Task
    from splinter.memory.session import Session

    monkeypatch.chdir(tmp_path)
    caplog.set_level("INFO", logger="splinter.loop")

    session = Session(str(tmp_path / "ses"))
    ladder = load_ladder()
    task = Task(description="Test task", acceptance="AC", effort="normal", id="T1")

    strategy = SprintStrategy()
    strategy._run_task_loop = MagicMock(  # type: ignore[method-assign]
        return_value=MagicMock(spec=RunResult, text="done")
    )
    strategy._run_plan_phase = MagicMock()  # type: ignore[method-assign]

    strategy.execute([task], session, ladder)

    log_messages = [r.message for r in caplog.records if r.name == "splinter.loop"]
    task_logs = [m for m in log_messages if "task 1/1" in m]
    assert task_logs, "Expected task log message not found"
    task_log = task_logs[0]

    assert task_log.startswith("sprint: task 1/1"), f"log={task_log}"
    assert "effort=normal" in task_log
    assert "[budget=" not in task_log, f"log should not contain [budget=...: {task_log}"
    assert "floor=" not in task_log, f"log should not contain floor=: {task_log}"


def test_adaptive_log_routing_detail_true() -> None:
    """AdaptiveStrategy sets _log_routing_detail to True."""
    from splinter.strategies.adaptive import AdaptiveStrategy

    strategy = AdaptiveStrategy()
    assert strategy._log_routing_detail is True


def test_adaptive_detailed_log_uses_name_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """When _log_routing_detail=True, adaptive logs detailed budget info with self.name prefix."""
    from unittest.mock import MagicMock

    from splinter.agents.runner import RunResult, Task
    from splinter.memory.session import Session
    from splinter.strategies.adaptive import AdaptiveStrategy

    monkeypatch.chdir(tmp_path)
    caplog.set_level("INFO", logger="splinter.loop")

    session = Session(str(tmp_path / "ses"))
    ladder = load_ladder()
    task = Task(description="Test task", acceptance="AC", effort="normal", id="T1")

    strategy = AdaptiveStrategy()
    strategy._run_task_loop = MagicMock(  # type: ignore[method-assign]
        return_value=MagicMock(spec=RunResult, text="done")
    )
    strategy._run_plan_phase = MagicMock()  # type: ignore[method-assign]

    strategy.execute([task], session, ladder, budget=10.0)

    log_messages = [r.message for r in caplog.records if r.name == "splinter.loop"]
    task_logs = [m for m in log_messages if "task 1/1" in m]
    assert task_logs, "Expected task log message not found"
    task_log = task_logs[0]

    assert task_log.startswith("adaptive: task 1/1"), f"log={task_log}"
    assert "effort=normal" in task_log
    assert "[budget=" in task_log, f"log should contain [budget=: {task_log}"
    assert "floor=" in task_log, f"log should contain floor=: {task_log}"


# ---------------------------------------------------------------------------
# US-004: Table-driven tests for Sprint tier routing
# ---------------------------------------------------------------------------

_SPRINT_ROUTE_CASES: list[tuple[str, str, float | None, list[str]]] = [
    # budget=None
    ("no_budget_trivial", "trivial", None, []),
    ("no_budget_normal", "normal", None, []),
    ("no_budget_hard", "hard", None, []),
    ("no_budget_critical", "critical", None, []),
    # budget=ample (10.0)
    ("ample_trivial", "trivial", 10.0, ["trivial"]),
    ("ample_normal", "normal", 10.0, ["normal"]),
    ("ample_hard", "hard", 10.0, ["hard"]),
    ("ample_critical", "critical", 10.0, ["critical"]),
    # budget=tight (0.008)
    ("tight_trivial", "trivial", 0.008, ["trivial"]),
    ("tight_normal", "normal", 0.008, ["normal"]),
    ("tight_hard", "hard", 0.008, ["hard"]),
    ("tight_critical", "critical", 0.008, ["critical"]),
    # budget=exhausted (0.0001)
    ("exhausted_trivial", "trivial", 0.0001, ["trivial"]),
    ("exhausted_normal", "normal", 0.0001, ["normal"]),
    ("exhausted_hard", "hard", 0.0001, ["hard"]),
    ("exhausted_critical", "critical", 0.0001, ["critical"]),
]


@pytest.mark.parametrize(
    "case_id,effort,budget,remaining",
    _SPRINT_ROUTE_CASES,
    ids=[c[0] for c in _SPRINT_ROUTE_CASES],
)
def test_sprint_routes_all_to_t0(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case_id: str,
    effort: str,
    budget: float | None,
    remaining: list[str],
) -> None:
    """Every effort × budget combination routes to T0."""
    monkeypatch.chdir(tmp_path)
    ladder = load_ladder()
    tier = SprintStrategy._route_tier(effort, ladder, budget, remaining)
    assert tier == 0, f"case={case_id}: expected T0, got T{tier}"


def test_sprint_routed_tier_equals_ladder_min(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sprint's routed tier equals min(t.level for t in ladder.tiers)."""
    monkeypatch.chdir(tmp_path)
    ladder = load_ladder()
    expected = min(t.level for t in ladder.tiers)
    for effort in ("trivial", "normal", "hard", "critical"):
        for budget in (None, 10.0, 0.008, 0.0001):
            remaining = [effort] if budget is not None else []
            tier = SprintStrategy._route_tier(effort, ladder, budget, remaining)
            assert tier == expected, (
                f"effort={effort}, budget={budget}: expected T{expected}, got T{tier}"
            )


def test_sprint_diverges_from_adaptive_on_ample_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sprint routes to T0; Adaptive respects effort floors under ample budget."""
    monkeypatch.chdir(tmp_path)
    ladder = load_ladder()

    for effort in ("trivial", "normal", "hard", "critical"):
        em = ladder.effort_mapping(effort)
        assert em is not None
        adaptive_tier = AdaptiveStrategy._route_tier(effort, ladder, 100.0, [effort])
        sprint_tier = SprintStrategy._route_tier(effort, ladder, 100.0, [effort])

        # Sprint always routes to T0
        assert sprint_tier == 0, f"effort={effort}: expected sprint T0, got T{sprint_tier}"

        # Adaptive routes to effort floor under ample budget
        assert adaptive_tier == em.start_tier, (
            f"effort={effort}: adaptive should route to floor T{em.start_tier}, "
            f"got T{adaptive_tier}"
        )

        # For non-trivial efforts, Sprint's tier < Adaptive's tier
        if em.start_tier > 0:
            assert sprint_tier < adaptive_tier, (
                f"effort={effort}: sprint should be cheaper; "
                f"sprint=T{sprint_tier}, adaptive=T{adaptive_tier}"
            )


# ---------------------------------------------------------------------------
# US-005: Sprint execute kwargs tests
# ---------------------------------------------------------------------------


def test_sprint_passes_start_tier_override_zero_for_all_efforts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sprint passes start_tier_override=0 for every effort level."""
    from unittest.mock import MagicMock

    from splinter.agents.runner import RunResult, Task
    from splinter.memory.session import Session

    monkeypatch.chdir(tmp_path)
    ladder = load_ladder()
    strategy = SprintStrategy()

    captured_start_tiers: list[int] = []

    def mock_loop(*args, **kwargs):  # type: ignore[no-untyped-def]
        captured_start_tiers.append(kwargs.get("start_tier_override"))
        rr = MagicMock(spec=RunResult)
        rr.text = "done"
        return rr

    strategy._run_task_loop = mock_loop  # type: ignore[method-assign]
    strategy._run_plan_phase = lambda *a, **kw: None  # type: ignore[method-assign]

    session = Session(str(tmp_path / "ses"))
    for effort in ("trivial", "normal", "hard", "critical"):
        captured_start_tiers.clear()
        task = Task(description="Test task", acceptance="AC", effort=effort, id="T1")
        strategy.execute([task], session, ladder)
        assert (
            captured_start_tiers[0] == 0
        ), f"effort={effort}: expected start_tier_override=0, got {captured_start_tiers[0]}"


def test_sprint_passes_configured_soft_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sprint passes soft_budget from config to _run_task_loop."""
    from unittest.mock import MagicMock

    from splinter.agents.runner import RunResult, Task
    from splinter.configure import configured_soft_budget
    from splinter.memory.session import Session

    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / ".splinter"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text("defaults:\n  soft_budget: false\n")

    ladder = load_ladder()
    strategy = SprintStrategy()

    captured_soft_budgets: list[bool] = []

    def mock_loop(*args, **kwargs):  # type: ignore[no-untyped-def]
        captured_soft_budgets.append(kwargs.get("soft_budget"))
        rr = MagicMock(spec=RunResult)
        rr.text = "done"
        return rr

    strategy._run_task_loop = mock_loop  # type: ignore[method-assign]
    strategy._run_plan_phase = lambda *a, **kw: None  # type: ignore[method-assign]

    session = Session(str(tmp_path / "ses"))
    task = Task(description="Test task", acceptance="AC", effort="trivial", id="T1")
    strategy.execute([task], session, ladder)

    assert (
        captured_soft_budgets[0] == configured_soft_budget()
    ), f"expected soft_budget={configured_soft_budget()}, got {captured_soft_budgets[0]}"
