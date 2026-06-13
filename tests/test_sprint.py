"""Tests for SprintStrategy (Leonardo — flash-first)."""

from __future__ import annotations

from pathlib import Path

import pytest

from splinter.models.roster import load_ladder
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
