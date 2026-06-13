"""Michelangelo — the ``sprint`` flash-first strategy.

Flow: topological sort (like adaptive/cascade), but every task starts at the
cheapest ladder tier regardless of estimated effort. Escalation happens only
when the eval decides it, via the existing US-001 ladder in
``DirectStrategy._run_task_loop``.
"""

from __future__ import annotations

from splinter.models.roster import Ladder
from splinter.strategies.adaptive import AdaptiveStrategy
from splinter.strategies.registry import register


@register
class SprintStrategy(AdaptiveStrategy):
    name = "sprint"
    aliases = ["michelangelo"]
    _log_prefix: str = "sprint"
    _log_routing_detail = False

    @staticmethod
    def _route_tier(
        effort: str,
        ladder: Ladder,
        remaining_budget: float | None = None,
        remaining_efforts: list[str] | None = None,
    ) -> int:
        """Always start at the cheapest (flash) tier; escalate only on eval failure."""
        return min(t.level for t in ladder.tiers)
