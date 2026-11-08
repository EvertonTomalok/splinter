"""Spend aggregation for the pure run view."""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

from splinter.models.roster import Ladder
from splinter.run_snapshot import Snapshot, cost_split

if TYPE_CHECKING:
    from splinter.run_view import SpendRow


def spend_rows(snap: Snapshot, ladder: Ladder | None) -> tuple[SpendRow, ...]:
    """Group observed spend by the C2 rail labels."""
    grouped: OrderedDict[str, list[float]] = OrderedDict()
    for entry in snap.entries:
        label = _label(entry.role, entry.tier, ladder)
        values = grouped.setdefault(label, [0.0, 0.0])
        values[0] += entry.cost
        values[1] += entry.latency_s
    _run_cost, pre_run, jev_cost, jev_calls = cost_split(snap)
    if pre_run:
        grouped["PRD + planner"] = [pre_run, 0.0]
    if jev_cost or jev_calls:
        grouped[f"jev · {jev_calls} calls"] = [jev_cost, 0.0]
    total = sum(values[0] for values in grouped.values())
    from splinter.run_view import SpendRow

    return tuple(
        SpendRow(label, values[0], values[1], values[0] / total if total else 0.0)
        for label, values in grouped.items()
    )


def _label(role: str, tier: int, ladder: Ladder | None) -> str:
    if role in ("localize", "filter") or role.startswith("recall"):
        return "localize"
    if role == "plan":
        if ladder is None:
            return "planner"
        return f"planner {ladder.planner_model}@{ladder.planner_effort}"
    if role == "run":
        return f"runner T{tier}"
    if role == "eval":
        if ladder is None:
            return "eval"
        return f"eval {ladder.eval_model}@{ladder.eval_effort}"
    return role
