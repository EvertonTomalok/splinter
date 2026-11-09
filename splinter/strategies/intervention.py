"""A person's decision about one story: hand it over, change its model, raise its tier."""

from __future__ import annotations

from dataclasses import dataclass

from splinter.agents.runner import resolve_model
from splinter.engines.runtime import report_model_failure
from splinter.models.roster import Ladder

ORCHESTRATOR = "orchestrator"
REROUTE = "reroute"
TIER_UP = "tier_up"
_HANDOVER = (
    "The user handed this story to the orchestrator. Finish it in this worktree: "
    "keep the work already done and complete what is missing."
)


@dataclass(frozen=True)
class Applied:
    tier: int
    note: str = ""


def apply_intervention(
    action: str, tier: int, ladder: Ladder, *, premium_tier: int, max_tier: int
) -> Applied:
    """The tier the story continues on, and what its next engine should be told."""
    if action == ORCHESTRATOR:
        return Applied(max(tier, premium_tier), _HANDOVER)
    if action == REROUTE and _has_sibling(tier, ladder):
        model, _variant = resolve_model(tier, ladder)
        report_model_failure(model)
        return Applied(tier)
    return Applied(min(tier + 1, max_tier))


def _has_sibling(tier: int, ladder: Ladder) -> bool:
    models = next((row.models for row in ladder.tiers if row.level == tier), [])
    return len(models) > 1
