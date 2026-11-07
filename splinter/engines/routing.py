"""Effort picks the model: a brief's effort maps to a ladder tier, and that
tier's ordered model chain is the list of engines the brief may run on.

The engine for each chain entry is derived from the model id prefix
(``mcode/…``, ``kimi/…``, ``agy/…``, ``codex/…``, ``opencode…/…``, bare claude
names), so there is a single source of truth for "which model runs": the
effort → model chains in ``splinter configure``.
"""

from __future__ import annotations

from splinter.engines.catalog import Engine
from splinter.enums import Effort
from splinter.models.roster import Ladder, provider_for

LAUNCHABLE_PROVIDERS: frozenset[str] = frozenset(
    {"mcode", "kimi", "agy", "claude", "codex", "opencode"}
)
_PREFIXED_PROVIDERS: frozenset[str] = frozenset({"mcode", "kimi", "agy"})
_CONTINUE_PROVIDERS: frozenset[str] = frozenset({"mcode", "kimi", "codex"})
_EFFORT_PROVIDERS: frozenset[str] = frozenset({"claude", "agy"})
DEFAULT_EFFORT = Effort.NORMAL


def engine_for_model(model_id: str, effort: str | None = None) -> Engine | None:
    """Build the engine that runs ``model_id``; None when no engine CLI can launch it."""
    model_id = model_id.strip()
    if not model_id:
        return None
    provider = provider_for(model_id)
    if provider not in LAUNCHABLE_PROVIDERS:
        return None
    bare = (
        model_id.removeprefix(f"{provider}/") if provider in _PREFIXED_PROVIDERS else model_id
    )
    return Engine(
        key=model_id,
        label=model_id,
        provider=provider,
        model=bare,
        role="effort chain",
        default_effort=effort or "medium",
        supports_continue=provider in _CONTINUE_PROVIDERS,
        supports_effort=provider in _EFFORT_PROVIDERS,
    )


def start_tier_for(ladder: Ladder, effort: str) -> int:
    """The ladder tier an effort starts on; unknown efforts start where ``normal`` does."""
    mapping = ladder.effort_mapping(effort) or ladder.effort_mapping(DEFAULT_EFFORT)
    return mapping.start_tier if mapping is not None else 0


def chain_for_effort(ladder: Ladder, effort: str) -> tuple[Engine, ...]:
    """Engines for ``effort`` in fallback order, each carrying its own effort variant."""
    level = start_tier_for(ladder, effort)
    try:
        tier = ladder.tier_by_level(level)
    except (KeyError, ValueError, IndexError):
        return ()
    mapping = ladder.effort_mapping(effort)
    tier_variant = ladder.tier_variant(level) or (mapping.variant if mapping else None)
    engines: list[Engine] = []
    for model in tier.models:
        engine = engine_for_model(model, tier.model_variants.get(model) or tier_variant)
        if engine is not None and all(e.key != engine.key for e in engines):
            engines.append(engine)
    return tuple(engines)
