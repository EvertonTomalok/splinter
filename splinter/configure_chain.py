"""Fallback chains for ``splinter configure``: pure state behind the TUI.

A chain is the ordered list of models one step (or one effort tier) may run on:
entry 1 is the primary, the rest are fallbacks tried in order when the one
before is rate-limited. Every entry carries its own effort. All helpers return
new tuples — chains are never mutated in place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from splinter.configure import EFFORT_CHOICES
from splinter.models.roster import Ladder, parse_pool_entry, provider_for

EFFORT_CYCLE: tuple[str, ...] = ("", *(e for e in EFFORT_CHOICES if e != "auto"))

#: Effort/variant names each provider CLI accepts natively; empty = no effort knob
#: (mcode/kimi have no flag; cursor encodes the variant in the model id).
PROVIDER_EFFORTS: dict[str, tuple[str, ...]] = {
    "claude": ("low", "medium", "high", "xhigh", "max"),
    "opencode": ("minimal", "low", "medium", "high", "max"),
    "codex": ("low", "medium", "high", "xhigh"),
    "agy": ("low", "medium", "high"),
    "mcode": (),
    "kimi": (),
    "cursor": (),
}
PROVIDERS: tuple[str, ...] = tuple(PROVIDER_EFFORTS)
_PREFIXED = frozenset({"opencode", "codex", "agy", "mcode", "kimi", "cursor"})

STEP_ATTRS: tuple[tuple[str, str, str, str], ...] = (
    ("localizer_recall", "localizer_recall_model", "localizer_recall_variant",
     "localizer_recall_timeout"),
    ("localizer_recall_large", "localizer_recall_large_model",
     "localizer_recall_large_variant", "localizer_recall_large_timeout"),
    ("localizer_precision", "localizer_precision_model", "localizer_precision_variant",
     "localizer_precision_timeout"),
    ("planner", "planner_model", "planner_effort", "planner_timeout"),
    ("prd", "prd_model", "prd_effort", "prd_timeout"),
    ("eval", "eval_model", "eval_effort", "eval_timeout"),
)
STEP_KEYS: tuple[str, ...] = tuple(key for key, *_ in STEP_ATTRS)


@dataclass(frozen=True)
class ChainEntry:
    """One model of a chain; ``effort == ""`` means the step/tier default."""

    model: str
    effort: str = ""

    def to_pool_entry(self) -> str:
        return f"{self.model}@{self.effort}" if self.effort else self.model


Chain = tuple[ChainEntry, ...]


def tier_key(level: int) -> str:
    return f"tier_{level}"


def _step_chain(ladder: Ladder, key: str, model_attr: str, effort_attr: str) -> Chain:
    effort = str(getattr(ladder, effort_attr, "") or "")
    pool = ladder.step_pool(key)
    if not pool:
        return (ChainEntry(str(getattr(ladder, model_attr)), effort),)
    entries = [parse_pool_entry(e) for e in pool]
    primary_model, primary_effort = entries[0]
    rest = tuple(ChainEntry(m, v or "") for m, v in entries[1:])
    return (ChainEntry(primary_model, primary_effort or effort), *rest)


def chains_from_ladder(ladder: Ladder) -> dict[str, Chain]:
    """Every step and tier chain, keyed by step key or ``tier_<level>``."""
    chains: dict[str, Chain] = {
        key: _step_chain(ladder, key, model_attr, effort_attr)
        for key, model_attr, effort_attr, _ in STEP_ATTRS
    }
    for tier in sorted(ladder.tiers, key=lambda t: t.level):
        primary_effort = ladder.tier_variant(tier.level) or ""
        chains[tier_key(tier.level)] = tuple(
            ChainEntry(m, tier.model_variants.get(m) or (primary_effort if i == 0 else ""))
            for i, m in enumerate(tier.models)
        )
    return chains


def timeouts_from_ladder(ladder: Ladder) -> dict[str, int]:
    timeouts = {key: int(getattr(ladder, attr)) for key, _, _, attr in STEP_ATTRS}
    for tier in ladder.tiers:
        timeouts[tier_key(tier.level)] = ladder.tier_timeout(tier.level)
    return timeouts


def efforts_for(model: str) -> tuple[str, ...]:
    """``""`` (default) plus the native effort names of ``model``'s provider."""
    return ("", *PROVIDER_EFFORTS.get(provider_for(model), EFFORT_CYCLE[1:]))


def fit_effort(model: str, effort: str) -> str:
    """Keep ``effort`` when ``model``'s provider accepts it, else fall back to default."""
    return effort if effort in efforts_for(model) else ""


def display_model(model: str) -> str:
    """Model id without a redundant ``<provider>/`` prefix (opencode-go/ stays)."""
    provider = provider_for(model)
    return model.removeprefix(f"{provider}/") if provider in _PREFIXED else model


def with_provider(model: str, provider: str) -> str:
    """Prefix a typed id with ``provider`` when it would otherwise route elsewhere."""
    model = model.strip()
    if not model or provider not in PROVIDERS or provider_for(model) == provider:
        return model
    return model if provider == "claude" else f"{provider}/{model}"


def append_entry(chain: Chain, model: str, effort: str = "") -> Chain:
    """Add ``model`` as the last fallback; a model already in the chain is ignored."""
    model = model.strip()
    if not model or any(e.model == model for e in chain):
        return chain
    return (*chain, ChainEntry(model, fit_effort(model, effort)))


def remove_entry(chain: Chain, index: int) -> Chain:
    """Drop entry ``index``; a chain always keeps at least its primary."""
    if len(chain) <= 1 or not 0 <= index < len(chain):
        return chain
    return chain[:index] + chain[index + 1 :]


def move_entry(chain: Chain, index: int, delta: int) -> Chain:
    target = index + delta
    if not (0 <= index < len(chain) and 0 <= target < len(chain)):
        return chain
    items = list(chain)
    items[index], items[target] = items[target], items[index]
    return tuple(items)


def replace_model(chain: Chain, index: int, model: str) -> Chain:
    model = model.strip()
    if not model or not 0 <= index < len(chain):
        return chain
    if any(e.model == model for i, e in enumerate(chain) if i != index):
        return chain
    entry = ChainEntry(model, fit_effort(model, chain[index].effort))
    return chain[:index] + (entry,) + chain[index + 1 :]


def set_effort(chain: Chain, index: int, effort: str) -> Chain:
    if not 0 <= index < len(chain) or chain[index].effort == effort:
        return chain
    return chain[:index] + (ChainEntry(chain[index].model, effort),) + chain[index + 1 :]


def cycle_effort(chain: Chain, index: int, step: int = 1) -> Chain:
    if not 0 <= index < len(chain):
        return chain
    current = chain[index].effort
    cycle = efforts_for(chain[index].model)
    if len(cycle) <= 1:
        return chain
    pos = cycle.index(current) if current in cycle else 0
    nxt = cycle[(pos + step) % len(cycle)]
    return chain[:index] + (ChainEntry(chain[index].model, nxt),) + chain[index + 1 :]


def short_model(model: str) -> str:
    """Last path segment of a model id: ``opencode-go/minimax-m3`` → ``minimax-m3``."""
    return model.rsplit("/", 1)[-1]


def chain_summary(chain: Chain, width: int = 18) -> str:
    """``minimax-m3 +1`` — the primary's short name plus the fallback count."""
    if not chain:
        return "—"
    extra = f" +{len(chain) - 1}" if len(chain) > 1 else ""
    name = short_model(chain[0].model)
    room = max(width - len(extra), 4)
    if len(name) > room:
        name = name[: room - 1] + "…"
    return f"{name}{extra}"


def chain_to_config(chain: Chain) -> tuple[Any, str, str]:
    """``(models value, primary effort, provider)`` in the config.yaml shape."""
    primary = chain[0]
    models: Any = (
        primary.model
        if len(chain) == 1
        else [primary.model, *(e.to_pool_entry() for e in chain[1:])]
    )
    return models, primary.effort, provider_for(primary.model)


def config_payload(
    chains: dict[str, Chain], timeouts: dict[str, int | None], tier_count: int
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build the ``models``/``efforts``/``timeouts``/``providers`` dicts to write."""
    models: dict[str, Any] = {}
    efforts: dict[str, Any] = {}
    out_timeouts: dict[str, Any] = {}
    providers: dict[str, Any] = {}
    for key in STEP_KEYS:
        chain = chains.get(key)
        if chain:
            models[key], efforts[key], providers[key] = chain_to_config(chain)
        out_timeouts[key] = timeouts.get(key)
    tier_models: list[Any] = []
    tier_efforts: list[str] = []
    tier_providers: list[str] = []
    for level in range(tier_count):
        chain = chains.get(tier_key(level))
        if not chain:
            continue
        m, e, p = chain_to_config(chain)
        tier_models.append(m)
        tier_efforts.append(e)
        tier_providers.append(p)
    models["tiers"] = tier_models
    efforts["tiers"] = tier_efforts
    providers["tiers"] = tier_providers
    out_timeouts["tiers"] = [timeouts.get(tier_key(i)) for i in range(tier_count)]
    return models, efforts, out_timeouts, providers


def efforts_starting_at(ladder: Ladder, level: int) -> list[str]:
    """Task efforts (trivial/normal/…) whose first tier is ``level``."""
    return [name for name, em in ladder.effort_map.items() if em.start_tier == level]
