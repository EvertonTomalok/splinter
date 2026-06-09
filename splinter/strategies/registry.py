"""Registry/factory for strategies (turtles).

Strategies register themselves with :func:`register` (by ``name`` plus any
``aliases``), and the pipeline resolves them with :func:`get_strategy`. This
replaces the hand-maintained ``STRATEGY_MAP`` dict and keeps the canonical name
and its turtle alias in sync automatically.
"""

from __future__ import annotations

from splinter.strategies.base import Strategy

_REGISTRY: dict[str, type[Strategy]] = {}


def register(cls: type[Strategy]) -> type[Strategy]:
    """Class decorator: register a strategy under its name and aliases."""
    for key in (cls.name, *cls.aliases):
        _REGISTRY[key.lower()] = cls
    return cls


def get_strategy(name: str) -> Strategy:
    """Instantiate the strategy registered under ``name`` (or a turtle alias)."""
    cls = _REGISTRY.get(name.lower())
    if cls is None:
        raise ValueError(
            f"unknown strategy '{name}'. Available: {', '.join(available_strategies())}"
        )
    return cls()


def available_strategies() -> list[str]:
    """All registered strategy names and aliases."""
    return sorted(_REGISTRY)
