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


def _ensure_registered() -> None:
    if not _REGISTRY:
        import splinter.strategies.cascade  # noqa: F401
        import splinter.strategies.direct  # noqa: F401


def get_strategy(name: str) -> Strategy:
    """Instantiate the strategy registered under ``name`` (or a turtle alias)."""
    _ensure_registered()
    cls = _REGISTRY.get(name.lower())
    if cls is None:
        raise ValueError(
            f"unknown strategy '{name}'. Available: {', '.join(available_strategies())}"
        )
    return cls()


def available_strategies() -> list[str]:
    """All registered strategy names and aliases."""
    _ensure_registered()
    return sorted(_REGISTRY)


def registered_strategies() -> list[type[Strategy]]:
    """Unique strategy classes (deduped across name + aliases), sorted by name."""
    _ensure_registered()
    seen: set[str] = set()
    out: list[type[Strategy]] = []
    for cls in _REGISTRY.values():
        if cls.name not in seen:
            seen.add(cls.name)
            out.append(cls)
    return sorted(out, key=lambda c: c.name)
