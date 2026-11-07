"""Process-wide model health so every strategy reroutes around failing models.

Tier model lists (``models.tiers[n].models``) are pools, not singletons: when a
model hits a transient provider gap (rate limit, overload) its breaker opens and
the next resolution picks a healthy sibling from the same tier — smarter model
calling for the existing strategies, no tier escalation burned.
"""

from __future__ import annotations

import threading

from splinter.engines.health import BreakerState, CircuitBreaker

_lock = threading.Lock()
_breakers: dict[str, CircuitBreaker] = {}
#: One gap opens a model's breaker — a 429 should reroute to a sibling on the
#: very next attempt; success closes it and the cooldown re-offers it.
_FAILURE_THRESHOLD = 1
_COOLDOWN_S = 300.0


def report_model_success(model: str) -> None:
    """Close the model's breaker after a good run."""
    with _lock:
        breaker = _breakers.get(model)
        if breaker is not None:
            breaker.record_success()


def report_model_failure(model: str, *, now: float | None = None) -> None:
    """Count an infra failure; the breaker opens at its threshold."""
    with _lock:
        breaker = _breakers.setdefault(
            model,
            CircuitBreaker(failure_threshold=_FAILURE_THRESHOLD, cooldown_s=_COOLDOWN_S),
        )
        breaker.record_failure(now)


def model_available(model: str, *, now: float | None = None) -> bool:
    """True while the model's breaker is closed (or half-open for one trial)."""
    with _lock:
        breaker = _breakers.get(model)
        return breaker.allow(now) if breaker is not None else True


def model_state(model: str, *, now: float | None = None) -> BreakerState:
    with _lock:
        breaker = _breakers.get(model)
        return breaker.state(now) if breaker is not None else BreakerState.CLOSED


def snapshot() -> dict[str, BreakerState]:
    with _lock:
        return {m: b.state() for m, b in _breakers.items()}


def reset_all() -> None:
    """Forget every breaker (tests, and fresh runs inside one process)."""
    with _lock:
        _breakers.clear()
