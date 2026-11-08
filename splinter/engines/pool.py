"""Engine pool: walk a brief's effort chain in order, skipping unhealthy engines."""

from __future__ import annotations

from collections.abc import Sequence

from splinter.engines.catalog import Engine
from splinter.engines.health import EngineHealth


class PoolExhausted(RuntimeError):
    """Raised when every engine in the chain is excluded or has an open breaker."""


class EnginePool:
    """Hands out the first healthy engine of the chain it is given."""

    def __init__(self, health: EngineHealth) -> None:
        self._health = health

    def acquire(self, chain: Sequence[Engine], exclude: set[str] | None = None) -> Engine:
        """Return the first engine of ``chain`` not excluded and allowed by health."""
        excluded = exclude if exclude is not None else set()
        tried: list[str] = []
        for engine in chain:
            if engine.key in excluded or not self._health.allow(engine.key):
                tried.append(engine.key)
                continue
            return engine
        raise PoolExhausted(f"no engine available; tried: {', '.join(tried) or 'none'}")

    def wait_s(self, chain: Sequence[Engine]) -> float:
        """Seconds until the first engine of ``chain`` leaves its cooldown."""
        return min((self._health.wait_s(engine.key) for engine in chain), default=0.0)

    def report_success(self, key: str) -> None:
        """Close the engine's breaker after a good run."""
        self._health.reset(key)

    def report_infra_failure(self, key: str) -> None:
        """Count an infra failure, tripping the breaker at the threshold."""
        self._health.trip(key)
