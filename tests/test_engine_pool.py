"""Tests for the effort-chain engine pool."""

from __future__ import annotations

import pytest

from splinter.engines.health import EngineHealth
from splinter.engines.pool import EnginePool, PoolExhausted
from splinter.engines.routing import engine_for_model


def _chain(*models: str) -> tuple:  # type: ignore[type-arg]
    engines = [engine_for_model(m) for m in models]
    return tuple(e for e in engines if e is not None)


CHAIN = _chain("mcode/minimax/MiniMax-M3", "kimi/kimi-code/k3.1-pro", "sonnet")


def test_acquire_follows_chain_order() -> None:
    assert EnginePool(EngineHealth()).acquire(CHAIN).key == "mcode/minimax/MiniMax-M3"


def test_acquire_respects_exclude() -> None:
    pool = EnginePool(EngineHealth())
    picked = pool.acquire(CHAIN, exclude={"mcode/minimax/MiniMax-M3"})
    assert picked.key == "kimi/kimi-code/k3.1-pro"


def test_acquire_skips_tripped_engine() -> None:
    registry = EngineHealth()
    registry.trip("mcode/minimax/MiniMax-M3")
    registry.trip("mcode/minimax/MiniMax-M3")
    assert EnginePool(registry).acquire(CHAIN).key == "kimi/kimi-code/k3.1-pro"


def test_acquire_exhausted_lists_tried_keys() -> None:
    pool = EnginePool(EngineHealth())
    with pytest.raises(PoolExhausted) as excinfo:
        pool.acquire(CHAIN, exclude={e.key for e in CHAIN})
    for engine in CHAIN:
        assert engine.key in str(excinfo.value)


def test_acquire_empty_chain_is_exhausted() -> None:
    with pytest.raises(PoolExhausted):
        EnginePool(EngineHealth()).acquire(())


def test_report_infra_failure_trips_after_threshold() -> None:
    registry = EngineHealth()
    pool = EnginePool(registry)
    key = CHAIN[0].key
    pool.report_infra_failure(key)
    assert registry.allow(key) is True
    pool.report_infra_failure(key)
    assert registry.allow(key) is False
    assert pool.acquire(CHAIN).key == CHAIN[1].key


def test_report_success_resets_breaker() -> None:
    registry = EngineHealth()
    pool = EnginePool(registry)
    key = CHAIN[0].key
    pool.report_infra_failure(key)
    pool.report_infra_failure(key)
    pool.report_success(key)
    assert pool.acquire(CHAIN).key == key
