"""Tests for project-level pricing store and roster merge."""

from __future__ import annotations

from pathlib import Path

import pytest

from splinter.models.pricing import _price_for
from splinter.models.pricing_store import (
    load_store,
    merge,
    price_for,
    save_store,
)
from splinter.models.roster import load_ladder
from splinter.providers.base import ModelPrice


def test_store_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store_file = tmp_path / "pricing.json"
    monkeypatch.setattr("splinter.models.pricing_store.STORE_PATH", store_file)
    prices = {"sonnet": ModelPrice(input=4.0, output=16.0, cache_read=0.4, cache_write=5.0)}
    save_store(prices)
    loaded = load_store()
    assert loaded["sonnet"] == prices["sonnet"]
    again = load_store()
    assert again == loaded


def test_bootstrap_fallback_when_store_entry_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("splinter.models.pricing_store.STORE_PATH", tmp_path / "missing.json")
    inp, out = _price_for("sonnet")
    assert inp == pytest.approx(3.0)
    assert out == pytest.approx(15.0)


def test_store_price_overrides_bootstrap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("splinter.models.pricing_store.STORE_PATH", tmp_path / "pricing.json")
    save_store({"sonnet": ModelPrice(input=9.0, output=45.0)})
    inp, out = _price_for("sonnet")
    assert inp == pytest.approx(9.0)
    assert out == pytest.approx(45.0)


def test_merge_is_immutable() -> None:
    existing = {"a": ModelPrice(input=1.0, output=2.0)}
    fetched = {"b": ModelPrice(input=3.0, output=4.0)}
    merged = merge(existing, fetched)
    assert "b" in merged
    assert "a" in merged
    assert "b" not in existing


def test_roster_load_attaches_synced_pricing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    store_path = tmp_path / ".splinter/pricing.json"
    monkeypatch.setattr("splinter.models.pricing_store.STORE_PATH", store_path)
    save_store({"sonnet": ModelPrice(input=2.5, output=12.5)})
    ladder = load_ladder()
    assert ladder.model_prices.get("sonnet") == ModelPrice(input=2.5, output=12.5)
    assert (
        ladder.eval_model in ladder.model_prices
        or ladder.planner_model in ladder.model_prices
        or ladder.model_prices
    )


def test_prefix_match_in_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("splinter.models.pricing_store.STORE_PATH", tmp_path / "pricing.json")
    save_store({"codex/gpt-5": ModelPrice(input=8.0, output=32.0)})
    assert price_for("codex/gpt-5-codex") == ModelPrice(input=8.0, output=32.0)
