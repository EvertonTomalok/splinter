"""Tests for Claude provider pricing — sourced from public endpoints, no API key."""

from __future__ import annotations

import pytest

from splinter.providers.base import ModelPrice
from splinter.providers.claude_cli import ClaudeProvider, fetch_pricing


def test_fetch_pricing_uses_public_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "splinter.models.public_pricing.fetch_public_pricing",
        lambda: {
            "claude-sonnet-4-6": ModelPrice(
                input=3.0, output=15.0, cache_read=0.3, cache_write=3.75
            )
        },
    )
    prices = fetch_pricing()
    assert prices["sonnet"] == ModelPrice(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75)
    assert prices["claude-sonnet-4-6"].input == 3.0


def test_fetch_pricing_falls_back_to_seed_when_public_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom() -> dict[str, ModelPrice]:
        raise RuntimeError("offline")

    monkeypatch.setattr("splinter.models.public_pricing.fetch_public_pricing", _boom)
    prices = fetch_pricing()
    # Haiku 4.5 is $1/$5 — the corrected seed, not the stale $0.25/$1.25.
    assert prices["haiku"].input == 1.0
    assert prices["haiku"].output == 5.0
    # Cache tiers are derived from the base rates when the source omits them.
    assert prices["haiku"].cache_read == pytest.approx(0.1)
    assert prices["haiku"].cache_write == pytest.approx(1.25)


def test_fetch_pricing_needs_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        "splinter.models.public_pricing.fetch_public_pricing",
        lambda: {},
    )
    prices = fetch_pricing()
    assert "sonnet" in prices and "opus" in prices


def test_claude_provider_exposes_fetch_pricing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "splinter.models.public_pricing.fetch_public_pricing",
        lambda: {},
    )
    provider = ClaudeProvider()
    assert provider.supports_pricing is True
    assert provider.fetch_pricing()["sonnet"].input == 3.0
