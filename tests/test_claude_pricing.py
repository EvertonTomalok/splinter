"""Tests for Claude provider pricing — sourced from public endpoints, no API key."""

from __future__ import annotations

import pytest

from splinter.providers.base import ModelPrice
from splinter.providers.claude_cli import ClaudeProvider, fetch_pricing


def test_fetch_pricing_enumerates_live_anthropic_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "splinter.models.public_pricing.fetch_public_catalog",
        lambda: {
            "claude-sonnet-4-6": {
                "litellm_provider": "anthropic",
                "input_cost_per_token": 3e-06,
                "output_cost_per_token": 15e-06,
                "cache_read_input_token_cost": 3e-07,
                "cache_creation_input_token_cost": 3.75e-06,
            },
            "claude-future-9": {
                "litellm_provider": "anthropic",
                "input_cost_per_token": 9e-06,
                "output_cost_per_token": 9e-05,
            },
            "gpt-5": {  # different provider — must be excluded
                "litellm_provider": "openai",
                "input_cost_per_token": 1e-06,
                "output_cost_per_token": 2e-06,
            },
        },
    )
    prices = fetch_pricing()
    # A brand-new Anthropic id appears with no code change.
    assert prices["claude-future-9"].input == 9.0
    # Seed alias resolves to the live rate.
    assert prices["sonnet"] == ModelPrice(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75)
    # Foreign-provider ids are not pulled in.
    assert "gpt-5" not in prices


def test_fetch_pricing_falls_back_to_seed_when_public_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom() -> dict[str, dict]:
        raise RuntimeError("offline")

    monkeypatch.setattr("splinter.models.public_pricing.fetch_public_catalog", _boom)
    prices = fetch_pricing()
    # Haiku 4.5 is $1/$5 — the corrected seed, not the stale $0.25/$1.25.
    assert prices["haiku"].input == 1.0
    assert prices["haiku"].output == 5.0
    # Cache tiers are derived from the base rates when the source omits them.
    assert prices["haiku"].cache_read == pytest.approx(0.1)
    assert prices["haiku"].cache_write == pytest.approx(1.25)


def test_fetch_pricing_needs_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("splinter.models.public_pricing.fetch_public_catalog", lambda: {})
    prices = fetch_pricing()
    assert "sonnet" in prices and "opus" in prices


def test_claude_provider_exposes_fetch_pricing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("splinter.models.public_pricing.fetch_public_catalog", lambda: {})
    provider = ClaudeProvider()
    assert provider.supports_pricing is True
    assert provider.fetch_pricing()["sonnet"].input == 3.0
