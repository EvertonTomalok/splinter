"""Tests for Claude provider pricing — sourced from public endpoints, no API key."""

from __future__ import annotations

import pytest

from splinter.providers.base import ModelPrice
from splinter.providers.claude_cli import ClaudeProvider, _calc_cost, fetch_pricing


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
    assert prices["haiku"].cache_read == 0.0
    assert prices["haiku"].cache_write == 0.0


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


# ── _calc_cost cache-exclusion ───────────────────────────────────────────


@pytest.mark.parametrize(
    "usage,expected_cost",
    [
        (
            {
                "input_tokens": 1_000_000,
                "output_tokens": 0,
                "cache_read_input_tokens": 5_000_000,
                "cache_creation_input_tokens": 2_000_000,
            },
            3.0,  # 1M * $3 input only; cache tokens not billed
        ),
        (
            {
                "input_tokens": 0,
                "output_tokens": 1_000_000,
                "cache_read_input_tokens": 9_000_000,
                "cache_creation_input_tokens": 9_000_000,
            },
            15.0,  # 1M * $15 output only
        ),
        (
            {
                "input_tokens": 500_000,
                "output_tokens": 500_000,
                "cache_read_input_tokens": 999_000_000,
                "cache_creation_input_tokens": 999_000_000,
            },
            pytest.approx(9.0),  # (0.5*3 + 0.5*15) = 1.5 + 7.5
        ),
        (
            {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 1_000_000},
            0.0,
        ),
    ],
)
def test_calc_cost_ignores_cache_tokens(
    tmp_path: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
    usage: dict,
    expected_cost: float,
) -> None:
    monkeypatch.chdir(tmp_path)
    cost, indeterminate = _calc_cost("sonnet", usage)
    assert indeterminate is False
    assert cost == expected_cost


def test_calc_cost_unknown_model_indeterminate(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
    cost, indeterminate = _calc_cost("nonexistent-model-xyz", usage)
    assert cost == 0.0
    assert indeterminate is True
