"""Fetch model pricing from a public endpoint — no API key required.

Source is the public LiteLLM pricing catalogue: a machine-readable JSON of
per-token input/output/cache rates covering Anthropic, OpenAI, and others. It
is served over plain HTTPS with no authentication, so ``splinter configure ->
Sync prices`` works without any provider API key.

Rates in the catalogue are USD *per token*; we convert to USD/MTok to match
:class:`ModelPrice`.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from splinter.providers.base import ModelPrice

_log = logging.getLogger("splinter.pricing")

PUBLIC_PRICING_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)


def _per_mtok(value: object) -> float:
    try:
        return float(value) * 1_000_000  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def fetch_public_pricing() -> dict[str, ModelPrice]:
    """Return the public catalogue as ``{model_id: ModelPrice}`` (USD/MTok).

    Raises :class:`RuntimeError` on network/parse failure so callers can decide
    whether to fall back to seed prices.
    """
    req = urllib.request.Request(
        PUBLIC_PRICING_URL,
        headers={"User-Agent": "splinter-pricing-sync"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"failed to fetch public pricing: {exc}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("public pricing source returned an unexpected payload")

    out: dict[str, ModelPrice] = {}
    for model_id, entry in payload.items():
        if not isinstance(model_id, str) or not isinstance(entry, dict):
            continue
        inp = _per_mtok(entry.get("input_cost_per_token"))
        out_cost = _per_mtok(entry.get("output_cost_per_token"))
        if inp <= 0 and out_cost <= 0:
            continue
        out[model_id] = ModelPrice(
            input=inp,
            output=out_cost,
            cache_read=_per_mtok(entry.get("cache_read_input_token_cost")),
            cache_write=_per_mtok(entry.get("cache_creation_input_token_cost")),
        )
    return out


def public_price_for(
    model_id: str,
    catalog: dict[str, ModelPrice],
    *,
    aliases: tuple[str, ...] = (),
) -> ModelPrice | None:
    """Look up *model_id* (or one of *aliases*) in the public *catalog*.

    Matching is exact only — we never guess across model versions, so a stale
    catalogue entry for a different release can't silently replace a seed price.
    """
    for candidate in (model_id, *aliases):
        price = catalog.get(candidate)
        if price is not None and (price.input > 0 or price.output > 0):
            return price
    return None
