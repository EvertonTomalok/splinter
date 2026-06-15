"""Fetch model pricing from a public endpoint — no API key required.

Source is the public LiteLLM pricing catalogue: a machine-readable JSON of
per-token input/output/cache rates covering Anthropic, OpenAI, and others. It
is served over plain HTTPS with no authentication, so ``splinter configure ->
Sync prices`` works without any provider API key.

Each entry carries a ``litellm_provider`` tag, so providers can enumerate their
own model ids live (new releases appear automatically) instead of relying on a
fixed seed list. Rates are USD *per token*; we convert to USD/MTok to match
:class:`ModelPrice`.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from collections.abc import Callable

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


def _entry_to_price(entry: dict) -> ModelPrice | None:
    inp = _per_mtok(entry.get("input_cost_per_token"))
    out = _per_mtok(entry.get("output_cost_per_token"))
    if inp <= 0 and out <= 0:
        return None
    return ModelPrice(
        input=inp,
        output=out,
        cache_read=_per_mtok(entry.get("cache_read_input_token_cost")),
        cache_write=_per_mtok(entry.get("cache_creation_input_token_cost")),
    )


def fetch_public_catalog() -> dict[str, dict]:
    """Return the raw public catalogue as ``{model_id: entry_dict}``.

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
    return {k: v for k, v in payload.items() if isinstance(k, str) and isinstance(v, dict)}


def fetch_public_pricing() -> dict[str, ModelPrice]:
    """Return the whole catalogue as ``{model_id: ModelPrice}`` (USD/MTok)."""
    out: dict[str, ModelPrice] = {}
    for model_id, entry in fetch_public_catalog().items():
        price = _entry_to_price(entry)
        if price is not None:
            out[model_id] = price
    return out


def provider_models(
    catalog: dict[str, dict],
    provider: str,
    *,
    predicate: Callable[[str], bool] | None = None,
) -> dict[str, ModelPrice]:
    """Return ``{model_id: ModelPrice}`` for every *provider* model in *catalog*.

    Filters by the entry's ``litellm_provider`` tag, optionally narrowed by a
    *predicate* on the model id. Lets a provider enumerate its live id set so
    newly published models are priced without a code change.
    """
    out: dict[str, ModelPrice] = {}
    for model_id, entry in catalog.items():
        if entry.get("litellm_provider") != provider:
            continue
        if predicate is not None and not predicate(model_id):
            continue
        price = _entry_to_price(entry)
        if price is not None:
            out[model_id] = price
    return out


def public_price_for(
    model_id: str,
    catalog: dict[str, ModelPrice],
    *,
    aliases: tuple[str, ...] = (),
) -> ModelPrice | None:
    """Look up *model_id* (or one of *aliases*) in a ``{id: ModelPrice}`` map.

    Matching is exact only — we never guess across model versions, so a stale
    catalogue entry for a different release can't silently replace a seed price.
    """
    for candidate in (model_id, *aliases):
        price = catalog.get(candidate)
        if price is not None and (price.input > 0 or price.output > 0):
            return price
    return None
