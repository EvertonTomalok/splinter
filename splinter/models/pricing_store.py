"""Project-level pricing store at ``.splinter/pricing.json``."""

from __future__ import annotations

import json
from pathlib import Path

from splinter.providers.base import ModelPrice

STORE_PATH = Path(".splinter") / "pricing.json"


def load_store() -> dict[str, ModelPrice]:
    if not STORE_PATH.exists():
        return {}
    try:
        raw: object = json.loads(STORE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, ModelPrice] = {}
    for model_id, entry in raw.items():
        if not isinstance(model_id, str) or not isinstance(entry, dict):
            continue
        try:
            out[model_id] = ModelPrice(
                input=float(entry.get("input", 0.0) or 0.0),
                output=float(entry.get("output", 0.0) or 0.0),
                cache_read=float(entry.get("cache_read", 0.0) or 0.0),
                cache_write=float(entry.get("cache_write", 0.0) or 0.0),
            )
        except (TypeError, ValueError):
            continue
    return out


def save_store(prices: dict[str, ModelPrice]) -> Path:
    payload: dict[str, dict[str, float]] = {
        model_id: {
            "input": price.input,
            "output": price.output,
            "cache_read": price.cache_read,
            "cache_write": price.cache_write,
        }
        for model_id, price in prices.items()
    }
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STORE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(STORE_PATH)
    return STORE_PATH


def merge(
    existing: dict[str, ModelPrice],
    fetched: dict[str, ModelPrice],
) -> dict[str, ModelPrice]:
    merged = dict(existing)
    merged.update(fetched)
    return merged


def price_for(model_id: str) -> ModelPrice | None:
    store = load_store()
    if model_id in store:
        return store[model_id]
    for prefix, price in store.items():
        if model_id.startswith(prefix):
            return price
    return None
