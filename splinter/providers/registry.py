"""Lookup of :class:`ModelProvider` strategies by name.

The runner resolves a provider name (from the active ladder tier) into a concrete
strategy here, so the call site never branches on the backend.
"""

from __future__ import annotations

from splinter.providers.base import ModelPrice, ModelProvider, PriceableProvider
from splinter.providers.claude_cli import ClaudeProvider
from splinter.providers.codex import CodexProvider
from splinter.providers.cursor import CursorProvider
from splinter.providers.opencode import OpencodeProvider

_PROVIDERS: dict[str, ModelProvider] = {
    p.name: p for p in (ClaudeProvider(), OpencodeProvider(), CodexProvider(), CursorProvider())
}


def get_provider(name: str) -> ModelProvider:
    """Return the provider strategy registered under ``name``."""
    try:
        return _PROVIDERS[name]
    except KeyError:
        raise ValueError(
            f"unknown provider '{name}'. Available: {', '.join(sorted(_PROVIDERS))}"
        ) from None


def available_providers() -> list[str]:
    """Names of all registered providers."""
    return sorted(_PROVIDERS)


def priceable_providers() -> list[ModelProvider]:
    """Providers that expose live :meth:`~PriceableProvider.fetch_pricing`."""
    return [p for p in _PROVIDERS.values() if getattr(p, "supports_pricing", False)]


def fetch_all_pricing() -> tuple[dict[str, ModelPrice], dict[str, str]]:
    """Fetch pricing from every priceable provider.

    Returns a merged model→price map and per-provider error messages. OpenCode is
    skipped (no ``fetch_pricing``); failures for other providers are collected
    without aborting the merge.
    """
    merged: dict[str, ModelPrice] = {}
    errors: dict[str, str] = {}
    for provider in priceable_providers():
        if not isinstance(provider, PriceableProvider):
            continue
        try:
            merged.update(provider.fetch_pricing())
        except Exception as exc:
            errors[provider.name] = str(exc)
    return merged, errors
