"""Lookup of :class:`ModelProvider` strategies by name.

The runner resolves a provider name (from the active ladder tier) into a concrete
strategy here, so the call site never branches on the backend.
"""

from __future__ import annotations

from splinter.providers.base import ModelProvider
from splinter.providers.claude_cli import ClaudeProvider
from splinter.providers.codex import CodexProvider
from splinter.providers.opencode import OpencodeProvider

_PROVIDERS: dict[str, ModelProvider] = {
    p.name: p for p in (ClaudeProvider(), OpencodeProvider(), CodexProvider())
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
