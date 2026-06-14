"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clear_config_cache() -> None:
    """Reset the configure module's config cache before each test."""
    from splinter.configure import invalidate_config_cache

    invalidate_config_cache()
    yield
    invalidate_config_cache()


@pytest.fixture(autouse=True)
def _mock_cursor_list_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent tests from spawning ``agent --list-models`` subprocess."""
    from splinter.providers import cursor as _cursor

    monkeypatch.setattr(_cursor, "list_models", lambda: ["cursor/test-model"])


@pytest.fixture(autouse=True)
def _mock_opencode_list_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent tests from shelling out to ``opencode models``."""
    from splinter.providers import opencode as _opencode

    monkeypatch.setattr(
        _opencode,
        "list_models",
        lambda timeout=30: [
            "opencode/test-model",
            "opencode-go/test-model",
            "openrouter/test/model",
        ],
    )
