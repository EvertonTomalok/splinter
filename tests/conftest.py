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
