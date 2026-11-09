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
def _isolate_ui_prefs(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Keep ``.splinter/ui.yaml`` writes out of the real working tree."""
    from splinter import ui_prefs

    root = tmp_path_factory.mktemp("ui-prefs")
    monkeypatch.setattr(ui_prefs, "_default_root", lambda: root)


@pytest.fixture(autouse=True)
def _block_orchestrator_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """A note sent from the composer must never reach a real model."""
    monkeypatch.setattr(
        "splinter.orchestrator_chat.run_text",
        lambda *args, **kwargs: '{"reply": "ok", "actions": []}',
    )


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


@pytest.fixture(autouse=True)
def _block_live_model_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    """Live model lists shell out (agy, kimi) and hit the network — never in tests."""
    from splinter.models import live_models

    def _blocked(*args: object, **kwargs: object) -> object:
        raise RuntimeError("live model lists are blocked in tests")

    monkeypatch.setattr(live_models, "_run", _blocked)
    monkeypatch.setattr(live_models, "_get_json", _blocked)


@pytest.fixture(autouse=True)
def _block_live_decisions(monkeypatch: pytest.MonkeyPatch) -> None:
    """jev/decision calls go over urllib — a repo config with jev enabled must not
    make the suite call OpenRouter/Vercel. Blocked as URLError so callers fail open."""
    import urllib.error
    import urllib.request

    def _blocked(*args: object, **kwargs: object) -> object:
        raise urllib.error.URLError("network calls are blocked in tests")

    monkeypatch.setattr(urllib.request, "urlopen", _blocked)
