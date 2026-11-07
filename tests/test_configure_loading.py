from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from splinter.providers import opencode
from splinter.tui_configure import ConfigureApp

ALL_PROVIDERS = {"claude", "opencode", "codex", "cursor", "mcode", "agy", "kimi"}


def test_models_fetched_in_background(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(opencode, "list_models", lambda timeout=30: ["opencode-go/test-model"])

    async def drive() -> None:
        app = ConfigureApp()
        async with app.run_test() as pilot:
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert set(app._models_by_provider) == ALL_PROVIDERS
            assert "opencode-go/test-model" in app._all_models()

    asyncio.run(drive())


def test_per_provider_failure_isolates(
    tmp_path: Path,
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    from splinter.configure import CODEX_MODELS, available_models_by_provider

    monkeypatch.chdir(tmp_path)

    def _raise(*args: object, **kwargs: object) -> list[str]:
        raise RuntimeError("opencode unavailable")

    monkeypatch.setattr(opencode, "list_models", _raise)
    result = available_models_by_provider()
    assert isinstance(result["opencode"], list)
    assert "sonnet" in result["claude"]
    assert result["codex"] == sorted(CODEX_MODELS)


def test_chains_render_before_models_arrive(
    tmp_path: Path,
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    from splinter.tui_configure_widgets import ChainList

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(opencode, "list_models", lambda timeout=30: ["opencode-go/test-model"])
    barrier = threading.Barrier(2, timeout=5)
    original_fetch = ConfigureApp._fetch_models

    def _slow_fetch(self: ConfigureApp) -> None:
        barrier.wait()
        original_fetch(self)

    monkeypatch.setattr(ConfigureApp, "_fetch_models", _slow_fetch)

    async def drive() -> None:
        app = ConfigureApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app._models_by_provider == {}
            assert app._all_models() == []
            planner = app.query_one("#chain-planner", ChainList)
            assert planner.option_count == len(app._chains["planner"])
            barrier.wait()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app._models_by_provider

    asyncio.run(drive())


def test_configure_sync_button(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(opencode, "list_models", lambda timeout=30: ["opencode-go/test-model"])
    monkeypatch.setattr(
        "splinter.configure.sync_prices",
        lambda: (3, {"cursor": "agent down"}),
    )

    async def drive() -> None:
        app = ConfigureApp()
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.pause()
            await pilot.click("#sync-prices")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert "synced 3 prices" in app._sync_message
            assert "cursor: agent down" in app._sync_message

    asyncio.run(drive())


def test_synced_catalogue_feeds_model_lists(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    import json

    from splinter.configure import available_models_by_provider, synced_models_by_provider

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(opencode, "list_models", lambda timeout=30: [])
    (tmp_path / ".splinter").mkdir()
    store = {
        mid: {"input": 1.0, "output": 2.0}
        for mid in (
            "claude-opus-5-5",
            "claude-mythos-5-1",
            "claude-opus-4-20250514",
            "claude-3-haiku",
            "codex/gpt-6",
        )
    }
    (tmp_path / ".splinter" / "pricing.json").write_text(json.dumps(store))
    synced = synced_models_by_provider()
    assert synced["claude"] == ["claude-mythos-5-1", "claude-opus-5-5"]
    groups = available_models_by_provider()
    assert {"claude-opus-5-5", "claude-mythos-5-1", "claude-sonnet-5-5"} <= set(groups["claude"])
    assert "claude-opus-4-20250514" not in groups["claude"]
    assert "codex/gpt-6" in groups["codex"]
