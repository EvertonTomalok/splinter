from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from splinter.providers import opencode
from splinter.tui import ConfigureApp


def test_loading_transition(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(opencode, "list_models", lambda timeout=30: ["opencode-go/test-model"])

    async def drive() -> None:
        app = ConfigureApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            rows = app.query_one("#rows")
            assert rows.loading is False
            assert app._models_by_provider
            assert set(app._models_by_provider.keys()) == {"claude", "opencode", "codex", "cursor"}

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


def test_configure_saves_after_loading(
    tmp_path: Path,
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    from textual.widgets import Select

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(opencode, "list_models", lambda timeout=30: ["opencode-go/test-model"])

    async def drive() -> None:
        app = ConfigureApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            app.query_one("#planner__eff", Select).value = "max"
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()
        assert app.saved
        assert app.saved_path
        from splinter.models.roster import load_ladder

        ladder = load_ladder()
        assert ladder.planner_effort == "max"

    asyncio.run(drive())


def test_loading_state_before_fetch(
    tmp_path: Path,
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    from textual.containers import VerticalScroll

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
            rows = app.query_one("#rows", VerticalScroll)
            assert rows.loading is True
            barrier.wait()
            await pilot.pause()
            await pilot.pause()
            rows = app.query_one("#rows", VerticalScroll)
            assert rows.loading is False
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
            await pilot.pause()
            await pilot.click("#sync-prices")
            await pilot.pause()
            assert "synced 3" in app._sync_message
            assert "cursor: agent down" in app._sync_message

    asyncio.run(drive())
