from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest


def _patch_list_models(monkeypatch: "pytest.MonkeyPatch") -> None:
    from splinter.providers import opencode

    monkeypatch.setattr(
        opencode, "list_models", lambda timeout=30: [
            "opencode-go/flash",
            "opencode-go/deepseek-v4-pro",
            "opencode-go/qwen-coder",
        ]
    )


async def _wait_worker(app: Any, pilot: Any) -> None:
    for _ in range(20):
        if app._models:
            break
        await pilot.pause()
    await pilot.pause()


def _apply_filter(app: Any, sid: str, text: str) -> None:
    from textual.widgets import Input

    inp = app.query_one(f"#{sid}__filter", Input)
    inp.value = text
    app.on_input_changed(Input.Changed(inp, text, inp))


def _set_provider_state(app: Any, sid: str, provider: str) -> None:
    full = [opt[0] for opt in app._model_opts_for(provider, "")]
    app._row_models[sid] = full
    app._repopulate_model(sid)


class TestFilterSubstringMatch:
    def test_substring_narrows_to_codex(
        self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
    ) -> None:
        from splinter.tui import ConfigureApp, _filter_models

        _patch_list_models(monkeypatch)
        monkeypatch.chdir(tmp_path)

        async def drive() -> None:
            app = ConfigureApp()
            async with app.run_test() as pilot:
                await _wait_worker(app, pilot)
                _set_provider_state(app, "planner", "(default)")
                _apply_filter(app, "planner", "codex")
                assert app._row_filters.get("planner") == "codex"
                full = app._row_models.get("planner", [])
                result = _filter_models(full, "codex")
                assert result
                assert "codex/gpt-5-codex" in result
                assert all("codex" in model for model in result)

        asyncio.run(drive())


class TestFilterClearRestore:
    def test_clear_restores_full_list(
        self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
    ) -> None:
        from splinter.tui import ConfigureApp, _filter_models

        _patch_list_models(monkeypatch)
        monkeypatch.chdir(tmp_path)

        async def drive() -> None:
            app = ConfigureApp()
            async with app.run_test() as pilot:
                await _wait_worker(app, pilot)
                _set_provider_state(app, "planner", "(default)")
                _apply_filter(app, "planner", "codex")
                _apply_filter(app, "planner", "")
                assert app._row_filters.get("planner") == ""
                full = app._row_models.get("planner", [])
                result = _filter_models(full, "")
                assert len(result) > 2
                assert "opus" in result
                assert "codex/gpt-5-codex" in result

        asyncio.run(drive())


class TestFilterSelectionPreserved:
    def test_selection_preserved_when_still_matches(
        self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
    ) -> None:
        from splinter.tui import ConfigureApp, _filter_models

        _patch_list_models(monkeypatch)
        monkeypatch.chdir(tmp_path)

        async def drive() -> None:
            app = ConfigureApp()
            async with app.run_test() as pilot:
                await _wait_worker(app, pilot)
                _set_provider_state(app, "planner", "(default)")
                _apply_filter(app, "planner", "op")
                assert app._row_filters.get("planner") == "op"
                full = app._row_models.get("planner", [])
                result = _filter_models(full, "op")
                assert "opus" in result
                assert all("op" in m.lower() for m in result)

        asyncio.run(drive())

    def test_selection_falls_back_when_no_longer_matches(
        self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
    ) -> None:
        from splinter.tui import ConfigureApp, _filter_models

        _patch_list_models(monkeypatch)
        monkeypatch.chdir(tmp_path)

        async def drive() -> None:
            app = ConfigureApp()
            async with app.run_test() as pilot:
                await _wait_worker(app, pilot)
                _set_provider_state(app, "planner", "(default)")
                _apply_filter(app, "planner", "zzz_nonexistent")
                assert app._row_filters.get("planner") == "zzz_nonexistent"
                full = app._row_models.get("planner", [])
                result = _filter_models(full, "zzz_nonexistent")
                assert result == []

        asyncio.run(drive())


class TestFilterPerRowIsolation:
    def test_per_row_isolation(
        self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
    ) -> None:
        from splinter.tui import ConfigureApp

        _patch_list_models(monkeypatch)
        monkeypatch.chdir(tmp_path)

        async def drive() -> None:
            app = ConfigureApp()
            async with app.run_test() as pilot:
                await _wait_worker(app, pilot)
                _set_provider_state(app, "planner", "(default)")
                _set_provider_state(app, "localizer_recall", "(default)")
                _apply_filter(app, "planner", "codex")
                assert app._row_filters.get("planner") == "codex"
                assert app._row_filters.get("localizer_recall", "") == ""
                planner_models = app._row_models.get("planner", [])
                localizer_models = app._row_models.get("localizer_recall", [])
                assert len(planner_models) > 0
                assert len(localizer_models) > 0

        asyncio.run(drive())


class TestFilterSurvivesProviderSwap:
    def test_filter_survives_provider_swap(
        self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
    ) -> None:
        from splinter.tui import ConfigureApp, _filter_models

        _patch_list_models(monkeypatch)
        monkeypatch.chdir(tmp_path)

        async def drive() -> None:
            app = ConfigureApp()
            async with app.run_test() as pilot:
                await _wait_worker(app, pilot)
                _set_provider_state(app, "planner", "(default)")
                _apply_filter(app, "planner", "deepseek")
                _set_provider_state(app, "planner", "claude")
                assert app._row_filters.get("planner") == "deepseek"
                full = app._row_models.get("planner", [])
                result = _filter_models(full, "deepseek")
                assert result == []

        asyncio.run(drive())


class TestFilterStateSurvivesRowRebuild:
    def test_state_survives_row_rebuild(
        self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
    ) -> None:
        from splinter.tui import ConfigureApp

        _patch_list_models(monkeypatch)
        monkeypatch.chdir(tmp_path)

        async def drive() -> None:
            app = ConfigureApp()
            async with app.run_test() as pilot:
                await _wait_worker(app, pilot)
                _set_provider_state(app, "planner", "(default)")
                _apply_filter(app, "planner", "sonnet")
                app._rebuild_rows()
                await pilot.pause()
                assert app._row_filters.get("planner") == "sonnet"
                restored = app.query_one("#planner__filter")
                assert restored.value == "sonnet"

        asyncio.run(drive())
