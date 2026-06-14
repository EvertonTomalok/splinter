"""Tests for pricing sync CLI, configure helpers, and run-start warnings."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from splinter.configure import run_configure, sync_prices, unpriced_models
from splinter.models.pricing_store import save_store
from splinter.providers import dispatch
from splinter.providers.base import ModelPrice, ProviderResponse
from splinter.providers.registry import fetch_all_pricing, priceable_providers


def test_priceable_providers_excludes_opencode() -> None:
    names = {p.name for p in priceable_providers()}
    assert "opencode" not in names
    assert {"claude", "codex", "cursor"}.issubset(names)


def test_fetch_all_pricing_collects_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    from splinter.providers.claude_cli import ClaudeProvider
    from splinter.providers.codex import CodexProvider
    from splinter.providers.cursor import CursorProvider

    monkeypatch.setattr(
        ClaudeProvider,
        "fetch_pricing",
        lambda self: {"sonnet": ModelPrice(input=3.0, output=15.0)},
    )

    def _fail(self: CodexProvider) -> dict[str, ModelPrice]:
        raise RuntimeError("codex offline")

    monkeypatch.setattr(CodexProvider, "fetch_pricing", _fail)
    monkeypatch.setattr(
        CursorProvider,
        "fetch_pricing",
        lambda self: {"cursor/auto": ModelPrice(input=1.0, output=2.0)},
    )
    merged, errors = fetch_all_pricing()
    assert merged["sonnet"].input == 3.0
    assert errors["codex"] == "codex offline"


def test_sync_prices_writes_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "splinter.providers.registry.fetch_all_pricing",
        lambda: ({"sonnet": ModelPrice(input=3.5, output=17.5)}, {}),
    )
    count, failures = sync_prices()
    assert count == 1
    assert failures == {}
    store = (tmp_path / ".splinter/pricing.json").read_text()
    assert "sonnet" in store


def test_run_configure_sync_prices_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "splinter.configure.sync_prices",
        lambda: (2, {"cursor": "agent down"}),
    )
    rc = run_configure(sync_prices_flag=True, interactive=False)
    out = capsys.readouterr().out
    assert rc == 0
    assert "synced 2" in out
    assert "cursor: agent down" in out


def test_unpriced_models_new_model_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    missing = unpriced_models(["sonnet", "opencode-go/foo"])
    assert "sonnet" in missing
    assert "opencode-go/foo" not in missing


def test_unpriced_models_all_priced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    save_store({"sonnet": ModelPrice(input=3.0, output=15.0)})
    assert unpriced_models(["sonnet"]) == []


def test_unpriced_models_zero_treated_as_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    save_store({"sonnet": ModelPrice(input=0.0, output=0.0)})
    assert "sonnet" in unpriced_models(["sonnet"])


def test_dispatch_warns_missing_price_and_continues(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _Provider:
        name = "claude"

        def run(self, *args: object, **kwargs: object) -> ProviderResponse:
            return ProviderResponse(text="ok")

    monkeypatch.setattr(dispatch, "get_provider", lambda _name: _Provider())
    monkeypatch.setattr(dispatch, "provider_for", lambda _m: "claude")
    with caplog.at_level(logging.WARNING, logger="splinter.pricing"):
        text = dispatch.run_text("hi", "brand-new-model")
    assert text == "ok"
    assert any("brand-new-model" in r.message for r in caplog.records)


def test_dispatch_skips_opencode_warning(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _Provider:
        name = "opencode"
        supports_pricing = False

        def run(self, *args: object, **kwargs: object) -> ProviderResponse:
            return ProviderResponse(text="ok")

    monkeypatch.setattr(dispatch, "get_provider", lambda _name: _Provider())
    monkeypatch.setattr(dispatch, "provider_for", lambda _m: "opencode")
    with caplog.at_level(logging.WARNING, logger="splinter.pricing"):
        text = dispatch.run_text("hi", "opencode/unknown", agent="build")
    assert text == "ok"
    assert not any("sync" in r.message.lower() for r in caplog.records)


def test_fully_priced_run_is_silent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _Provider:
        name = "claude"

        def run(self, *args: object, **kwargs: object) -> ProviderResponse:
            return ProviderResponse(text="ok")

    monkeypatch.chdir(tmp_path)
    save_store({"priced-model": ModelPrice(input=1.0, output=2.0)})
    monkeypatch.setattr(dispatch, "get_provider", lambda _name: _Provider())
    monkeypatch.setattr(dispatch, "provider_for", lambda _m: "claude")
    with caplog.at_level(logging.WARNING, logger="splinter.pricing"):
        dispatch.run_text("hi", "priced-model")
    assert not caplog.records


def test_trace_uses_synced_provider_cost(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from splinter.providers import claude_cli as claude_module

    monkeypatch.chdir(tmp_path)
    save_store({"sonnet": ModelPrice(input=10.0, output=40.0)})
    cost, indeterminate = claude_module._calc_cost(
        "sonnet",
        {"input_tokens": 1_000_000, "output_tokens": 0},
    )
    assert cost == pytest.approx(10.0)
    assert indeterminate is False
