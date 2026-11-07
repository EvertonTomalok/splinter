"""Live model lists (agy, kimi, decision models) and their cache — all I/O mocked."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from splinter.models import live_models
from splinter.models.live_models import (
    cached_decision_models,
    cached_provider_models,
    parse_agy_models,
    parse_kimi_models,
    parse_openrouter_decisions,
    parse_vercel_decisions,
    sync_model_lists,
)

AGY_OUT = (
    "Fetching available models...\n"
    "gemini-4-argon-medium\tGemini 4 Argon (Medium)\n"
    "gemini-4-argon-high\tGemini 4 Argon (High)\n"
    "claude-opus-5-5-low\tClaude Opus 5.5 (Low)\n"
    "gpt-oss-120b-medium\tGPT-OSS 120B (Medium)\n"
)
KIMI_OUT = json.dumps(
    {
        "providers": {"managed:kimi-code": {"apiKey": "secret"}},
        "models": {"kimi-code/k3": {"model": "k3"}, "kimi-code/kimi-for-coding": {}},
    }
)
OPENROUTER = {
    "data": [
        {"id": "typesafe/jev-1.14", "architecture": {"output_modalities": ["decisions"]}},
        {"id": "openai/gpt-6", "architecture": {"output_modalities": ["text"]}},
    ]
}
VERCEL = {
    "data": [
        {"id": "typesafe-ai/jev-1.14.0", "type": "evaluation"},
        {"id": "anthropic/claude-opus-5-5", "type": "language"},
    ]
}


def test_parse_agy_strips_effort_suffix_and_dedupes() -> None:
    assert parse_agy_models(AGY_OUT) == [
        "agy/gemini-4-argon",
        "agy/claude-opus-5-5",
        "agy/gpt-oss-120b",
    ]


def test_parse_kimi_reads_model_aliases_only() -> None:
    assert parse_kimi_models(KIMI_OUT) == ["kimi/kimi-code/k3", "kimi/kimi-code/kimi-for-coding"]


def test_parse_decision_catalogues() -> None:
    assert parse_openrouter_decisions(OPENROUTER) == ["typesafe/jev-1.14"]
    assert parse_vercel_decisions(VERCEL) == ["typesafe-ai/jev-1.14.0"]


def test_sync_writes_cache_and_keeps_old_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    def fake_run(argv: list[str], timeout: int = 30) -> str:
        if argv[0] == "agy":
            return AGY_OUT
        raise RuntimeError("kimi exited 1")

    def fake_get(url: str, timeout: int = 30) -> Any:
        return OPENROUTER if "openrouter" in url else VERCEL

    monkeypatch.setattr(live_models, "_run", fake_run)
    monkeypatch.setattr(live_models, "_get_json", fake_get)
    (tmp_path / ".splinter").mkdir()
    (tmp_path / ".splinter" / "models.json").write_text(
        json.dumps({"providers": {"kimi": ["kimi/old"]}})
    )
    _, failures = sync_model_lists()
    assert set(failures) == {"kimi"}
    assert cached_provider_models("agy")[0] == "agy/gemini-4-argon"
    assert cached_provider_models("kimi") == ["kimi/old"]
    assert cached_decision_models("vercel") == ["typesafe-ai/jev-1.14.0"]


def test_cache_feeds_picker_and_jev(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from splinter.configure import available_models_by_provider
    from splinter.tui_configure_jev import model_options

    monkeypatch.chdir(tmp_path)
    (tmp_path / ".splinter").mkdir()
    (tmp_path / ".splinter" / "models.json").write_text(
        json.dumps(
            {
                "providers": {"agy": ["agy/gemini-4-argon"], "kimi": ["kimi/kimi-code/k3"]},
                "decisions": {"vercel": ["typesafe-ai/jev", "liquid/d1-ultra"]},
            }
        )
    )
    groups = available_models_by_provider()
    assert "agy/gemini-4-argon" in groups["agy"]
    assert "kimi/kimi-code/k3" in groups["kimi"]
    values = [v for _, v in model_options("vercel")]
    assert values[0] == "typesafe-ai/jev"
    assert "liquid/d1-ultra" in values
