"""Tests for the static engine catalog."""

from __future__ import annotations

from splinter.engines.catalog import (
    CATALOG,
    PROBE_PROMPT,
    PROBE_UP_PATTERN,
    Engine,
    catalog_table,
    engine_by_key,
)

_EXPECTED_ORDER = (
    "mcode",
    "kimi",
    "kimi-flash",
    "mcode-flash",
    "agy",
    "sonnet",
    "claude-plan",
    "codex",
    "glm",
    "opencode-minimax",
    "reviewer-opus",
    "reviewer-fable",
    "reviewer-sonnet",
)


def test_catalog_completeness_and_order() -> None:
    assert tuple(engine.key for engine in CATALOG) == _EXPECTED_ORDER
    assert len(CATALOG) == 13


def test_mcode_fields() -> None:
    assert engine_by_key("mcode") == Engine(
        key="mcode",
        label="MiniMax M3 (mcode)",
        provider="mcode",
        model="minimax/MiniMax-M3",
        role="main workhorse: multi-file wiring and implementation",
        default_effort="low",
        supports_continue=True,
        supports_model_override=True,
        supports_effort=False,
        unset_env=(),
    )


def test_mcode_flash_fields() -> None:
    engine = engine_by_key("mcode-flash")
    assert engine is not None
    assert engine.provider == "mcode"
    assert engine.model == "minimax/MiniMax-M3.1-Flash"
    assert engine.supports_continue is True


def test_kimi_fields() -> None:
    engine = engine_by_key("kimi")
    assert engine is not None
    assert engine.provider == "kimi"
    assert engine.model == "kimi-code/k3.1-pro"
    assert engine.supports_continue is True


def test_kimi_flash_fields() -> None:
    engine = engine_by_key("kimi-flash")
    assert engine is not None
    assert engine.provider == "kimi"
    assert engine.model == "kimi-code/kimi-for-coding-highspeed"
    assert engine.supports_continue is True


def test_agy_fields() -> None:
    engine = engine_by_key("agy")
    assert engine is not None
    assert engine.provider == "agy"
    assert engine.model == "gemini-3.8-flash"
    assert engine.supports_model_override is False
    assert engine.supports_effort is True
    assert engine.supports_continue is False


def test_sonnet_fields() -> None:
    engine = engine_by_key("sonnet")
    assert engine is not None
    assert engine.provider == "claude"
    assert engine.model == "claude-sonnet-5-5"
    assert engine.supports_effort is True


def test_claude_plan_strips_credentials() -> None:
    engine = engine_by_key("claude-plan")
    assert engine is not None
    assert engine.provider == "claude"
    assert engine.model == "claude-sonnet-5-5"
    assert engine.supports_effort is True
    assert engine.unset_env == ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")


def test_codex_uses_own_config() -> None:
    engine = engine_by_key("codex")
    assert engine is not None
    assert engine.provider == "codex"
    assert engine.model == ""
    assert engine.supports_continue is True


def test_opencode_engines_pin_models() -> None:
    glm = engine_by_key("glm")
    assert glm is not None
    assert glm.provider == "opencode"
    assert glm.model == "zai-coding-plan/glm-5.3-flash"
    assert glm.supports_model_override is False
    minimax = engine_by_key("opencode-minimax")
    assert minimax is not None
    assert minimax.provider == "opencode"
    assert minimax.model == "minimax-coding-plan/MiniMax-M3"
    assert minimax.supports_model_override is False


def test_reviewers_default_to_high_effort() -> None:
    models = {
        "reviewer-opus": "claude-opus-5-5",
        "reviewer-fable": "claude-fable-5",
        "reviewer-sonnet": "claude-sonnet-5-5",
    }
    for key, model in models.items():
        engine = engine_by_key(key)
        assert engine is not None
        assert engine.provider == "claude"
        assert engine.model == model
        assert engine.default_effort == "high"
        assert engine.supports_effort is True
        assert engine.role == "final review with fixes"


def test_engine_by_key_unknown_returns_none() -> None:
    assert engine_by_key("gpt-turbo") is None


def test_probe_constants() -> None:
    assert PROBE_PROMPT == "Reply with the word ok only."
    assert PROBE_UP_PATTERN.search("OK") is not None
    assert PROBE_UP_PATTERN.search("everything is Ok here") is not None
    assert PROBE_UP_PATTERN.search("token expired") is None


def test_catalog_table_lists_every_engine() -> None:
    table = catalog_table()
    assert table.startswith("| Engine | Launch | State |")
    assert table.count("| catalog |") == 13
    for key in _EXPECTED_ORDER:
        assert f"launch {key} <wt> <brief> <log>" in table
    for engine in CATALOG:
        assert engine.label in table
        assert engine.role in table
