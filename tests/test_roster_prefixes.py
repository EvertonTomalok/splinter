"""Unit tests for provider prefix routing (mcode/agy additions)."""

from __future__ import annotations

from splinter.models.roster import provider_for


class TestProviderPrefixes:
    def test_mcode_prefix(self) -> None:
        assert provider_for("mcode/minimax/MiniMax-M3") == "mcode"

    def test_kimi_prefix(self) -> None:
        assert provider_for("kimi/kimi-code/k3.1-pro") == "kimi"

    def test_agy_prefix(self) -> None:
        assert provider_for("agy/gemini-3.8-flash") == "agy"

    def test_existing_prefixes_unchanged(self) -> None:
        assert provider_for("codex/gpt-5.5") == "codex"
        assert provider_for("opencode/deepseek-v4-flash-free") == "opencode"
        assert provider_for("cursor/claude-4") == "cursor"

    def test_bare_ids_still_claude(self) -> None:
        assert provider_for("sonnet") == "claude"
        assert provider_for("claude/sonnet") == "claude"

    def test_opencode_subscription_ids_route_to_opencode(self) -> None:
        for model in (
            "zai-coding-plan/glm-5.3-flash",
            "minimax-coding-plan/MiniMax-M3",
            "minimax/MiniMax-M3",
            "openrouter/qwen/qwen3.7-plus",
        ):
            assert provider_for(model) == "opencode", model
