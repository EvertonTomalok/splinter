from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# configure.py helpers
# ---------------------------------------------------------------------------


class TestTemplateOverrideRoundTrip:
    def test_write_then_read(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        from splinter.configure import (
            template_current_text,
            template_is_overridden,
            write_template_override,
        )

        assert not template_is_overridden("plan")
        write_template_override("plan", "custom planner text")
        assert template_is_overridden("plan")
        assert template_current_text("plan") == "custom planner text"
        assert (tmp_path / ".splinter" / "prompts" / "plan.md").exists()

    def test_reset_removes_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        from splinter.configure import (
            reset_template_override,
            template_current_text,
            template_is_overridden,
            write_template_override,
        )

        write_template_override("eval", "custom eval")
        assert template_is_overridden("eval")
        result = reset_template_override("eval")
        assert result is True
        assert not template_is_overridden("eval")
        # falls back to packaged default
        default = template_current_text("eval")
        assert "VERDICT" in default  # packaged eval.md has VERDICT token

    def test_reset_nonexistent_returns_false(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        from splinter.configure import reset_template_override

        assert reset_template_override("plan") is False


class TestAgentsTemplate:
    def test_agents_path_is_splinter_agents_md(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        from splinter.configure import template_override_path

        p = template_override_path("agents")
        assert str(p) == ".splinter/AGENTS.md"

    def test_agents_write_and_read(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        from splinter.configure import (
            template_current_text,
            template_is_overridden,
            write_template_override,
        )

        write_template_override("agents", "# My project guide")
        assert template_is_overridden("agents")
        assert template_current_text("agents") == "# My project guide"
        assert (tmp_path / ".splinter" / "AGENTS.md").exists()


# ---------------------------------------------------------------------------
# prd.py — PRD override
# ---------------------------------------------------------------------------


class TestPrdOverride:
    def test_override_wins_over_skill_md(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        # Create packaged fallback so we can verify it's NOT returned
        skill_dir = tmp_path / "skills" / "prd"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("packaged skill text")

        # Create override
        override_dir = tmp_path / ".splinter" / "prompts"
        override_dir.mkdir(parents=True)
        (override_dir / "prd.md").write_text("custom prd override")

        from splinter.prd import _load_prd_skill

        assert _load_prd_skill() == "custom prd override"

    def test_fallback_to_skill_md_when_no_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        skill_dir = tmp_path / "skills" / "prd"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("packaged skill text")

        from splinter.prd import _load_prd_skill

        assert _load_prd_skill() == "packaged skill text"

    def test_empty_string_when_no_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        from splinter.prd import _load_prd_skill

        assert _load_prd_skill() == ""


# ---------------------------------------------------------------------------
# TemplatesScreen TUI
# ---------------------------------------------------------------------------


def _patch_list_models(monkeypatch: pytest.MonkeyPatch) -> None:
    from splinter.providers import opencode

    monkeypatch.setattr(opencode, "list_models", lambda timeout=30: [])


class TestTemplatesScreenNav:
    def test_templates_screen_loads_and_dismisses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_list_models(monkeypatch)
        monkeypatch.chdir(tmp_path)

        async def drive() -> None:
            from splinter.tui import ConfigureApp

            app = ConfigureApp()
            async with app.run_test(size=(160, 40)) as pilot:
                await pilot.pause()
                await pilot.click("#goto-templates")
                await pilot.pause()
                from splinter.tui import TemplatesScreen

                assert isinstance(app.screen, TemplatesScreen)
                await pilot.press("escape")
                await pilot.pause()
                from splinter.tui import TemplatesScreen as _TS

                assert not isinstance(app.screen, _TS)

        asyncio.run(drive())


class TestTemplatesScreenSaveDefault:
    def test_save_helper_writes_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """write_template_override + template_current_text round-trips (no TUI)."""
        monkeypatch.chdir(tmp_path)
        from splinter.configure import template_current_text, write_template_override

        write_template_override("plan", "my custom planner")
        assert template_current_text("plan") == "my custom planner"
        assert (tmp_path / ".splinter" / "prompts" / "plan.md").read_text() == "my custom planner"

    def test_default_helper_removes_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """reset_template_override removes the file; current_text falls back to default."""
        monkeypatch.chdir(tmp_path)
        from splinter.configure import (
            reset_template_override,
            template_current_text,
            write_template_override,
        )

        write_template_override("eval", "custom eval text")
        reset_template_override("eval")
        default = template_current_text("eval")
        assert "VERDICT" in default  # packaged eval.md has VERDICT token
        assert not (tmp_path / ".splinter" / "prompts" / "eval.md").exists()

    def test_templates_screen_visible_after_nav(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """TemplatesScreen is the active screen after clicking Templates ▸."""
        _patch_list_models(monkeypatch)
        monkeypatch.chdir(tmp_path)

        async def drive() -> None:
            from splinter.tui import ConfigureApp, TemplatesScreen

            app = ConfigureApp()
            async with app.run_test(size=(160, 40)) as pilot:
                await pilot.pause()
                await pilot.click("#goto-templates")
                await pilot.pause()
                assert isinstance(app.screen, TemplatesScreen)
                assert app.screen.query_one("#tmpl-list") is not None
                assert app.screen.query_one("#tmpl-editor") is not None

        asyncio.run(drive())
