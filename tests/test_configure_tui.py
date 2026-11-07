"""List + detail configure screen: navigation, chain editing, jev, agents, save."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from textual.pilot import Pilot
from textual.widgets import Checkbox, ContentSwitcher, Input, OptionList, Select

from splinter.configure_chain import ChainEntry, tier_key
from splinter.providers import opencode
from splinter.tui_configure import ConfigureApp
from splinter.tui_configure_widgets import ChainList, ModelPicker, match_models, newest_first

Drive = Callable[[ConfigureApp, Pilot[Any]], Awaitable[None]]


@pytest.fixture
def run_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[[Drive], ConfigureApp]:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(opencode, "list_models", lambda timeout=30: ["opencode-go/m1"])

    def _run(drive: Drive) -> ConfigureApp:
        app = ConfigureApp()

        async def main() -> None:
            async with app.run_test(size=(150, 45)) as pilot:
                await app.workers.wait_for_complete()
                await pilot.pause()
                await drive(app, pilot)

        asyncio.run(main())
        return app

    return _run


def test_nav_lists_steps_tiers_and_run_settings(run_app: Callable[[Drive], ConfigureApp]) -> None:
    async def drive(app: ConfigureApp, pilot: Pilot[Any]) -> None:
        nav = app.query_one("#nav", OptionList)
        ids = {nav.get_option_at_index(i).id for i in range(nav.option_count)}
        expected = {"planner", "eval", "prd", tier_key(0), tier_key(5), "gates", "agents", "jev"}
        assert expected <= ids
        assert not app.query("#engine-rows")
        nav.highlighted = nav.get_option_index("jev")
        await pilot.pause()
        assert app.query_one("#switcher", ContentSwitcher).current == "pane-jev"

    run_app(drive)


def test_chain_keys_edit_effort_order_and_remove(
    run_app: Callable[[Drive], ConfigureApp],
) -> None:
    async def drive(app: ConfigureApp, pilot: Pilot[Any]) -> None:
        app._set_chain("eval", (ChainEntry("opus", "high"), ChainEntry("sonnet", "low")), 0)
        app.query_one("#chain-eval", ChainList).focus()
        await pilot.press("down", "shift+up")
        assert [e.model for e in app._chains["eval"]] == ["sonnet", "opus"]
        assert app.query_one("#chain-eval", ChainList).highlighted == 0
        await pilot.press("right")
        assert app._chains["eval"][0].effort == "medium"
        await pilot.press("x")
        assert app._chains["eval"] == (ChainEntry("opus", "high"),)
        await pilot.press("x")
        assert len(app._chains["eval"]) == 1

    run_app(drive)


def test_add_fallback_through_picker(run_app: Callable[[Drive], ConfigureApp]) -> None:
    async def drive(app: ConfigureApp, pilot: Pilot[Any]) -> None:
        before = len(app._chains["planner"])
        app.query_one("#chain-planner", ChainList).focus()
        await pilot.press("a")
        await pilot.pause()
        assert isinstance(app.screen, ModelPicker)
        picker_provider = app.screen.query_one("#picker-provider", Select)
        assert picker_provider.value == "claude"
        picker_provider.value = "opencode"
        await pilot.pause()
        await pilot.press("m", "1", "enter")
        await pilot.pause()
        chain = app._chains["planner"]
        assert len(chain) == before + 1
        assert chain[-1].model == "opencode-go/m1"
        nav = app.query_one("#nav", OptionList)
        prompt = str(nav.get_option("planner").prompt)
        assert f"+{len(chain) - 1}" in prompt

    run_app(drive)


def test_picker_accepts_custom_model_id(run_app: Callable[[Drive], ConfigureApp]) -> None:
    async def drive(app: ConfigureApp, pilot: Pilot[Any]) -> None:
        app.query_one("#chain-planner", ChainList).focus()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press(*"acme/x", "enter")
        await pilot.pause()
        assert app._chains["planner"][0].model == "acme/x"

    run_app(drive)


def test_save_roundtrips_chains_agents_and_jev(
    run_app: Callable[[Drive], ConfigureApp],
) -> None:
    async def drive(app: ConfigureApp, pilot: Pilot[Any]) -> None:
        app._set_chain(
            tier_key(1),
            (ChainEntry("opencode-go/minimax-m3", "high"), ChainEntry("mcode/minimax/M3", "low")),
            0,
        )
        app.query_one("#timeout-planner", Input).value = "900"
        app.query_one("#agents-acceptance", Select).value = "gate"
        app.query_one("#jev-recall", Checkbox).value = True
        app.query_one("#jev-provider-0", Select).value = "vercel"
        await pilot.pause()
        assert app.query_one("#jev-model-0", Select).value == "typesafe-ai/jev"
        assert "AI_GATEWAY_API_KEY" in app.query_one("#jev-token-0", Input).placeholder
        app.query_one("#jev-token-0", Input).value = "vk-secret"
        app.query_one("#jev-provider-1", Select).value = "openrouter"
        await pilot.pause()
        app.query_one("#jev-token-1", Input).value = "$MY_OR_KEY"
        await pilot.pause()
        nav = app.query_one("#nav", OptionList)
        assert "· jev" in str(nav.get_option("localizer_recall").prompt)
        assert "· jev" not in str(nav.get_option("localizer_precision").prompt)
        await pilot.press("s")

    app = run_app(drive)
    assert app.saved

    from splinter.configure import configured_agents_options, configured_decisions, load_config
    from splinter.configure_chain import chains_from_ladder
    from splinter.models.roster import load_ladder
    from splinter.providers.decisions import DecisionSlot

    ladder = load_ladder()
    assert chains_from_ladder(ladder)[tier_key(1)] == (
        ChainEntry("opencode-go/minimax-m3", "high"),
        ChainEntry("mcode/minimax/M3", "low"),
    )
    assert ladder.planner_timeout == 900
    assert configured_agents_options().acceptance_check == "gate"
    decisions = configured_decisions()
    assert decisions.use_for == ("recall",)
    assert decisions.providers == (
        DecisionSlot(provider="vercel", model="typesafe-ai/jev", api_key="vk-secret"),
        DecisionSlot(provider="openrouter", model="typesafe/jev-1.14", api_key_env="MY_OR_KEY"),
    )
    assert not {"engines", "engine_models", "engine_efforts"} & set(load_config()["agents"])


def test_jev_toggles_live_on_recall_and_filter(run_app: Callable[[Drive], ConfigureApp]) -> None:
    async def drive(app: ConfigureApp, pilot: Pilot[Any]) -> None:
        recall = app.query_one("#jev-recall", Checkbox)
        filt = app.query_one("#jev-filter", Checkbox)
        assert recall in app.query_one("#pane-localizer_recall").query(Checkbox)
        assert filt in app.query_one("#pane-localizer_precision").query(Checkbox)
        large = app.query_one("#jev-recall_large", Checkbox)
        assert large in app.query_one("#pane-localizer_recall_large").query(Checkbox)
        large.value = True
        await pilot.pause()
        nav = app.query_one("#nav", OptionList)
        assert "· jev" in str(nav.get_option("localizer_recall_large").prompt)
        large.value = False
        await pilot.pause()
        assert not app.query_one("#pane-jev").query(Checkbox)
        filt.value = True
        await pilot.pause()
        assert app._jev_targets() == ["filter"]
        app.query_one("#jev-provider-1", Select).value = "none"
        await pilot.pause()
        assert app.query_one("#jev-token-1", Input).disabled is True

    run_app(drive)


def test_token_masked_until_eye_pressed(run_app: Callable[[Drive], ConfigureApp]) -> None:
    from textual.widgets import Button

    async def drive(app: ConfigureApp, pilot: Pilot[Any]) -> None:
        nav = app.query_one("#nav", OptionList)
        nav.highlighted = nav.get_option_index("jev")
        await pilot.pause()
        token = app.query_one("#jev-token-0", Input)
        assert token.password is True
        await pilot.click("#jev-eye-0")
        await pilot.pause()
        assert token.password is False
        assert "hide" in str(app.query_one("#jev-eye-0", Button).label)
        await pilot.pause(0.5)
        await pilot.click("#jev-eye-0")
        await pilot.pause()
        assert token.password is True
        app.query_one("#jev-provider-1", Select).value = "none"
        await pilot.pause()
        assert app.query_one("#jev-eye-1", Button).disabled is True
        assert app.query_one("#jev-token-1", Input).password is True

    run_app(drive)


def test_picker_prefixes_typed_id_with_provider(
    run_app: Callable[[Drive], ConfigureApp],
) -> None:
    async def drive(app: ConfigureApp, pilot: Pilot[Any]) -> None:
        app.query_one("#chain-planner", ChainList).focus()
        await pilot.press("a")
        await pilot.pause()
        app.screen.query_one("#picker-provider", Select).value = "mcode"
        await pilot.pause()
        app.screen.query_one("#picker-filter", Input).value = "minimax/brand-new"
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert app._chains["planner"][-1].model == "mcode/minimax/brand-new"
        assert app._chains["planner"][-1].effort == ""

    run_app(drive)


def test_clicked_row_opens_picker_on_its_own_model(
    run_app: Callable[[Drive], ConfigureApp],
) -> None:
    async def drive(app: ConfigureApp, pilot: Pilot[Any]) -> None:
        nav = app.query_one("#nav", OptionList)
        nav.highlighted = nav.get_option_index("planner")
        await pilot.pause()
        app._set_chain(
            "planner", (ChainEntry("opencode-go/m1", "low"), ChainEntry("fable", "low")), 0
        )
        await pilot.pause()
        await pilot.click("#chain-planner", offset=(10, 1))
        await pilot.pause()
        assert isinstance(app.screen, ModelPicker)
        picker_list = app.screen.query_one("#picker-list", OptionList)
        assert picker_list.highlighted is not None
        assert str(picker_list.get_option_at_index(picker_list.highlighted).prompt) == "fable"
        await pilot.press("enter")
        await pilot.pause()
        assert [e.model for e in app._chains["planner"]] == ["opencode-go/m1", "fable"]

    run_app(drive)


def test_effort_select_follows_row_and_provider(
    run_app: Callable[[Drive], ConfigureApp],
) -> None:
    async def drive(app: ConfigureApp, pilot: Pilot[Any]) -> None:
        app._set_chain(
            "eval",
            (ChainEntry("opus", "high"), ChainEntry("mcode/minimax/M3"), ChainEntry("codex/x")),
            0,
        )
        select = app.query_one("#effort-eval", Select)
        assert select.value == "high"
        select.value = "max"
        await pilot.pause()
        assert app._chains["eval"][0].effort == "max"
        chain_list = app.query_one("#chain-eval", ChainList)
        chain_list.highlighted = 1
        await pilot.pause()
        assert select.disabled is True
        chain_list.highlighted = 2
        await pilot.pause()
        assert select.disabled is False
        values = [str(v) for _, v in select._options]  # type: ignore[attr-defined]
        assert "max" not in values and "xhigh" in values
        assert app._chains["eval"][2].effort == ""

    run_app(drive)


def test_picker_click_picks_the_row_under_the_pointer(
    run_app: Callable[[Drive], ConfigureApp],
) -> None:
    async def drive(app: ConfigureApp, pilot: Pilot[Any]) -> None:
        many = {f"opencode/m{i:02d}" for i in range(30)}
        app._models_by_provider = {**app._models_by_provider, "opencode": sorted(many)}
        nav = app.query_one("#nav", OptionList)
        nav.highlighted = nav.get_option_index("planner")
        await pilot.pause()
        app._set_chain("planner", (ChainEntry("opencode/m20", "low"),), 0)
        app.query_one("#chain-planner", ChainList).focus()
        await pilot.press("enter")
        await pilot.pause()
        picker_list = app.screen.query_one("#picker-list", OptionList)
        row = 3
        target = str(picker_list.get_option_at_index(row + round(picker_list.scroll_y)).prompt)
        await pilot.click("#picker-list", offset=(4, row))
        await pilot.pause()
        assert app._chains["planner"][0].model == target

    run_app(drive)


def test_picker_tab_cycles_provider_and_keeps_typing(
    run_app: Callable[[Drive], ConfigureApp],
) -> None:
    from splinter.models.roster import provider_for

    async def drive(app: ConfigureApp, pilot: Pilot[Any]) -> None:
        app.query_one("#chain-planner", ChainList).focus()
        await pilot.press("a")
        await pilot.pause()
        picker = app.screen
        provider = picker.query_one("#picker-provider", Select)
        start = provider.value
        await pilot.press("tab")
        await pilot.pause()
        assert provider.value != start
        assert picker.focused is picker.query_one("#picker-filter", Input)
        listed = picker.query_one("#picker-list", OptionList)
        prompts = [str(listed.get_option_at_index(i).prompt) for i in range(listed.option_count)]
        assert all(provider_for(m) == provider.value for m in prompts)
        await pilot.press("shift+tab")
        await pilot.pause()
        assert provider.value == start

    run_app(drive)


def test_match_models_needs_every_token() -> None:
    models = ["codex/gpt-5.5", "opencode-go/deepseek-v4-pro", "opencode-go/minimax-m3"]
    assert match_models(models, "") == models
    assert match_models(models, "open MINI") == ["opencode-go/minimax-m3"]
    assert match_models(models, "zzz") == []


def test_newest_first_puts_subscriptions_before_openrouter() -> None:
    models = ["openrouter/a/b", "zai-coding-plan/glm-5.3", "minimax-coding-plan/MiniMax-M3"]
    assert newest_first(models)[-1] == "openrouter/a/b"


def test_newest_first_orders_versions_descending() -> None:
    models = ["claude-opus-4-1", "claude-opus-5-5", "claude-opus-4-8", "opus", "claude-haiku-5-5"]
    assert newest_first(models) == [
        "claude-haiku-5-5",
        "claude-opus-5-5",
        "claude-opus-4-8",
        "claude-opus-4-1",
        "opus",
    ]
