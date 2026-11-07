"""Pure fallback-chain helpers behind the configure TUI."""

from __future__ import annotations

from pathlib import Path

import pytest

from splinter.configure_chain import (
    EFFORT_CYCLE,
    ChainEntry,
    append_entry,
    chain_summary,
    chain_to_config,
    chains_from_ladder,
    config_payload,
    cycle_effort,
    display_model,
    efforts_for,
    fit_effort,
    move_entry,
    remove_entry,
    replace_model,
    short_model,
    tier_key,
    timeouts_from_ladder,
    with_provider,
)
from splinter.models.roster import load_ladder

A = ChainEntry("opus", "high")
B = ChainEntry("sonnet", "low")
C = ChainEntry("codex/gpt-5.5", "")


def test_append_adds_last_and_ignores_duplicates() -> None:
    assert append_entry((A,), "sonnet", "low") == (A, B)
    assert append_entry((A, B), "opus") == (A, B)
    assert append_entry((A,), "  ") == (A,)


def test_remove_keeps_primary_when_alone() -> None:
    assert remove_entry((A, B, C), 1) == (A, C)
    assert remove_entry((A,), 0) == (A,)
    assert remove_entry((A, B), 5) == (A, B)


def test_move_swaps_within_bounds() -> None:
    assert move_entry((A, B, C), 2, -1) == (A, C, B)
    assert move_entry((A, B), 0, -1) == (A, B)
    assert move_entry((A, B), 1, 1) == (A, B)


def test_replace_model_keeps_effort_and_rejects_duplicates() -> None:
    assert replace_model((A, B), 0, "fable") == (ChainEntry("fable", "high"), B)
    assert replace_model((A, B), 0, "sonnet") == (A, B)


def test_cycle_effort_wraps_both_ways() -> None:
    last = EFFORT_CYCLE[-1]
    chain = (ChainEntry("m", last),)
    assert cycle_effort(chain, 0)[0].effort == EFFORT_CYCLE[0]
    assert cycle_effort((ChainEntry("m", ""),), 0, -1)[0].effort == last
    assert "auto" not in EFFORT_CYCLE


@pytest.mark.parametrize(
    ("model", "short"),
    [("opencode-go/minimax-m3", "minimax-m3"), ("opus", "opus"), ("mcode/minimax/M3", "M3")],
)
def test_short_model(model: str, short: str) -> None:
    assert short_model(model) == short


def test_chain_summary_counts_fallbacks_and_truncates() -> None:
    assert chain_summary((A, B, C)) == "opus +2"
    assert chain_summary((ChainEntry("x" * 40),), width=10) == "xxxxxxxxx…"
    assert chain_summary(()) == "—"


def test_chain_to_config_shapes() -> None:
    assert chain_to_config((A,)) == ("opus", "high", "claude")
    assert chain_to_config((A, B, C)) == (
        ["opus", "sonnet@low", "codex/gpt-5.5"],
        "high",
        "claude",
    )


def test_payload_roundtrips_through_ladder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from splinter.configure import write_model_config

    monkeypatch.chdir(tmp_path)
    ladder = load_ladder()
    chains = chains_from_ladder(ladder)
    chains["planner"] = (A, B, C)
    chains[tier_key(1)] = (ChainEntry("opencode-go/minimax-m3", "high"), B)
    timeouts: dict[str, int | None] = dict(timeouts_from_ladder(ladder))
    timeouts["planner"] = 900
    models, efforts, outs, providers = config_payload(chains, timeouts, len(ladder.tiers))
    write_model_config(models, efforts, timeouts=outs, providers=providers)

    reloaded = chains_from_ladder(load_ladder())
    assert reloaded["planner"] == (A, B, C)
    assert reloaded[tier_key(1)] == (ChainEntry("opencode-go/minimax-m3", "high"), B)
    assert timeouts_from_ladder(load_ladder())["planner"] == 900
    assert providers["tiers"][1] == "opencode"


@pytest.mark.parametrize(
    ("model", "efforts"),
    [
        ("opus", ("", "low", "medium", "high", "xhigh", "max")),
        ("codex/gpt-5.5", ("", "low", "medium", "high", "xhigh")),
        ("opencode-go/minimax-m3", ("", "minimal", "low", "medium", "high", "max")),
        ("agy/gemini-3.8-flash", ("", "low", "medium", "high")),
        ("mcode/minimax/MiniMax-M3", ("",)),
        ("kimi/kimi-code/k3.1-pro", ("",)),
        ("cursor/auto", ("",)),
    ],
)
def test_efforts_follow_provider_names(model: str, efforts: tuple[str, ...]) -> None:
    assert efforts_for(model) == efforts


def test_effort_fits_new_provider_on_swap() -> None:
    assert fit_effort("codex/gpt-5.5", "max") == ""
    assert replace_model((ChainEntry("opus", "max"),), 0, "codex/gpt-5.5")[0].effort == ""
    assert replace_model((ChainEntry("opus", "high"),), 0, "agy/g")[0].effort == "high"
    assert append_entry((A,), "mcode/minimax/M3", "high")[-1].effort == ""


def test_cycle_effort_is_noop_without_effort_knob() -> None:
    chain = (ChainEntry("mcode/minimax/M3"),)
    assert cycle_effort(chain, 0) == chain


@pytest.mark.parametrize(
    ("typed", "provider", "expected"),
    [
        ("minimax/X", "mcode", "mcode/minimax/X"),
        ("gemini-3.8-flash", "agy", "agy/gemini-3.8-flash"),
        ("opencode-go/m1", "opencode", "opencode-go/m1"),
        ("sonnet", "claude", "sonnet"),
        ("acme/x", "*", "acme/x"),
    ],
)
def test_with_provider(typed: str, provider: str, expected: str) -> None:
    assert with_provider(typed, provider) == expected


def test_display_model_drops_redundant_prefix() -> None:
    assert display_model("mcode/minimax/M3") == "minimax/M3"
    assert display_model("opencode-go/minimax-m3") == "opencode-go/minimax-m3"
    assert display_model("opus") == "opus"
