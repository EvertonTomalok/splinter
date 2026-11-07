"""Effort → tier → model chain → engine routing for the agents strategy."""

from __future__ import annotations

import pytest

from splinter.engines.routing import chain_for_effort, engine_for_model, start_tier_for
from splinter.models.roster import EffortMapping, Ladder, Tier


def _ladder() -> Ladder:
    return Ladder(
        tiers=[
            Tier(name="easy", level=0, models=["opencode-go/deepseek-v4-pro"]),
            Tier(
                name="moderate",
                level=1,
                models=["mcode/minimax/MiniMax-M3", "kimi/kimi-code/k3.1-pro", "cursor/gpt"],
                model_variants={"kimi/kimi-code/k3.1-pro": "low"},
            ),
            Tier(name="hard", level=3, models=["sonnet", "sonnet"], provider="claude"),
        ],
        effort_map={
            "trivial": EffortMapping(start_tier=0, variant="low"),
            "normal": EffortMapping(start_tier=1, variant="high"),
            "hard": EffortMapping(start_tier=3, variant="max"),
        },
        eval_model="opus",
        eval_effort="high",
        planner_model="opus",
        planner_effort="high",
        localizer_recall_model="haiku",
        localizer_recall_large_model="haiku",
        localizer_precision_model="haiku",
    )


@pytest.mark.parametrize(
    ("model_id", "provider", "bare", "resumes"),
    [
        ("mcode/minimax/MiniMax-M3", "mcode", "minimax/MiniMax-M3", True),
        ("kimi/kimi-code/k3.1-pro", "kimi", "kimi-code/k3.1-pro", True),
        ("agy/gemini-3.8-flash", "agy", "gemini-3.8-flash", False),
        ("codex/gpt-5.5", "codex", "codex/gpt-5.5", True),
        ("opencode-go/minimax-m3", "opencode", "opencode-go/minimax-m3", False),
        ("sonnet", "claude", "sonnet", False),
    ],
)
def test_engine_for_model_derives_provider(
    model_id: str, provider: str, bare: str, resumes: bool
) -> None:
    engine = engine_for_model(model_id, "high")
    assert engine is not None
    assert (engine.key, engine.provider, engine.model) == (model_id, provider, bare)
    assert engine.supports_continue is resumes
    assert engine.default_effort == "high"


@pytest.mark.parametrize("model_id", ["", "  ", "cursor/gpt-5"])
def test_engine_for_model_rejects_unlaunchable(model_id: str) -> None:
    assert engine_for_model(model_id) is None


def test_start_tier_unknown_effort_uses_normal() -> None:
    ladder = _ladder()
    assert start_tier_for(ladder, "hard") == 3
    assert start_tier_for(ladder, "bogus") == 1


def test_chain_for_effort_orders_and_carries_variants() -> None:
    chain = chain_for_effort(_ladder(), "normal")
    assert [e.key for e in chain] == ["mcode/minimax/MiniMax-M3", "kimi/kimi-code/k3.1-pro"]
    assert [e.default_effort for e in chain] == ["high", "low"]


def test_chain_for_effort_prefers_tier_variant_and_dedupes() -> None:
    ladder = _ladder()
    ladder.tier_variants[3] = "xhigh"
    chain = chain_for_effort(ladder, "hard")
    assert [(e.key, e.default_effort) for e in chain] == [("sonnet", "xhigh")]


def test_chain_for_effort_missing_tier_is_empty() -> None:
    ladder = _ladder()
    ladder.effort_map["critical"] = EffortMapping(start_tier=9, variant="high")
    assert chain_for_effort(ladder, "critical") == ()


@pytest.mark.parametrize("model", ["codex/gpt-5.5", "opencode-go/model"])
def test_launch_preserves_effort(model: str) -> None:
    from unittest.mock import MagicMock, patch

    from splinter.engines.launch import run_engine

    engine = engine_for_model(model, "xhigh")
    assert engine is not None
    result = MagicMock(text="ok", tokens={}, cost=0, raw={}, session_id=None)
    with patch(f"splinter.providers.{engine.provider}.run", return_value=result) as run:
        run_engine(engine, "prompt")
    key = "effort" if engine.provider == "codex" else "variant"
    assert run.call_args.kwargs[key] == "xhigh"


@pytest.mark.parametrize("model", [
    "codex/gpt-5.5", "opencode-go/model", "mcode/minimax/MiniMax-M3",
    "kimi/kimi-code/k3.1-pro", "agy/gemini-3.8-flash", "claude/sonnet",
])
def test_probe_model_derived_engine(model: str) -> None:
    from unittest.mock import MagicMock, patch

    from splinter.engines.health import probe

    engine = engine_for_model(model)
    assert engine is not None
    with (
        patch("splinter.engines.health.shutil.which", return_value="/bin/provider"),
        patch("splinter.engines.health.run_subprocess", return_value=MagicMock(
            returncode=0, stdout="ok", stderr="",
        )) as run,
    ):
        assert probe(engine).up
        run.assert_called_once()
