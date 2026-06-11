"""Tests for AdaptiveStrategy and supporting helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from splinter.agents.runner import RunResult, Task
from splinter.configure import configured_budget
from splinter.enums import Decision
from splinter.models.pricing import estimate_tier_cost
from splinter.models.roster import Tier, load_ladder
from splinter.strategies.adaptive import AdaptiveStrategy
from splinter.strategies.registry import get_strategy

# ---------------------------------------------------------------------------
# _route_tier: cheapest capable tier selection
# ---------------------------------------------------------------------------


def test_route_tier_trivial_picks_floor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    ladder = load_ladder()
    strategy = AdaptiveStrategy()
    # trivial floor = T0; T0 is cheapest in T0+
    tier = strategy._route_tier("trivial", ladder)
    assert tier == 0


def test_route_tier_critical_respects_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    ladder = load_ladder()
    strategy = AdaptiveStrategy()
    # critical floor = T4; routed tier must be >= floor
    tier = strategy._route_tier("critical", ladder)
    floor = ladder.effort_mapping("critical")
    assert floor is not None
    assert tier >= floor.start_tier


def test_route_tier_returns_cheapest_among_capable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    ladder = load_ladder()
    strategy = AdaptiveStrategy()
    for effort in ("trivial", "normal", "hard", "critical"):
        tier = strategy._route_tier(effort, ladder)
        em = ladder.effort_mapping(effort)
        assert em is not None
        assert tier >= em.start_tier
        candidates = [t for t in ladder.tiers if t.level >= em.start_tier]
        cheapest = min(candidates, key=lambda t: estimate_tier_cost(t, effort))
        assert tier == cheapest.level


# ---------------------------------------------------------------------------
# estimate_tier_cost: sanity checks
# ---------------------------------------------------------------------------


def test_estimate_tier_cost_cheapest_is_flash() -> None:
    flash = Tier(name="t", level=0, models=["opencode-go/deepseek-v4-flash"], provider="opencode")
    sonnet = Tier(name="t2", level=5, models=["sonnet"], provider="claude")
    assert estimate_tier_cost(flash, "normal") < estimate_tier_cost(sonnet, "normal")


def test_estimate_tier_cost_scales_with_effort() -> None:
    tier = Tier(name="t", level=0, models=["opencode-go/deepseek-v4-pro"], provider="opencode")
    trivial = estimate_tier_cost(tier, "trivial")
    critical = estimate_tier_cost(tier, "critical")
    assert critical > trivial


# ---------------------------------------------------------------------------
# configured_budget: reads from config
# ---------------------------------------------------------------------------


def test_configured_budget_default_is_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert configured_budget() is None


def test_configured_budget_reads_from_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / ".splinter"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text("defaults:\n  budget: 2.50\n")
    assert configured_budget() == pytest.approx(2.50)


def test_adaptive_uses_config_budget_when_no_cli_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When budget=None is passed, effective_budget should come from config."""
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / ".splinter"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text("defaults:\n  budget: 1.00\n")

    ladder = load_ladder()
    strategy = AdaptiveStrategy()

    captured: list[float | None] = []

    def mock_loop(*args, **kwargs):  # type: ignore[no-untyped-def]
        captured.append(kwargs.get("budget"))
        rr = MagicMock(spec=RunResult)
        rr.text = "done"
        return rr

    strategy._run_task_loop = mock_loop  # type: ignore[method-assign]
    strategy._run_plan_phase = lambda *a, **kw: None  # type: ignore[method-assign]

    task = Task(description="Test task", acceptance="AC", effort="trivial", id="T1")
    session = MagicMock()
    session.read.return_value = ""
    session.read_status.return_value = {}

    strategy.execute([task], session, ladder, budget=None)

    assert captured and captured[0] == pytest.approx(1.00)


# ---------------------------------------------------------------------------
# soft_budget: caps escalation but run continues
# ---------------------------------------------------------------------------


def test_soft_budget_caps_escalation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Over soft budget → tier must not increase after the cap is triggered."""
    monkeypatch.chdir(tmp_path)
    from splinter.memory.knowledge import KnowledgeStore
    from splinter.memory.session import Session
    from splinter.obs.trace import Trace
    from splinter.strategies.base import EvalVerdict

    session = Session(str(tmp_path / "ses"))

    ladder = load_ladder()
    trace = Trace()
    knowledge = KnowledgeStore(session)
    task = Task(description="soft budget test", acceptance="AC", effort="trivial")

    fake_run = RunResult(
        text="ok",
        model="opencode-go/deepseek-v4-pro",
        tier=0,
        tokens={"input": 100, "output": 50},
        cost=0.0,
        raw={},
    )
    escalate_verdict = EvalVerdict(decision=Decision.ESCALATE, reason="escalate please")

    iteration_tiers: list[int] = []

    def fake_chain_handle(ctx: object) -> None:
        iteration_tiers.append(ctx.tier)  # type: ignore[attr-defined]
        ctx.run_result = fake_run  # type: ignore[attr-defined]
        ctx.gate_output = ""  # type: ignore[attr-defined]
        ctx.verdict = escalate_verdict  # type: ignore[attr-defined]
        ctx.oc_session = None  # type: ignore[attr-defined]
        ctx.eval_session = None  # type: ignore[attr-defined]
        # pump trace cost above the budget
        from splinter.obs.trace import log_run

        log_run(trace, fake_run, iteration=1, task=0)
        for entry in trace.entries:
            object.__setattr__(entry, "cost", 5.0)

    from splinter.strategies import direct as direct_mod

    with patch.object(direct_mod, "_make_plan", return_value="plan"):
        with patch.object(direct_mod, "build_chain") as mock_build:
            chain = MagicMock()
            chain.handle.side_effect = fake_chain_handle
            mock_build.return_value = chain

            from splinter.agents.evaluator import Evaluator

            with patch.object(Evaluator, "next_action") as mock_action:
                action = MagicMock()
                action.stop = False
                action.next_tier = 3  # wants to escalate to T3
                action.ask_user = False
                mock_action.return_value = action

                strategy = AdaptiveStrategy()
                strategy._run_task_loop(
                    task,
                    session,
                    ladder,
                    trace,
                    knowledge,
                    effort=None,
                    budget=0.01,  # tiny budget → immediately over
                    max_iterations=3,
                    localization="",
                    soft_budget=True,
                    start_tier_override=0,
                )

    # With soft_budget, once over budget escalation is suppressed.
    # All iterations should stay at tier 0, never reaching tier 3.
    assert all(t == 0 for t in iteration_tiers), f"tiers: {iteration_tiers}"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_adaptive_registered() -> None:
    strategy = get_strategy("adaptive")
    assert isinstance(strategy, AdaptiveStrategy)


def test_donatello_alias_registered() -> None:
    strategy = get_strategy("donatello")
    assert isinstance(strategy, AdaptiveStrategy)
