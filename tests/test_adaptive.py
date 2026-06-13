"""Tests for AdaptiveStrategy and supporting helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from splinter.agents.runner import RunResult, Task
from splinter.configure import configured_budget, configured_soft_budget
from splinter.enums import Decision
from splinter.models.pricing import estimate_tier_cost
from splinter.models.roster import Tier, load_ladder
from splinter.strategies.adaptive import AdaptiveStrategy, _effort_weight
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


def test_route_tier_no_budget_returns_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a budget, _route_tier returns the effort floor for all effort levels."""
    monkeypatch.chdir(tmp_path)
    ladder = load_ladder()
    for effort in ("trivial", "normal", "hard", "critical"):
        tier = AdaptiveStrategy._route_tier(effort, ladder)
        em = ladder.effort_mapping(effort)
        assert em is not None
        assert tier == em.start_tier, f"effort={effort}: expected T{em.start_tier}, got T{tier}"


# ---------------------------------------------------------------------------
# _route_tier: budget-aware routing (US-002, US-003)
# ---------------------------------------------------------------------------

# Tier/cost constants derived from ladder.yaml + pricing.py for normal effort:
#   T1 (minimax): (8000*0.80 + 1500*4.50) / 1_000_000 = 0.01315
#   T0 (deepseek_pro): (8000*0.27 + 1500*1.10) / 1_000_000 = 0.00381
# For hard effort (T3 floor, same deepseek_pro model as T0):
#   T3 cost: (16000*0.27 + 3000*1.10) / 1_000_000 = 0.00762

_BUDGET_ROUTE_CASES: list[tuple[str, str, float, list[str], int]] = [
    # ample budget → effort floor (same as cascade start)
    ("ample_normal", "normal", 10.0, ["normal"], 1),
    ("ample_hard", "hard", 10.0, ["hard"], 3),
    # tight budget (floor over share) → down-route below floor
    # normal floor T1 cost=0.01315 > share=0.008, T0 cost=0.00381 ≤ 0.008 → T0
    ("tight_normal_down_routes", "normal", 0.008, ["normal"], 0),
    # critical always at floor regardless of budget (hard capability floor)
    ("critical_tight_budget_keeps_floor", "critical", 0.0001, ["critical"], 4),
    # none fits → globally cheapest tier (T0, level 0)
    ("none_fits_returns_cheapest", "normal", 0.001, ["normal"], 0),
]


@pytest.mark.parametrize(
    "case_id,effort,budget,remaining,expected",
    _BUDGET_ROUTE_CASES,
    ids=[c[0] for c in _BUDGET_ROUTE_CASES],
)
def test_route_tier_budget_aware(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case_id: str,
    effort: str,
    budget: float,
    remaining: list[str],
    expected: int,
) -> None:
    monkeypatch.chdir(tmp_path)
    ladder = load_ladder()
    tier = AdaptiveStrategy._route_tier(effort, ladder, budget, remaining)
    assert tier == expected, f"case={case_id}: expected T{expected}, got T{tier}"


def test_route_tier_ample_budget_equals_cascade_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ample budget: donatello routes to the same floor tier cascade would start at."""
    monkeypatch.chdir(tmp_path)
    ladder = load_ladder()
    for effort in ("normal", "hard"):
        em = ladder.effort_mapping(effort)
        assert em is not None
        cascade_floor = em.start_tier
        donatello_tier = AdaptiveStrategy._route_tier(effort, ladder, 100.0, [effort])
        assert donatello_tier == cascade_floor, (
            f"effort={effort}: expected T{cascade_floor}, got T{donatello_tier}"
        )


def test_route_tier_tight_budget_routes_below_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tight budget: donatello routes strictly cheaper than cascade's start tier."""
    monkeypatch.chdir(tmp_path)
    ladder = load_ladder()
    # normal floor is T1 (minimax, cost 0.01315); budget of 0.008 → share < T1 cost
    em = ladder.effort_mapping("normal")
    assert em is not None
    cascade_floor = em.start_tier  # T1
    donatello_tier = AdaptiveStrategy._route_tier("normal", ladder, 0.008, ["normal"])
    assert donatello_tier < cascade_floor, (
        f"expected down-route below T{cascade_floor}, got T{donatello_tier}"
    )


def test_route_tier_routed_cost_le_share_when_affordable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Routed tier's estimated cost ≤ per-task share when an affordable tier exists."""
    monkeypatch.chdir(tmp_path)
    ladder = load_ladder()
    budget = 0.008
    remaining = ["normal"]
    total_w = sum(_effort_weight(e) for e in remaining)
    per_task_share = budget * _effort_weight("normal") / total_w
    tier_level = AdaptiveStrategy._route_tier("normal", ladder, budget, remaining)
    tier = ladder.tier_by_level(tier_level)
    assert estimate_tier_cost(tier, "normal") <= per_task_share


def test_route_tier_effort_weighting_hard_gt_trivial() -> None:
    """Hard effort gets a larger budget share than trivial effort in the same remaining set."""
    budget = 0.020
    remaining = ["trivial", "hard"]
    total_w = sum(_effort_weight(e) for e in remaining)
    trivial_share = budget * _effort_weight("trivial") / total_w
    hard_share = budget * _effort_weight("hard") / total_w
    assert hard_share > trivial_share
    assert hard_share == pytest.approx(trivial_share * 3)  # weights 1 vs 3


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
# configured_soft_budget: reads from config
# ---------------------------------------------------------------------------


def test_configured_soft_budget_default_true(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no config file, configured_soft_budget() defaults to True."""
    monkeypatch.chdir(tmp_path)
    assert configured_soft_budget() is True


def test_configured_soft_budget_reads_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """configured_soft_budget() reads soft_budget from config."""
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / ".splinter"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text("defaults:\n  soft_budget: false\n")
    assert configured_soft_budget() is False


def test_configured_soft_budget_invalid_returns_true(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed soft_budget value returns True (covers exception branch)."""
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / ".splinter"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text("defaults:\n  soft_budget: [not, a, bool]\n")
    assert configured_soft_budget() is True


def test_adaptive_passes_config_soft_budget_to_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AdaptiveStrategy.execute passes configured soft_budget to _run_task_loop."""
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / ".splinter"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text("defaults:\n  soft_budget: false\n")

    ladder = load_ladder()
    strategy = AdaptiveStrategy()

    captured: list[bool] = []

    def mock_loop(*args, **kwargs):  # type: ignore[no-untyped-def]
        captured.append(kwargs.get("soft_budget"))
        rr = MagicMock(spec=RunResult)
        rr.text = "done"
        return rr

    strategy._run_task_loop = mock_loop  # type: ignore[method-assign]
    strategy._run_plan_phase = lambda *a, **kw: None  # type: ignore[method-assign]

    task = Task(description="Test task", acceptance="AC", effort="trivial", id="T1")
    session = MagicMock()
    session.read.return_value = ""
    session.read_status.return_value = {}

    strategy.execute([task], session, ladder)

    assert captured and captured[0] is False


def test_adaptive_default_soft_budget_true(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no config, AdaptiveStrategy passes soft_budget=True to _run_task_loop."""
    monkeypatch.chdir(tmp_path)

    ladder = load_ladder()
    strategy = AdaptiveStrategy()

    captured: list[bool] = []

    def mock_loop(*args, **kwargs):  # type: ignore[no-untyped-def]
        captured.append(kwargs.get("soft_budget"))
        rr = MagicMock(spec=RunResult)
        rr.text = "done"
        return rr

    strategy._run_task_loop = mock_loop  # type: ignore[method-assign]
    strategy._run_plan_phase = lambda *a, **kw: None  # type: ignore[method-assign]

    task = Task(description="Test task", acceptance="AC", effort="trivial", id="T1")
    session = MagicMock()
    session.read.return_value = ""
    session.read_status.return_value = {}

    strategy.execute([task], session, ladder)

    assert captured and captured[0] is True


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


# ---------------------------------------------------------------------------
# US-002: Sprint lines logging
# ---------------------------------------------------------------------------


def test_adaptive_log_routing_detail_true() -> None:
    """AdaptiveStrategy._log_routing_detail is True by default."""
    strategy = AdaptiveStrategy()
    assert strategy._log_routing_detail is True


def test_adaptive_detailed_log_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Detailed log includes [budget=...] when _log_routing_detail=True."""
    from unittest.mock import MagicMock

    from splinter.agents.runner import RunResult, Task
    from splinter.memory.session import Session

    monkeypatch.chdir(tmp_path)
    caplog.set_level("INFO", logger="splinter.loop")

    session = Session(str(tmp_path / "ses"))
    ladder = load_ladder()
    task = Task(description="Test task", acceptance="AC", effort="normal", id="T1")

    strategy = AdaptiveStrategy()
    strategy._run_task_loop = MagicMock(
        return_value=MagicMock(spec=RunResult, text="done")
    )
    strategy._run_plan_phase = MagicMock()

    strategy.execute([task], session, ladder, budget=10.0)

    log_messages = [r.message for r in caplog.records if r.name == "splinter.loop"]
    task_logs = [m for m in log_messages if "task 1/1" in m]
    assert task_logs, "Expected task log message not found"
    task_log = task_logs[0]

    assert task_log.startswith("adaptive: task 1/1"), f"log={task_log}"
    assert "effort=normal" in task_log
    assert "[budget=" in task_log, f"log should contain [budget=: {task_log}"
    assert "floor=" in task_log, f"log should contain floor=: {task_log}"


def test_adaptive_uses_self_name_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Detailed log prefix uses self.name, not literal 'adaptive'."""
    from unittest.mock import MagicMock

    from splinter.agents.runner import RunResult, Task
    from splinter.memory.session import Session

    monkeypatch.chdir(tmp_path)
    caplog.set_level("INFO", logger="splinter.loop")

    session = Session(str(tmp_path / "ses"))
    ladder = load_ladder()
    task = Task(description="Test task", acceptance="AC", effort="normal", id="T1")

    strategy = AdaptiveStrategy()
    strategy._run_task_loop = MagicMock(
        return_value=MagicMock(spec=RunResult, text="done")
    )
    strategy._run_plan_phase = MagicMock()

    strategy.execute([task], session, ladder, budget=10.0)

    log_messages = [r.message for r in caplog.records if r.name == "splinter.loop"]
    task_logs = [m for m in log_messages if "task 1/1" in m]
    assert task_logs
    task_log = task_logs[0]

    prefix = f"{strategy.name}: task"
    assert task_log.startswith(prefix), f"Expected prefix {strategy.name}, got {task_log}"


def test_adaptive_logs_adaptive_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Regression test: AdaptiveStrategy logs with 'adaptive: task' prefix."""
    from unittest.mock import MagicMock

    from splinter.agents.runner import RunResult, Task
    from splinter.memory.session import Session

    monkeypatch.chdir(tmp_path)
    caplog.set_level("INFO", logger="splinter.loop")

    session = Session(str(tmp_path / "ses"))
    ladder = load_ladder()
    task = Task(description="Test task", acceptance="AC", effort="normal", id="T1")

    strategy = AdaptiveStrategy()
    strategy._run_task_loop = MagicMock(
        return_value=MagicMock(spec=RunResult, text="done")
    )
    strategy._run_plan_phase = MagicMock()

    strategy.execute([task], session, ladder)

    log_messages = [r.message for r in caplog.records if r.name == "splinter.loop"]
    task_logs = [m for m in log_messages if "task 1/1" in m]
    assert task_logs, "Expected task log message not found"
    task_log = task_logs[0]

    assert task_log.startswith("adaptive: task"), f"log={task_log}"
