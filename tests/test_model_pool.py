"""Tests for the configure-driven model pool: breakers, reroute, writers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from splinter.agents.runner import Task, resolve_model, run_task
from splinter.engines import runtime
from splinter.engines.health import BreakerState
from splinter.models.roster import Tier, load_ladder
from splinter.providers.base import ProviderGapError, ProviderResponse


@pytest.fixture(autouse=True)
def _clean_runtime() -> None:
    runtime.reset_all()
    yield
    runtime.reset_all()


def _ladder(models: list[str]) -> Any:
    from splinter.models.roster import EffortMapping, Ladder

    ladder = Ladder.__new__(Ladder)
    ladder.tiers = [Tier(name="T1", level=1, models=list(models), provider="opencode")]
    ladder.tier_variants = {}
    ladder.tier_timeouts = {}
    ladder.default_timeout = 60
    ladder.effort_map = {"normal": EffortMapping(start_tier=1, variant="medium")}
    return ladder


class TestRuntimeBreaker:
    def test_healthy_by_default(self) -> None:
        assert runtime.model_available("opencode/foo") is True

    def test_trips_on_first_failure(self) -> None:
        runtime.report_model_failure("m", now=0.0)
        assert runtime.model_available("m", now=1.0) is False
        assert runtime.model_state("m", now=1.0) == BreakerState.OPEN

    def test_cooldown_half_open_then_reset(self) -> None:
        runtime.report_model_failure("m", now=0.0)
        assert runtime.model_available("m", now=400.0) is True
        assert runtime.model_state("m", now=400.0) == BreakerState.HALF_OPEN
        runtime.report_model_success("m")
        assert runtime.model_state("m", now=401.0) == BreakerState.CLOSED

    def test_snapshot_and_reset(self) -> None:
        runtime.report_model_failure("m", now=0.0)
        assert "m" in runtime.snapshot()
        runtime.reset_all()
        assert runtime.snapshot() == {}


class TestResolveModelPool:
    def test_first_healthy_model_wins(self) -> None:
        assert resolve_model(1, _ladder(["m1", "m2"])) == ("m1", "opencode")

    def test_skips_tripped_sibling(self) -> None:
        runtime.report_model_failure("m1")
        assert resolve_model(1, _ladder(["m1", "m2"]))[0] == "m2"

    def test_falls_back_to_first_when_all_tripped(self) -> None:
        runtime.report_model_failure("m1")
        runtime.report_model_failure("m2")
        assert resolve_model(1, _ladder(["m1", "m2"]))[0] == "m1"


class TestRunTaskReroute:
    def test_gap_reroutes_to_tier_sibling(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import splinter.agents.runner as runner_mod
        from splinter.providers import dispatch

        calls: list[str] = []
        sleeps: list[float] = []

        def fake_session(prompt: str, model: str, **kw: Any) -> tuple[ProviderResponse, None]:
            calls.append(model)
            if len(calls) == 1:
                raise ProviderGapError(
                    "rate_limit", "opencode", model, RuntimeError("429 rate limit")
                )
            return ProviderResponse(text="ok"), None

        monkeypatch.setattr(dispatch, "run_provider_session", fake_session)
        monkeypatch.setattr(runner_mod.time, "sleep", lambda s: sleeps.append(s))
        task = Task(description="do", acceptance="- [ ] it works")
        result = run_task(task, "plan", 1, _ladder(["m1", "m2"]))

        assert calls == ["m1", "m2"]
        assert result.model == "m2"
        assert runtime.model_state("m1") == BreakerState.OPEN
        assert sleeps == [2]

    def test_gap_without_sibling_keeps_backoff(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import splinter.agents.runner as runner_mod
        from splinter.providers import dispatch

        calls: list[str] = []
        sleeps: list[float] = []

        def fake_session(prompt: str, model: str, **kw: Any) -> tuple[ProviderResponse, None]:
            calls.append(model)
            if len(calls) == 1:
                raise ProviderGapError(
                    "rate_limit", "opencode", model, RuntimeError("429 rate limit")
                )
            return ProviderResponse(text="ok"), None

        monkeypatch.setattr(dispatch, "run_provider_session", fake_session)
        monkeypatch.setattr(runner_mod.time, "sleep", lambda s: sleeps.append(s))
        task = Task(description="do", acceptance="- [ ] it works")
        run_task(task, "plan", 1, _ladder(["m1"]))

        assert calls == ["m1", "m1"]
        assert sleeps == [5]

    def test_success_reports_health(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from splinter.providers import dispatch

        runtime.report_model_failure("m1", now=0.0)
        monkeypatch.setattr(
            dispatch,
            "run_provider_session",
            lambda prompt, model, **kw: (ProviderResponse(text="ok"), None),
        )
        task = Task(description="do", acceptance="- [ ] it works")
        run_task(task, "plan", 1, _ladder(["m1"]))
        assert runtime.model_state("m1") == BreakerState.CLOSED


class TestWriteAgentsConfig:
    def test_roundtrip(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from splinter.configure import configured_agents_options, write_agents_config

        monkeypatch.chdir(tmp_path)
        write_agents_config(
            acceptance_check="gate",
            stall_min=5,
            max_attempts=2,
            probe_on_start=True,
        )
        opts = configured_agents_options()
        assert opts.acceptance_check == "gate"
        assert opts.stall_min == 5.0
        assert opts.max_attempts == 2
        assert opts.probe_on_start is True

    def test_legacy_engine_keys_dropped_on_write(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import yaml

        from splinter.configure import load_config, write_agents_config

        monkeypatch.chdir(tmp_path)
        cfg_dir = Path(".splinter")
        cfg_dir.mkdir()
        legacy = {"engines": ["mcode"], "engine_models": {"mcode": "x"}, "stall_min": 4}
        (cfg_dir / "config.yaml").write_text(yaml.dump({"agents": legacy}))
        write_agents_config(max_attempts=2)
        section = load_config()["agents"]
        assert not {"engines", "engine_models", "engine_efforts"} & set(section)
        assert section["stall_min"] == 4
        assert section["max_attempts"] == 2

    def test_invalid_acceptance_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from splinter.configure import write_agents_config

        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError):
            write_agents_config(acceptance_check="nope")

    def test_none_fields_preserved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from splinter.configure import configured_agents_options, write_agents_config

        monkeypatch.chdir(tmp_path)
        write_agents_config(stall_min=3)
        write_agents_config(max_attempts=4)
        opts = configured_agents_options()
        assert opts.stall_min == 3.0
        assert opts.max_attempts == 4

    def test_run_configure_agents_flags(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from splinter.configure import configured_agents_options, run_configure

        monkeypatch.chdir(tmp_path)
        rc = run_configure(
            interactive=False,
            agents_acceptance="gate+orchestrator",
            agents_stall_min=6,
            agents_max_attempts=5,
            agents_probe_on_start=True,
        )
        assert rc == 0
        opts = configured_agents_options()
        assert opts.acceptance_check == "gate+orchestrator"
        assert opts.stall_min == 6.0
        assert opts.max_attempts == 5
        assert opts.probe_on_start is True


class TestTierPoolConfig:
    def test_config_tier_pool_applies_to_ladder(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import yaml

        monkeypatch.chdir(tmp_path)
        cfg_dir = Path(".splinter")
        cfg_dir.mkdir(exist_ok=True)
        (cfg_dir / "config.yaml").write_text(
            yaml.dump({"models": {"tiers": [["opencode/a", "mcode/b", "opencode/c"]]}})
        )
        tier = load_ladder().tier_by_level(0)
        assert tier.models == ["opencode/a", "mcode/b", "opencode/c"]
        assert tier.provider == "opencode"

    def test_single_model_entry_still_works(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import yaml

        monkeypatch.chdir(tmp_path)
        cfg_dir = Path(".splinter")
        cfg_dir.mkdir(exist_ok=True)
        (cfg_dir / "config.yaml").write_text(
            yaml.dump({"models": {"tiers": ["opencode/z"]}})
        )
        tier = load_ladder().tier_by_level(0)
        assert tier.models[0] == "opencode/z"
