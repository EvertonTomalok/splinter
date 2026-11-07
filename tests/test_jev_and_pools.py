"""Tests for jev-assisted localization and step-pool fallback everywhere."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import splinter.providers.decisions as decisions_mod
import splinter.providers.dispatch as dispatch
from splinter.agents.localizer import (
    CodeAnchor,
    _jev_rank_anchors,
    _jev_rank_files,
    filter_task_context,
)
from splinter.agents.runner import Task, resolve_variant, run_task
from splinter.engines import runtime
from splinter.models.roster import Tier, parse_pool_entry
from splinter.providers.base import ProviderGapError, ProviderResponse


@pytest.fixture(autouse=True)
def _clean_runtime() -> None:
    runtime.reset_all()
    yield
    runtime.reset_all()


def _ladder(models: list[str], model_variants: dict[str, str] | None = None) -> Any:
    from splinter.models.roster import EffortMapping, Ladder

    ladder = Ladder.__new__(Ladder)
    ladder.tiers = [
        Tier(
            name="T1",
            level=1,
            models=list(models),
            provider="opencode",
            model_variants=model_variants or {},
        )
    ]
    ladder.tier_variants = {}
    ladder.tier_timeouts = {}
    ladder.default_timeout = 60
    ladder.effort_map = {"normal": EffortMapping(start_tier=1, variant="medium")}
    return ladder


def _decisions_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    use_for: tuple[str, ...],
    threshold: float = 0.5,
) -> None:
    from splinter.configure import write_decisions_config

    monkeypatch.chdir(tmp_path)
    write_decisions_config(use_for=list(use_for), threshold=threshold)


def _noul_response(probs: list[float]) -> dict[str, Any]:
    return {
        "answers": {
            f"q{i}": {"type": "noul", "noul": p} for i, p in enumerate(probs)
        },
        "usage": {"input_tokens": 10, "output_tokens": 2, "cost": 0.0},
    }


class TestParsePoolEntry:
    def test_fallback_receives_full_prompt(self) -> None:
        from unittest.mock import MagicMock, patch

        provider = MagicMock()
        provider.run.return_value = ProviderResponse(text="ok")
        with (
            patch.object(runtime, "model_available", side_effect=lambda m: m != "sonnet"),
            patch.object(dispatch, "get_provider", return_value=provider),
        ):
            dispatch.run_provider_session(
                "delta", "sonnet", session="old", pool=["codex/gpt-5.5"],
                fallback_prompt="task acceptance plan output",
            )
        assert provider.run.call_args.args[0] == "task acceptance plan output"
        assert provider.run.call_args.kwargs["session"] is None

    def test_runner_discards_session_owned_by_other_model(self) -> None:
        from unittest.mock import patch

        with patch.object(dispatch, "run_provider_session", return_value=(
            ProviderResponse(text="ok"), None,
        )) as run:
            run_task(
                Task(description="task", acceptance="ok"), "implementation plan", 1,
                _ladder(["sonnet"]), opencode_session="codex-session",
                session_model="codex/gpt-5.5",
            )
        assert run.call_args.kwargs["session"] is None
        assert "implementation plan" in run.call_args.args[0]

    def test_primary_variant_preserved(self) -> None:
        assert dispatch._pool_candidates("sonnet", ["sonnet@high", "opus@low"]) == [
            ("sonnet", "high"), ("opus", "low"),
        ]

    def test_open_primary_does_not_forward_foreign_session(self) -> None:
        from unittest.mock import MagicMock, patch

        provider = MagicMock()
        provider.run.return_value = ProviderResponse(text="ok", session_id="codex-session")
        with (
            patch.object(runtime, "model_available", side_effect=lambda m: m != "sonnet"),
            patch.object(dispatch, "get_provider", return_value=provider),
        ):
            text, sid = dispatch.run_text_session(
                "full prompt", "sonnet", session="claude-session", pool=["codex/gpt-5.5"],
            )
        assert text == "ok"
        assert provider.run.call_args.kwargs["session"] is None
        assert sid is None

    def test_variant(self) -> None:
        assert parse_pool_entry("m@high") == ("m", "high")

    def test_plain(self) -> None:
        assert parse_pool_entry("m") == ("m", None)

    def test_empty_variant(self) -> None:
        assert parse_pool_entry("m@") == ("m", None)


class TestStepPoolConfig:
    def test_step_list_pool(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import yaml

        from splinter.models.roster import load_ladder

        monkeypatch.chdir(tmp_path)
        cfg = Path(".splinter")
        cfg.mkdir(exist_ok=True)
        (cfg / "config.yaml").write_text(
            yaml.dump(
                {
                    "models": {
                        "planner": ["sonnet@high", "opencode/deepseek-v4-pro@medium"],
                        "eval": ["opus@high", "sonnet@low"],
                    }
                }
            )
        )
        ladder = load_ladder()
        assert ladder.planner_model == "sonnet"
        assert ladder.step_pool("planner") == [
            "sonnet@high",
            "opencode/deepseek-v4-pro@medium",
        ]
        assert ladder.eval_model == "opus"
        assert ladder.step_pool("eval") == ["opus@high", "sonnet@low"]

    def test_single_string_no_pool(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import yaml

        from splinter.models.roster import load_ladder

        monkeypatch.chdir(tmp_path)
        cfg = Path(".splinter")
        cfg.mkdir(exist_ok=True)
        (cfg / "config.yaml").write_text(yaml.dump({"models": {"planner": "sonnet"}}))
        ladder = load_ladder()
        assert ladder.planner_model == "sonnet"
        assert ladder.step_pool("planner") == []


class TestResolveVariantModelPool:
    def test_pool_variant_wins_over_tier(self) -> None:
        task = Task(description="x", acceptance="- [ ] y")
        ladder = _ladder(["m1", "m2"], {"m2": "high"})
        ladder.tier_variants[1] = "low"
        assert resolve_variant(task, None, ladder, 1, model="m2") == "high"
        assert resolve_variant(task, None, ladder, 1, model="m1") == "low"

    def test_tier_pool_sibling_variant_in_run_task(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[str, str]] = []

        def fake_session(prompt: str, model: str, **kw: Any) -> tuple[ProviderResponse, None]:
            calls.append((model, str(kw.get("variant"))))
            if len(calls) == 1:
                assert kw["session"] == "original"
                raise ProviderGapError(
                    "rate_limit", "opencode", model, RuntimeError("429")
                )
            assert kw["session"] is None
            assert "plan" in prompt
            return ProviderResponse(text="ok"), None

        monkeypatch.setattr(dispatch, "run_provider_session", fake_session)
        import splinter.agents.runner as runner_mod

        sleeps: list[float] = []
        monkeypatch.setattr(runner_mod.time, "sleep", lambda s: sleeps.append(s))
        task = Task(description="x", acceptance="- [ ] y")
        ladder = _ladder(["m1", "m2"], {"m1": "low", "m2": "high"})
        result = run_task(task, "plan", 1, ladder, opencode_session="original")
        assert calls == [("m1", "low"), ("m2", "high")]
        assert result.model == "m2"


class TestDispatchPoolReroute:
    def test_run_text_reroutes_across_providers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        used: list[str] = []

        class FakeProvider:
            name = "fake"

            def run(self, prompt: str, model: str, **kw: Any) -> ProviderResponse:
                used.append(model)
                if model == "sonnet":
                    raise ProviderGapError(
                        "rate_limit", "claude", model, RuntimeError("429 rate limit")
                    )
                return ProviderResponse(text=f"ok:{model}")

        monkeypatch.setattr(dispatch, "get_provider", lambda name: FakeProvider())
        monkeypatch.setattr(dispatch.time, "sleep", lambda s: None)
        out = dispatch.run_text(
            "hi", "sonnet", variant="high", pool=["opencode/deepseek-v4-pro@medium"]
        )
        assert used == ["sonnet", "opencode/deepseek-v4-pro"]
        assert out == "ok:opencode/deepseek-v4-pro"
        assert runtime.model_state("sonnet") .value == "open"

    def test_exhausted_pool_reraises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeProvider:
            name = "fake"

            def run(self, prompt: str, model: str, **kw: Any) -> ProviderResponse:
                raise ProviderGapError(
                    "rate_limit", "claude", model, RuntimeError("429")
                )

        monkeypatch.setattr(dispatch, "get_provider", lambda name: FakeProvider())
        monkeypatch.setattr(dispatch.time, "sleep", lambda s: None)
        with pytest.raises(ProviderGapError):
            dispatch.run_text("hi", "sonnet", variant="high", pool=["haiku"])

    def test_non_transient_raises_immediately(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[str] = []

        class FakeProvider:
            name = "fake"

            def run(self, prompt: str, model: str, **kw: Any) -> ProviderResponse:
                calls.append(model)
                raise ProviderGapError(
                    "insufficient_balance", "claude", model, RuntimeError("broke")
                )

        monkeypatch.setattr(dispatch, "get_provider", lambda name: FakeProvider())
        with pytest.raises(ProviderGapError):
            dispatch.run_text("hi", "sonnet", variant="high", pool=["haiku"])
        assert calls == ["sonnet"]

    def test_skips_tripped_sibling(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runtime.report_model_failure("haiku")
        calls: list[str] = []

        class FakeProvider:
            name = "fake"

            def run(self, prompt: str, model: str, **kw: Any) -> ProviderResponse:
                calls.append(model)
                if model == "sonnet":
                    raise ProviderGapError(
                        "overload", "claude", model, RuntimeError("503")
                    )
                return ProviderResponse(text="ok")

        monkeypatch.setattr(dispatch, "get_provider", lambda name: FakeProvider())
        monkeypatch.setattr(dispatch.time, "sleep", lambda s: None)
        dispatch.run_text("hi", "sonnet", variant="high", pool=["haiku", "opus"])
        assert calls == ["sonnet", "opus"]


class TestJevRankAnchors:
    def test_drop_sort_and_confidence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _decisions_config(monkeypatch, tmp_path, ("recall",), threshold=0.5)
        captured: dict[str, Any] = {}

        def fake_post(
            url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int
        ) -> dict[str, Any]:
            captured.update(payload)
            return _noul_response([0.2, 0.9, 0.6])

        monkeypatch.setattr(decisions_mod, "_post_json", fake_post)
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        anchors = [
            CodeAnchor(file="a.py", symbol="", reason="ra", confidence=0.9),
            CodeAnchor(file="b.py", symbol="", reason="rb", confidence=0.9),
            CodeAnchor(file="c.py", symbol="", reason="rc", confidence=0.9),
        ]
        ranked = _jev_rank_anchors("do the thing", anchors, hot=0.8, medium=0.4)
        assert [a.file for a in ranked] == ["b.py", "c.py"]
        assert ranked[0].confidence == 0.9
        assert ranked[1].relevance == "medium"
        assert "b.py" in captured["state"]
        assert "`b.py`" in captured["questions"]["q1"]["instructions"]

    def test_fail_open_on_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _decisions_config(monkeypatch, tmp_path, ("recall",))

        def boom(*args: Any, **kw: Any) -> Any:
            raise decisions_mod.DecisionError("down")

        monkeypatch.setattr(decisions_mod, "_post_json", boom)
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        anchors = [CodeAnchor(file="a.py", symbol="", reason="r", confidence=0.9)]
        assert _jev_rank_anchors("t", anchors, hot=0.8, medium=0.4) == anchors

    def test_all_dropped_keeps_original(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _decisions_config(monkeypatch, tmp_path, ("recall",), threshold=0.99)
        monkeypatch.setattr(
            decisions_mod, "_post_json", lambda *a, **kw: _noul_response([0.1])
        )
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        anchors = [CodeAnchor(file="a.py", symbol="", reason="r", confidence=0.9)]
        assert _jev_rank_anchors("t", anchors, hot=0.8, medium=0.4) == anchors


class TestJevRankFiles:
    def test_rank_and_filter(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _decisions_config(monkeypatch, tmp_path, ("filter",), threshold=0.5)
        monkeypatch.setattr(
            decisions_mod, "_post_json", lambda *a, **kw: _noul_response([0.1, 0.9])
        )
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        assert _jev_rank_files("task", ["x.py", "y.py"]) == ["y.py"]

    def test_disabled_passthrough(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _decisions_config(monkeypatch, tmp_path, ())
        called: list[Any] = []

        def spy(*args: Any, **kw: Any) -> Any:
            called.append(args)
            raise AssertionError("decide must not be called when disabled")

        monkeypatch.setattr(decisions_mod, "_post_json", spy)
        assert _jev_rank_files("task", ["x.py"]) == ["x.py"]
        assert called == []


class TestFilterTaskContextJev:
    def test_filter_ranks_files_before_summary(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _decisions_config(monkeypatch, tmp_path, ("filter",), threshold=0.5)
        monkeypatch.setattr(
            decisions_mod, "_post_json", lambda *a, **kw: _noul_response([0.2, 0.9])
        )
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")

        (tmp_path / "x.py").write_text("x = 1\n")
        (tmp_path / "y.py").write_text("y = 2\n")
        monkeypatch.chdir(tmp_path)

        import splinter.agents.localizer as loc

        prompted: dict[str, Any] = {}

        def fake_run_text(prompt: str, model: str, **kw: Any) -> str:
            prompted["prompt"] = prompt
            return "summary"

        monkeypatch.setattr(loc, "run_text", fake_run_text)
        task = Task(
            description="task",
            acceptance="- [ ] a",
            target_files=["x.py", "y.py"],
        )
        ladder = _ladder(["m"])
        ladder.localizer_recall_model = "m"
        ladder.localizer_recall_variant = "low"
        ladder.localizer_recall_timeout = 30
        ladder.localizer_agent = "explore"
        ladder.localizer_recall_fallback_model = "m2"
        ladder.step_pools = {}
        out = filter_task_context(task, ladder)
        assert out == "summary"
        assert "y.py" in prompted["prompt"]
        assert "x.py" not in prompted["prompt"]
