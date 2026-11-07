"""Tests for the Decisions API client (jev) and its localizer integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import splinter.providers.decisions as decisions_mod
from splinter.providers.decisions import (
    DecisionAnswer,
    DecisionError,
    decide,
    noul_questions,
    parse_noul_answers,
    resolve_provider,
)


def _fake_post(payloads: list[dict[str, Any]], response: dict[str, Any]) -> Any:
    captured: dict[str, Any] = {}

    def post(
        url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int
    ) -> dict[str, Any]:
        captured["url"] = url
        captured["payload"] = payload
        captured["headers"] = headers
        captured["timeout"] = timeout
        payloads.append(payload)
        return response

    return post, captured


class TestDecide:
    def test_payload_and_parse(self, monkeypatch: pytest.MonkeyPatch) -> None:
        response = {
            "model": "typesafe/jev-1.14-20261014",
            "answers": {
                "q0": {"type": "noul", "noul": 0.9},
                "q1": {"type": "noul", "noul": 0.1},
            },
            "usage": {"input_tokens": 357, "output_tokens": 38, "cost": 1.5e-05},
        }
        payloads: list[dict[str, Any]] = []
        post, captured = _fake_post(payloads, response)
        monkeypatch.setattr(decisions_mod, "_post_json", post)
        monkeypatch.setenv("OPENROUTER_API_KEY", "k-1")

        result = decide("state text", noul_questions(2, "relevant?"), model="typesafe/jev-1.14")

        assert captured["url"] == "https://openrouter.ai/api/alpha/decisions"
        assert captured["headers"]["Authorization"] == "Bearer k-1"
        assert captured["payload"]["model"] == "typesafe/jev-1.14"
        assert captured["payload"]["state"] == "state text"
        assert set(captured["payload"]["questions"]) == {"q0", "q1"}
        assert parse_noul_answers(result, 2) == [0.9, 0.1]
        assert result.input_tokens == 357
        assert result.cost == 1.5e-05

    def test_custom_url_and_key_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payloads: list[dict[str, Any]] = []
        post, captured = _fake_post(
            payloads, {"answers": {"q0": {"type": "noul", "noul": 1}}}
        )
        monkeypatch.setattr(decisions_mod, "_post_json", post)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("MY_JEV_KEY", "k-2")

        decide(
            "s",
            {"q0": {"type": "noul", "instructions": "x"}},
            url="https://example.com/api/decisions",
            api_key_env="MY_JEV_KEY",
        )
        assert captured["url"] == "https://example.com/api/decisions"
        assert captured["headers"]["Authorization"] == "Bearer k-2"

    def test_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(DecisionError):
            resolve_provider()

    def test_http_error_wrapped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import urllib.error

        def boom(*args: Any, **kw: Any) -> Any:
            raise urllib.error.HTTPError(
                "url", 401, "Unauthorized", None, None  # type: ignore[arg-type]
            )

        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        monkeypatch.setattr(decisions_mod.urllib.request, "urlopen", boom)
        with pytest.raises(DecisionError):
            decide("s", {"q0": {"type": "noul", "instructions": "x"}})

    def test_missing_answers_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        post, _ = _fake_post([], {"model": "m"})
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        monkeypatch.setattr(decisions_mod, "_post_json", post)
        with pytest.raises(DecisionError):
            decide("s", {"q0": {"type": "noul", "instructions": "x"}})


class TestAnswerShapes:
    def test_choice_and_score(self) -> None:
        choice = DecisionAnswer(
            kind="choice",
            raw={"choice": "billing", "probabilities": {"billing": 0.9}, "confidence": 0.8},
        )
        assert choice.choice == "billing"
        assert choice.confidence == 0.8
        assert choice.probabilities == {"billing": 0.9}
        score = DecisionAnswer(kind="score", raw={"score": 1.21})
        assert score.score == 1.21
        assert score.noul is None


class TestConfigureDecisions:
    def test_roundtrip(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from splinter.configure import configured_decisions, write_decisions_config

        monkeypatch.chdir(tmp_path)
        write_decisions_config(
            model="~typesafe/jev-latest",
            use_for=["recall", "filter", "bogus"],
            threshold=0.7,
        )
        opts = configured_decisions()
        assert opts.model == "~typesafe/jev-latest"
        assert opts.use_for == ("recall", "filter")
        assert opts.threshold == 0.7

    def test_run_configure_flags(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from splinter.configure import configured_decisions, run_configure

        monkeypatch.chdir(tmp_path)
        rc = run_configure(
            interactive=False,
            decisions_model="typesafe/jev-1.14",
            jev_recall=True,
            decisions_threshold=0.6,
        )
        assert rc == 0
        opts = configured_decisions()
        assert opts.model == "typesafe/jev-1.14"
        assert opts.use_for == ("recall",)
        assert opts.threshold == 0.6

    def test_flags_only_touch_named_targets(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from splinter.configure import configured_decisions, run_configure

        monkeypatch.chdir(tmp_path)
        run_configure(interactive=False, jev_recall=True, jev_recall_large=True)
        run_configure(interactive=False, jev_filter=True)
        assert configured_decisions().use_for == ("recall", "recall_large", "filter")
        run_configure(interactive=False, jev_recall_large=False)
        assert configured_decisions().use_for == ("recall", "filter")

    def test_defaults(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from splinter.configure import configured_decisions

        monkeypatch.chdir(tmp_path)
        opts = configured_decisions()
        assert opts.provider == "openrouter"
        assert opts.model == "typesafe/jev-1.14"
        assert opts.use_for == ()


class TestNoulQuestions:
    def test_items_are_named_in_each_question(self) -> None:
        qs = noul_questions(2, "relevant?", items=["a.py", "b.py"])
        assert qs["q0"]["instructions"] == "[0] `a.py` — relevant?"
        assert qs["q1"]["instructions"] == "[1] `b.py` — relevant?"

    def test_without_items_keeps_shared_instruction(self) -> None:
        assert noul_questions(1, "x") == {"q0": {"type": "noul", "instructions": "x"}}

    def test_mismatched_items_fall_back_to_shared(self) -> None:
        assert noul_questions(2, "x", items=["only-one"])["q1"]["instructions"] == "x"


class TestVercelAdapterAndChain:
    def test_vercel_wire_translates_request_and_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        response = {
            "model": "typesafe-ai/jev",
            "answers": [
                {"type": "predicate", "name": "q0", "probability": 0.92},
                {"type": "predicate", "name": "q1", "probability": 0.08},
            ],
            "usage": {"input_tokens": 282, "output_tokens": 20},
            "provider_metadata": {"gateway": {"cost": "0.000011844"}},
        }
        payloads: list[dict[str, Any]] = []
        post, captured = _fake_post(payloads, response)
        monkeypatch.setattr(decisions_mod, "_post_json", post)
        result = decide(
            "state",
            noul_questions(2, "relevant?"),
            model="typesafe-ai/jev",
            provider="vercel",
            api_key="vk",
        )
        assert captured["url"] == "https://ai-gateway.vercel.sh/v1/decisions"
        assert captured["headers"]["Authorization"] == "Bearer vk"
        assert captured["payload"] == {
            "model": "typesafe-ai/jev",
            "input": "state",
            "questions": [
                {"type": "predicate", "name": "q0", "instructions": "relevant?"},
                {"type": "predicate", "name": "q1", "instructions": "relevant?"},
            ],
        }
        assert parse_noul_answers(result, 2) == [0.92, 0.08]
        assert result.cost == pytest.approx(0.000011844)

    def test_chain_falls_back_to_next_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from splinter.providers.decisions import DecisionSlot, decide_chain

        urls: list[str] = []

        def post(
            url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int
        ) -> dict[str, Any]:
            urls.append(url)
            if "vercel" in url:
                raise DecisionError("decisions HTTP 429: limit")
            return {"answers": {"q0": {"type": "noul", "noul": 0.7}}}

        monkeypatch.setattr(decisions_mod, "_post_json", post)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        slots = (
            DecisionSlot(provider="vercel", api_key="vk"),
            DecisionSlot(provider="openrouter", api_key="ok"),
        )
        result = decide_chain("s", noul_questions(1, "x"), slots)
        assert parse_noul_answers(result, 1) == [0.7]
        assert ["vercel" in u for u in urls] == [True, False]

    def test_chain_raises_when_every_slot_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from splinter.providers.decisions import DecisionSlot, decide_chain

        monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        slots = (DecisionSlot(provider="vercel"), DecisionSlot(provider="openrouter"))
        with pytest.raises(DecisionError, match="AI_GATEWAY_API_KEY"):
            decide_chain("s", noul_questions(1, "x"), slots)

    def test_slot_default_model_per_provider(self) -> None:
        from splinter.providers.decisions import DecisionSlot

        assert DecisionSlot(provider="vercel").resolved_model() == "typesafe-ai/jev"
        assert DecisionSlot(provider="openrouter").resolved_model() == "typesafe/jev-1.14"


class TestDecisionsConfig:
    def test_providers_chain_roundtrip_and_legacy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from splinter.configure import configured_decisions, write_decisions_config
        from splinter.providers.decisions import DecisionSlot

        monkeypatch.chdir(tmp_path)
        write_decisions_config(provider="openrouter", model="typesafe/jev-1.13")
        legacy = configured_decisions()
        assert legacy.providers == (DecisionSlot(provider="openrouter", model="typesafe/jev-1.13"),)

        write_decisions_config(
            providers=[
                {"provider": "vercel", "model": "typesafe-ai/jev", "api_key": "vk"},
                {"provider": "openrouter", "api_key_env": "MY_KEY", "model": ""},
            ],
            use_for=["filter"],
        )
        opts = configured_decisions()
        assert opts.providers == (
            DecisionSlot(provider="vercel", model="typesafe-ai/jev", api_key="vk"),
            DecisionSlot(provider="openrouter", api_key_env="MY_KEY"),
        )
        assert opts.provider == "vercel"
        assert opts.use_for == ("filter",)

        write_decisions_config(model="openai/gpt-6-luna-decisions")
        assert configured_decisions().providers[0].model == "openai/gpt-6-luna-decisions"
