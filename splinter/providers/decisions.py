"""Decisions API client (TypeSafe Jev and friends) with per-provider adapters.

Decision models answer typed questions with calibrated probabilities — no
generated text. Splinter uses them to sharpen the localizer: ranking recall
candidates and gating the per-task filter.

Two wire formats are spoken:

- ``typesafe`` (OpenRouter): ``{state, questions: {q0: {type: noul}}}`` →
  ``answers`` keyed by question id.
- ``openai`` (Vercel AI Gateway ``/v1/decisions``): ``{input, questions: [{name,
  type: predicate}]}`` → ``answers`` array; translated back to the noul shape.

:func:`decide_chain` walks an ordered list of :class:`DecisionSlot` (primary,
then fallback): any error on one slot moves to the next.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("splinter.providers")


#: Curated decision-model ids on OpenRouter (Jev first).
DECISION_MODELS: tuple[str, ...] = (
    "typesafe/jev-1.14",
    "~typesafe/jev-latest",
    "typesafe/jev-1.13",
    "openai/gpt-6-luna-decisions",
    "microsoft/microsoft-decision-1",
    "perplexity/pplx-decider-v1.1-27b",
    "cloudflare/clef",
    "cloudflare/clef-flash",
    "inception/mercury-decide",
    "liquid/d1",
    "upstage/solar-decide",
)

DEFAULT_DECISION_MODEL = "typesafe/jev-1.14"

#: Decision-model slugs on Vercel AI Gateway (Jev first).
VERCEL_DECISION_MODELS: tuple[str, ...] = (
    "typesafe-ai/jev",
    "openai/gpt-6-luna-decisions",
)


@dataclass(frozen=True)
class DecisionProvider:
    key: str
    label: str
    url: str
    api_key_env: str
    wire: str = "typesafe"
    models: tuple[str, ...] = ()

    @property
    def default_model(self) -> str:
        return self.models[0] if self.models else DEFAULT_DECISION_MODEL


DECISION_PROVIDERS: dict[str, DecisionProvider] = {
    "openrouter": DecisionProvider(
        key="openrouter",
        label="OpenRouter",
        url="https://openrouter.ai/api/alpha/decisions",
        api_key_env="OPENROUTER_API_KEY",
        wire="typesafe",
        models=DECISION_MODELS,
    ),
    "vercel": DecisionProvider(
        key="vercel",
        label="Vercel AI Gateway",
        url="https://ai-gateway.vercel.sh/v1/decisions",
        api_key_env="AI_GATEWAY_API_KEY",
        wire="openai",
        models=VERCEL_DECISION_MODELS,
    ),
}


@dataclass(frozen=True)
class DecisionSlot:
    """One provider in the jev chain; ``api_key`` wins over ``api_key_env``."""

    provider: str = "openrouter"
    model: str = ""
    api_key: str = ""
    api_key_env: str = ""
    url: str = ""

    def spec(self) -> DecisionProvider:
        return DECISION_PROVIDERS.get(self.provider, DECISION_PROVIDERS["openrouter"])

    def resolved_model(self) -> str:
        return self.model or self.spec().default_model


@dataclass(frozen=True)
class DecisionAnswer:
    kind: str
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def choice(self) -> str | None:
        value = self.raw.get("choice")
        return str(value) if value is not None else None

    @property
    def score(self) -> float | None:
        value = self.raw.get("score")
        return float(value) if isinstance(value, (int, float)) else None

    @property
    def noul(self) -> float | None:
        value = self.raw.get("noul")
        return float(value) if isinstance(value, (int, float)) else None

    @property
    def confidence(self) -> float | None:
        value = self.raw.get("confidence")
        return float(value) if isinstance(value, (int, float)) else None

    @property
    def probabilities(self) -> dict[str, float]:
        raw = self.raw.get("probabilities")
        if not isinstance(raw, dict):
            return {}
        return {str(k): float(v) for k, v in raw.items() if isinstance(v, (int, float))}


@dataclass(frozen=True)
class DecisionResult:
    model: str
    answers: dict[str, DecisionAnswer]
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


class DecisionError(RuntimeError):
    """Raised when the Decisions API call fails (network, auth, bad payload)."""


def _post_json(
    url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    for name, value in headers.items():
        req.add_header(name, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        raise DecisionError(f"decisions HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        raise DecisionError(f"decisions call failed: {exc}") from exc
    if not isinstance(data, dict):
        raise DecisionError("decisions response was not a JSON object")
    return data


def resolve_provider(
    *, provider: str = "openrouter", url: str = "", api_key_env: str = "", api_key: str = ""
) -> tuple[str, str]:
    """Return (url, api_key) for the configured decision provider."""
    spec = DECISION_PROVIDERS.get(provider, DECISION_PROVIDERS["openrouter"])
    final_url = url or spec.url
    if api_key:
        return final_url, api_key
    key_name = api_key_env or spec.api_key_env
    api_key = os.environ.get(key_name, "")
    if not api_key:
        raise DecisionError(
            f"missing {key_name} — set it to use decision model provider '{provider}'"
        )
    return final_url, api_key


def decide(
    state: str,
    questions: dict[str, dict[str, Any]],
    *,
    model: str = DEFAULT_DECISION_MODEL,
    provider: str = "openrouter",
    url: str = "",
    api_key_env: str = "",
    api_key: str = "",
    timeout: int = 90,
) -> DecisionResult:
    """Ask a decision model typed ``questions`` about ``state`` in one round trip."""
    final_url, key = resolve_provider(
        provider=provider, url=url, api_key_env=api_key_env, api_key=api_key
    )
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    spec = DECISION_PROVIDERS.get(provider, DECISION_PROVIDERS["openrouter"])
    if spec.wire == "openai":
        body = _post_json(final_url, _openai_payload(state, questions, model), headers, timeout)
        return _openai_result(body, model)
    payload: dict[str, Any] = {
        "model": model,
        "state": state,
        "questions": questions,
    }
    body = _post_json(final_url, payload, headers, timeout)
    answers_raw = body.get("answers")
    if not isinstance(answers_raw, dict):
        raise DecisionError(f"decisions response missing answers: {body}")
    answers = {
        str(key): DecisionAnswer(kind=str(value.get("type", "")), raw=dict(value))
        for key, value in answers_raw.items()
        if isinstance(value, dict)
    }
    usage = body.get("usage")
    usage_dict = usage if isinstance(usage, dict) else {}
    return DecisionResult(
        model=str(body.get("model", model)),
        answers=answers,
        input_tokens=int(usage_dict.get("input_tokens", 0) or 0),
        output_tokens=int(usage_dict.get("output_tokens", 0) or 0),
        cost=float(usage_dict.get("cost", 0.0) or 0.0),
    )


_OPENAI_TYPES = {"noul": "predicate", "boolean": "predicate"}


def _openai_payload(
    state: str, questions: dict[str, dict[str, Any]], model: str
) -> dict[str, Any]:
    """Keyed typesafe questions → the OpenAI ``/decisions`` array shape."""
    return {
        "model": model,
        "input": state,
        "questions": [
            {
                "type": _OPENAI_TYPES.get(str(q.get("type", "")), str(q.get("type", ""))),
                "name": name,
                "instructions": str(q.get("instructions", "")),
            }
            for name, q in questions.items()
        ],
    }


def _openai_result(body: dict[str, Any], model: str) -> DecisionResult:
    """OpenAI ``/decisions`` response → :class:`DecisionResult` in the noul shape."""
    answers_raw = body.get("answers")
    if not isinstance(answers_raw, list):
        raise DecisionError(f"decisions response missing answers: {body}")
    answers: dict[str, DecisionAnswer] = {}
    for item in answers_raw:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        raw = dict(item)
        if item.get("type") == "predicate":
            raw = {**raw, "type": "noul", "noul": item.get("probability")}
        answers[str(item["name"])] = DecisionAnswer(kind=str(raw.get("type", "")), raw=raw)
    usage_raw = body.get("usage")
    usage: dict[str, Any] = usage_raw if isinstance(usage_raw, dict) else {}
    meta = body.get("provider_metadata") or {}
    gateway = meta.get("gateway") if isinstance(meta, dict) else None
    try:
        cost = float((gateway or {}).get("cost", 0.0) or 0.0)
    except (TypeError, ValueError):
        cost = 0.0
    return DecisionResult(
        model=str(body.get("model", model)),
        answers=answers,
        input_tokens=int(usage.get("input_tokens", 0) or 0),
        output_tokens=int(usage.get("output_tokens", 0) or 0),
        cost=cost,
    )


def decide_chain(
    state: str,
    questions: dict[str, dict[str, Any]],
    slots: tuple[DecisionSlot, ...] | list[DecisionSlot],
    *,
    timeout: int = 90,
) -> DecisionResult:
    """Try each slot in order (primary, then fallback); raise only when all fail."""
    errors: list[str] = []
    for slot in slots:
        try:
            return decide(
                state,
                questions,
                model=slot.resolved_model(),
                provider=slot.provider,
                url=slot.url,
                api_key_env=slot.api_key_env,
                api_key=slot.api_key,
                timeout=timeout,
            )
        except DecisionError as exc:
            log.warning("jev provider %s failed: %s", slot.provider, exc)
            errors.append(f"{slot.provider}: {exc}")
    raise DecisionError("every decision provider failed — " + "; ".join(errors or ["none"]))


def noul_questions(
    count: int, instructions: str, items: list[str] | None = None
) -> dict[str, dict[str, Any]]:
    """Build one noul question per item key (``q0..qN``) for a shared state.

    Pass ``items`` so each question names its item (``[i] `label` — …``): with a
    bare shared instruction the model cannot tell ``q0`` from ``q3`` and answers
    every item alike (live jev scored four files 0.66–0.68 that way, vs
    0.93/0.60/0.23/0.14 with the item named).
    """
    labels = items if items is not None and len(items) == count else None
    return {
        f"q{i}": {
            "type": "noul",
            "instructions": f"[{i}] `{labels[i]}` — {instructions}" if labels else instructions,
        }
        for i in range(count)
    }


def parse_noul_answers(result: DecisionResult, count: int) -> list[float]:
    """Extract per-item noul probabilities (``q0..qN``); missing → 0.5."""
    values: list[float] = []
    for i in range(count):
        answer = result.answers.get(f"q{i}")
        noul = answer.noul if answer is not None else None
        values.append(float(noul) if noul is not None else 0.5)
    return values
