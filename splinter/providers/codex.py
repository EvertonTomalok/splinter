"""Subprocess adapter and provider strategy for the ``codex`` CLI."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from splinter.procreg import run_subprocess
from splinter.providers.base import ModelProvider, ProviderResponse

_stream_log = logging.getLogger("splinter.live")
_log = logging.getLogger("splinter.providers")


@dataclass(frozen=True)
class CodexResult:
    text: str
    tokens: dict[str, int]
    cost: float
    raw: dict[str, Any]
    session_id: str | None
    cost_indeterminate: bool = False


_CLI_EFFORTS = {"low", "medium", "high", "xhigh"}
_EFFORT_ALIASES: dict[str, str | None] = {
    "minimal": "low",
    "auto": None,
    "max": "xhigh",
}

# $/1M tokens — input, output (confirmed model id: gpt-5-codex; rates for US-005)
_PRICING: dict[str, tuple[float, float]] = {
    "gpt-5-codex": (10.00, 40.00),
}


def _calc_cost(model: str, tokens: dict[str, int]) -> tuple[float, bool]:
    """Return (cost_usd, indeterminate). indeterminate=True when model not in pricing table."""
    prices = _PRICING.get(model)
    if not prices:
        _log.warning("cost indeterminate: unknown model %r not in pricing table", model)
        return 0.0, True
    inp = tokens.get("input", 0) or 0
    out = tokens.get("output", 0) or 0
    return (inp * prices[0] + out * prices[1]) / 1_000_000, False


def _normalize_effort(effort: str | None) -> str | None:
    if effort is None:
        return None
    if effort in _EFFORT_ALIASES:
        return _EFFORT_ALIASES[effort]
    if effort in _CLI_EFFORTS:
        return effort
    return None


def _strip_prefix(model: str) -> str:
    return model[len("codex/"):] if model.startswith("codex/") else model


def _stream_codex_event(line: str) -> None:
    line = line.strip()
    if not line:
        return
    try:
        obj = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return
    if not isinstance(obj, dict):
        return
    if obj.get("type") == "item.completed":
        item = obj.get("item")
        if isinstance(item, dict) and item.get("type") == "agent_message":
            txt = str(item.get("text", "")).strip().replace("\n", " ")
            if txt:
                _stream_log.info("  \U0001f4ac %s", txt[:120])


def _parse_jsonl(stdout: str) -> dict[str, Any]:
    session_id: str | None = None
    text_parts: list[str] = []
    tokens: dict[str, int] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        event_type = obj.get("type")
        if event_type == "thread.started":
            session_id = str(obj["thread_id"]) if obj.get("thread_id") else None
        elif event_type == "item.completed":
            item = obj.get("item")
            if isinstance(item, dict) and item.get("type") == "agent_message":
                text_parts.append(str(item.get("text", "")))
        elif event_type == "turn.completed":
            raw_usage = obj.get("usage")
            if isinstance(raw_usage, dict):
                tokens = {
                    "input": int(raw_usage.get("input_tokens", 0) or 0),
                    "output": int(raw_usage.get("output_tokens", 0) or 0),
                    "cached_input": int(raw_usage.get("cached_input_tokens", 0) or 0),
                    "reasoning": int(raw_usage.get("reasoning_output_tokens", 0) or 0),
                }
    return {
        "text": "\n".join(text_parts),
        "session_id": session_id,
        "tokens": tokens,
    }


def run(
    prompt: str,
    model: str,
    *,
    effort: str | None = None,
    resume: str | None = None,
    timeout: int | None = None,
) -> CodexResult:
    if timeout is None:
        from splinter.configure import configured_timeout

        timeout = configured_timeout()

    bare_model = _strip_prefix(model)
    normalized_effort = _normalize_effort(effort)

    base_flags: list[str] = [
        "--json",
        "--skip-git-repo-check",
        "--dangerously-bypass-approvals-and-sandbox",
        "-m",
        bare_model,
    ]
    if normalized_effort is not None:
        base_flags.extend(["-c", f"model_reasoning_effort={normalized_effort}"])

    if resume is not None:
        cmd: list[str] = ["codex", "exec", "resume", *base_flags, resume, prompt]
    else:
        cmd = ["codex", "exec", *base_flags, prompt]

    proc = run_subprocess(cmd, timeout=timeout, on_line=_stream_codex_event)
    if proc.returncode != 0:
        raise RuntimeError(f"codex exited {proc.returncode}: {proc.stderr.strip()}")

    parsed = _parse_jsonl(proc.stdout)
    tokens = parsed["tokens"]
    cost, cost_indeterminate = _calc_cost(bare_model, tokens)

    return CodexResult(
        text=parsed["text"],
        tokens=tokens,
        cost=cost,
        raw={**parsed, "_session_id": parsed["session_id"]},
        session_id=parsed["session_id"],
        cost_indeterminate=cost_indeterminate,
    )


def ping(model: str = "gpt-5-codex", timeout: int = 30) -> bool:
    try:
        result = run("respond with only the word ok", f"codex/{model}", timeout=timeout)
        return "ok" in result.text.lower()
    except Exception:
        return False


class CodexProvider(ModelProvider):
    """Routes runs through ``codex exec``; ``variant`` maps onto ``model_reasoning_effort``."""

    name = "codex"

    def run(
        self,
        prompt: str,
        model: str,
        *,
        variant: str | None = None,
        output_format: str = "json",
        session: str | None = None,
        timeout: int | None = None,
        agent: str = "build",
    ) -> ProviderResponse:
        from splinter.providers.base import detect_provider_gap

        try:
            result = run(prompt, model, effort=variant, resume=session, timeout=timeout)
        except Exception as exc:
            gap = detect_provider_gap(exc, self.name, model)
            if gap:
                raise gap from exc
            raise
        gap = detect_provider_gap(RuntimeError(result.text), self.name, model)
        if gap:
            raise gap from RuntimeError(result.text)
        return ProviderResponse(
            text=result.text,
            tokens=result.tokens,
            cost=result.cost,
            raw=result.raw,
            session_id=result.session_id,
            cost_indeterminate=result.cost_indeterminate,
        )
