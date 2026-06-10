"""Subprocess adapter and provider strategy for the ``claude`` CLI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from splinter.procreg import run_subprocess
from splinter.providers.base import ModelProvider, ProviderResponse


@dataclass(frozen=True)
class ClaudeResult:
    text: str
    usage: dict[str, Any]
    raw: dict[str, Any]


#: The ``claude`` CLI accepts only these reasoning-effort values. The harness
#: speaks a slightly different vocabulary (``minimal``/``auto``); map onto the
#: CLI's set here so an unknown value never reaches the subprocess.
_CLI_EFFORTS = {"low", "medium", "high", "xhigh", "max"}
_EFFORT_ALIASES = {"minimal": "low", "auto": None}

# $/1M tokens — input, output
_PRICING: dict[str, tuple[float, float]] = {
    "sonnet": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "opus": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
}


def _calc_cost(model: str, usage: dict[str, Any]) -> float:
    prices = _PRICING.get(model)
    if not prices:
        return 0.0
    inp = usage.get("input_tokens", 0) or 0
    out = usage.get("output_tokens", 0) or 0
    return (inp * prices[0] + out * prices[1]) / 1_000_000


def _normalize_effort(effort: str | None) -> str | None:
    if effort is None:
        return None
    if effort in _EFFORT_ALIASES:
        return _EFFORT_ALIASES[effort]
    if effort in _CLI_EFFORTS:
        return effort
    return None  # unknown → let the CLI use its default rather than crash


def run(
    prompt: str,
    model: str,
    *,
    effort: str | None = None,
    output_format: str = "json",
    resume: str | None = None,
    session_id: str | None = None,
    timeout: int | None = None,
) -> ClaudeResult:
    if timeout is None:
        from splinter.configure import configured_timeout

        timeout = configured_timeout()
    cmd: list[str] = [
        "claude", "-p", "--model", model,
        "--output-format", output_format,
        "--dangerously-skip-permissions",
    ]
    effort = _normalize_effort(effort)
    if effort is not None:
        cmd.extend(["--effort", effort])
    if resume is not None:
        cmd.extend(["--resume", resume])
    if session_id is not None:
        cmd.extend(["--session-id", session_id])
    # `--` terminates option parsing so a prompt starting with '-' (e.g. a PRD
    # whose first line is the '---' YAML fence) isn't mistaken for a CLI flag.
    cmd.extend(["--", prompt])

    proc = run_subprocess(cmd, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"claude exited {proc.returncode}: {proc.stderr.strip()}")

    raw: dict[str, Any] = {}
    text = proc.stdout.strip()
    if output_format == "json":
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            raw = {"result": text}
        # The CLI exits 0 even on API errors (e.g. a bad --model 404s) and tucks
        # the message into `result`. Surface it as a failure instead of letting
        # the error string flow downstream as if it were a real response.
        if raw.get("is_error"):
            status = raw.get("api_error_status", "")
            msg = raw.get("result", "claude returned an error")
            raise RuntimeError(f"claude API error{f' {status}' if status else ''}: {msg}")
    else:
        raw = {"result": text}

    result_text = raw.get("result", raw.get("content", text))
    if isinstance(result_text, list):
        parts = []
        for block in result_text:
            if isinstance(block, dict):
                parts.append(block.get("text", str(block)))
            else:
                parts.append(str(block))
        result_text = "\n".join(parts)

    usage = raw.get("usage", {})
    sid = raw.get("session_id", "")

    return ClaudeResult(text=str(result_text), usage=usage, raw={**raw, "_session_id": sid})


def ping(model: str = "sonnet", timeout: int = 30) -> bool:
    try:
        result = run("respond with only the word ok", model, timeout=timeout)
        return "ok" in result.text.lower()
    except Exception:
        return False


class ClaudeProvider(ModelProvider):
    """Routes runs through ``claude -p``; ``variant`` maps onto ``--effort``."""

    name = "claude"

    def run(
        self,
        prompt: str,
        model: str,
        *,
        variant: str | None = None,
        session: str | None = None,
        timeout: int | None = None,
    ) -> ProviderResponse:
        from splinter.providers.base import detect_provider_gap
        try:
            result = run(prompt, model, effort=variant, resume=session, timeout=timeout)
        except Exception as exc:
            gap = detect_provider_gap(exc, self.name, model)
            if gap:
                raise gap from exc
            raise
        return ProviderResponse(
            text=result.text,
            tokens=result.usage,
            cost=_calc_cost(model, result.usage),
            raw=result.raw,
            session_id=result.raw.get("_session_id") or None,
        )
