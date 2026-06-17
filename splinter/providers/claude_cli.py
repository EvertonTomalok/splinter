"""Subprocess adapter and provider strategy for the ``claude`` CLI."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from splinter.procreg import run_subprocess
from splinter.providers.base import ModelPrice, ModelProvider, ProviderResponse

# Child of "splinter" so the run-pane log handler surfaces these live.
_stream_log = logging.getLogger("splinter.live")
_log = logging.getLogger("splinter.providers")


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

# Bootstrap seed USD/MTok (input, output) — used when the public catalogue has
# no entry for a model id.
_PRICING: dict[str, tuple[float, float]] = {
    "haiku": (1.00, 5.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "sonnet": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "opus": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
}

# Public-catalogue ids to try for each seed alias (exact match only).
_PUBLIC_ALIASES: dict[str, tuple[str, ...]] = {
    "haiku": ("claude-haiku-4-5",),
    "sonnet": ("claude-sonnet-4-6",),
    "opus": ("claude-opus-4-8",),
}

def _seed_price(model: str) -> ModelPrice | None:
    if model in _PRICING:
        inp, out = _PRICING[model]
        return ModelPrice(input=inp, output=out)
    for prefix, prices in _PRICING.items():
        if model.startswith(prefix):
            return ModelPrice(input=prices[0], output=prices[1])
    return None


def _lookup_price(model: str) -> ModelPrice | None:
    from splinter.models.pricing_store import price_for

    synced = price_for(model)
    if synced is not None and (synced.input > 0 or synced.output > 0):
        return synced
    return _seed_price(model)


def _calc_cost(model: str, usage: dict[str, Any]) -> tuple[float, bool]:
    """Return (cost_usd, indeterminate). indeterminate=True when model not in pricing table."""
    price = _lookup_price(model)
    if price is None or (price.input <= 0 and price.output <= 0):
        _log.warning("cost indeterminate: unknown model %r not in pricing table", model)
        return 0.0, True
    inp = int(usage.get("input_tokens", 0) or 0)
    out = int(usage.get("output_tokens", 0) or 0)
    cost = (inp * price.input + out * price.output) / 1_000_000
    return cost, False


def fetch_pricing() -> dict[str, ModelPrice]:
    """Return Anthropic model pricing (USD/MTok). No API key required.

    Enumerates every Anthropic model id live from the public pricing catalogue
    (new releases appear automatically), then layers the harness's seed aliases
    (``sonnet``/``opus``/``haiku``) on top — preferring the live rate, falling
    back to the bundled seed when the catalogue lacks the model or the network
    is unavailable.
    """
    from splinter.models.public_pricing import (
        fetch_public_catalog,
        provider_models,
        public_price_for,
    )

    live: dict[str, ModelPrice] = {}
    try:
        live = provider_models(fetch_public_catalog(), "anthropic")
    except RuntimeError as exc:
        _log.warning("public pricing unavailable (%s); using seed rates", exc)

    prices: dict[str, ModelPrice] = dict(live)
    for alias, (inp, out) in _PRICING.items():
        public = public_price_for(alias, live, aliases=_PUBLIC_ALIASES.get(alias, ()))
        price = public if public is not None else ModelPrice(input=inp, output=out)
        prices[alias] = price
    return prices


def _normalize_effort(effort: str | None) -> str | None:
    if effort is None:
        return None
    if effort in _EFFORT_ALIASES:
        return _EFFORT_ALIASES[effort]
    if effort in _CLI_EFFORTS:
        return effort
    return None  # unknown → let the CLI use its default rather than crash


def _tool_detail(inp: dict[str, Any]) -> str:
    for key in ("description", "command", "file_path", "path", "pattern", "filePath"):
        val = inp.get(key)
        if val:
            return str(val)
    return ""


def _event_summaries(line: str) -> list[tuple[str, str]]:
    """Parse a claude stream-json event line into (kind, summary) pairs.

    Returns list of (kind, summary) tuples where kind ∈ ("tool_use", "text").
    Returns [] on malformed JSON, non-dict, or non-assistant events.
    """
    line = line.strip()
    if not line:
        return []
    try:
        obj = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(obj, dict):
        return []

    msg_type = obj.get("type")
    if msg_type != "assistant":
        return []

    message = obj.get("message")
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if not isinstance(content, list):
        return []

    summaries: list[tuple[str, str]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_use":
            tool = block.get("name") or "tool"
            raw_inp = block.get("input")
            inp: dict[str, Any] = raw_inp if isinstance(raw_inp, dict) else {}
            detail = _tool_detail(inp)
            summary = f"🔧 {tool} {detail[:90].replace(chr(10), ' ')}"
            summaries.append(("tool_use", summary))
        elif block.get("type") == "text":
            txt = str(block.get("text", "")).strip().replace("\n", " ")
            if txt:
                summary = f"💬 {txt}"
                summaries.append(("text", summary))
    return summaries


def _stream_claude_event(line: str) -> None:
    """Log a one-line, human-readable summary of a claude stream-json event."""
    summaries = _event_summaries(line)
    for kind, summary in summaries:
        _stream_log.info("  %s", summary)
        try:
            from splinter.obs.agentic import record_action

            record_action(kind, summary)
        except Exception:
            pass


def _parse_stream_json(stdout: str) -> dict[str, Any]:
    """Collect the final ``result`` event from an NDJSON stream-json run."""
    last_result: dict[str, Any] | None = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("type") == "result":
            last_result = obj
    if last_result is not None:
        return last_result
    text = stdout.strip()
    if not text:
        return {}
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return {"result": text}
    return obj if isinstance(obj, dict) else {"result": text}


def _parse_json_output(text: str) -> dict[str, Any]:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        raw = {"result": text}
    return raw if isinstance(raw, dict) else {"result": text}


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
    stream_json = output_format == "json"
    cli_format = "stream-json" if stream_json else output_format
    cmd: list[str] = [
        "claude",
        "-p",
        "--model",
        model,
        "--output-format",
        cli_format,
        "--dangerously-skip-permissions",
    ]
    if stream_json:
        cmd.append("--verbose")
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

    proc = run_subprocess(
        cmd,
        timeout=timeout,
        on_line=_stream_claude_event if stream_json else None,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"claude exited {proc.returncode}: {proc.stderr.strip()}")

    text = proc.stdout.strip()
    if output_format == "json":
        raw = _parse_stream_json(proc.stdout) if stream_json else _parse_json_output(text)
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
    supports_pricing = True

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
            result = run(
                prompt,
                model,
                effort=variant,
                output_format=output_format,
                resume=session,
                timeout=timeout,
            )
        except Exception as exc:
            gap = detect_provider_gap(exc, self.name, model)
            if gap:
                raise gap from exc
            raise
        cost, cost_indeterminate = _calc_cost(model, result.usage)
        return ProviderResponse(
            text=result.text,
            tokens={
                "input": result.usage.get("input_tokens", 0) or 0,
                "output": result.usage.get("output_tokens", 0) or 0,
            },
            cost=cost,
            raw=result.raw,
            session_id=result.raw.get("_session_id") or None,
            cost_indeterminate=cost_indeterminate,
        )

    def fetch_pricing(self) -> dict[str, ModelPrice]:
        return fetch_pricing()
