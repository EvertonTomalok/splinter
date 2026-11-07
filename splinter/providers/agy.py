"""Subprocess adapter and provider strategy for the ``agy`` CLI."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Sequence

from splinter.procreg import run_subprocess
from splinter.providers.base import ModelProvider, ProviderResponse

_stream_log = logging.getLogger("splinter.live")
_log = logging.getLogger("splinter.providers")

_ERROR_TYPES = frozenset({"error", "result_error"})
_EFFORTS = frozenset({"low", "medium", "high"})
_EFFORT_SUFFIXES = ("-low", "-medium", "-high")
_DEFAULT_MODEL = "gemini-3.8-flash"


def _strip_prefix(model: str) -> str:
    return model[len("agy/") :] if model.startswith("agy/") else model


def _final_model(model: str, variant: str | None) -> str:
    """Effort rides in the model id: ``gemini-3.8-flash`` → ``…-high``."""
    base = _strip_prefix(model) or _DEFAULT_MODEL
    for suffix in _EFFORT_SUFFIXES:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    effort = variant if variant is not None and variant in _EFFORTS else "low"
    return f"{base}-{effort}"


def _coerce_int(value: Any) -> int | None:
    """Best-effort int from a scalar; None for nested/non-numeric values."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return None
    return None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _log_block(block: dict[str, Any]) -> None:
    btype = block.get("type")
    if btype in {"tool_use", "tool"}:
        tool = block.get("name") or block.get("tool") or "tool"
        detail = block.get("description") or block.get("command") or block.get("filePath") or ""
        _stream_log.info("  🔧 %s %s", tool, str(detail)[:90].replace("\n", " "))
    elif btype == "text" and isinstance(block.get("text"), str):
        txt = str(block["text"]).strip().replace("\n", " ")
        if txt:
            _stream_log.info("  💬 %s", txt)


def _stream_event(line: str) -> None:
    """Log a one-line, human-readable summary of an agy NDJSON event."""
    line = line.strip()
    if not line:
        return
    try:
        obj = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return
    if not isinstance(obj, dict):
        return
    message = obj.get("message")
    if isinstance(message, dict):
        blocks = message.get("content")
        if isinstance(blocks, list):
            for block in blocks:
                if isinstance(block, dict):
                    _log_block(block)
    _log_block(obj)


def _usage_tokens(usage: dict[str, Any]) -> dict[str, int]:
    raw_in = usage.get("input_tokens")
    if raw_in is None:
        raw_in = usage.get("input")
    raw_out = usage.get("output_tokens")
    if raw_out is None:
        raw_out = usage.get("output")
    tokens: dict[str, int] = {}
    inp = _coerce_int(raw_in)
    out = _coerce_int(raw_out)
    if inp is not None:
        tokens["input"] = inp
    if out is not None:
        tokens["output"] = out
    return tokens


def _event_cost(obj: dict[str, Any], usage: dict[str, Any]) -> float | None:
    for candidate in (obj.get("cost"), obj.get("total_cost_usd"), usage.get("cost")):
        coerced = _coerce_float(candidate)
        if coerced is not None:
            return coerced
    return None


def _parse_stream(stdout: str) -> dict[str, Any]:
    """Parse agy NDJSON stream-json into text/tokens/cost/session/error."""
    text_parts: list[str] = []
    tokens: dict[str, int] = {}
    session_id: str | None = None
    cost: float | None = None
    error_text: str | None = None
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

        message = obj.get("message")
        message_dict: dict[str, Any] = message if isinstance(message, dict) else {}
        raw_content = obj.get("content")
        if not isinstance(raw_content, list):
            raw_content = message_dict.get("content")
        if isinstance(raw_content, list):
            for block in raw_content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                text = block.get("text")
                if (btype == "text" or btype is None) and isinstance(text, str) and text:
                    text_parts.append(text)
        elif obj.get("type") == "text" and isinstance(obj.get("text"), str) and obj["text"]:
            text_parts.append(str(obj["text"]))

        usage = obj.get("usage")
        usage_dict: dict[str, Any] = usage if isinstance(usage, dict) else {}
        if not usage_dict:
            nested = message_dict.get("usage")
            usage_dict = nested if isinstance(nested, dict) else {}
        if usage_dict:
            tokens.update(_usage_tokens(usage_dict))
        if cost is None:
            cost = _event_cost(obj, usage_dict)

        sid = obj.get("session_id") or obj.get("sessionId")
        if isinstance(sid, str) and sid and session_id is None:
            session_id = sid

        if obj.get("type") in _ERROR_TYPES or obj.get("is_error") is True:
            raw_msg = obj.get("message") or obj.get("error") or obj.get("text")
            if isinstance(raw_msg, str) and raw_msg.strip():
                error_text = raw_msg.strip()
            else:
                error_text = json.dumps(obj)
    return {
        "text": "\n".join(text_parts),
        "tokens": tokens,
        "cost": cost,
        "session_id": session_id,
        "error": error_text,
    }


@dataclass(frozen=True)
class AgyResult:
    text: str
    tokens: dict[str, int]
    cost: float
    raw: dict[str, Any]
    session_id: str | None = None
    cost_indeterminate: bool = False


def run(
    prompt: str,
    model: str,
    *,
    variant: str | None = None,
    fmt: str = "json",
    session: str | None = None,
    timeout: int | None = None,
    agent: str = "build",
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    idle_timeout: float | None = None,
    add_dirs: Sequence[str] = (),
) -> AgyResult:
    """Run ``agy`` (edit-only, sandboxed) and parse its stream-json NDJSON.

    ``fmt``/``agent`` exist for interface parity; agy always streams json and
    has no resume flag — a passed ``session`` is echoed in the result only.
    """
    if timeout is None:
        from splinter.configure import configured_timeout

        timeout = configured_timeout()

    cmd: list[str] = [
        "agy",
        "-p",
        prompt,
        "--model",
        _final_model(model, variant),
        "--mode",
        "accept-edits",
        "--sandbox",
        "--output-format",
        "stream-json",
    ]
    for directory in add_dirs:
        cmd.extend(["--add-dir", directory])

    proc = run_subprocess(
        cmd,
        timeout=timeout,
        cwd=cwd,
        env=env,
        idle_timeout=idle_timeout,
        on_line=_stream_event,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"agy exited {proc.returncode}: {proc.stderr.strip()[-2000:]}")

    parsed = _parse_stream(proc.stdout)
    if parsed.get("error"):
        raise RuntimeError(str(parsed["error"]))
    raw_cost = parsed.get("cost")
    if raw_cost is None:
        _log.warning("cost indeterminate: no cost field in agy response payload")
        cost, cost_indeterminate = 0.0, True
    else:
        cost, cost_indeterminate = float(raw_cost), False
    session_id = session or parsed.get("session_id")
    return AgyResult(
        text=str(parsed.get("text") or ""),
        tokens=parsed["tokens"],
        cost=cost,
        raw=parsed,
        session_id=session_id,
        cost_indeterminate=cost_indeterminate,
    )


class AgyProvider(ModelProvider):
    """Routes runs through ``agy`` in accept-edits sandbox mode.

    ``variant`` maps onto the model effort suffix (low|medium|high, default
    low). agy reports billed cost in its stream payload; no pricing lookup.
    """

    name = "agy"
    supports_pricing = False

    def run(
        self,
        prompt: str,
        model: str,
        *,
        variant: str | None = None,
        output_format: str = "json",
        session: str | None = None,
        session_id: str | None = None,
        timeout: int | None = None,
        agent: str = "build",
        cwd: str | None = None,
    ) -> ProviderResponse:
        from splinter.providers.base import detect_provider_gap

        try:
            result = run(
                prompt,
                model,
                variant=variant,
                fmt=output_format,
                session=session,
                timeout=timeout,
                agent=agent,
                cwd=cwd,
            )
        except Exception as exc:
            gap = detect_provider_gap(exc, self.name, model)
            if gap:
                raise gap from exc
            raise
        return ProviderResponse(
            text=result.text,
            tokens=result.tokens,
            cost=result.cost,
            raw=result.raw,
            session_id=result.session_id,
            cost_indeterminate=result.cost_indeterminate,
        )
