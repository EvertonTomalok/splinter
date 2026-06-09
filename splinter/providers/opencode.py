"""Subprocess adapter and provider strategy for the ``opencode`` CLI."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Any

from splinter.enums import Variant
from splinter.procreg import run_subprocess
from splinter.providers.base import ModelProvider, ProviderResponse

VALID_VARIANTS = {v.value for v in Variant}


@dataclass(frozen=True)
class OpencodeResult:
    text: str
    tokens: dict[str, int]
    cost: float
    raw: dict[str, Any]


def run(
    prompt: str,
    model: str,
    *,
    variant: str | None = None,
    fmt: str = "json",
    session: str | None = None,
    timeout: int | None = None,
) -> OpencodeResult:
    if timeout is None:
        from splinter.configure import configured_timeout

        timeout = configured_timeout()
    cmd: list[str] = [
        "opencode",
        "run",
        "-m",
        model,
        "--agent",
        "build",  # the build agent can create/edit files; "plan" is read-only
        "--format",
        fmt,
        "--dangerously-skip-permissions",
    ]
    if variant is not None and variant != "auto":
        cmd.extend(["--variant", variant])
    if session is not None:
        cmd.extend(["-s", session])

    # The prompt is the positional message (NOT `-c`, which means --continue).
    cmd.append(prompt)

    proc = run_subprocess(cmd, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"opencode exited {proc.returncode}: {proc.stderr.strip()}")

    raw = _parse_output(proc.stdout, fmt)
    text = _extract_text(raw)
    tokens = _extract_tokens(raw)
    cost = _extract_cost(raw)

    return OpencodeResult(text=text, tokens=tokens, cost=cost, raw=raw)


def list_models(timeout: int = 30) -> list[str]:
    proc = subprocess.run(["opencode", "models"], capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"opencode models exited {proc.returncode}: {proc.stderr.strip()}")
    return [line.strip() for line in proc.stdout.strip().splitlines() if line.strip()]


def _parse_output(stdout: str, fmt: str) -> dict[str, Any]:
    if fmt != "json":
        return {"text": stdout.strip()}
    stdout = stdout.strip()
    if not stdout:
        return {}

    # Try single JSON first
    try:
        data = json.loads(stdout)
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            merged: dict[str, Any] = {}
            for item in data:
                if isinstance(item, dict):
                    merged.update(item)
            return merged
        return {"text": str(data)}
    except json.JSONDecodeError:
        pass

    # Parse NDJSON line-by-line, extracting text from message events and
    # metadata from step_finish events
    lines = stdout.splitlines()
    text_parts: list[str] = []
    metadata: dict[str, Any] = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue

        # Skip empty events
        if not any(k for k in obj if k not in ("type", "timestamp", "sessionID")):
            continue

        # Extract message text
        if "text" in obj:
            text_parts.append(str(obj["text"]))
        elif "content" in obj:
            text_parts.append(_extract_text(obj))
        elif "result" in obj:
            text_parts.append(str(obj["result"]))
        elif "messages" in obj:
            text_parts.append(_extract_text(obj))

        # Extract metadata from step_finish
        if obj.get("type") == "step_finish" or obj.get("type") == "step-finish":
            part = obj.get("part", {})
            if isinstance(part, dict):
                sid = part.get("sessionID") or part.get("session_id") or part.get("session")
                metadata["session_id"] = sid
                if "tokens" in part:
                    metadata["tokens"] = part["tokens"]
                if "cost" in part:
                    metadata["cost"] = part["cost"]
            metadata.update(obj)

    return {"text": "\n".join(text_parts), **metadata}


def _extract_text(raw: dict[str, Any]) -> str:
    if "text" in raw:
        return str(raw["text"])
    if "result" in raw:
        return str(raw["result"])
    if "content" in raw:
        c = raw["content"]
        if isinstance(c, list):
            parts = []
            for block in c:
                if isinstance(block, dict):
                    parts.append(block.get("text", str(block)))
                else:
                    parts.append(str(block))
            return "\n".join(parts)
        return str(c)
    if "messages" in raw:
        msgs = raw["messages"]
        if isinstance(msgs, list) and msgs:
            last = msgs[-1]
            if isinstance(last, dict):
                content = last.get("content", "")
                if isinstance(content, list):
                    parts = []
                    for block in content:
                        if isinstance(block, dict):
                            parts.append(block.get("text", str(block)))
                        else:
                            parts.append(str(block))
                    return "\n".join(parts)
                return str(content)
    return json.dumps(raw)


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


def _extract_tokens(raw: dict[str, Any]) -> dict[str, int]:
    tokens: dict[str, int] = {}
    if "usage" in raw and isinstance(raw["usage"], dict):
        u = raw["usage"]
        inp = _coerce_int(u.get("input_tokens") or u.get("prompt_tokens")) or 0
        out = _coerce_int(u.get("output_tokens") or u.get("completion_tokens")) or 0
        tokens["input"] = inp
        tokens["output"] = out
    if "tokens" in raw and isinstance(raw["tokens"], dict):
        # opencode nests some values (e.g. cache: {...}); keep only scalar counts.
        for key, value in raw["tokens"].items():
            coerced = _coerce_int(value)
            if coerced is not None:
                tokens[key] = coerced
    return tokens


def _extract_cost(raw: dict[str, Any]) -> float:
    if "cost" in raw:
        try:
            return float(raw["cost"])
        except (TypeError, ValueError):
            pass
    if "usage" in raw and isinstance(raw["usage"], dict):
        c = raw["usage"].get("cost")
        if c is not None:
            try:
                return float(c)
            except (TypeError, ValueError):
                pass
    return 0.0


class OpencodeProvider(ModelProvider):
    """Routes runs through ``opencode run`` with ``--variant`` and session reuse."""

    name = "opencode"

    def run(
        self,
        prompt: str,
        model: str,
        *,
        variant: str | None = None,
        session: str | None = None,
        timeout: int | None = None,
    ) -> ProviderResponse:
        result = run(prompt, model, variant=variant, session=session, timeout=timeout)
        session_id = session or result.raw.get("session_id") or result.raw.get("session")
        return ProviderResponse(
            text=result.text,
            tokens=result.tokens,
            cost=result.cost,
            raw=result.raw,
            session_id=session_id,
        )
