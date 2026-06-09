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


def run(
    prompt: str,
    model: str,
    *,
    effort: str | None = None,
    output_format: str = "json",
    resume: str | None = None,
    session_id: str | None = None,
    timeout: int = 300,
) -> ClaudeResult:
    cmd: list[str] = [
        "claude", "-p", prompt, "--model", model,
        "--output-format", output_format,
        "--dangerously-skip-permissions",
    ]
    if effort is not None:
        cmd.extend(["--effort", effort])
    if resume is not None:
        cmd.extend(["--resume", resume])
    if session_id is not None:
        cmd.extend(["--session-id", session_id])

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
        timeout: int = 600,
    ) -> ProviderResponse:
        result = run(prompt, model, effort=variant, resume=session, timeout=timeout)
        return ProviderResponse(
            text=result.text,
            tokens=result.usage,
            cost=0.0,
            raw=result.raw,
            session_id=result.raw.get("_session_id") or None,
        )
