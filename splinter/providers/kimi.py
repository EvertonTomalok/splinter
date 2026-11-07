"""Subprocess adapter and provider strategy for the ``kimi`` CLI (kimi-code)."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from splinter.procreg import run_subprocess
from splinter.providers.base import ModelProvider, ProviderResponse

_stream_log = logging.getLogger("splinter.live")
_log = logging.getLogger("splinter.providers")


def _stream_event(line: str) -> None:
    """Log a one-line, human-readable summary of a kimi NDJSON event."""
    line = line.strip()
    if not line:
        return
    try:
        obj = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return
    if not isinstance(obj, dict):
        return
    etype = obj.get("type")
    raw_msg = obj.get("message")
    message: dict[str, Any] = raw_msg if isinstance(raw_msg, dict) else {}
    content = message.get("content") if isinstance(message.get("content"), list) else None
    for block in content or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_use":
            name = block.get("name") or "tool"
            _stream_log.info("  🔧 %s", str(name)[:90])
        elif block.get("type") == "text" and isinstance(block.get("text"), str):
            txt = block["text"].strip().replace("\n", " ")
            if txt:
                _stream_log.info("  💬 %s", txt)
    if etype == "text" and isinstance(obj.get("text"), str):
        txt = obj["text"].strip().replace("\n", " ")
        if txt:
            _stream_log.info("  💬 %s", txt)


@dataclass(frozen=True)
class KimiResult:
    text: str
    tokens: dict[str, int]
    cost: float
    raw: dict[str, Any]
    session_id: str | None = None
    cost_indeterminate: bool = False


def _coerce_int(value: Any) -> int | None:
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


def _usage_object(obj: dict[str, Any]) -> dict[str, Any]:
    raw = obj.get("usage")
    if isinstance(raw, dict):
        return dict(raw)
    message = obj.get("message")
    if isinstance(message, dict) and isinstance(message.get("usage"), dict):
        return dict(message["usage"])
    return {}


def _parse_output(stdout: str) -> KimiResult:
    """Parse kimi stream-json NDJSON defensively (text, usage, session id)."""
    text_parts: list[str] = []
    tokens: dict[str, int] = {}
    session_id: str | None = None
    cost = 0.0
    cost_indeterminate = True
    error_text: str | None = None

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue
        sid = obj.get("session_id") or obj.get("sessionId")
        if isinstance(sid, str) and sid:
            session_id = sid
        message = obj.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), list):
            for block in message["content"]:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    text_parts.append(block["text"])
        elif obj.get("type") == "text" and isinstance(obj.get("text"), str):
            text_parts.append(obj["text"])
        if obj.get("type") in {"error", "result_error"} or obj.get("is_error") is True:
            raw_msg = obj.get("error") or obj.get("message") or obj.get("text")
            if isinstance(raw_msg, str) and raw_msg.strip():
                error_text = raw_msg
            else:
                error_text = json.dumps(obj)
        usage = _usage_object(obj)
        if usage:
            inp = _coerce_int(usage.get("input_tokens") or usage.get("input"))
            out = _coerce_int(usage.get("output_tokens") or usage.get("output"))
            if inp is not None:
                tokens.setdefault("input", inp)
            if out is not None:
                tokens.setdefault("output", out)
            raw_cost = usage.get("cost", obj.get("cost"))
            if isinstance(raw_cost, (int, float)) and not isinstance(raw_cost, bool):
                cost = float(raw_cost)
                cost_indeterminate = False

    if error_text:
        raise RuntimeError(error_text.strip())
    return KimiResult(
        text="\n".join(text_parts).strip(),
        tokens=tokens,
        cost=cost,
        raw={"session_id": session_id} if session_id else {},
        session_id=session_id,
        cost_indeterminate=cost_indeterminate,
    )


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
) -> KimiResult:
    """Run one headless prompt through ``kimi`` (kimi-code CLI).

    ``kimi/`` prefix is stripped from ``model``; an empty model omits ``-m`` so
    the CLI's ``default_model`` (config.toml) applies. ``session`` resumes via
    ``-S``. ``variant``/``fmt``/``agent`` are accepted for interface parity.
    """
    if timeout is None:
        from splinter.configure import configured_timeout

        timeout = configured_timeout()
    bare = model.removeprefix("kimi/")
    cmd: list[str] = ["kimi", "-p", prompt, "--output-format", "stream-json"]
    if bare:
        cmd.extend(["-m", bare])
    if session is not None:
        cmd.extend(["-S", session])
    for extra_dir in add_dirs:
        cmd.extend(["--add-dir", extra_dir])

    proc = run_subprocess(
        cmd,
        timeout=timeout,
        cwd=cwd,
        env=env,
        idle_timeout=idle_timeout,
        on_line=_stream_event,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"kimi exited {proc.returncode}: {proc.stderr.strip()[-2000:]}"
        )
    return _parse_output(proc.stdout)


class KimiProvider(ModelProvider):
    """Routes runs through ``kimi -p`` with ``-S`` session resume."""

    name = "kimi"
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
            session_id=result.session_id or session,
            cost_indeterminate=result.cost_indeterminate,
        )
