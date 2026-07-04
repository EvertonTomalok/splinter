"""Subprocess adapter for the agent CLI (print-mode, non-interactive).

All CLI flags live inside :func:`run` so correcting the real flag set is a
one-function edit — callers never reference flag strings directly.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass
from typing import Any

from splinter.procreg import run_subprocess
from splinter.providers.base import ModelPrice, ModelProvider, ProviderResponse

_stream_log = logging.getLogger("splinter.live")
_log = logging.getLogger("splinter.providers")

_MODEL_PREFIX = "cursor/"

# Cursor exposes each underlying model family under many effort/thinking/fast
# variants (e.g. ``claude-opus-4-8-thinking-high-fast``). All variants of a
# family share the family's base rate, so we price by longest-prefix match on
# the family name. USD/MTok (input, output) — relative cost proxies for ranking
# tiers, not billing (cursor is subscription-metered). Ordered longest-first at
# match time so ``gpt-5.2-codex`` wins over ``gpt-5.2``.
_FAMILY_PRICING: dict[str, tuple[float, float]] = {
    "auto": (3.0, 15.0),
    "composer-2.5": (1.0, 5.0),
    # Claude
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-4.6-opus": (5.0, 25.0),
    "claude-4.5-opus": (5.0, 25.0),
    "claude-4.6-sonnet": (3.0, 15.0),
    "claude-4.5-sonnet": (3.0, 15.0),
    "claude-4-sonnet": (3.0, 15.0),
    # OpenAI / Codex
    "gpt-5.5": (12.0, 48.0),
    "gpt-5.4-mini": (2.0, 8.0),
    "gpt-5.4-nano": (0.5, 2.0),
    "gpt-5.4": (10.0, 40.0),
    "gpt-5.3-codex": (10.0, 40.0),
    "gpt-5.2-codex": (8.0, 32.0),
    "gpt-5.2": (8.0, 32.0),
    "gpt-5.1-codex-max": (10.0, 40.0),
    "gpt-5.1-codex-mini": (2.0, 8.0),
    "gpt-5.1": (8.0, 32.0),
    "gpt-5-mini": (0.5, 2.0),
    # Other vendors Cursor proxies
    "gemini-3.5-flash": (0.5, 2.5),
    "gemini-3.1-pro": (2.0, 10.0),
    "gemini-3-flash": (0.5, 2.5),
    "grok-build-0.1": (3.0, 15.0),
    "grok-4.3": (3.0, 15.0),
    "kimi-k2.5": (0.55, 2.5),
}


def _family_price(bare_id: str) -> ModelPrice | None:
    """Resolve a cursor model id to its family rate by longest-prefix match."""
    for prefix in sorted(_FAMILY_PRICING, key=len, reverse=True):
        if bare_id == prefix or bare_id.startswith(prefix):
            inp, out = _FAMILY_PRICING[prefix]
            return ModelPrice(input=inp, output=out)
    return None

_PRICE_LINE_RE = re.compile(
    r"\$?\s*([\d.]+)\s*/?\s*(?:MTok\s*)?(?:in|input).{0,20}?\$?\s*([\d.]+)\s*/?\s*(?:MTok\s*)?(?:out|output)",
    re.IGNORECASE,
)


def _agent_list_models_stdout() -> str:
    proc = subprocess.run(
        ["agent", "--list-models"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"agent --list-models exited {proc.returncode}: {proc.stderr.strip()}"
        )
    return proc.stdout


def _parse_model_lines(stdout: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line or " - " not in line:
            continue
        model_id, desc = line.split(" - ", 1)
        model_id = model_id.strip()
        if model_id:
            rows.append((model_id, desc.strip()))
    return rows


def _price_from_description(desc: str) -> ModelPrice | None:
    match = _PRICE_LINE_RE.search(desc)
    if not match:
        return None
    return ModelPrice(input=float(match.group(1)), output=float(match.group(2)))


def _tool_detail(args: dict[str, Any]) -> str:
    for key in (
        "description",
        "command",
        "globPattern",
        "pattern",
        "path",
        "filePath",
        "targetDirectory",
    ):
        val = args.get(key)
        if val:
            return str(val)
    return ""


def _tool_summary(obj: dict[str, Any]) -> str | None:
    subtype = str(obj.get("subtype", ""))
    raw_tool_call = obj.get("tool_call")
    tool_call: dict[str, Any] = raw_tool_call if isinstance(raw_tool_call, dict) else {}
    tool_name = ""
    args: dict[str, Any] = {}
    for key, value in tool_call.items():
        if not key.endswith("ToolCall") or not isinstance(value, dict):
            continue
        tool_name = key[: -len("ToolCall")]
        raw_args = value.get("args")
        args = raw_args if isinstance(raw_args, dict) else {}
        break
    if not tool_name:
        return None
    detail = _tool_detail(args)
    if subtype == "started":
        state = "started"
    elif subtype == "completed":
        state = "completed"
    else:
        state = subtype
    return f"🔧 {tool_name} [{state}] {detail}".strip()


def _assistant_text(obj: dict[str, Any]) -> str:
    raw_message = obj.get("message")
    message: dict[str, Any] = raw_message if isinstance(raw_message, dict) else {}
    raw_content = message.get("content")
    content: list[Any] = raw_content if isinstance(raw_content, list) else []
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            txt = str(block.get("text", "")).strip()
            if txt:
                parts.append(txt)
    return "\n".join(parts).strip()


def _stream_cursor_line(line: str) -> None:
    text = line.strip()
    if not text:
        return
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        _stream_log.info("  %s", text)
        return
    if not isinstance(obj, dict):
        return

    etype = str(obj.get("type", ""))
    if etype == "tool_call":
        summary = _tool_summary(obj)
        if summary:
            _stream_log.info("  %s", summary)
            try:
                from splinter.obs.agentic import record_action

                record_action("tool_use", summary, provider="cursor")
            except Exception:
                pass
        return

    if etype == "assistant":
        summary = _assistant_text(obj)
        if not summary:
            return
        one_line = summary.splitlines()[0][:160]
        _stream_log.info("  💬 %s", one_line)
        try:
            from splinter.obs.agentic import record_action

            record_action("text", f"💬 {one_line}", provider="cursor")
        except Exception:
            pass


def list_models() -> list[str]:
    """Return model ids available via ``agent --list-models``, prefixed with ``cursor/``."""
    try:
        return sorted(
            [f"{_MODEL_PREFIX}{mid}" for mid, _ in _parse_model_lines(_agent_list_models_stdout())]
        )
    except Exception:
        return [f"{_MODEL_PREFIX}auto"]


def fetch_pricing() -> dict[str, ModelPrice]:
    """Price every model from ``agent --list-models`` (USD/MTok). No API key.

    Each cursor model is an alias of an underlying family; its rate comes from
    an inline price in the listing (rare), the public pricing catalogue, or the
    family table — so coverage tracks the actual model list, not a fixed seed.
    """
    from splinter.models.public_pricing import fetch_public_pricing, public_price_for

    try:
        rows = _parse_model_lines(_agent_list_models_stdout())
    except RuntimeError as exc:
        _log.warning("cursor agent --list-models failed (%s); pricing 'auto' only", exc)
        rows = [("auto", "")]

    catalog: dict[str, ModelPrice] = {}
    try:
        catalog = fetch_public_pricing()
    except RuntimeError as exc:
        _log.warning("public pricing unavailable (%s); using family rates", exc)

    prices: dict[str, ModelPrice] = {}
    for bare_id, desc in rows:
        model_id = f"{_MODEL_PREFIX}{bare_id}"
        price = (
            _price_from_description(desc)
            or public_price_for(bare_id, catalog)
            or _family_price(bare_id)
        )
        if price is not None:
            prices[model_id] = price
    if not prices:
        raise RuntimeError("agent --list-models returned no models with pricing")
    return prices


def _extract_tokens(raw_usage: object) -> dict[str, int]:
    usage = raw_usage if isinstance(raw_usage, dict) else {}

    def _n(key: str) -> int:
        value = usage.get(key, 0)
        if isinstance(value, bool):
            return 0
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            try:
                return int(float(value))
            except ValueError:
                return 0
        return 0

    return {
        "input": _n("inputTokens"),
        "output": _n("outputTokens"),
        "cache_read": _n("cacheReadTokens"),
        "cache_write": _n("cacheWriteTokens"),
    }


def _calc_cost(model: str, tokens: dict[str, int]) -> tuple[float, bool]:
    bare = model.removeprefix(_MODEL_PREFIX)
    price = _family_price(bare)
    if price is None:
        return 0.0, True
    inp = int(tokens.get("input", 0) or 0)
    out = int(tokens.get("output", 0) or 0)
    total = (inp * price.input + out * price.output) / 1_000_000
    return total, False


def _parse_stream_json(stdout: str) -> tuple[str, dict[str, int], str | None, dict[str, Any]]:
    final_text = ""
    tokens: dict[str, int] = {}
    session_id: str | None = None
    init_session_id: str | None = None
    last_assistant = ""
    last_result: dict[str, Any] = {}

    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        etype = str(obj.get("type", ""))
        raw_sid_any = obj.get("session_id")
        if isinstance(raw_sid_any, str) and raw_sid_any.strip():
            sid_any = raw_sid_any.strip()
            session_id = session_id or sid_any
            if etype == "system" and str(obj.get("subtype", "")) == "init":
                init_session_id = sid_any
        if etype == "assistant":
            txt = _assistant_text(obj)
            if txt:
                last_assistant = txt
        elif etype == "result":
            last_result = obj
            raw_result = obj.get("result")
            if isinstance(raw_result, str):
                final_text = raw_result.strip()
            raw_sid = obj.get("session_id")
            if isinstance(raw_sid, str) and raw_sid.strip():
                session_id = raw_sid.strip()
            tokens = _extract_tokens(obj.get("usage"))

    if init_session_id:
        session_id = init_session_id

    text = final_text or last_assistant or stdout.strip()
    return text, tokens, session_id, last_result


@dataclass(frozen=True)
class CursorResult:
    text: str
    tokens: dict[str, int]
    raw: dict[str, Any]
    session_id: str | None
    cost: float
    cost_indeterminate: bool = False


def run(
    prompt: str,
    *,
    model: str | None = None,
    session: str | None = None,
    timeout: int | None = None,
    project_dir: str = ".",
) -> CursorResult:
    """Execute ``prompt`` via ``agent -p`` and return the text response.

    Raises :class:`RuntimeError` on non-zero exit so callers can treat it as
    a transient failure without inspecting returncode themselves.
    """
    if timeout is None:
        from splinter.configure import configured_timeout

        timeout = configured_timeout()
    bare_model = model.removeprefix(_MODEL_PREFIX) if model else None
    cmd: list[str] = ["agent", "-p", "--trust", "--output-format", "stream-json"]
    if session is not None:
        cmd += ["--resume", session]
    if bare_model and bare_model != "auto":
        cmd += ["--model", bare_model]
    cmd += ["--", prompt]
    proc = run_subprocess(
        cmd,
        timeout=timeout,
        cwd=project_dir,
        on_line=_stream_cursor_line,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"agent exited {proc.returncode}: {proc.stderr.strip()}")
    text, tokens, session_id, parsed = _parse_stream_json(proc.stdout)
    resolved_model = model or f"{_MODEL_PREFIX}auto"
    cost, cost_indeterminate = _calc_cost(resolved_model, tokens)
    return CursorResult(
        text=text,
        tokens=tokens,
        raw={"returncode": proc.returncode, "parsed": parsed},
        session_id=session_id,
        cost=cost,
        cost_indeterminate=cost_indeterminate,
    )


def ping(timeout: int = 30) -> bool:
    try:
        result = run("respond with only the word ok", timeout=timeout)
        return "ok" in result.text.lower()
    except Exception:
        return False


class CursorProvider(ModelProvider):
    """Routes runs through ``agent -p``; passes ``--model`` when a specific model is set."""

    name = "cursor"
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
        cwd: str | None = None,
    ) -> ProviderResponse:
        from splinter.providers.base import detect_provider_gap

        try:
            result = run(
                prompt,
                model=model or None,
                session=session,
                timeout=timeout,
                project_dir=cwd or ".",
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

    def fetch_pricing(self) -> dict[str, ModelPrice]:
        return fetch_pricing()
