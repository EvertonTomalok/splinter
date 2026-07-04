"""Subprocess adapter and provider strategy for the ``codex`` CLI."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from splinter.procreg import run_subprocess
from splinter.providers.base import ModelPrice, ModelProvider, ProviderResponse

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
_BUILD_AGENTS = frozenset({"build", "run"})
_MESSAGE_KINDS = frozenset({"agent_message", "assistant_message"})
_EFFORT_ALIASES: dict[str, str | None] = {
    "minimal": "low",
    "auto": None,
    "max": "xhigh",
}

# Bootstrap seed USD/MTok — used when the public catalogue has no entry.
_PRICING: dict[str, tuple[float, float]] = {
    "gpt-5-codex": (10.00, 40.00),
    "gpt-5.5": (12.00, 48.00),
    "gpt-5.4": (10.00, 40.00),
    "gpt-5.4-mini": (2.00, 8.00),
    "gpt-5.3-codex": (10.00, 40.00),
    "gpt-5.2": (8.00, 32.00),
    "codex-auto-review": (5.00, 20.00),
}


def _seed_price(model: str) -> ModelPrice | None:
    bare = _strip_prefix(model)
    if bare in _PRICING:
        inp, out = _PRICING[bare]
        return ModelPrice(input=inp, output=out)
    return None


def _lookup_price(model: str) -> ModelPrice | None:
    from splinter.models.pricing_store import price_for

    candidates = [model]
    if not model.startswith("codex/"):
        candidates.append(f"codex/{model}")
    for candidate in candidates:
        synced = price_for(candidate)
        if synced is not None and (synced.input > 0 or synced.output > 0):
            return synced
    return _seed_price(model)


def _calc_cost(model: str, tokens: dict[str, int]) -> tuple[float, bool]:
    """Return (cost_usd, indeterminate). indeterminate=True when model not in pricing table."""
    price = _lookup_price(model)
    if price is None or (price.input <= 0 and price.output <= 0):
        _log.warning("cost indeterminate: unknown model %r not in pricing table", model)
        return 0.0, True
    inp = int(tokens.get("input", 0) or 0)
    out = int(tokens.get("output", 0) or 0)
    cost = (inp * price.input + out * price.output) / 1_000_000
    return cost, False


def _is_codex_model(model_id: str) -> bool:
    """OpenAI catalogue ids the codex backend can drive (gpt-5 family + codex)."""
    return model_id.startswith("gpt-5") or "codex" in model_id


def fetch_pricing() -> dict[str, ModelPrice]:
    """Return Codex model pricing (USD/MTok). No API key required.

    Enumerates the relevant OpenAI model ids live from the public pricing
    catalogue (new gpt-5 / codex releases appear automatically), keyed with the
    ``codex/`` prefix, then fills any seed id the catalogue lacks. Falls back to
    seeds entirely when the network is unavailable.
    """
    from splinter.models.public_pricing import fetch_public_catalog, provider_models

    live: dict[str, ModelPrice] = {}
    try:
        live = provider_models(
            fetch_public_catalog(), "openai", predicate=_is_codex_model
        )
    except RuntimeError as exc:
        _log.warning("public pricing unavailable (%s); using seed rates", exc)

    prices: dict[str, ModelPrice] = {}
    for model_id, price in live.items():
        codex_id = model_id if model_id.startswith("codex/") else f"codex/{model_id}"
        prices[codex_id] = price
    for bare, (inp, out) in _PRICING.items():
        prices.setdefault(f"codex/{bare}", ModelPrice(input=inp, output=out))
    return prices


def _normalize_effort(effort: str | None) -> str | None:
    if effort is None:
        return None
    if effort in _EFFORT_ALIASES:
        return _EFFORT_ALIASES[effort]
    if effort in _CLI_EFFORTS:
        return effort
    return None


def _strip_prefix(model: str) -> str:
    return model[len("codex/") :] if model.startswith("codex/") else model


def _use_read_only_sandbox(agent: str) -> bool:
    """Text-only roles (PRD/plan/eval) must not let Codex edit files on disk."""
    return agent not in _BUILD_AGENTS


def _item_kind(item: dict[str, Any]) -> str:
    kind = item.get("type") or item.get("item_type")
    return str(kind) if kind else ""


def _item_text(item: dict[str, Any]) -> str:
    return str(item.get("text") or "")


def _is_message_item(item: dict[str, Any]) -> bool:
    return _item_kind(item) in _MESSAGE_KINDS


def _build_base_flags(bare_model: str, *, effort: str | None, agent: str) -> list[str]:
    flags: list[str] = ["--json", "--skip-git-repo-check", "-m", bare_model]
    if _use_read_only_sandbox(agent):
        flags.extend(["-s", "read-only"])
    else:
        flags.append("--dangerously-bypass-approvals-and-sandbox")
    if effort is not None:
        flags.extend(["-c", f"model_reasoning_effort={effort}"])
    return flags


def _record_message(
    item: dict[str, Any],
    *,
    message_texts: dict[str, str],
    message_order: list[str],
) -> None:
    text = _item_text(item)
    if not text:
        return
    item_id = str(item.get("id") or "")
    if item_id:
        if item_id not in message_texts:
            message_order.append(item_id)
        message_texts[item_id] = text
        return
    anon = f"__msg_{len(message_order)}"
    message_order.append(anon)
    message_texts[anon] = text


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
    if obj.get("type") in {"item.updated", "item.completed"}:
        item = obj.get("item")
        if isinstance(item, dict) and _is_message_item(item):
            txt = _item_text(item).strip().replace("\n", " ")
            if txt:
                _stream_log.info("  \U0001f4ac %s", txt)


def _parse_jsonl(stdout: str) -> dict[str, Any]:
    session_id: str | None = None
    message_texts: dict[str, str] = {}
    message_order: list[str] = []
    tokens: dict[str, int] = {}
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
        event_type = obj.get("type")
        if event_type == "thread.started":
            session_id = str(obj["thread_id"]) if obj.get("thread_id") else None
        elif event_type == "error":
            msg = obj.get("message") or obj.get("error") or obj.get("text")
            if isinstance(msg, str) and msg.strip().startswith("Reconnecting..."):
                continue
            if isinstance(msg, str) and msg.strip():
                error_text = msg.strip()
            else:
                error_text = json.dumps(obj)
        elif event_type in {"turn.failed", "response.error"}:
            nested = obj.get("error")
            msg = (
                nested.get("message")
                if isinstance(nested, dict)
                else obj.get("message") or obj.get("error") or obj.get("text")
            )
            if isinstance(msg, str) and msg.strip():
                error_text = msg.strip()
            else:
                error_text = json.dumps(obj)
        elif event_type in {"item.updated", "item.completed"}:
            item = obj.get("item")
            if isinstance(item, dict) and _is_message_item(item):
                _record_message(item, message_texts=message_texts, message_order=message_order)
        elif event_type == "turn.completed":
            raw_usage = obj.get("usage")
            if isinstance(raw_usage, dict):
                tokens = {
                    "input": int(raw_usage.get("input_tokens", 0) or 0),
                    "output": int(raw_usage.get("output_tokens", 0) or 0),
                    "cached_input": int(raw_usage.get("cached_input_tokens", 0) or 0),
                    "reasoning": int(raw_usage.get("reasoning_output_tokens", 0) or 0),
                }
    text_parts = [message_texts[item_id] for item_id in message_order if message_texts.get(item_id)]
    return {
        "text": "\n".join(text_parts),
        "session_id": session_id,
        "tokens": tokens,
        "error": error_text,
    }


def run(
    prompt: str,
    model: str,
    *,
    effort: str | None = None,
    resume: str | None = None,
    timeout: int | None = None,
    agent: str = "build",
    cwd: str | None = None,
) -> CodexResult:
    if timeout is None:
        from splinter.configure import configured_timeout

        timeout = configured_timeout()

    bare_model = _strip_prefix(model)
    normalized_effort = _normalize_effort(effort)
    base_flags = _build_base_flags(bare_model, effort=normalized_effort, agent=agent)

    last_msg_path: Path | None = None
    fd, last_msg_name = tempfile.mkstemp(suffix=".md")
    os.close(fd)
    last_msg_path = Path(last_msg_name)
    try:
        base_flags.extend(["-o", str(last_msg_path)])

        # `--` terminates option parsing so a prompt starting with '-' (e.g. a PRD
        # whose first line is the '---' YAML fence) isn't mistaken for a CLI flag.
        if resume is not None:
            cmd: list[str] = ["codex", "exec", "resume", *base_flags, "--", resume, prompt]
        else:
            cmd = ["codex", "exec", *base_flags, "--", prompt]

        proc = run_subprocess(cmd, timeout=timeout, cwd=cwd, on_line=_stream_codex_event)
        if proc.returncode != 0:
            raise RuntimeError(f"codex exited {proc.returncode}: {proc.stderr.strip()}")

        parsed = _parse_jsonl(proc.stdout)
        if parsed.get("error"):
            raise RuntimeError(str(parsed["error"]))
        text = str(parsed.get("text") or "")
        if not text.strip() and last_msg_path.exists():
            text = last_msg_path.read_text()
        session_id = parsed.get("session_id") or resume
        tokens = parsed["tokens"]
        cost, cost_indeterminate = _calc_cost(bare_model, tokens)

        return CodexResult(
            text=text,
            tokens=tokens,
            cost=cost,
            raw={**parsed, "_session_id": session_id},
            session_id=session_id,
            cost_indeterminate=cost_indeterminate,
        )
    finally:
        if last_msg_path is not None:
            last_msg_path.unlink(missing_ok=True)


def ping(model: str = "gpt-5-codex", timeout: int = 30) -> bool:
    try:
        result = run("respond with only the word ok", f"codex/{model}", timeout=timeout)
        return "ok" in result.text.lower()
    except Exception:
        return False


class CodexProvider(ModelProvider):
    """Routes runs through ``codex exec``; ``variant`` maps onto ``model_reasoning_effort``."""

    name = "codex"
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
                prompt, model, effort=variant, resume=session, timeout=timeout, agent=agent, cwd=cwd
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
