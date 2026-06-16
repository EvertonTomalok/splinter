"""Subprocess adapter for the agent CLI (print-mode, non-interactive).

All CLI flags live inside :func:`run` so correcting the real flag set is a
one-function edit — callers never reference flag strings directly.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from typing import Any

from splinter.procreg import run_subprocess
from splinter.providers.base import ModelPrice, ModelProvider, ProviderResponse

_stream_log = logging.getLogger("splinter.live")
_log = logging.getLogger("splinter.providers")

_DEFAULT_TIMEOUT = 180

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
            return ModelPrice(
                input=inp,
                output=out,
                cache_read=round(inp * 0.1, 6),
                cache_write=round(inp * 1.25, 6),
            )
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


def _stream_cursor_line(line: str) -> None:
    text = line.strip()
    if not text:
        return
    _stream_log.info("  %s", text)


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


@dataclass(frozen=True)
class CursorResult:
    text: str
    raw: dict[str, Any]


def run(
    prompt: str,
    *,
    model: str | None = None,
    timeout: int | None = None,
    project_dir: str = ".",
) -> CursorResult:
    """Execute ``prompt`` via ``agent -p`` and return the text response.

    Raises :class:`RuntimeError` on non-zero exit so callers can treat it as
    a transient failure without inspecting returncode themselves.
    """
    bare_model = model.removeprefix(_MODEL_PREFIX) if model else None
    cmd: list[str] = ["agent", "-p", "--trust"]
    if bare_model and bare_model != "auto":
        cmd += ["--model", bare_model]
    cmd += ["--", prompt]
    proc = run_subprocess(
        cmd,
        timeout=timeout or _DEFAULT_TIMEOUT,
        cwd=project_dir,
        on_line=_stream_cursor_line,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"agent exited {proc.returncode}: {proc.stderr.strip()}")
    text = proc.stdout.strip()
    return CursorResult(text=text, raw={"returncode": proc.returncode})


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
    ) -> ProviderResponse:
        result = run(prompt, model=model or None, timeout=timeout)
        return ProviderResponse(text=result.text, raw=result.raw)

    def fetch_pricing(self) -> dict[str, ModelPrice]:
        return fetch_pricing()
