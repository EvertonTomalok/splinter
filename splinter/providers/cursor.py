"""Subprocess adapter for the Cursor CLI (print-mode, non-interactive).

All CLI flags live inside :func:`run` so correcting the real flag set is a
one-function edit — callers never reference flag strings directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from splinter.procreg import run_subprocess
from splinter.providers.base import ModelProvider, ProviderResponse

_stream_log = logging.getLogger("splinter.live")

_DEFAULT_TIMEOUT = 180


@dataclass(frozen=True)
class CursorResult:
    text: str
    raw: dict[str, Any]


def run(
    prompt: str,
    *,
    timeout: int | None = None,
    project_dir: str = ".",
) -> CursorResult:
    """Execute ``prompt`` via ``cursor --print`` and return the text response.

    Raises :class:`RuntimeError` on non-zero exit so callers can treat it as
    a transient failure without inspecting returncode themselves.
    """
    cmd: list[str] = [
        "cursor",
        "--print",
        "--",
        prompt,
    ]
    proc = run_subprocess(
        cmd,
        timeout=timeout or _DEFAULT_TIMEOUT,
        cwd=project_dir,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"cursor exited {proc.returncode}: {proc.stderr.strip()}")
    text = proc.stdout.strip()
    return CursorResult(text=text, raw={"returncode": proc.returncode})


def ping(timeout: int = 30) -> bool:
    try:
        result = run("respond with only the word ok", timeout=timeout)
        return "ok" in result.text.lower()
    except Exception:
        return False


class CursorProvider(ModelProvider):
    """Routes runs through ``cursor --print``; ``model`` and ``variant`` are ignored
    (Cursor controls its own model selection via the IDE profile)."""

    name = "cursor"

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
        result = run(prompt, timeout=timeout)
        return ProviderResponse(text=result.text, raw=result.raw)
