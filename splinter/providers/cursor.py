"""Subprocess adapter for the agent CLI (print-mode, non-interactive).

All CLI flags live inside :func:`run` so correcting the real flag set is a
one-function edit — callers never reference flag strings directly.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from typing import Any

from splinter.procreg import run_subprocess
from splinter.providers.base import ModelProvider, ProviderResponse

_stream_log = logging.getLogger("splinter.live")

_DEFAULT_TIMEOUT = 180

_MODEL_PREFIX = "cursor/"


def list_models() -> list[str]:
    """Return model ids available via ``agent --list-models``, prefixed with ``cursor/``."""
    try:
        proc = subprocess.run(
            ["agent", "--list-models"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        models: list[str] = []
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line or " - " not in line:
                continue
            model_id = line.split(" - ", 1)[0].strip()
            if model_id:
                models.append(f"{_MODEL_PREFIX}{model_id}")
        return sorted(models)
    except Exception:
        return [f"{_MODEL_PREFIX}auto"]


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
    # Strip the ``cursor/`` namespace prefix before passing to the CLI.
    bare_model = model.removeprefix(_MODEL_PREFIX) if model else None
    cmd: list[str] = ["agent", "-p"]
    if bare_model and bare_model != "auto":
        cmd += ["--model", bare_model]
    cmd += ["--", prompt]
    proc = run_subprocess(
        cmd,
        timeout=timeout or _DEFAULT_TIMEOUT,
        cwd=project_dir,
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
