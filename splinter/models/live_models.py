"""Live model lists the pricing catalogue does not carry, cached per project.

- agy: ``agy models`` (ids carry the effort suffix; the base id is kept)
- kimi: ``kimi provider list --json`` (configured model aliases)
- decision models: OpenRouter ``/api/v1/models?output_modalities=all`` filtered
  to ``decisions`` output, Vercel AI Gateway ``/v1/models`` filtered to
  ``type == evaluation``.

``sync_model_lists`` refreshes ``.splinter/models.json``; readers only load the
cache, so opening ``splinter configure`` never shells out or hits the network
for these.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

log = logging.getLogger("splinter.models")

CACHE_PATH = Path(".splinter") / "models.json"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models?output_modalities=all"
VERCEL_MODELS_URL = "https://ai-gateway.vercel.sh/v1/models"
KIMI_LIST_ARGV = ["kimi", "provider", "list", "--json"]
_AGY_EFFORT_SUFFIX = re.compile(r"-(low|medium|high)$")


def _get_json(url: str, timeout: int = 30) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "splinter-configure"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _run(argv: list[str], timeout: int = 30) -> str:
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(argv)} exited {proc.returncode}: {proc.stderr.strip()}")
    return proc.stdout


def parse_agy_models(output: str) -> list[str]:
    """``gemini-3.8-flash-high\\tGemini…`` lines → unique ``agy/<base>`` ids, in order."""
    seen: list[str] = []
    for line in output.splitlines():
        model_id = line.split("\t", 1)[0].strip()
        if not model_id or " " in model_id:
            continue
        base = f"agy/{_AGY_EFFORT_SUFFIX.sub('', model_id)}"
        if base not in seen:
            seen.append(base)
    return seen


def parse_kimi_models(output: str) -> list[str]:
    """Model aliases from ``kimi provider list --json`` as ``kimi/<alias>`` ids."""
    data = json.loads(output)
    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, dict):
        return []
    return [f"kimi/{alias}" for alias in models if isinstance(alias, str) and alias]


def parse_openrouter_decisions(data: Any) -> list[str]:
    items = data.get("data") if isinstance(data, dict) else None
    out: list[str] = []
    for item in items or []:
        arch = item.get("architecture") if isinstance(item, dict) else None
        outputs = arch.get("output_modalities") if isinstance(arch, dict) else None
        if isinstance(outputs, list) and "decisions" in outputs and item.get("id"):
            out.append(str(item["id"]))
    return sorted(out)


def parse_vercel_decisions(data: Any) -> list[str]:
    items = data.get("data") if isinstance(data, dict) else None
    return sorted(
        str(item["id"])
        for item in items or []
        if isinstance(item, dict) and item.get("type") == "evaluation" and item.get("id")
    )


def sync_model_lists() -> tuple[dict[str, Any], dict[str, str]]:
    """Fetch every live list; failures keep the previous cached value."""
    cache = load_model_lists()
    providers: dict[str, list[str]] = dict(cache.get("providers", {}))
    decisions: dict[str, list[str]] = dict(cache.get("decisions", {}))
    failures: dict[str, str] = {}
    sources: list[tuple[str, dict[str, list[str]], Any]] = [
        ("agy", providers, lambda: parse_agy_models(_run(["agy", "models"]))),
        ("kimi", providers, lambda: parse_kimi_models(_run(KIMI_LIST_ARGV))),
        ("openrouter", decisions,
         lambda: parse_openrouter_decisions(_get_json(OPENROUTER_MODELS_URL))),
        ("vercel", decisions, lambda: parse_vercel_decisions(_get_json(VERCEL_MODELS_URL))),
    ]
    for name, target, fetch in sources:
        try:
            models = fetch()
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError,
                urllib.error.URLError) as exc:
            failures[name] = str(exc)[:200]
            continue
        if models:
            target[name] = models
    payload = {"providers": providers, "decisions": decisions}
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload, failures


def load_model_lists() -> dict[str, Any]:
    """The cached lists (``{"providers": {...}, "decisions": {...}}``); empty when absent."""
    if not CACHE_PATH.exists():
        return {}
    try:
        data = json.loads(CACHE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def cached_provider_models(provider: str) -> list[str]:
    models = load_model_lists().get("providers", {}).get(provider, [])
    return [str(m) for m in models] if isinstance(models, list) else []


def cached_decision_models(provider: str) -> list[str]:
    models = load_model_lists().get("decisions", {}).get(provider, [])
    return [str(m) for m in models] if isinstance(models, list) else []
