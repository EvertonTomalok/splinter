"""Deterministic mechanical gate (build/lint/typecheck/test) run before eval.

The gate is project-agnostic: its checks come from (in precedence order) the
session's own ``gate.json`` (what the planner detected or the user confirmed),
the project ``.splinter/config.yaml`` ``gate_checks``, or — only as a last
resort — the Python defaults below.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger("splinter.gate")


@dataclass(frozen=True)
class GateResult:
    passed: bool
    checks: list[tuple[str, bool, str]] = field(default_factory=list)


DEFAULT_CHECKS = [
    {"name": "ruff", "cmd": "uv run ruff check", "when": "always"},
    {"name": "mypy", "cmd": "uv run mypy splinter", "when": "always"},
    {"name": "pytest", "cmd": "uv run pytest", "when": "tests_exist"},
]

#: File under the session dir holding the resolved gate checks for this run.
GATE_FILE = "gate.json"


def _config_gate_checks(project_dir: str) -> list[dict[str, str]] | None:
    config_path = Path(project_dir) / ".splinter" / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            cfg: dict[str, Any] = yaml.safe_load(f) or {}
        checks: list[dict[str, str]] | None = cfg.get("gate_checks")
        if checks:
            return checks
    return None


def configured_gate_checks(
    project_dir: str = ".", session_dir: str | Path | None = None
) -> list[dict[str, str]] | None:
    """Gate checks explicitly set for this run (session gate.json > config.yaml).

    Returns ``None`` when nothing is configured — the caller then detects/asks
    rather than silently assuming the Python defaults.
    """
    if session_dir is not None:
        p = Path(session_dir) / GATE_FILE
        if p.exists():
            try:
                loaded = json.loads(p.read_text())
                # File present == explicitly configured; [] means "no checks".
                if isinstance(loaded, list):
                    return loaded
            except (json.JSONDecodeError, ValueError):
                pass
    return _config_gate_checks(project_dir)


def save_gate_checks(session_dir: str | Path, checks: list[dict[str, str]]) -> None:
    """Persist resolved gate checks for this session so run_gate picks them up."""
    Path(session_dir, GATE_FILE).write_text(json.dumps(checks, indent=2))


def parse_gate_spec(spec: str) -> list[dict[str, str]]:
    """Turn a free-form gate spec into checks.

    Accepts a JSON array, or a simple ``cmd1; cmd2`` / newline-separated list of
    shell commands. ``none``/``skip`` yields an empty gate (no mechanical checks).
    """
    spec = spec.strip()
    if not spec or spec.lower() in ("none", "skip", "no gate"):
        return []
    try:
        data = json.loads(spec)
        if isinstance(data, list):
            return [c for c in data if isinstance(c, dict) and c.get("cmd")]
    except (json.JSONDecodeError, ValueError):
        pass
    checks: list[dict[str, str]] = []
    for cmd in re.split(r"[;\n]+", spec):
        cmd = cmd.strip()
        if cmd:
            checks.append({"name": cmd.split()[0], "cmd": cmd, "when": "always"})
    return checks


def _load_gate_checks(
    project_dir: str = ".", session_dir: str | Path | None = None
) -> list[dict[str, str]]:
    configured = configured_gate_checks(project_dir, session_dir)
    return configured if configured is not None else DEFAULT_CHECKS


def _should_run(check: dict[str, str], project_dir: str) -> bool:
    when = check.get("when", "always")
    if when == "always":
        return True
    if when == "tests_exist":
        tests_dir = Path(project_dir) / "tests"
        return tests_dir.exists() and any(tests_dir.rglob("test_*.py"))
    return True


def run_gate(project_dir: str = ".", session_dir: str | Path | None = None) -> GateResult:
    checks = _load_gate_checks(project_dir, session_dir)
    results: list[tuple[str, bool, str]] = []
    all_passed = True

    for check in checks:
        if not _should_run(check, project_dir):
            continue
        name = check["name"]
        cmd = check["cmd"]
        try:
            proc = subprocess.run(
                cmd.split(),
                capture_output=True,
                text=True,
                timeout=120,
                cwd=project_dir,
            )
            passed = proc.returncode == 0
            output = proc.stdout + proc.stderr
            if not passed:
                all_passed = False
            results.append((name, passed, output.strip()[:500]))
        except subprocess.TimeoutExpired:
            results.append((name, False, "timed out"))
            all_passed = False
        except FileNotFoundError:
            results.append((name, False, f"command not found: {cmd}"))
            all_passed = False

    return GateResult(passed=all_passed, checks=results)


def detect_gate_checks(ladder: Any, project_dir: str = ".") -> list[dict[str, str]]:
    """Ask the model to inspect the repo and propose gate commands.

    Returns ``[]`` when it can't determine them — the caller then asks the user.
    """
    from splinter.providers.dispatch import run_text

    prompt = (
        "Inspect THIS repository (manifests, lockfiles, scripts, CI config) and "
        "determine the mechanical gate commands a change must pass: lint, "
        "typecheck, build, and tests as applicable to this project's stack.\n\n"
        "Output ONLY a JSON array, no prose. Each item: "
        '{"name": "<short>", "cmd": "<exact shell command>", '
        '"when": "always" | "tests_exist"}.\n'
        "Use the project's real tooling (e.g. npm/pnpm/yarn scripts, make targets, "
        "cargo, go, gradle, pytest, etc.). Use \"when\":\"tests_exist\" for the test "
        "command. If you cannot determine any, output []."
    )
    try:
        raw = run_text(
            prompt, ladder.localizer_precision_model,
            variant=getattr(ladder, "localizer_precision_variant", "high"),
            timeout=getattr(ladder, "localizer_precision_timeout", 600),
        )
    except Exception as exc:  # noqa: BLE001 — detection is best-effort
        log.warning("gate detection failed: %s", exc)
        return []
    return _parse_detected(raw)


def _parse_detected(raw: str) -> list[dict[str, str]]:
    """Pull the JSON array of checks out of a model response."""
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return []
    out: list[dict[str, str]] = []
    for item in data if isinstance(data, list) else []:
        if isinstance(item, dict) and item.get("cmd"):
            out.append({
                "name": str(item.get("name") or str(item["cmd"]).split()[0]),
                "cmd": str(item["cmd"]),
                "when": str(item.get("when") or "always"),
            })
    return out
