"""Deterministic mechanical gate (build/lint/typecheck/test) run before eval."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class GateResult:
    passed: bool
    checks: list[tuple[str, bool, str]] = field(default_factory=list)


DEFAULT_CHECKS = [
    {"name": "ruff", "cmd": "uv run ruff check", "when": "always"},
    {"name": "mypy", "cmd": "uv run mypy splinter", "when": "always"},
    {"name": "pytest", "cmd": "uv run pytest", "when": "tests_exist"},
]


def _load_gate_checks(project_dir: str = ".") -> list[dict[str, str]]:
    config_path = Path(project_dir) / ".splinter" / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            cfg: dict[str, Any] = yaml.safe_load(f) or {}
        checks: list[dict[str, str]] = cfg.get("gate_checks", DEFAULT_CHECKS)
        return checks
    return DEFAULT_CHECKS


def _should_run(check: dict[str, str], project_dir: str) -> bool:
    when = check.get("when", "always")
    if when == "always":
        return True
    if when == "tests_exist":
        tests_dir = Path(project_dir) / "tests"
        return tests_dir.exists() and any(tests_dir.rglob("test_*.py"))
    return True


def run_gate(project_dir: str = ".") -> GateResult:
    checks = _load_gate_checks(project_dir)
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
