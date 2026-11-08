"""Deterministic mechanical gate (build/lint/typecheck/test) run before eval.

The gate is project-agnostic: its checks come from (in precedence order) the
session's own ``gate.json`` (what the planner detected or the user confirmed),
the project ``.splinter/config.yaml`` ``gate_checks``, or — only as a last
resort — the Python defaults below.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from splinter.agents.runner import Task

import yaml

log = logging.getLogger("splinter.gate")


@dataclass(frozen=True)
class GateResult:
    passed: bool
    checks: list[tuple[str, bool, str]] = field(default_factory=list)


#: File under the session dir holding the resolved gate checks for this run.
GATE_FILE = "gate.json"

_EXT_LANG: dict[str, str] = {
    ".py": "python",
    ".go": "go",
    ".rs": "rust",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript-npm",
    ".jsx": "javascript-npm",
    ".rb": "ruby",
    ".java": "java",
    ".kt": "kotlin",
    ".swift": "swift",
    ".cs": "csharp",
    ".php": "php",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".h": "cpp",
}


def task_languages(task: Task) -> set[str]:
    langs: set[str] = set()
    for f in task.target_files or []:
        lang = _EXT_LANG.get(Path(f).suffix)
        if lang:
            langs.add(lang)
    return langs


def _with_language(checks: list[dict[str, str]]) -> list[dict[str, str]]:
    """Normalize checks by adding 'language' field if missing."""
    return [{**check, "language": check.get("language", "all")} for check in checks]


def _config_gate_checks(project_dir: str) -> list[dict[str, str]] | None:
    config_path = Path(project_dir) / ".splinter" / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            cfg: dict[str, Any] = yaml.safe_load(f) or {}
        checks: list[dict[str, str]] | None = cfg.get("gate_checks")
        if checks:
            return _with_language(checks)
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
                    return _with_language(loaded)
            except (json.JSONDecodeError, ValueError):
                pass
    return _config_gate_checks(project_dir)


def save_gate_checks(session_dir: str | Path, checks: list[dict[str, str]]) -> None:
    """Persist resolved gate checks for this session so run_gate picks them up."""
    Path(session_dir, GATE_FILE).write_text(json.dumps(checks, indent=2))


def parse_gate_spec(spec: str, language: str) -> list[dict[str, str]]:
    """Turn a free-form gate spec into checks.

    Accepts a JSON array, or a simple ``cmd1; cmd2`` / newline-separated list of
    shell commands. ``none``/``skip`` yields an empty gate (no mechanical checks).
    Each check is stamped with the provided language.
    """
    spec = spec.strip()
    if not spec or spec.lower() in ("none", "skip", "no gate"):
        return []
    try:
        data = json.loads(spec)
        if isinstance(data, list):
            return [
                {**c, "language": c.get("language", language)}
                for c in data
                if isinstance(c, dict) and c.get("cmd")
            ]
    except (json.JSONDecodeError, ValueError):
        pass
    checks: list[dict[str, str]] = []
    for cmd in re.split(r"[;\n]+", spec):
        cmd = cmd.strip()
        if cmd:
            checks.append(
                {
                    "name": cmd.split()[0],
                    "cmd": cmd,
                    "when": "always",
                    "language": language,
                }
            )
    return checks


def _load_gate_checks(
    project_dir: str = ".", session_dir: str | Path | None = None
) -> list[dict[str, str]]:
    configured = configured_gate_checks(project_dir, session_dir)
    # Never invent checks — if nothing is configured, return empty so run_gate
    # passes with a warning. The pipeline's _resolve_gate() is responsible for
    # populating gate.json before the run loop starts.
    return configured if configured is not None else []


_PROTO_GEN_PATTERNS = (
    "*.pb.go",
    "*_grpc.pb.go",
    "*.twirp.go",
    "wire_gen.go",
    "*_pb2.py",
    "*_pb2_grpc.py",
    "*.pb.py",
    "*.pb.js",
    "*.pb.ts",
    "*.grpc.pb.js",
    "*.grpc.pb.ts",
    "*_pb.rb",
    "*_services_pb.rb",
    "*.pb.cc",
    "*.pb.h",
    "*.grpc.pb.cc",
    "*.grpc.pb.h",
    "*.pb.swift",
    "*.grpc.swift",
    "*.g.cs",
    "*_grpc.php",
    "*.java",
)

_PROTO_GEN_RE = re.compile("|".join(fnmatch.translate(p) for p in _PROTO_GEN_PATTERNS))
_PROTO_PRUNE_DIRS = frozenset({"node_modules", ".git", "target", "dist"})
_RUST_GEN_DIRS = frozenset({"gen", "generated", "proto_gen"})


def _proto_changed(project_dir: str, memo: dict[tuple[str, float], bool]) -> bool:
    try:
        key = (project_dir, os.stat(project_dir).st_mtime)
    except OSError:
        return False
    cached = memo.get(key)
    if cached is not None:
        return cached
    newest_proto: float | None = None
    oldest_gen: float | None = None
    for dirpath, dirnames, filenames in os.walk(project_dir, topdown=True, followlinks=False):
        if dirpath != project_dir and os.path.basename(dirpath) == "build":
            dirnames[:] = [d for d in dirnames if d == "generated"]
        else:
            dirnames[:] = [d for d in dirnames if d not in _PROTO_PRUNE_DIRS]
        rel_parts = Path(os.path.relpath(dirpath, project_dir)).parts
        in_rust_gen = any(part in _RUST_GEN_DIRS for part in rel_parts)
        for name in filenames:
            is_proto = name.endswith(".proto")
            is_gen = not is_proto and (
                _PROTO_GEN_RE.match(name) is not None or (in_rust_gen and name.endswith(".rs"))
            )
            if not (is_proto or is_gen):
                continue
            try:
                mtime = os.stat(os.path.join(dirpath, name)).st_mtime
            except OSError:
                continue
            if is_proto:
                newest_proto = mtime if newest_proto is None else max(newest_proto, mtime)
            else:
                oldest_gen = mtime if oldest_gen is None else min(oldest_gen, mtime)
    if newest_proto is None:
        decision = False
    elif oldest_gen is None:
        decision = True
    else:
        decision = newest_proto > oldest_gen
    memo[key] = decision
    return decision


def _should_run(
    check: dict[str, str],
    project_dir: str,
    memo: dict[tuple[str, float], bool] | None = None,
) -> bool:
    when = check.get("when", "always")
    if when == "always":
        return True
    if when == "tests_exist":
        return _tests_exist(project_dir)
    if when == "proto_changed":
        return _proto_changed(project_dir, {} if memo is None else memo)
    return True


#: Test-file patterns per ecosystem — ``tests_exist`` is used by non-Python
#: presets too (ruby/java/swift/…), so detection must not be Python-only.
_TEST_FILE_PATTERNS = (
    "test_*.py",
    "*_test.py",
    "*_test.go",
    "*.test.ts",
    "*.test.tsx",
    "*.test.js",
    "*.spec.ts",
    "*.spec.js",
    "*_spec.rb",
    "*_test.rb",
    "*Test.java",
    "*Test.kt",
    "*Tests.swift",
    "*Test.cs",
    "*Tests.cs",
    "*Test.php",
    "*_test.rs",
)


def _tests_exist(project_dir: str) -> bool:
    root = Path(project_dir)
    for d in ("tests", "test", "spec", "src/test"):
        p = root / d
        if p.is_dir() and any(f.is_file() for f in p.rglob("*")):
            return True
    return any(
        f
        for pat in _TEST_FILE_PATTERNS
        for f in root.rglob(pat)
        if "node_modules" not in f.parts and "target" not in f.parts
    )


#: Seconds a gate check may run unless the check sets its own ``timeout``.
DEFAULT_CHECK_TIMEOUT = 120


def _check_timeout(check: dict[str, str]) -> int:
    try:
        value = int(check.get("timeout", DEFAULT_CHECK_TIMEOUT))
    except (TypeError, ValueError):
        return DEFAULT_CHECK_TIMEOUT
    return value if value > 0 else DEFAULT_CHECK_TIMEOUT


def _lang_match(check: dict[str, str], languages: set[str]) -> bool:
    if not languages:
        return True
    lang = check.get("language", "all")
    return lang == "all" or lang in languages


def run_gate(
    project_dir: str = ".",
    session_dir: str | Path | None = None,
    languages: set[str] | None = None,
) -> GateResult:
    checks = _load_gate_checks(project_dir, session_dir)
    active_langs = languages or set()
    results: list[tuple[str, bool, str]] = []
    all_passed = True
    proto_memo: dict[tuple[str, float], bool] = {}

    for check in checks:
        if not _should_run(check, project_dir, proto_memo):
            continue
        if not _lang_match(check, active_langs):
            continue
        name = check["name"]
        cmd = check["cmd"]
        try:
            # shell=True: gate cmds are "exact shell commands" (quoting, &&,
            # backtick substitution in the presets) — naive .split() breaks them.
            proc = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=_check_timeout(check),
                cwd=project_dir,
            )
            passed = proc.returncode == 0
            output = proc.stdout + proc.stderr
            if not passed:
                all_passed = False
            results.append((name, passed, output.strip()))
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
        '"when": "always" | "tests_exist" | "proto_changed", '
        '"language": "all" | "<language key e.g. python/go/rust/typescript>"}.\n'
        "Use the project's real tooling (e.g. npm/pnpm/yarn scripts, make targets, "
        'cargo, go, gradle, pytest, etc.). Use "when":"tests_exist" for the test '
        'command. Use "when":"proto_changed" for proto codegen commands (e.g. wire gen, protoc). '
        "If you cannot determine any, output []."
    )
    try:
        raw = run_text(
            prompt,
            ladder.localizer_precision_model,
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
            out.append(
                {
                    "name": str(item.get("name") or str(item["cmd"]).split()[0]),
                    "cmd": str(item["cmd"]),
                    "when": str(item.get("when") or "always"),
                    "language": str(item.get("language") or "all"),
                }
            )
    return out
