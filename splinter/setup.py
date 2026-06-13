from __future__ import annotations

import shutil
import subprocess
import sys
from typing import Callable

from splinter.models.roster import load_ladder
from splinter.providers import claude_cli, codex, cursor, opencode


def _check(label: str, fn: Callable[[], tuple[bool, str]]) -> tuple[bool, str]:
    try:
        ok, detail = fn()
        status = "OK" if ok else "FALHA"
        msg = f"  {label} {'.' * max(1, 30 - len(label))} {status}"
        if detail:
            msg += f" ({detail})"
        print(msg)
        return ok, ""
    except Exception as e:
        print(f"  {label} {'.' * max(1, 30 - len(label))} FALHA ({e})")
        return False, str(e)


def _check_cursor() -> tuple[bool, str]:
    path = shutil.which("agent")
    if not path:
        return False, "cursor agent CLI not found in PATH (install Cursor and run 'agent login')"
    ok = cursor.ping()
    if not ok:
        return False, "agent -p did not respond (try 'agent login')"
    return True, path


def _check_codex() -> tuple[bool, str]:
    path = shutil.which("codex")
    if not path:
        return False, "codex not found in PATH"
    ok = codex.ping()
    if not ok:
        return False, "codex exec did not respond"
    return True, path


def _check_claude() -> tuple[bool, str]:
    path = shutil.which("claude")
    if not path:
        return False, "claude not found in PATH"
    ok = claude_cli.ping()
    if not ok:
        return False, "claude -p did not respond"
    return True, path


def _check_opencode() -> tuple[bool, str]:
    path = shutil.which("opencode")
    if not path:
        return False, "opencode not found in PATH"
    models = opencode.list_models()
    oc_models = [m for m in models if m.startswith("opencode-go/")]
    if not oc_models:
        return False, "no opencode-go/* models found"
    return True, f"{len(oc_models)} models"


def _check_ladder() -> tuple[bool, str]:
    ladder = load_ladder()
    try:
        available = set(opencode.list_models())
    except Exception:
        return False, "could not list opencode models"
    missing: list[str] = []
    for mid in ladder.opencode_model_ids():
        if mid not in available:
            missing.append(mid)
    if missing:
        return False, f"missing: {', '.join(missing)}"
    return True, f"{len(ladder.opencode_model_ids())} models verified"


def _check_python() -> tuple[bool, str]:
    proc = subprocess.run([sys.executable, "--version"], capture_output=True, text=True, timeout=10)
    if proc.returncode != 0:
        return False, "python not responding"
    ver = proc.stdout.strip()
    return True, ver


def _check_rtk() -> tuple[bool, str]:
    path = shutil.which("rtk")
    if not path:
        return False, "rtk not found in PATH"
    proc = subprocess.run(["rtk", "--version"], capture_output=True, text=True, timeout=10)
    ver = proc.stdout.strip()
    if "rust type kit" in ver.lower():
        return False, "wrong rtk (Rust Type Kit, need Rust Token Killer)"
    proc2 = subprocess.run(["rtk", "gain"], capture_output=True, text=True, timeout=10)
    if proc2.returncode != 0 and proc2.returncode != 1:
        return False, "rtk gain not working"
    return True, ver


def run_setup() -> int:
    print("checking providers...")
    results: list[bool] = []

    ok, _ = _check("cursor agent CLI", _check_cursor)
    results.append(ok)

    ok, _ = _check("codex exec (gpt-5-codex)", _check_codex)
    results.append(ok)

    ok, _ = _check("claude -p (sonnet)", _check_claude)
    results.append(ok)

    ok, _ = _check("opencode models", _check_opencode)
    results.append(ok)

    ok, _ = _check("ladder vs roster", _check_ladder)
    results.append(ok)

    ok, _ = _check("python (uv run)", _check_python)
    results.append(ok)

    ok, _ = _check("rtk", _check_rtk)
    results.append(ok)

    if all(results):
        print("environment ready.")
        return 0
    else:
        print("environment has issues — fix the FALHA items above.")
        return 1
