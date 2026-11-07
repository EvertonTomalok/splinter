"""Run guard: PATH shims that keep writes, builds and tests out of workhorse agents.

Port of the agent-engine run guard (guard.sh + guardedNames + permission deny
lists): the workhorse edits files; the orchestrator builds, tests and commits.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

GUARDED_NAMES: tuple[str, ...] = (
    "git", "go", "make", "pnpm", "npm", "npx", "yarn", "vitest", "tsc", "eslint", "next",
    "golangci-lint", "uv", "pytest", "mypy", "ruff", "pip", "docker", "docker-compose",
    "colima", "gh", "terraform", "claude", "claude-credit", "opencode", "agy", "mcode",
    "codex", "kill", "pkill", "killall", "brew",
)

GUARD_SH = (
    "#!/bin/sh\n"
    "# The run guard behind every shim in $RUN/shims: agents of any engine get this on PATH\n"
    "# first, so builds, tests, package managers, containers, git writes and nested agents stay\n"
    "# the orchestrator's. Reading commands pass through to the real binary.\n"
    "# Spawned by agent-engine init; identical in behavior to the plan-with-agents kit guard.\n"
    'name=$(/usr/bin/basename "$0")\n'
    'SHIMS=$(/usr/bin/dirname "$0")\n'
    "deny() {\n"
    '  echo "BLOCKED by the run guard: $name $* — the orchestrator builds, tests, generates '
    'and commits after you. Edit files and finish." >&2\n'
    "  exit 126\n"
    "}\n"
    'case "$name" in\n'
    "  git)\n"
    '    case "${1:-}" in\n'
    "      status|diff|log|show|blame|ls-files|rev-parse|grep|cat-file) ;;\n"
    '      *) deny "$@" ;;\n'
    "    esac ;;\n"
    "  go)\n"
    '    case "${1:-}" in\n'
    "      doc|env|version|list|fmt) ;;\n"
    '      *) deny "$@" ;;\n'
    "    esac ;;\n"
    '  *) deny "$@" ;;\n'
    "esac\n"
    "PATH=$(echo \"$PATH\" | /usr/bin/tr ':' '\\n' | /usr/bin/grep -vxF \"$SHIMS\" "
    "| /usr/bin/paste -sd: -)\n"
    "export PATH\n"
    'exec "$name" "$@"\n'
)


@dataclass(frozen=True)
class GuardEnv:
    run_dir: Path
    guard_script: Path
    shims_dir: Path


def _replace_symlink(target: Path, path: Path) -> None:
    if path.is_symlink() and os.readlink(path) == str(target):
        return
    path.unlink(missing_ok=True)
    os.symlink(target, path)


def setup_guard(run_dir: Path) -> GuardEnv:
    """Materialize guard.sh (0755) and one shim symlink per guarded name.

    Idempotent: an existing shim pointing at the current guard script is kept;
    anything else (stale link, regular file) is replaced.
    """
    guard_script = run_dir / "guard.sh"
    shims_dir = run_dir / "shims"
    run_dir.mkdir(parents=True, exist_ok=True)
    shims_dir.mkdir(parents=True, exist_ok=True)
    guard_script.write_text(GUARD_SH, encoding="utf-8")
    guard_script.chmod(0o755)
    target = guard_script if guard_script.is_absolute() else guard_script.resolve()
    for name in GUARDED_NAMES:
        _replace_symlink(target, shims_dir / name)
    return GuardEnv(run_dir=run_dir, guard_script=guard_script, shims_dir=shims_dir)


def guarded_env(
    guard: GuardEnv,
    *,
    base: Mapping[str, str] | None = None,
    unset: Iterable[str] = (),
    set_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Environment for a guarded agent process: shims first on PATH.

    Keys in ``unset`` and ``set_env`` are dropped from the base (or
    ``os.environ``) before ``set_env`` is applied, so overrides win. ``PATH`` is
    rebuilt with the shims dir first; ``PWD`` stays out — Popen ``cwd`` wins.
    """
    source: Mapping[str, str] = os.environ if base is None else base
    original_path = source.get("PATH", "")
    drop = set(unset) | set(set_env or ()) | {"PATH", "PWD"}
    env = {key: value for key, value in source.items() if key not in drop}
    if set_env:
        env.update(set_env)
    env.pop("PWD", None)
    if original_path:
        env["PATH"] = str(guard.shims_dir) + os.pathsep + original_path
    else:
        env["PATH"] = str(guard.shims_dir)
    env["SPLINTER_LAUNCH_PATH"] = original_path
    return env


def reviewer_deny() -> list[str]:
    """Claude ``--disallowedTools`` deny list for reviewer agents."""
    return [
        "Bash(git push:*)", "Bash(git fetch:*)", "Bash(git pull:*)", "Bash(git rebase:*)",
        "Bash(git merge:*)", "Bash(git stash:*)", "Bash(git worktree:*)", "Bash(git branch:*)",
        "Bash(gh:*)", "Bash(docker:*)", "Bash(pkill:*)", "Bash(kill:*)", "Bash(claude:*)",
        "Bash(claude-credit:*)", "Bash(opencode:*)", "Bash(agy:*)", "Bash(mcode:*)",
        "Bash(codex:*)", "Agent", "Task",
    ]


def workhorse_deny() -> list[str]:
    """Claude ``--disallowedTools`` deny list for workhorse agents."""
    return reviewer_deny() + [
        "Bash(git commit:*)", "Bash(git add:*)", "Bash(git reset:*)", "Bash(git clean:*)",
        "Bash(git checkout:*)", "Bash(git switch:*)", "Bash(go test:*)", "Bash(go build:*)",
        "Bash(go vet:*)", "Bash(golangci-lint:*)", "Bash(go tool:*)", "Bash(make:*)",
        "Bash(pnpm:*)", "Bash(npx:*)", "Bash(vitest:*)", "Bash(tsc:*)", "Bash(uv run:*)",
        "Bash(pytest:*)",
    ]


def opencode_permission_json() -> str:
    """Opencode permission policy: single-line ordered JSON, last rule wins."""
    return (
        '{"permission":{"bash":{"*":"allow",'
        '"git push*":"deny","git fetch*":"deny","git pull*":"deny","git rebase*":"deny",'
        '"git merge*":"deny","git commit*":"deny","git add*":"deny","git stash*":"deny",'
        '"git worktree*":"deny","git reset*":"deny","git clean*":"deny","git branch*":"deny",'
        '"git checkout*":"deny","git switch*":"deny","gh *":"deny","go test*":"deny",'
        '"go build*":"deny","go vet*":"deny","go tool*":"deny","golangci-lint*":"deny",'
        '"make *":"deny","pnpm *":"deny","npx *":"deny","vitest*":"deny","tsc*":"deny",'
        '"uv run*":"deny","pytest*":"deny","docker *":"deny","pkill*":"deny",'
        '"kill *":"deny","claude*":"deny","opencode*":"deny","agy*":"deny",'
        '"rm -rf /*":"deny","rm -rf ~*":"deny"}}}'
    )
