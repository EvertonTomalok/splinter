"""Run the project's generators in a brief worktree, outside the run guard.

The workhorse contract forbids generators and the guard refuses them, so the
orchestrator runs them itself after the workhorse edits and before acceptance:
a brief that changes a ``.proto`` gets its stubs here, never from the engine.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from splinter.configure import CodegenStep

_OUTPUT_LIMIT = 1_500


@dataclass(frozen=True)
class CodegenResult:
    ran: tuple[str, ...] = ()
    failure: str = ""

    @property
    def ok(self) -> bool:
        return not self.failure


def changed_files(worktree: Path) -> list[str]:
    out = subprocess.run(
        ["git", "status", "--porcelain", "-z", "--untracked-files=all"],
        cwd=worktree, capture_output=True, text=True, check=False,
    ).stdout
    found: list[str] = []
    fields = iter(out.split("\0"))
    for entry in fields:
        if len(entry) <= 3:
            continue
        found.append(entry[3:])
        if {"R", "C"} & set(entry[:2]):
            next(fields, None)
    return found


def run_codegen(worktree: Path, steps: tuple[CodegenStep, ...]) -> CodegenResult:
    """Run each step whose paths the change touched; stop at the first failure."""
    if not steps:
        return CodegenResult()
    changed = changed_files(worktree)
    ran: list[str] = []
    for step in steps:
        if step.paths and not any(f.startswith(step.paths) for f in changed):
            continue
        try:
            proc = subprocess.run(
                step.cmd, shell=True, cwd=worktree, capture_output=True, text=True,
                timeout=step.timeout, check=False,
            )
        except subprocess.TimeoutExpired:
            return CodegenResult(tuple(ran), f"{step.cmd}: timed out after {step.timeout}s")
        ran.append(step.cmd)
        if proc.returncode != 0:
            output = (proc.stderr or proc.stdout).strip()[-_OUTPUT_LIMIT:]
            return CodegenResult(
                tuple(ran), f"{step.cmd} exited {proc.returncode}: {output}"
            )
    return CodegenResult(tuple(ran))
