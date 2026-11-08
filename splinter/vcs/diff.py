"""The git diff of one story, behind an injectable seam.

The run screen shows the changes of the selected story. Git runs through a
``Runner`` so the tests never spawn a subprocess. The patch is plain ``git diff``
on purpose: ``rtk git diff`` compacts it into a summary the view cannot colour.
Files the runner created and git does not track yet are listed in the stat as
``new``, since ``git diff`` alone leaves them out.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

#: Runs one git command inside a directory and returns its stdout.
Runner = Callable[[list[str], Path], str]

_GIT_TIMEOUT = 30


class _StorySession(Protocol):
    """The session slice the diff view needs."""

    def read_worktrees(self) -> dict[str, dict[str, str]]: ...

    def read_run_live(self, task_no: int) -> dict[str, Any] | None: ...


@dataclass(frozen=True)
class StoryDiff:
    """The stat summary and the patch of one story."""

    stat: str
    patch: str
    error: str = ""


def git_runner(argv: list[str], cwd: Path) -> str:
    """Run one plain git command in ``cwd``; a failing git raises with its stderr."""
    cmd = ["git", "--no-pager", *argv]
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT,
        check=False,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip().splitlines()
        raise RuntimeError(detail[-1] if detail else f"git {argv[0]} failed in {cwd}")
    return proc.stdout


def story_diff(cwd: Path, run: Runner = git_runner) -> StoryDiff:
    """The ``--stat`` summary, new untracked files included, and the patch at ``cwd``."""
    try:
        stat = run(["diff", "--no-color", "--stat"], cwd)
        patch = run(["diff", "--no-color"], cwd)
        untracked = run(["ls-files", "--others", "--exclude-standard"], cwd)
    except Exception as exc:
        return StoryDiff(stat="", patch="", error=str(exc))
    new = [f" {path} | new" for path in untracked.splitlines() if path.strip()]
    if new:
        stat = "\n".join([*new, stat.rstrip()]).rstrip() + "\n"
    return StoryDiff(stat=stat, patch=patch)


def _worktree_keys(story: int, task_id: str) -> tuple[str, ...]:
    keys = (task_id, f"T{story}", f"US-{story:03d}", str(story))
    return tuple(dict.fromkeys(key for key in keys if key))


def story_worktree(session: _StorySession, story: int, task_id: str = "") -> Path | None:
    """The worktree a story runs in, or ``None`` when it runs in the main tree."""
    try:
        worktrees = session.read_worktrees()
    except Exception:
        worktrees = {}
    for key in _worktree_keys(story, task_id):
        path = (worktrees.get(key) or {}).get("path", "")
        if path and Path(path).exists():
            return Path(path)
    try:
        descriptor = session.read_run_live(story)
    except Exception:
        descriptor = None
    for key in ("worktree", "worktree_path", "cwd"):
        value = descriptor.get(key) if isinstance(descriptor, dict) else None
        if isinstance(value, str) and value and Path(value).exists():
            return Path(value)
    return None


def story_cwd(session: _StorySession, story: int, task_id: str = "") -> Path:
    """The tree a story works in: its worktree when it has one, else the main tree."""
    return story_worktree(session, story, task_id) or Path.cwd()


def shown_tree(path: Path) -> str:
    """``main tree`` for the cwd, a path relative to it when inside, else ``~/…``."""
    here = Path.cwd().resolve()
    target = path.resolve()
    if target == here:
        return "main tree"
    if target.is_relative_to(here):
        return str(target.relative_to(here))
    home = Path.home()
    return f"~/{target.relative_to(home)}" if target.is_relative_to(home) else str(target)
