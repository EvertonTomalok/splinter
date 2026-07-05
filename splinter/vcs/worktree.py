"""Git worktree capability detection and lifecycle management."""

from __future__ import annotations

import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

#: Process-wide guard so only ONE squash-merge (plus its conflict resolution) runs
#: at a time. Parallel workers each merge their task branch into the shared main
#: tree; a merge that stops to resolve conflicts leaves the index half-staged, so
#: a second merge starting concurrently would build on a poisoned index. Holding
#: this for the whole merge+resolve+commit keeps every merge atomic against the
#: others regardless of which caller invokes it.
_MERGE_SEMAPHORE = threading.Semaphore(1)


class WorktreeMergeConflict(Exception):
    """Raised when a squash-merge produces conflicts that need manual resolution."""


@dataclass(frozen=True)
class WorktreeHandle:
    path: Path
    branch: str
    task_id: str


def worktree_supported(cwd: Path | None = None) -> bool:
    """Return True only if cwd is inside a git repo AND `git worktree list` exits 0."""
    base = cwd or Path.cwd()
    if not (base / ".git").exists():
        return False
    try:
        result = subprocess.run(
            ["git", "worktree", "list"],
            cwd=base,
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, OSError, subprocess.SubprocessError, subprocess.TimeoutExpired):
        return False


def _worktree_path(task_id: str, base_dir: Path) -> Path:
    safe = task_id.replace("/", "_").replace("\\", "_")
    return base_dir / ".splinter" / "worktrees" / safe


def _branch_name(task_id: str) -> str:
    return f"splinter/{task_id}"


def create_worktree(task_id: str, base_dir: Path | None = None) -> WorktreeHandle:
    """Create a git worktree for task_id; return its handle."""
    cwd = base_dir or Path.cwd()
    path = _worktree_path(task_id, cwd)
    path.parent.mkdir(parents=True, exist_ok=True)
    branch = _branch_name(task_id)
    subprocess.run(
        ["git", "worktree", "add", str(path), "-b", branch],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return WorktreeHandle(path=path, branch=branch, task_id=task_id)


def commit_worktree(handle: WorktreeHandle, message: str = "") -> bool:
    """Stage and commit all changes in the worktree's own working dir.

    The coder edits files in the worktree tree but does not commit, so its work
    lives only as uncommitted changes on ``handle.branch``. ``squash_merge`` pulls
    committed history, so without this the branch is empty and the merge a no-op.
    Runs against the worktree (its own index — no race with the main repo index).
    Returns ``True`` if a commit was created, ``False`` when there was nothing to
    commit (a clean tree exits non-zero, which is expected, not an error).
    """
    msg = message or f"splinter: {handle.task_id} work"
    subprocess.run(
        ["git", "add", "-A"],
        cwd=handle.path,
        capture_output=True,
        text=True,
        check=True,
    )
    result = subprocess.run(
        ["git", "commit", "-m", msg],
        cwd=handle.path,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _unmerged_paths(cwd: Path) -> list[str]:
    """Paths left in a conflicted (unmerged) state in the index."""
    r = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=U"],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    return [p for p in r.stdout.splitlines() if p.strip()]


def branch_has_unmerged_commits(
    handle: WorktreeHandle, base_branch: str = "", base_dir: Path | None = None
) -> bool:
    """True if ``handle.branch`` has commits not on ``base_branch``.

    The merge decision must key on "does the branch carry work" — not on whether
    *this* invocation of ``commit_worktree`` created a fresh commit. On resume a
    task's work is already committed on its branch from the prior run: without
    this check the merge is skipped and the branch discarded on teardown,
    silently losing a PASSed task's output.
    """
    cwd = base_dir or Path.cwd()
    base = base_branch or "HEAD"
    r = subprocess.run(
        ["git", "rev-list", "--count", f"{base}..{handle.branch}"],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        return False
    return r.stdout.strip() not in ("", "0")


def squash_merge(
    handle: WorktreeHandle, base_branch: str = "", base_dir: Path | None = None
) -> None:
    """Squash-merge handle.branch into base_branch (default: current HEAD branch).

    Serialised process-wide by ``_MERGE_SEMAPHORE`` so no two merges interleave.
    On conflict the merge is **resolved, never abandoned**: the task branch was
    validated by eval in isolation, so its version wins on conflicting paths.
    Only a non-content failure (index locked, unconcluded merge) is rolled back
    and re-raised as ``WorktreeMergeConflict``.
    """
    cwd = base_dir or Path.cwd()
    if not base_branch:
        r = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        base_branch = r.stdout.strip()

    with _MERGE_SEMAPHORE:
        result = subprocess.run(
            ["git", "merge", "--squash", handle.branch],
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            conflicted = _unmerged_paths(cwd)
            if not conflicted:
                # Not a content conflict (dirty/locked index, unconcluded merge):
                # restore the tree and surface it — nothing to auto-resolve.
                subprocess.run(
                    ["git", "reset", "--hard", "HEAD"], cwd=cwd, capture_output=True, text=True
                )
                raise WorktreeMergeConflict(
                    f"squash merge of {handle.branch!r} into {base_branch!r} failed:\n"
                    f"{result.stderr.strip()}"
                )
            # Resolve toward the task branch ("theirs" in a squash merge): each
            # task PASSed against its own tree, so that tree is the validated one.
            for path in conflicted:
                co = subprocess.run(
                    ["git", "checkout", "--theirs", "--", path],
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                )
                if co.returncode == 0:
                    subprocess.run(
                        ["git", "add", "--", path], cwd=cwd, capture_output=True, text=True
                    )
                else:
                    # modify/delete conflict: the task removed the file — honour that.
                    subprocess.run(
                        ["git", "rm", "--force", "--", path],
                        cwd=cwd,
                        capture_output=True,
                        text=True,
                    )
            still = _unmerged_paths(cwd)
            if still:
                subprocess.run(
                    ["git", "reset", "--hard", "HEAD"], cwd=cwd, capture_output=True, text=True
                )
                raise WorktreeMergeConflict(
                    f"squash merge of {handle.branch!r} into {base_branch!r} left "
                    f"unresolved paths: {', '.join(still)}"
                )

        # Commit only if the merge actually staged something — an ahead branch whose
        # diff is already in base squashes to an empty index (no-op, not an error).
        if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=cwd).returncode != 0:
            subprocess.run(
                ["git", "commit", "-m", f"squash: {handle.task_id} results"],
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True,
            )


def teardown_worktree(handle: WorktreeHandle, base_dir: Path | None = None) -> None:
    """Remove the worktree directory and delete its branch."""
    cwd = base_dir or Path.cwd()
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(handle.path)],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "branch", "-D", handle.branch],
        cwd=cwd,
        capture_output=True,
        text=True,
    )


def reattach_worktree(task_id: str, base_dir: Path | None = None) -> WorktreeHandle | None:
    """Return a handle for an existing worktree (pause/resume), or None if absent."""
    cwd = base_dir or Path.cwd()
    path = _worktree_path(task_id, cwd)
    if not path.exists():
        return None
    branch = _branch_name(task_id)
    try:
        result = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    if str(path) in result.stdout:
        return WorktreeHandle(path=path, branch=branch, task_id=task_id)
    return None
