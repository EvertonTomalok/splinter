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


_DEP_DIR = "node_modules"
_DEP_DEPTH = 4


def _workspace_dirs(wt_dir: Path, depth: int = _DEP_DEPTH) -> list[Path]:
    """Worktree dirs (relative, root included) up to ``depth`` levels, hidden ones pruned."""
    found: list[Path] = [Path()]
    frontier: list[Path] = [Path()]
    for _ in range(depth):
        nxt: list[Path] = []
        for rel in frontier:
            try:
                children = sorted((wt_dir / rel).iterdir())
            except OSError:
                continue
            nxt.extend(
                rel / c.name
                for c in children
                if c.is_dir()
                and not c.is_symlink()
                and not c.name.startswith(".")
                and c.name != _DEP_DIR
            )
        found.extend(nxt)
        frontier = nxt
    return found


def _exclude_dep_links(main_dir: Path) -> None:
    """Keep the ``node_modules`` symlinks out of ``git add -A``.

    A ``node_modules/`` gitignore pattern matches directories only; a symlink is
    a file to git, so without this every brief commit would carry the link.
    """
    git_dir = main_dir / ".git"
    if not git_dir.is_dir():
        return
    exclude = git_dir / "info" / "exclude"
    current = exclude.read_text() if exclude.exists() else ""
    if _DEP_DIR in current.split():
        return
    exclude.parent.mkdir(parents=True, exist_ok=True)
    sep = "" if not current or current.endswith("\n") else "\n"
    exclude.write_text(f"{current}{sep}{_DEP_DIR}\n")


def _link_deps_from_main(main_dir: Path, wt_dir: Path) -> None:
    """Symlink dependency directories from the main repo into the worktree.

    Git worktrees share tracked files but not gitignored directories like
    ``node_modules``. Without them, gate commands (``yarn lint``, ``tsc``,
    ``vitest``) fail. Every workspace up to four levels deep (``apps/web``,
    ``web/apps/hub``, ``mocks/asaas``) gets its own link to the main repo's
    already-installed dependencies.
    """
    linked = False
    for rel in _workspace_dirs(wt_dir):
        src = main_dir / rel / _DEP_DIR
        dst = wt_dir / rel / _DEP_DIR
        if src.is_dir() and not dst.exists() and not dst.is_symlink():
            dst.symlink_to(src, target_is_directory=True)
            linked = True
    if linked:
        _exclude_dep_links(main_dir)


def create_worktree(
    task_id: str, base_dir: Path | None = None, base_ref: str | None = None
) -> WorktreeHandle:
    """Create a git worktree for task_id; return its handle.

    ``base_ref`` is the start-point the new branch forks from — the integration
    branch for brief worktrees; default HEAD (the previous behaviour).
    """
    cwd = base_dir or Path.cwd()
    path = _worktree_path(task_id, cwd)
    path.parent.mkdir(parents=True, exist_ok=True)
    branch = _branch_name(task_id)
    cmd = ["git", "worktree", "add", str(path), "-b", branch]
    if base_ref:
        cmd.append(base_ref)
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0 and "already exists" in result.stderr:
        result = subprocess.run(
            ["git", "worktree", "add", str(path), branch],
            cwd=cwd,
            capture_output=True,
            text=True,
        )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, output=result.stdout, stderr=result.stderr
        )
    _link_deps_from_main(cwd, path)
    return WorktreeHandle(path=path, branch=branch, task_id=task_id)


def create_integration_worktree(
    slug: str, base_dir: Path | None = None, base_ref: str | None = None
) -> WorktreeHandle:
    """Create the run's integration worktree (branch ``splinter/<slug>-int``).

    Brief worktrees fork from this branch and merge back into it with ``--no-ff``,
    so main history only ever sees one merge per run.
    """
    return create_worktree(f"{slug}-int", base_dir=base_dir, base_ref=base_ref)


def merge_no_ff(
    handle: WorktreeHandle, target_dir: Path, message: str = ""
) -> bool:
    """Merge ``handle.branch`` into the branch checked out at ``target_dir`` with --no-ff.

    Serialised process-wide by ``_MERGE_SEMAPHORE``. Content conflicts are
    resolved toward the brief branch (it passed acceptance in isolation); a
    non-content failure rolls the tree back and raises
    :class:`WorktreeMergeConflict`. Returns ``False`` when the branch carries
    nothing to merge.
    """
    if not branch_has_unmerged_commits(handle, base_dir=target_dir):
        return False
    msg = message or f"merge: {handle.task_id}"
    with _MERGE_SEMAPHORE:
        result = subprocess.run(
            ["git", "merge", "--no-ff", handle.branch, "-m", msg],
            cwd=target_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            conflicted = _unmerged_paths(target_dir)
            if not conflicted:
                subprocess.run(
                    ["git", "merge", "--abort"],
                    cwd=target_dir,
                    capture_output=True,
                    text=True,
                )
                raise WorktreeMergeConflict(
                    f"no-ff merge of {handle.branch!r} into {target_dir} failed:\n"
                    f"{result.stderr.strip()}"
                )
            for path in conflicted:
                co = subprocess.run(
                    ["git", "checkout", "--theirs", "--", path],
                    cwd=target_dir,
                    capture_output=True,
                    text=True,
                )
                if co.returncode == 0:
                    subprocess.run(
                        ["git", "add", "--", path],
                        cwd=target_dir,
                        capture_output=True,
                        text=True,
                    )
                else:
                    subprocess.run(
                        ["git", "rm", "--force", "--", path],
                        cwd=target_dir,
                        capture_output=True,
                        text=True,
                    )
            still = _unmerged_paths(target_dir)
            if still:
                subprocess.run(
                    ["git", "reset", "--hard", "HEAD"],
                    cwd=target_dir,
                    capture_output=True,
                    text=True,
                )
                raise WorktreeMergeConflict(
                    f"no-ff merge of {handle.branch!r} left unresolved paths: "
                    f"{', '.join(still)}"
                )
            commit = subprocess.run(
                ["git", "commit", "--no-edit"],
                cwd=target_dir,
                capture_output=True,
                text=True,
            )
            if commit.returncode != 0:
                subprocess.run(
                    ["git", "reset", "--hard", "HEAD"],
                    cwd=target_dir,
                    capture_output=True,
                    text=True,
                )
                raise WorktreeMergeConflict(
                    f"no-ff merge of {handle.branch!r} could not be committed:\n"
                    f"{commit.stderr.strip()}"
                )
    return True


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
