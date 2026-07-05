"""Regression: parallel squash-merges must resolve conflicts, never abandon work.

The bug: when two parallel tasks edited the same file, the second task's
squash-merge conflicted, the code re-raised and abandoned it, and on resume the
already-committed branch had "nothing new to commit" so it was skipped and torn
down — a PASSed task's output silently vanished from master.

These use REAL git repos (not mocks) because the defect lives in the exact git
state transitions: conflict staging, resolution, and ahead-of-base detection.
"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path

import pytest

from splinter.vcs.worktree import (
    WorktreeHandle,
    WorktreeMergeConflict,
    branch_has_unmerged_commits,
    squash_merge,
)


def _git(cwd: Path, *args: str) -> str:
    r = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )
    return r.stdout.strip()


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "master")
    _git(repo, "config", "user.email", "t@t.com")
    _git(repo, "config", "user.name", "t")
    (repo / "f.py").write_text("base\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "base")
    return repo


def _branch_with_change(repo: Path, branch: str, content: str) -> WorktreeHandle:
    """Cut a branch off current master, change f.py, commit, return to master."""
    _git(repo, "checkout", "-b", branch)
    (repo / "f.py").write_text(content)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", f"{branch} work")
    _git(repo, "checkout", "master")
    return WorktreeHandle(path=repo, branch=branch, task_id=branch)


def test_conflicting_merge_is_resolved_not_raised(tmp_path: Path) -> None:
    """Second, conflicting task branch is auto-resolved toward its own version."""
    repo = _init_repo(tmp_path)
    # Both branches edit the SAME line off the same base → guaranteed conflict.
    a = _branch_with_change(repo, "splinter/A", "from-A\n")
    b = _branch_with_change(repo, "splinter/B", "from-B\n")

    squash_merge(a, base_branch="master", base_dir=repo)  # clean
    squash_merge(b, base_branch="master", base_dir=repo)  # would conflict

    # Resolved toward B (the task branch just merged), tree clean, both committed.
    assert (repo / "f.py").read_text() == "from-B\n"
    assert _git(repo, "status", "--porcelain") == ""
    assert not _git(repo, "diff", "--name-only", "--diff-filter=U")
    subjects = _git(repo, "log", "--format=%s").splitlines()
    assert "squash: splinter/A results" in subjects
    assert "squash: splinter/B results" in subjects


def test_resume_merges_already_committed_branch(tmp_path: Path) -> None:
    """A branch committed on a prior run still merges — no fresh commit required.

    Reproduces the resume path: the worktree is clean (work already committed),
    so ``commit_worktree`` would no-op; the merge must key on the branch being
    ahead of base, or the work is lost.
    """
    repo = _init_repo(tmp_path)
    b = _branch_with_change(repo, "splinter/US-003", "US-003-impl\n")

    assert branch_has_unmerged_commits(b, base_branch="master", base_dir=repo) is True
    squash_merge(b, base_branch="master", base_dir=repo)

    assert (repo / "f.py").read_text() == "US-003-impl\n"
    assert "squash: splinter/US-003 results" in _git(repo, "log", "--format=%s")


def test_branch_has_unmerged_commits_false_when_equal(tmp_path: Path) -> None:
    """No commits ahead of base → nothing to merge (guards the no-op path)."""
    repo = _init_repo(tmp_path)
    _git(repo, "branch", "splinter/empty")  # points at master, 0 ahead
    handle = WorktreeHandle(path=repo, branch="splinter/empty", task_id="empty")
    assert branch_has_unmerged_commits(handle, base_branch="master", base_dir=repo) is False


def test_no_op_merge_of_ahead_branch_already_in_base(tmp_path: Path) -> None:
    """Ahead branch whose diff is already in base squashes empty — no error."""
    repo = _init_repo(tmp_path)
    b = _branch_with_change(repo, "splinter/dup", "dup\n")
    squash_merge(b, base_branch="master", base_dir=repo)
    head_before = _git(repo, "rev-parse", "HEAD")
    # Merging the same branch again: it is still "ahead" (its commit isn't on
    # master by sha) but its content is identical → staged diff empty → no commit.
    squash_merge(b, base_branch="master", base_dir=repo)
    assert _git(repo, "rev-parse", "HEAD") == head_before
    assert _git(repo, "status", "--porcelain") == ""


def test_concurrent_merges_serialized_by_semaphore(tmp_path: Path) -> None:
    """Two threads merging conflicting branches at once must not corrupt the repo.

    The semaphore forces one merge (plus its resolution) to finish before the
    next starts; both land, the tree ends clean, no unmerged paths leak.
    """
    repo = _init_repo(tmp_path)
    a = _branch_with_change(repo, "splinter/T1", "T1\n")
    b = _branch_with_change(repo, "splinter/T2", "T2\n")

    errors: list[Exception] = []
    barrier = threading.Barrier(2)

    def _merge(h: WorktreeHandle) -> None:
        try:
            barrier.wait()
            squash_merge(h, base_branch="master", base_dir=repo)
        except Exception as exc:  # noqa: BLE001 — collect for assertion
            errors.append(exc)

    t1 = threading.Thread(target=_merge, args=(a,))
    t2 = threading.Thread(target=_merge, args=(b,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert errors == []
    assert _git(repo, "status", "--porcelain") == ""
    assert not _git(repo, "diff", "--name-only", "--diff-filter=U")
    subjects = _git(repo, "log", "--format=%s")
    assert "squash: splinter/T1 results" in subjects
    assert "squash: splinter/T2 results" in subjects


def test_non_content_failure_still_raises(tmp_path: Path) -> None:
    """A merge that fails with no conflicted paths (bad ref) is surfaced, not hidden."""
    repo = _init_repo(tmp_path)
    bad = WorktreeHandle(path=repo, branch="splinter/does-not-exist", task_id="x")
    with pytest.raises(WorktreeMergeConflict):
        squash_merge(bad, base_branch="master", base_dir=repo)
    # Repo left clean after the rollback.
    assert _git(repo, "status", "--porcelain") == ""
