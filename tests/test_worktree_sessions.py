from __future__ import annotations

import fcntl
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from splinter.tui_run_iteration import _WORKTREE
from splinter.vcs import worktree
from splinter.vcs.worktree import (
    WorktreeHandle,
    create_integration_worktree,
    create_worktree,
    reattach_worktree,
    squash_merge,
)


@pytest.fixture
def git_calls(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def run(cmd: list[str], **_kw: Any) -> SimpleNamespace:
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="main\n", stderr="")

    monkeypatch.setattr(worktree.subprocess, "run", run)
    return calls


def test_two_sessions_never_share_a_worktree_or_branch(
    tmp_path: Path, git_calls: list[list[str]]
) -> None:
    first = create_worktree("US-001", base_dir=tmp_path, session_id="ses_a")
    second = create_worktree("US-001", base_dir=tmp_path, session_id="ses_b")
    assert first.branch == "splinter/wt/ses_a/US-001"
    assert first.path == tmp_path / ".splinter" / "worktrees" / "ses_a" / "US-001"
    assert second.branch == "splinter/wt/ses_b/US-001"
    assert first.path != second.path


def test_integration_worktree_belongs_to_its_session(
    tmp_path: Path, git_calls: list[list[str]]
) -> None:
    handle = create_integration_worktree("run", base_dir=tmp_path, session_id="ses_a")
    assert handle.branch == "splinter/wt/ses_a/run-int"


def test_without_a_session_the_old_layout_stays(
    tmp_path: Path, git_calls: list[list[str]]
) -> None:
    handle = create_worktree("US-001", base_dir=tmp_path)
    assert handle.branch == "splinter/US-001"
    assert handle.path == tmp_path / ".splinter" / "worktrees" / "US-001"


def _listed(monkeypatch: pytest.MonkeyPatch, *paths: Path) -> None:
    stdout = "".join(f"worktree {path}\n" for path in paths)
    monkeypatch.setattr(
        worktree.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(returncode=0, stdout=stdout, stderr=""),
    )


def test_reattach_prefers_the_session_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / ".splinter" / "worktrees"
    scoped, legacy = root / "ses_a" / "US-001", root / "US-001"
    scoped.mkdir(parents=True)
    legacy.mkdir(parents=True)
    _listed(monkeypatch, scoped, legacy)
    handle = reattach_worktree("US-001", base_dir=tmp_path, session_id="ses_a")
    assert handle is not None and handle.path == scoped
    assert handle.branch == "splinter/wt/ses_a/US-001"


def test_reattach_falls_back_to_the_old_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy = tmp_path / ".splinter" / "worktrees" / "US-001"
    legacy.mkdir(parents=True)
    _listed(monkeypatch, legacy)
    handle = reattach_worktree("US-001", base_dir=tmp_path, session_id="ses_a")
    assert handle is not None and handle.path == legacy
    assert handle.branch == "splinter/US-001"


def test_merge_into_the_main_checkout_locks_other_processes_out(
    tmp_path: Path, git_calls: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".git").mkdir()
    locks: list[int] = []
    monkeypatch.setattr(worktree.fcntl, "flock", lambda _file, op: locks.append(op))
    handle = WorktreeHandle(tmp_path / "wt", "splinter/wt/ses_a/US-001", "US-001")
    squash_merge(handle, base_dir=tmp_path)
    assert locks == [fcntl.LOCK_EX, fcntl.LOCK_UN]


@pytest.mark.parametrize(
    "line",
    [
        "/repo/.splinter/worktrees/US-009/src/a.py",
        "/repo/.splinter/worktrees/ses_a/US-009/src/a.py",
    ],
)
def test_activity_paths_lose_the_worktree_prefix_in_both_layouts(line: str) -> None:
    assert _WORKTREE.sub("", line) == "src/a.py"
