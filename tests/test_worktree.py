"""Tests for VCS worktree capability detection and lifecycle (US-001, US-004)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from splinter.vcs.worktree import (
    WorktreeHandle,
    WorktreeMergeConflict,
    commit_worktree,
    create_worktree,
    reattach_worktree,
    squash_merge,
    teardown_worktree,
    worktree_supported,
)


class TestWorktreeSupported:
    def test_repo_present_returns_true(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert worktree_supported(cwd=tmp_path) is True

    def test_no_git_dir_returns_false(self, tmp_path: Path) -> None:
        assert worktree_supported(cwd=tmp_path) is False

    def test_git_binary_missing_returns_false(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        with patch("subprocess.run", side_effect=FileNotFoundError()):
            assert worktree_supported(cwd=tmp_path) is False

    def test_git_command_fails_returns_false(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=128)
            assert worktree_supported(cwd=tmp_path) is False

    def test_oserror_returns_false(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        with patch("subprocess.run", side_effect=OSError()):
            assert worktree_supported(cwd=tmp_path) is False

    def test_subprocess_error_returns_false(self, tmp_path: Path) -> None:
        import subprocess

        (tmp_path / ".git").mkdir()
        with patch("subprocess.run", side_effect=subprocess.SubprocessError()):
            assert worktree_supported(cwd=tmp_path) is False


class TestWorktreeLifecycle:
    def test_create_worktree_calls_git(self, tmp_path: Path) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            handle = create_worktree("US-001", base_dir=tmp_path)
        assert handle.task_id == "US-001"
        assert handle.branch == "splinter/US-001"
        assert "US-001" in str(handle.path)
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "worktree" in cmd
        assert "add" in cmd

    def test_teardown_worktree_calls_remove_and_delete(self, tmp_path: Path) -> None:
        handle = WorktreeHandle(
            path=tmp_path / ".splinter" / "worktrees" / "US-001",
            branch="splinter/US-001",
            task_id="US-001",
        )
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            teardown_worktree(handle, base_dir=tmp_path)
        assert mock_run.call_count == 2
        cmds = [mock_run.call_args_list[i][0][0] for i in range(2)]
        assert any("remove" in c for c in cmds)
        assert any("-D" in c or "branch" in c for c in cmds)

    def test_squash_merge_success(self, tmp_path: Path) -> None:
        handle = WorktreeHandle(
            path=tmp_path / "wt",
            branch="splinter/US-001",
            task_id="US-001",
        )

        def fake_run(cmd: list[str], **kw: object) -> MagicMock:
            # `git diff --cached --quiet` exits non-zero when something is staged,
            # which is what gates the commit — model a clean merge with changes.
            rc = 1 if ("diff" in cmd and "--quiet" in cmd) else 0
            return MagicMock(returncode=rc, stdout="main\n", stderr="")

        with patch("subprocess.run", side_effect=fake_run) as mock_run:
            squash_merge(handle, base_branch="main", base_dir=tmp_path)
        cmds = [c[0][0] for c in mock_run.call_args_list]
        assert ["--squash" in c for c in cmds].count(True) == 1
        assert any("commit" in c for c in cmds)  # staged merge is committed

    def test_commit_worktree_stages_and_commits_returns_true(self, tmp_path: Path) -> None:
        handle = WorktreeHandle(path=tmp_path / "wt", branch="splinter/US-001", task_id="US-001")
        with patch("subprocess.run") as mock_run:
            # git add → 0, git commit → 0 (a commit was created)
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            created = commit_worktree(handle)
        assert created is True
        assert mock_run.call_count == 2
        add_cmd = mock_run.call_args_list[0][0][0]
        commit_cmd = mock_run.call_args_list[1][0][0]
        assert "add" in add_cmd
        assert "commit" in commit_cmd
        # both ops run inside the worktree, not the main repo
        assert mock_run.call_args_list[0][1]["cwd"] == handle.path
        assert mock_run.call_args_list[1][1]["cwd"] == handle.path

    def test_commit_worktree_clean_tree_returns_false(self, tmp_path: Path) -> None:
        handle = WorktreeHandle(path=tmp_path / "wt", branch="splinter/US-001", task_id="US-001")

        def fake_run(cmd: list[str], **kw: object) -> MagicMock:
            # git commit on a clean tree exits non-zero ("nothing to commit").
            rc = 1 if "commit" in cmd else 0
            return MagicMock(returncode=rc, stdout="", stderr="nothing to commit")

        with patch("subprocess.run", side_effect=fake_run):
            created = commit_worktree(handle)
        assert created is False

    def test_squash_merge_conflict_raises(self, tmp_path: Path) -> None:
        handle = WorktreeHandle(path=tmp_path / "wt", branch="splinter/US-001", task_id="US-001")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="CONFLICT")
            with pytest.raises(WorktreeMergeConflict):
                squash_merge(handle, base_branch="main", base_dir=tmp_path)

    def test_reattach_returns_none_when_path_absent(self, tmp_path: Path) -> None:
        result = reattach_worktree("US-001", base_dir=tmp_path)
        assert result is None

    def test_reattach_returns_handle_when_path_present(self, tmp_path: Path) -> None:
        wt_path = tmp_path / ".splinter" / "worktrees" / "US-001"
        wt_path.mkdir(parents=True)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=str(wt_path) + "\n")
            result = reattach_worktree("US-001", base_dir=tmp_path)
        assert result is not None
        assert result.task_id == "US-001"
        assert result.branch == "splinter/US-001"


class TestSessionWorktreeTracking:
    def test_set_and_read_worktrees(self, tmp_path: Path) -> None:
        from splinter.memory.session import Session

        session = Session.__new__(Session)
        session.id = "test"
        session.dir = tmp_path
        (tmp_path / "knowledge").mkdir()

        session.set_worktree("US-001", "/path/to/wt", "splinter/US-001")
        worktrees = session.read_worktrees()
        assert "US-001" in worktrees
        assert worktrees["US-001"]["path"] == "/path/to/wt"
        assert worktrees["US-001"]["branch"] == "splinter/US-001"

    def test_read_worktrees_empty_when_absent(self, tmp_path: Path) -> None:
        from splinter.memory.session import Session

        session = Session.__new__(Session)
        session.id = "test"
        session.dir = tmp_path
        assert session.read_worktrees() == {}

    def test_set_worktree_persists_multiple(self, tmp_path: Path) -> None:
        from splinter.memory.session import Session

        session = Session.__new__(Session)
        session.id = "test"
        session.dir = tmp_path
        (tmp_path / "knowledge").mkdir()

        session.set_worktree("US-001", "/wt1", "splinter/US-001")
        session.set_worktree("US-002", "/wt2", "splinter/US-002")
        worktrees = session.read_worktrees()
        assert len(worktrees) == 2
        assert worktrees["US-002"]["path"] == "/wt2"
