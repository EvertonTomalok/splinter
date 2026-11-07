"""Tests for agents-strategy worktree additions: base_ref, integration, --no-ff merge."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from splinter.vcs.worktree import (
    WorktreeHandle,
    WorktreeMergeConflict,
    create_integration_worktree,
    create_worktree,
    merge_no_ff,
)


def _handle(task_id: str = "run-q1", tmp: Path | None = None) -> WorktreeHandle:
    return WorktreeHandle(
        path=(tmp or Path("/tmp/wt")) / task_id,
        branch=f"splinter/{task_id}",
        task_id=task_id,
    )


def _fake_git(
    responses: dict[tuple[str, ...], list[MagicMock]] | None = None,
    default: MagicMock | None = None,
) -> MagicMock:
    store: dict[tuple[str, ...], list[MagicMock]] = {
        k: list(v) for k, v in (responses or {}).items()
    }

    def run(cmd: list[str], **_kwargs: Any) -> MagicMock:
        key = tuple(cmd[:2])
        queue = store.get(key)
        if queue:
            result = queue.pop(0)
            if not queue:
                store[key] = [result]
            return result
        return default if default is not None else MagicMock(returncode=0, stdout="", stderr="")

    return MagicMock(side_effect=run)


def _ok(stdout: str = "") -> MagicMock:
    return MagicMock(returncode=0, stdout=stdout, stderr="")


def _fail(stderr: str = "") -> MagicMock:
    return MagicMock(returncode=1, stdout="", stderr=stderr)


class TestCreateWorktreeBaseRef:
    def test_base_ref_appended(self, tmp_path: Path) -> None:
        mock = _fake_git()
        with patch("subprocess.run", mock):
            create_worktree("run-q1", base_dir=tmp_path, base_ref="splinter/run-int")
        cmd = mock.call_args[0][0]
        assert cmd[:3] == ["git", "worktree", "add"]
        assert cmd[4:6] == ["-b", "splinter/run-q1"]
        assert cmd[-1] == "splinter/run-int"

    def test_no_base_ref_keeps_old_shape(self, tmp_path: Path) -> None:
        mock = _fake_git()
        with patch("subprocess.run", mock):
            create_worktree("US-001", base_dir=tmp_path)
        cmd = mock.call_args[0][0]
        assert cmd[-1] == "splinter/US-001"


class TestIntegrationWorktree:
    def test_branch_and_dir_use_slug(self, tmp_path: Path) -> None:
        mock = _fake_git()
        with patch("subprocess.run", mock):
            handle = create_integration_worktree("us-004", base_dir=tmp_path)
        assert handle.branch == "splinter/us-004-int"
        assert handle.task_id == "us-004-int"
        assert "us-004-int" in str(handle.path)


class TestMergeNoFF:
    def test_no_commits_returns_false(self, tmp_path: Path) -> None:
        handle = _handle(tmp=tmp_path)
        mock = _fake_git({("git", "rev-list"): [_ok("0\n")]})
        with patch("subprocess.run", mock):
            assert merge_no_ff(handle, tmp_path) is False

    def test_clean_merge_runs_no_ff_with_message(self, tmp_path: Path) -> None:
        handle = _handle(tmp=tmp_path)
        mock = _fake_git({("git", "rev-list"): [_ok("2\n")]})
        with patch("subprocess.run", mock):
            assert merge_no_ff(handle, tmp_path, message="merge: run-q1") is True
        merge_cmd = [c for c in mock.call_args_list if c[0][0][:2] == ["git", "merge"]][0][0][0]
        assert "--no-ff" in merge_cmd
        assert "merge: run-q1" in merge_cmd
        assert handle.branch in merge_cmd

    def test_content_conflict_resolved_toward_brief(self, tmp_path: Path) -> None:
        handle = _handle(tmp=tmp_path)
        mock = _fake_git(
            {
                ("git", "rev-list"): [_ok("1\n")],
                ("git", "merge"): [_fail("conflict")],
                ("git", "diff"): [_ok("src/a.py\n"), _ok("")],
            }
        )
        with patch("subprocess.run", mock):
            assert merge_no_ff(handle, tmp_path) is True
        cmds = [c[0][0] for c in mock.call_args_list]
        assert ["git", "checkout", "--theirs", "--", "src/a.py"] in cmds
        assert any(c[:2] == ["git", "commit"] for c in cmds)

    def test_non_content_failure_preserves_local_changes(self, tmp_path: Path) -> None:
        handle = _handle(tmp=tmp_path)
        mock = _fake_git(
            {
                ("git", "rev-list"): [_ok("1\n")],
                ("git", "merge"): [_fail("index locked")],
                ("git", "diff"): [_ok("")],
            }
        )
        with patch("subprocess.run", mock):
            with pytest.raises(WorktreeMergeConflict):
                merge_no_ff(handle, tmp_path)
        cmds = [c[0][0] for c in mock.call_args_list]
        assert ["git", "reset", "--hard", "HEAD"] not in cmds
