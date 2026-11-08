"""Which files a brief changed, read from ``git status --porcelain -z``."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import splinter.agents.codegen as codegen_mod


class TestChangedFiles:
    def _changed(self, tmp_path: Path, stdout: str) -> tuple[list[str], list[str]]:
        with patch.object(codegen_mod.subprocess, "run") as run:
            run.return_value = MagicMock(stdout=stdout)
            found = codegen_mod.changed_files(tmp_path)
        return found, run.call_args.args[0]

    def test_spaces_and_renames(self, tmp_path: Path) -> None:
        found, cmd = self._changed(tmp_path, "?? src/a b.py\0R  new.py\0old.py\0 M c.py\0")
        assert found == ["src/a b.py", "new.py", "c.py"]
        assert cmd == ["git", "status", "--porcelain", "-z", "--untracked-files=all"]

    def test_copy_and_worktree_rename_skip_their_source(self, tmp_path: Path) -> None:
        found, _ = self._changed(tmp_path, "C  c2.py\0c1.py\0 R r2.py\0r1.py\0A  ação.py\0")
        assert found == ["c2.py", "r2.py", "ação.py"]

    def test_nothing_changed(self, tmp_path: Path) -> None:
        assert self._changed(tmp_path, "")[0] == []
