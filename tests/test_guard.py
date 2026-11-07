"""Tests for the run guard: shims, guarded env and permission deny lists."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

from splinter.vcs.guard import (
    GUARD_SH,
    GUARDED_NAMES,
    guarded_env,
    opencode_permission_json,
    reviewer_deny,
    setup_guard,
    workhorse_deny,
)


class TestSetupGuard:
    def test_writes_executable_guard_script(self, tmp_path: Path) -> None:
        env = setup_guard(tmp_path)
        assert env.run_dir == tmp_path
        assert env.guard_script == tmp_path / "guard.sh"
        assert env.guard_script.read_text(encoding="utf-8") == GUARD_SH
        assert stat.S_IMODE(env.guard_script.stat().st_mode) == 0o755

    def test_creates_shims_dir_with_symlink_per_name(self, tmp_path: Path) -> None:
        env = setup_guard(tmp_path)
        assert env.shims_dir == tmp_path / "shims"
        assert env.shims_dir.is_dir()
        assert len(GUARDED_NAMES) == 32
        for name in GUARDED_NAMES:
            shim = env.shims_dir / name
            assert shim.is_symlink(), name
            assert os.readlink(shim) == str(env.guard_script)

    def test_idempotent_second_call(self, tmp_path: Path) -> None:
        first = setup_guard(tmp_path)
        second = setup_guard(tmp_path)
        assert second == first
        assert second.guard_script.read_text(encoding="utf-8") == GUARD_SH
        assert stat.S_IMODE(second.guard_script.stat().st_mode) == 0o755
        for name in GUARDED_NAMES:
            shim = second.shims_dir / name
            assert shim.is_symlink(), name
            assert os.readlink(shim) == str(second.guard_script)

    def test_replaces_stale_symlink(self, tmp_path: Path) -> None:
        shims = tmp_path / "shims"
        shims.mkdir(parents=True)
        os.symlink("/missing/guard", shims / "git")
        env = setup_guard(tmp_path)
        assert (shims / "git").is_symlink()
        assert os.readlink(shims / "git") == str(env.guard_script)

    def test_replaces_regular_file_shim(self, tmp_path: Path) -> None:
        shims = tmp_path / "shims"
        shims.mkdir(parents=True)
        (shims / "brew").write_text("#!/bin/sh\n", encoding="utf-8")
        env = setup_guard(tmp_path)
        assert (shims / "brew").is_symlink()
        assert os.readlink(shims / "brew") == str(env.guard_script)

    def test_creates_missing_run_dir(self, tmp_path: Path) -> None:
        env = setup_guard(tmp_path / "nested" / "run")
        assert env.guard_script.exists()
        assert env.shims_dir.is_dir()


class TestGuardedEnv:
    def test_puts_shims_first_on_path(self, tmp_path: Path) -> None:
        guard = setup_guard(tmp_path)
        out = guarded_env(guard, base={"PATH": "/usr/bin:/bin", "HOME": "/home/dev"})
        assert out["PATH"] == str(guard.shims_dir) + os.pathsep + "/usr/bin:/bin"
        assert out["HOME"] == "/home/dev"

    def test_removes_unset_keys(self, tmp_path: Path) -> None:
        guard = setup_guard(tmp_path)
        base = {"PATH": "/usr/bin", "ANTHROPIC_API_KEY": "secret", "KEEP": "yes"}
        out = guarded_env(guard, base=base, unset=["ANTHROPIC_API_KEY"])
        assert "ANTHROPIC_API_KEY" not in out
        assert out["KEEP"] == "yes"

    def test_applies_set_env(self, tmp_path: Path) -> None:
        guard = setup_guard(tmp_path)
        out = guarded_env(
            guard,
            base={"PATH": "/usr/bin", "OPENCODE_CONFIG_CONTENT": "old"},
            set_env={"OPENCODE_CONFIG_CONTENT": "new"},
        )
        assert out["OPENCODE_CONFIG_CONTENT"] == "new"

    def test_set_env_overrides_base_value(self, tmp_path: Path) -> None:
        guard = setup_guard(tmp_path)
        out = guarded_env(guard, base={"PATH": "/usr/bin", "FOO": "1"}, set_env={"FOO": "2"})
        assert out["FOO"] == "2"

    def test_drops_pwd(self, tmp_path: Path) -> None:
        guard = setup_guard(tmp_path)
        out = guarded_env(guard, base={"PATH": "/usr/bin", "PWD": "/somewhere"})
        assert "PWD" not in out

    def test_base_without_path_yields_shims_only(self, tmp_path: Path) -> None:
        guard = setup_guard(tmp_path)
        out = guarded_env(guard, base={"HOME": "/home/dev"})
        assert out["PATH"] == str(guard.shims_dir)

    def test_base_none_uses_os_environ(self, tmp_path: Path) -> None:
        guard = setup_guard(tmp_path)
        out = guarded_env(guard)
        assert out["PATH"].startswith(str(guard.shims_dir) + os.pathsep)
        assert out.get("HOME") == os.environ.get("HOME")
        assert "PWD" not in out


class TestGuardShContent:
    def test_shebang_and_deny(self) -> None:
        assert GUARD_SH.startswith("#!/bin/sh\n")
        assert "BLOCKED by the run guard: $name $*" in GUARD_SH
        assert "exit 126" in GUARD_SH

    def test_git_read_allowlist(self) -> None:
        assert "status|diff|log|show|blame|ls-files|rev-parse|grep|cat-file) ;;" in GUARD_SH

    def test_go_inspection_allowlist(self) -> None:
        assert "doc|env|version|list|fmt) ;;" in GUARD_SH

    def test_pass_through_removes_shims_from_path(self) -> None:
        assert "/usr/bin/grep -vxF \"$SHIMS\"" in GUARD_SH
        assert 'exec "$name" "$@"' in GUARD_SH

    def test_guarded_names_exact(self) -> None:
        assert GUARDED_NAMES == (
            "git", "go", "make", "pnpm", "npm", "npx", "yarn", "vitest", "tsc", "eslint",
            "next", "golangci-lint", "uv", "pytest", "mypy", "ruff", "pip", "docker",
            "docker-compose", "colima", "gh", "terraform", "claude", "claude-credit",
            "opencode", "agy", "mcode", "codex", "kill", "pkill", "killall", "brew",
        )


class TestDenyLists:
    def test_reviewer_deny_exact(self) -> None:
        assert reviewer_deny() == [
            "Bash(git push:*)", "Bash(git fetch:*)", "Bash(git pull:*)", "Bash(git rebase:*)",
            "Bash(git merge:*)", "Bash(git stash:*)", "Bash(git worktree:*)",
            "Bash(git branch:*)", "Bash(gh:*)", "Bash(docker:*)", "Bash(pkill:*)",
            "Bash(kill:*)", "Bash(claude:*)", "Bash(claude-credit:*)", "Bash(opencode:*)",
            "Bash(agy:*)", "Bash(mcode:*)", "Bash(codex:*)", "Agent", "Task",
        ]

    def test_workhorse_deny_extends_reviewer(self) -> None:
        assert workhorse_deny() == reviewer_deny() + [
            "Bash(git commit:*)", "Bash(git add:*)", "Bash(git reset:*)", "Bash(git clean:*)",
            "Bash(git checkout:*)", "Bash(git switch:*)", "Bash(go test:*)",
            "Bash(go build:*)", "Bash(go vet:*)", "Bash(golangci-lint:*)",
            "Bash(go tool:*)", "Bash(make:*)", "Bash(pnpm:*)", "Bash(npx:*)",
            "Bash(vitest:*)", "Bash(tsc:*)", "Bash(uv run:*)", "Bash(pytest:*)",
        ]

    def test_opencode_permission_json_single_ordered_line(self) -> None:
        text = opencode_permission_json()
        assert "\n" not in text
        policy = json.loads(text)["permission"]["bash"]
        items = list(policy.items())
        assert len(items) == 36
        assert items[0] == ("*", "allow")
        assert items[-1] == ("rm -rf ~*", "deny")
        assert policy["git push*"] == "deny"
        assert policy["git commit*"] == "deny"
        assert policy["pytest*"] == "deny"
        assert policy["rm -rf /*"] == "deny"
