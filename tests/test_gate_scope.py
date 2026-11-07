"""Changed-scope gate kinds: plans only, no real git or tool subprocess."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from splinter.agents import gate_scope
from splinter.agents.gate import _check_timeout
from splinter.agents.gate_scope import base_commit, changed_files, plan, run_kind


def _fake_git(answers: dict[str, str]) -> gate_scope.Git:
    def git(args: list[str], cwd: Path) -> str:
        return answers.get(" ".join(args), "")

    return git


def _touch(root: Path, *paths: str) -> None:
    for rel in paths:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("")


# --- base + changed files ---------------------------------------------------


def test_base_is_head_outside_splinter_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(gate_scope.BASE_ENV, raising=False)
    git = _fake_git({"symbolic-ref --short -q HEAD": "master\n"})
    assert base_commit(Path("."), git) == "HEAD"


def test_base_is_branch_creation_point(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(gate_scope.BASE_ENV, raising=False)
    git = _fake_git(
        {
            "symbolic-ref --short -q HEAD": "splinter/b1\n",
            "reflog show --format=%H refs/heads/splinter/b1": "ccc\nbbb\naaa\n",
        }
    )
    assert base_commit(Path("."), git) == "aaa"


def test_base_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(gate_scope.BASE_ENV, "origin/dev")
    assert base_commit(Path("."), _fake_git({})) == "origin/dev"


def test_changed_files_merges_committed_dirty_and_untracked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(gate_scope.BASE_ENV, raising=False)
    git = _fake_git(
        {
            "symbolic-ref --short -q HEAD": "splinter/b1\n",
            "reflog show --format=%H refs/heads/splinter/b1": "aaa\n",
            "diff --name-only -z aaa HEAD": "api/a.go\0web/x.ts\0",
            "diff --name-only -z HEAD": "api/a.go\0README.md\0",
            "ls-files --others --exclude-standard -z": "new.py\0web/node_modules/m.js\0",
        }
    )
    assert changed_files(Path("."), git) == ["README.md", "api/a.go", "new.py", "web/x.ts"]


# --- go ---------------------------------------------------------------------


def test_go_groups_packages_by_module(tmp_path: Path) -> None:
    _touch(tmp_path, "apps/api/go.mod", "apps/api/internal/svc/a.go", "apps/api/main.go")
    _touch(tmp_path, "apps/cron/go.mod", "apps/cron/job/j.go")
    files = ["apps/api/internal/svc/a.go", "apps/api/main.go", "apps/cron/job/j.go"]
    steps = plan("go-test", tmp_path, files)
    assert [(s.argv, s.cwd) for s in steps] == [
        (("go", "test", ".", "./internal/svc"), tmp_path / "apps/api"),
        (("go", "test", "./job"), tmp_path / "apps/cron"),
    ]
    assert steps[0].env["GOMAXPROCS"] == "4"


def test_go_skips_deleted_package(tmp_path: Path) -> None:
    _touch(tmp_path, "go.mod")
    assert plan("go-vet", tmp_path, ["gone/a.go"]) == []


def test_go_lint_needs_a_golangci_config(tmp_path: Path) -> None:
    _touch(tmp_path, "go.mod", "pkg/a.go")
    assert plan("go-lint", tmp_path, ["pkg/a.go"]) == []
    _touch(tmp_path, ".golangci.yml")
    (step,) = plan("go-lint", tmp_path, ["pkg/a.go"])
    assert step.argv == ("golangci-lint", "run", "--concurrency=2", "./pkg")


def test_go_fmt_lists_changed_files(tmp_path: Path) -> None:
    _touch(tmp_path, "go.mod", "pkg/a.go")
    (step,) = plan("go-fmt", tmp_path, ["pkg/a.go", "README.md"])
    assert step.argv == ("gofmt", "-l", "pkg/a.go")
    assert step.fail_on_output


# --- ts ---------------------------------------------------------------------


def _workspace(root: Path, rel: str, scripts: dict[str, str], *bins: str) -> None:
    (root / rel).mkdir(parents=True, exist_ok=True)
    (root / rel / "package.json").write_text(json.dumps({"scripts": scripts}))
    for tool in bins:
        _touch(root, str(Path(rel) / "node_modules" / ".bin" / tool))


def test_ts_vitest_runs_related_on_changed_files(tmp_path: Path) -> None:
    _workspace(tmp_path, "apps/web", {"test": "vitest run"}, "vitest")
    _touch(tmp_path, "apps/web/yarn.lock", "apps/web/lib/a.ts")
    (step,) = plan("ts-test", tmp_path, ["apps/web/lib/a.ts"])
    vitest = str(tmp_path / "apps/web/node_modules/.bin/vitest")
    assert step.argv == (
        vitest, "related", "--run", "--passWithNoTests", "--maxWorkers=2", "lib/a.ts"
    )
    assert step.cwd == tmp_path / "apps/web"


def test_ts_bun_workspace_uses_root_bins_and_bun_runtime(tmp_path: Path) -> None:
    _touch(tmp_path, "bun.lock", "eslint.config.mjs", "web/apps/hub/src/a.tsx")
    _workspace(tmp_path, "", {}, "eslint")
    _workspace(tmp_path, "web/apps/hub", {"typecheck": "tsc --noEmit", "lint": "eslint ."})
    (lint,) = plan("ts-lint", tmp_path, ["web/apps/hub/src/a.tsx"])
    assert lint.argv == ("bun", "--bun", str(tmp_path / "node_modules/.bin/eslint"), "src/a.tsx")
    (check,) = plan("ts-typecheck", tmp_path, ["web/apps/hub/src/a.tsx"])
    assert check.argv == ("bun", "run", "typecheck")


def test_ts_typecheck_uses_tsc_lint_script(tmp_path: Path) -> None:
    _touch(tmp_path, "bun.lock", "mocks/sms/src/a.ts")
    _workspace(tmp_path, "mocks/sms", {"lint": "tsc --noEmit", "test": "bun test"})
    (check,) = plan("ts-typecheck", tmp_path, ["mocks/sms/src/a.ts"])
    assert check.argv == ("bun", "run", "lint")
    (test,) = plan("ts-test", tmp_path, ["mocks/sms/src/a.ts"])
    assert test.argv == ("bun", "test")


def test_ts_lint_skips_without_eslint(tmp_path: Path) -> None:
    _touch(tmp_path, "pkg/a.ts")
    _workspace(tmp_path, "pkg", {})
    assert plan("ts-lint", tmp_path, ["pkg/a.ts"]) == []


# --- python -----------------------------------------------------------------


def test_py_kinds_with_uv(tmp_path: Path) -> None:
    _touch(tmp_path, "uv.lock", "apps/worker/pyproject.toml", "apps/worker/src/w.py")
    _touch(tmp_path, "apps/worker/tests/test_w.py", "packages/x/test_x.py")
    (tmp_path / "pyproject.toml").write_text("[tool.mypy]\nstrict = true\n")
    files = ["apps/worker/src/w.py", "packages/x/test_x.py", "notes.md"]
    (lint,) = plan("py-lint", tmp_path, files)
    assert lint.argv == (
        "uv", "run", "ruff", "check", "apps/worker/src/w.py", "packages/x/test_x.py"
    )
    (typed,) = plan("py-type", tmp_path, files)
    assert typed.argv[:3] == ("uv", "run", "mypy")
    (test,) = plan("py-test", tmp_path, files)
    assert test.argv[-2:] == ("apps/worker/tests", "packages/x/test_x.py")
    assert 5 in test.ok_codes


def test_py_test_drops_files_inside_a_tests_dir_target(tmp_path: Path) -> None:
    _touch(tmp_path, "app/pyproject.toml", "app/src/m.py", "app/tests/test_m.py")
    (test,) = plan("py-test", tmp_path, ["app/src/m.py", "app/tests/test_m.py"])
    assert test.cwd == tmp_path / "app"
    assert test.argv[-1] == "tests"
    assert "tests/test_m.py" not in test.argv


def test_py_nested_uv_project_runs_in_its_own_dir(tmp_path: Path) -> None:
    _touch(tmp_path, "services/agents-py/uv.lock", "services/agents-py/pyproject.toml")
    _touch(tmp_path, "services/agents-py/src/a.py", "services/agents-py/tests/test_a.py")
    (lint,) = plan("py-lint", tmp_path, ["services/agents-py/src/a.py"])
    assert lint.argv == ("uv", "run", "ruff", "check", "src/a.py")
    assert lint.cwd == tmp_path / "services/agents-py"
    (test,) = plan("py-test", tmp_path, ["services/agents-py/src/a.py"])
    assert test.argv[-1] == "tests"


def test_py_type_skips_without_mypy_config(tmp_path: Path) -> None:
    _touch(tmp_path, "a.py")
    assert plan("py-type", tmp_path, ["a.py"]) == []


# --- run --------------------------------------------------------------------


def test_unknown_kind_is_usage_error(tmp_path: Path) -> None:
    assert run_kind("rust-fmt", tmp_path, git=_fake_git({"rev-parse --show-toplevel": "/r"})) == 2


def test_nothing_touched_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(gate_scope.BASE_ENV, raising=False)
    git = _fake_git({"rev-parse --show-toplevel": str(tmp_path)})
    with patch("splinter.agents.gate_scope.subprocess.run") as run:
        assert run_kind("go-test", tmp_path, git=git) == 0
    run.assert_not_called()


def test_failing_step_fails_the_kind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(gate_scope.BASE_ENV, raising=False)
    _touch(tmp_path, "go.mod", "pkg/a.go")
    git = _fake_git(
        {"rev-parse --show-toplevel": str(tmp_path), "diff --name-only -z HEAD": "pkg/a.go\0"}
    )
    with (
        patch("splinter.agents.gate_scope.shutil.which", return_value="/usr/bin/go"),
        patch("splinter.agents.gate_scope.subprocess.run") as run,
    ):
        run.return_value.returncode = 1
        assert run_kind("go-vet", tmp_path, git=git) == 1
    assert run.call_args.args[0] == ("go", "vet", "./pkg")


def test_gofmt_output_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(gate_scope.BASE_ENV, raising=False)
    _touch(tmp_path, "go.mod", "pkg/a.go")
    git = _fake_git(
        {"rev-parse --show-toplevel": str(tmp_path), "diff --name-only -z HEAD": "pkg/a.go\0"}
    )
    with (
        patch("splinter.agents.gate_scope.shutil.which", return_value="/usr/bin/gofmt"),
        patch("splinter.agents.gate_scope.subprocess.run") as run,
    ):
        run.return_value.returncode = 0
        run.return_value.stdout = "pkg/a.go\n"
        assert run_kind("go-fmt", tmp_path, git=git) == 1


# --- per-check timeout ------------------------------------------------------


@pytest.mark.parametrize(
    ("check", "expected"),
    [({}, 120), ({"timeout": "600"}, 600), ({"timeout": "x"}, 120), ({"timeout": "0"}, 120)],
)
def test_check_timeout(check: dict[str, str], expected: int) -> None:
    assert _check_timeout(check) == expected
