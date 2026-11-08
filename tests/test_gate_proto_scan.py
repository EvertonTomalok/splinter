from __future__ import annotations

import os
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from splinter.agents import gate

OLD = 1_000_000_000
NEW = 2_000_000_000


def _touch(path: Path, mtime: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")
    os.utime(path, (mtime, mtime))
    return path


def _record_walks(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    roots: list[str] = []
    real_walk = os.walk

    def wrapper(top: Any, *args: Any, **kwargs: Any) -> Iterator[tuple[str, list[str], list[str]]]:
        for item in real_walk(top, *args, **kwargs):
            roots.append(item[0])
            yield item

    monkeypatch.setattr(gate.os, "walk", wrapper)
    return roots


def _count_walk_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    real_walk = os.walk

    def wrapper(top: Any, *args: Any, **kwargs: Any) -> Iterator[tuple[str, list[str], list[str]]]:
        calls.append(str(top))
        return real_walk(top, *args, **kwargs)

    monkeypatch.setattr(gate.os, "walk", wrapper)
    return calls


def test_node_modules_is_not_visited_and_alone_gives_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _touch(tmp_path / "node_modules" / "x.proto", NEW)
    _touch(tmp_path / "src" / "main.py", OLD)
    roots = _record_walks(monkeypatch)

    assert gate._proto_changed(str(tmp_path), {}) is False
    relative = [os.path.relpath(r, tmp_path) for r in roots]
    assert relative
    assert not any("node_modules" in r for r in relative)


def test_proto_without_generated_is_true(tmp_path: Path) -> None:
    _touch(tmp_path / "api" / "a.proto", OLD)

    assert gate._proto_changed(str(tmp_path), {}) is True


def test_generated_older_than_proto_is_true(tmp_path: Path) -> None:
    _touch(tmp_path / "api" / "a.proto", NEW)
    _touch(tmp_path / "gen" / "a.pb.go", OLD)

    assert gate._proto_changed(str(tmp_path), {}) is True


def test_generated_newer_than_proto_is_false(tmp_path: Path) -> None:
    _touch(tmp_path / "api" / "a.proto", OLD)
    _touch(tmp_path / "gen" / "a.pb.go", NEW)

    assert gate._proto_changed(str(tmp_path), {}) is False


def test_build_classes_alone_is_not_generated(tmp_path: Path) -> None:
    _touch(tmp_path / "api" / "a.proto", OLD)
    _touch(tmp_path / "build" / "classes" / "B.java", NEW)

    assert gate._proto_changed(str(tmp_path), {}) is True


def test_build_generated_counts_but_build_classes_does_not(tmp_path: Path) -> None:
    _touch(tmp_path / "api" / "a.proto", OLD)
    _touch(tmp_path / "build" / "generated" / "A.java", NEW)
    _touch(tmp_path / "build" / "classes" / "B.java", OLD - 100)

    assert gate._proto_changed(str(tmp_path), {}) is False


def test_build_classes_is_not_visited(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _touch(tmp_path / "api" / "a.proto", OLD)
    _touch(tmp_path / "build" / "classes" / "B.java", NEW)
    _touch(tmp_path / "build" / "generated" / "A.java", NEW)
    roots = _record_walks(monkeypatch)

    gate._proto_changed(str(tmp_path), {})

    relative = [os.path.relpath(r, tmp_path) for r in roots]
    assert not any("classes" in r for r in relative)
    assert os.path.join("build", "generated") in relative


def test_target_generated_is_ignored(tmp_path: Path) -> None:
    _touch(tmp_path / "api" / "a.proto", OLD)
    _touch(tmp_path / "target" / "x.pb.go", NEW)

    assert gate._proto_changed(str(tmp_path), {}) is True


def test_rust_files_count_only_in_relative_gen_dirs(tmp_path: Path) -> None:
    _touch(tmp_path / "api" / "a.proto", OLD + 100)
    _touch(tmp_path / "src" / "lib.rs", NEW)
    assert gate._proto_changed(str(tmp_path), {}) is True

    _touch(tmp_path / "src" / "gen" / "a.rs", OLD)
    assert gate._proto_changed(str(tmp_path), {}) is True

    _touch(tmp_path / "src" / "gen" / "a.rs", NEW)
    assert gate._proto_changed(str(tmp_path), {}) is False


def test_gen_part_above_project_dir_is_ignored(tmp_path: Path) -> None:
    project = tmp_path / "generated" / "proj"
    _touch(project / "api" / "a.proto", OLD)
    _touch(project / "src" / "lib.rs", NEW)

    assert gate._proto_changed(str(project), {}) is True


def test_symlinked_directory_is_not_descended(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside"
    _touch(outside / "linked" / "z.pb.go", NEW)
    project = tmp_path / "project"
    _touch(project / "api" / "a.proto", OLD)
    (project / "link").symlink_to(outside / "linked", target_is_directory=True)
    roots = _record_walks(monkeypatch)

    assert gate._proto_changed(str(project), {}) is True
    relative = [os.path.relpath(r, project) for r in roots]
    assert sorted(relative) == [".", "api"]


def test_memo_makes_second_check_skip_the_walk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _touch(tmp_path / "api" / "a.proto", OLD)
    calls = _count_walk_calls(monkeypatch)
    memo: dict[tuple[str, float], bool] = {}
    check = {"name": "gen", "cmd": "true", "when": "proto_changed"}

    first = gate._should_run(check, str(tmp_path), memo)
    second = gate._should_run(check, str(tmp_path), memo)

    assert first is True
    assert second is True
    assert len(calls) == 1
    assert len(memo) == 1


def test_should_run_without_memo_walks_each_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _touch(tmp_path / "api" / "a.proto", OLD)
    calls = _count_walk_calls(monkeypatch)
    check = {"name": "gen", "cmd": "true", "when": "proto_changed"}

    gate._should_run(check, str(tmp_path))
    gate._should_run(check, str(tmp_path))

    assert len(calls) == 2


def test_run_gate_with_two_proto_checks_walks_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _touch(tmp_path / "api" / "a.proto", OLD)
    checks = [
        {"name": "gen-a", "cmd": "echo a", "when": "proto_changed"},
        {"name": "gen-b", "cmd": "echo b", "when": "proto_changed"},
    ]
    monkeypatch.setattr(gate, "_load_gate_checks", lambda *a, **k: checks)
    run = MagicMock(
        return_value=subprocess.CompletedProcess(args="", returncode=0, stdout="", stderr="")
    )
    monkeypatch.setattr(gate.subprocess, "run", run)
    calls = _count_walk_calls(monkeypatch)

    result = gate.run_gate(str(tmp_path))

    assert result.passed is True
    assert [name for name, _, _ in result.checks] == ["gen-a", "gen-b"]
    assert run.call_count == 2
    assert len(calls) == 1


def test_project_root_named_build_is_walked_normally(tmp_path: Path) -> None:
    root = tmp_path / "build"
    (root / "api").mkdir(parents=True)
    (root / "api" / "a.proto").write_text("")
    assert gate._proto_changed(str(root), {}) is True
