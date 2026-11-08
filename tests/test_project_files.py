"""Only the project's own files reach search and localization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from splinter.agents.localizer import (
    CodeAnchor,
    _find_meta_files,
    _meta_anchors,
    _project_anchors,
    _read_candidate_files,
)
from splinter.tools.project_files import (
    is_project_path,
    looks_binary,
    project_paths,
    walk_project_files,
)


@pytest.mark.parametrize(
    "path",
    [
        "apps/api/internal/services/billing.go",
        "app/agenda/internal/adapters/out/repo.go",
        "web/apps/hub/src/App.tsx",
        ".claude/rules/backend.md",
        ".github/workflows/ci.yml",
        "internal/build/version.go",
        "AGENTS.md",
        "go.mod",
        "package.json",
        "apps/web/new-file-to-create.ts",
    ],
)
def test_project_paths_kept(path: str) -> None:
    assert is_project_path(path)


@pytest.mark.parametrize(
    "path",
    [
        "apps/web/node_modules/recharts/AGENTS.md",
        "node_modules/react/index.js",
        "vendor/github.com/pkg/errors/errors.go",
        "home/go/pkg/mod/github.com/gin-gonic/gin@v1.9.1/gin.go",
        ".venv/lib/python3.12/site-packages/x.py",
        "apps/web/.next/server/page.js",
        "src/__pycache__/m.cpython-312.pyc",
        "go.sum",
        "apps/web/yarn.lock",
        "bun.lock",
        "uv.lock",
        "public/logo.png",
        "assets/fonts/Inter.woff2",
        "docs/manual.pdf",
        "bin/server.exe",
        "static/app.min.js",
        "static/app.js.map",
        "data/local.sqlite3",
        "video/out.mp4",
        ".DS_Store",
        "../outside/file.go",
    ],
)
def test_non_project_paths_dropped(path: str) -> None:
    assert not is_project_path(path)


def test_absolute_paths_must_sit_inside_the_repo(tmp_path: Path) -> None:
    assert is_project_path(str(tmp_path / "pkg" / "a.go"), tmp_path)
    assert not is_project_path("/Users/x/go/pkg/mod/github.com/a/b.go", tmp_path)


def test_project_paths_keeps_order() -> None:
    files = ["b.go", "node_modules/x.js", "a.go", "logo.png"]
    assert project_paths(files) == ["b.go", "a.go"]


def test_looks_binary(tmp_path: Path) -> None:
    (tmp_path / "bin.dat").write_bytes(b"\x7fELF\x00\x01")
    (tmp_path / "text.go").write_text("package main\n")
    assert looks_binary(tmp_path / "bin.dat")
    assert not looks_binary(tmp_path / "text.go")
    assert not looks_binary(tmp_path / "missing")


def test_walk_prunes_dependencies_and_build_output(tmp_path: Path) -> None:
    for rel in (
        "src/a.py",
        "app/x/adapters/out/repo.go",
        "node_modules/pkg/index.js",
        "vendor/mod/m.go",
        "dist/bundle.js",
        ".venv/lib/x.py",
        "img/logo.png",
        "go.sum",
    ):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("")
    assert walk_project_files(tmp_path) == ["app/x/adapters/out/repo.go", "src/a.py"]


# --- localizer --------------------------------------------------------------


def _tree(root: Path, *paths: str) -> None:
    for rel in paths:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("# doc")


def test_meta_files_skip_dependency_trees(tmp_path: Path) -> None:
    _tree(tmp_path, "AGENTS.md", "apps/web/AGENTS.md", "apps/web/node_modules/recharts/AGENTS.md")
    _tree(tmp_path, "vendor/x/CLAUDE.md", ".venv/lib/CLAUDE.md")
    assert _find_meta_files(str(tmp_path)) == ["AGENTS.md", "apps/web/AGENTS.md"]


def test_nested_meta_file_injected_only_for_its_subtree(tmp_path: Path) -> None:
    _tree(tmp_path, "AGENTS.md", "CLAUDE.md", "apps/web/AGENTS.md", "scripts/ralph/CLAUDE.md")
    injected = _meta_anchors(str(tmp_path), existing_files={"apps/api/main.go"})
    assert sorted(a.file for a in injected) == ["AGENTS.md", "CLAUDE.md"]
    injected = _meta_anchors(str(tmp_path), existing_files={"apps/web/app/page.tsx"})
    assert sorted(a.file for a in injected) == ["AGENTS.md", "CLAUDE.md", "apps/web/AGENTS.md"]


def test_recall_anchors_outside_project_dropped(tmp_path: Path) -> None:
    anchors = [
        CodeAnchor(file="apps/api/a.go", symbol="", reason="r", confidence=0.9),
        CodeAnchor(file="apps/web/node_modules/x/AGENTS.md", symbol="", reason="r", confidence=1),
        CodeAnchor(file="go.sum", symbol="", reason="r", confidence=0.5),
        CodeAnchor(file="public/hero.png", symbol="", reason="r", confidence=0.5),
    ]
    assert [a.file for a in _project_anchors(anchors, str(tmp_path))] == ["apps/api/a.go"]


def test_filter_reads_skip_binary_and_noise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.go").write_text("package a\n")
    (tmp_path / "blob.txt").write_bytes(b"ab\x00cd")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "m.js").write_text("x")
    block = _read_candidate_files(["a.go", "blob.txt", "node_modules/m.js"])
    assert "### a.go" in block
    assert "blob.txt" not in block
    assert "node_modules" not in block


def test_meta_anchors_go_through_jev_and_stay_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import patch

    from splinter.agents import localizer
    from splinter.configure import DecisionsOptions
    from splinter.memory.session import Session
    from splinter.models.roster import Ladder

    _tree(tmp_path, "AGENTS.md", "apps/web/AGENTS.md", "apps/web/page.tsx")
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path / "home"))
    ladder = Ladder(
        tiers=[],
        effort_map={},
        eval_model="sonnet",
        eval_effort="high",
        planner_model="sonnet",
        planner_effort="high",
        localizer_recall_model="recall-model",
        localizer_recall_large_model="",
        localizer_precision_model="precision-model",
    )
    recall = json.dumps([{"file": "apps/web/page.tsx", "symbol": "Page", "confidence": 0.9}])
    seen: list[list[str]] = []

    def fake_rank(item: str, anchors: list[CodeAnchor], **_: object) -> list[CodeAnchor]:
        seen.append([a.file for a in anchors])
        return [a for a in anchors if not a.file.endswith("AGENTS.md")]

    with (
        patch.object(localizer, "_run_search_tools", return_value="files"),
        patch.object(localizer, "_recall_phase", return_value=recall),
        patch.object(localizer, "_jev_rank_anchors", side_effect=fake_rank),
        patch(
            "splinter.configure.configured_decisions",
            return_value=DecisionsOptions(use_for=("recall",)),
        ),
    ):
        out = localizer._localize_item(
            0, "task", ladder, repo_path=str(tmp_path), session=Session("s")
        )
    assert sorted(seen[0]) == ["AGENTS.md", "apps/web/AGENTS.md", "apps/web/page.tsx"]
    assert [a.file for a in out] == ["apps/web/page.tsx"]
