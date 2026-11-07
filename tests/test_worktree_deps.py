"""Worktrees reuse the main repo's node_modules at every workspace level."""

from __future__ import annotations

from pathlib import Path

from splinter.vcs.worktree import _link_deps_from_main


def _repo(tmp_path: Path) -> tuple[Path, Path]:
    main = tmp_path / "main"
    wt = tmp_path / "wt"
    for rel in ("", "apps/web", "web/apps/hub", "mocks/asaas"):
        (main / rel / "node_modules").mkdir(parents=True)
        (wt / rel).mkdir(parents=True, exist_ok=True)
    (main / ".git" / "info").mkdir(parents=True)
    return main, wt


def test_links_root_and_nested_workspaces(tmp_path: Path) -> None:
    main, wt = _repo(tmp_path)
    _link_deps_from_main(main, wt)
    for rel in ("", "apps/web", "web/apps/hub", "mocks/asaas"):
        link = wt / rel / "node_modules"
        assert link.is_symlink()
        assert link.resolve() == (main / rel / "node_modules").resolve()


def test_skips_workspace_missing_in_worktree(tmp_path: Path) -> None:
    main, wt = _repo(tmp_path)
    (main / "only-main" / "node_modules").mkdir(parents=True)
    _link_deps_from_main(main, wt)
    assert not (wt / "only-main").exists()


def test_keeps_existing_node_modules(tmp_path: Path) -> None:
    main, wt = _repo(tmp_path)
    (wt / "apps/web/node_modules").mkdir()
    _link_deps_from_main(main, wt)
    assert not (wt / "apps/web/node_modules").is_symlink()


def test_excludes_links_from_git_once(tmp_path: Path) -> None:
    main, wt = _repo(tmp_path)
    exclude = main / ".git" / "info" / "exclude"
    exclude.write_text("# local\n*.log")
    _link_deps_from_main(main, wt)
    _link_deps_from_main(main, tmp_path / "wt")
    assert exclude.read_text() == "# local\n*.log\nnode_modules\n"


def test_no_links_leaves_exclude_untouched(tmp_path: Path) -> None:
    main = tmp_path / "main"
    (main / ".git" / "info").mkdir(parents=True)
    wt = tmp_path / "wt"
    wt.mkdir()
    _link_deps_from_main(main, wt)
    assert not (main / ".git" / "info" / "exclude").exists()
