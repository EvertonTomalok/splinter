"""Changed-scope gate checks: lint, typecheck and test only what a change touched.

``splinter gate <kind>`` is meant as a ``gate_checks`` command. It finds the files
the change touched — commits since the ``splinter/*`` branch was created plus
uncommitted and untracked files; on any other branch only the uncommitted ones —
and runs the kind's tool on just those packages, workspaces or files, with
capped parallelism (``GOMAXPROCS=4``, ``-p=2``, ``--concurrency=2``,
``--maxWorkers=2``). Nothing touched means nothing to check, and it passes.

Kinds: ``go-fmt go-vet go-lint go-test ts-typecheck ts-lint ts-test py-lint
py-type py-test``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

SPLINTER_BRANCH_PREFIX = "splinter/"
BASE_ENV = "SPLINTER_GATE_BASE"

_SKIP_PARTS = frozenset({"node_modules", "vendor", "dist", ".next", "build", ".venv"})
_TS_SUFFIXES = frozenset({".ts", ".tsx", ".mts", ".cts", ".js", ".jsx", ".mjs", ".cjs"})
_ESLINT_CONFIGS = (
    "eslint.config.js",
    "eslint.config.mjs",
    "eslint.config.cjs",
    "eslint.config.ts",
    ".eslintrc",
    ".eslintrc.js",
    ".eslintrc.cjs",
    ".eslintrc.json",
    ".eslintrc.yml",
)
_GOLANGCI_CONFIGS = (".golangci.yml", ".golangci.yaml", ".golangci.toml", ".golangci.json")
_LOCKFILES = (
    ("bun.lock", "bun"),
    ("bun.lockb", "bun"),
    ("pnpm-lock.yaml", "pnpm"),
    ("yarn.lock", "yarn"),
    ("package-lock.json", "npm"),
)
_CAPPED_ENV = {"GOMAXPROCS": "4", "GOFLAGS": "-p=2"}
_PYTEST_NO_TESTS = 5

Git = Callable[[list[str], Path], str]


@dataclass(frozen=True)
class Step:
    """One command of a gate kind; ``fail_on_output`` fails it when stdout is non-empty."""

    label: str
    argv: tuple[str, ...]
    cwd: Path
    fail_on_output: bool = False
    ok_codes: tuple[int, ...] = (0,)
    env: dict[str, str] = field(default_factory=dict)


def _git(args: list[str], cwd: Path) -> str:
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    return proc.stdout if proc.returncode == 0 else ""


def base_commit(root: Path, git: Git = _git) -> str:
    """Where the change starts: the ``splinter/*`` branch's creation point, else HEAD."""
    override = os.environ.get(BASE_ENV, "").strip()
    if override:
        return override
    branch = git(["symbolic-ref", "--short", "-q", "HEAD"], root).strip()
    if branch.startswith(SPLINTER_BRANCH_PREFIX):
        entries = git(["reflog", "show", "--format=%H", f"refs/heads/{branch}"], root).split()
        if entries:
            return entries[-1]
    return "HEAD"


def changed_files(root: Path, git: Git = _git) -> list[str]:
    """Repo-relative paths the change touched (committed since base, staged, unstaged, new)."""
    base = base_commit(root, git)
    chunks = [
        git(["diff", "--name-only", "-z", base, "HEAD"], root) if base != "HEAD" else "",
        git(["diff", "--name-only", "-z", "HEAD"], root),
        git(["ls-files", "--others", "--exclude-standard", "-z"], root),
    ]
    paths = {p for chunk in chunks for p in chunk.split("\0") if p}
    return sorted(p for p in paths if not _SKIP_PARTS.intersection(Path(p).parts))


def _nearest(root: Path, rel_dir: Path, names: Iterable[str]) -> Path | None:
    """Closest dir from ``rel_dir`` up to ``root`` holding one of ``names`` (relative)."""
    wanted = tuple(names)
    current = rel_dir
    while True:
        if any((root / current / name).exists() for name in wanted):
            return current
        if current == Path():
            return None
        current = current.parent


def _rel(path: Path, base: Path) -> str:
    return str(path.relative_to(base)) if base != Path() else str(path)


def _pkg_pattern(pkg_dir: Path, module: Path) -> str:
    inner = pkg_dir.relative_to(module) if module != Path() else pkg_dir
    return "." if inner == Path() else f"./{inner}"


def _go_modules(root: Path, files: list[str]) -> dict[Path, list[Path]]:
    """Module dir → package dirs that still hold Go files."""
    grouped: dict[Path, set[Path]] = {}
    for f in files:
        if not f.endswith(".go"):
            continue
        pkg = Path(f).parent
        if not (root / pkg).is_dir() or not any((root / pkg).glob("*.go")):
            continue
        module = _nearest(root, pkg, ("go.mod",))
        if module is not None:
            grouped.setdefault(module, set()).add(pkg)
    return {m: sorted(pkgs) for m, pkgs in sorted(grouped.items())}


def plan_go(kind: str, root: Path, files: list[str]) -> list[Step]:
    if kind == "go-fmt":
        go_files = [f for f in files if f.endswith(".go") and (root / f).is_file()]
        if not go_files:
            return []
        return [Step("gofmt -l", ("gofmt", "-l", *go_files), root, fail_on_output=True)]
    steps: list[Step] = []
    for module, pkgs in _go_modules(root, files).items():
        patterns = tuple(_pkg_pattern(p, module) for p in pkgs)
        cwd = root / module
        if kind == "go-vet":
            steps.append(Step(f"go vet ({module})", ("go", "vet", *patterns), cwd, env=_CAPPED_ENV))
        elif kind == "go-test":
            steps.append(
                Step(f"go test ({module})", ("go", "test", *patterns), cwd, env=_CAPPED_ENV)
            )
        elif kind == "go-lint" and _nearest(root, module, _GOLANGCI_CONFIGS) is not None:
            argv = ("golangci-lint", "run", "--concurrency=2", *patterns)
            steps.append(Step(f"golangci-lint ({module})", argv, cwd, env=_CAPPED_ENV))
    return steps


def _package_json(root: Path, workspace: Path) -> dict[str, object]:
    try:
        data = json.loads((root / workspace / "package.json").read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _scripts(root: Path, workspace: Path) -> dict[str, str]:
    raw = _package_json(root, workspace).get("scripts")
    return {str(k): str(v) for k, v in raw.items()} if isinstance(raw, dict) else {}


def _runner(root: Path, workspace: Path) -> str:
    current = workspace
    while True:
        for lockfile, runner in _LOCKFILES:
            if (root / current / lockfile).exists():
                return runner
        if current == Path():
            return "npm"
        current = current.parent


def _bin(root: Path, workspace: Path, tool: str) -> Path | None:
    found = _nearest(root, workspace, (f"node_modules/.bin/{tool}",))
    return root / found / "node_modules" / ".bin" / tool if found is not None else None


def _run_script(runner: str, name: str) -> tuple[str, ...]:
    return ("npm", "run", name) if runner == "npm" else (runner, "run", name)


def _ts_workspaces(root: Path, files: list[str]) -> dict[Path, list[str]]:
    grouped: dict[Path, list[str]] = {}
    for f in files:
        if Path(f).suffix not in _TS_SUFFIXES or not (root / f).is_file():
            continue
        workspace = _nearest(root, Path(f).parent, ("package.json",))
        if workspace is not None:
            grouped.setdefault(workspace, []).append(_rel(Path(f), workspace))
    return dict(sorted(grouped.items()))


def _ts_step(kind: str, root: Path, workspace: Path, rel_files: list[str]) -> Step | None:
    scripts = _scripts(root, workspace)
    runner = _runner(root, workspace)
    cwd = root / workspace
    bun = ("bun", "--bun") if runner == "bun" else ()
    label = f"{kind} ({workspace})"
    if kind == "ts-typecheck":
        if "typecheck" in scripts:
            return Step(label, _run_script(runner, "typecheck"), cwd)
        if "tsc" in scripts.get("lint", ""):
            return Step(label, _run_script(runner, "lint"), cwd)
        tsc = _bin(root, workspace, "tsc")
        if tsc is not None and (cwd / "tsconfig.json").exists():
            return Step(label, (*bun, str(tsc), "--noEmit"), cwd)
        return None
    if kind == "ts-lint":
        eslint = _bin(root, workspace, "eslint")
        if eslint is None or _nearest(root, workspace, _ESLINT_CONFIGS) is None:
            return None
        return Step(label, (*bun, str(eslint), *rel_files), cwd)
    test = scripts.get("test", "")
    if "vitest" in test:
        vitest = _bin(root, workspace, "vitest")
        if vitest is None:
            return None
        flags = ("related", "--run", "--passWithNoTests", "--maxWorkers=2")
        return Step(label, (*bun, str(vitest), *flags, *rel_files), cwd)
    if test.strip() == "bun test":
        return Step(label, ("bun", "test"), cwd)
    return Step(label, _run_script(runner, "test"), cwd) if test else None


def plan_ts(kind: str, root: Path, files: list[str]) -> list[Step]:
    steps = (_ts_step(kind, root, ws, rels) for ws, rels in _ts_workspaces(root, files).items())
    return [s for s in steps if s is not None]


def _has_mypy_config(project: Path) -> bool:
    if (project / "mypy.ini").exists() or (project / ".mypy.ini").exists():
        return True
    pyproject = project / "pyproject.toml"
    return pyproject.exists() and "[tool.mypy]" in pyproject.read_text()


def _is_test_file(path: Path) -> bool:
    return path.name.startswith("test_") or path.name.endswith("_test.py")


def _py_projects(root: Path, files: list[str]) -> dict[Path, list[str]]:
    """Python project (closest ``uv.lock``, else ``pyproject.toml``, else root) → files."""
    grouped: dict[Path, list[str]] = {}
    for f in files:
        if not f.endswith(".py") or not (root / f).is_file():
            continue
        parent = Path(f).parent
        project = (
            _nearest(root, parent, ("uv.lock",))
            or _nearest(root, parent, ("pyproject.toml",))
            or Path()
        )
        grouped.setdefault(project, []).append(_rel(Path(f), project))
    return dict(sorted(grouped.items()))


def _py_test_targets(root: Path, rel_files: list[str]) -> list[str]:
    """Changed test files plus the ``tests/`` dir of each member a source file sits in."""
    targets: set[str] = set()
    for f in rel_files:
        path = Path(f)
        if _is_test_file(path):
            targets.add(f)
            continue
        member = _nearest(root, path.parent, ("pyproject.toml",))
        if member is not None and (root / member / "tests").is_dir():
            targets.add(str(member / "tests"))
    dirs = [Path(t) for t in targets if (root / t).is_dir()]
    return sorted(t for t in targets if not any(d in Path(t).parents for d in dirs))


def _py_step(kind: str, project_dir: Path, rel_files: list[str]) -> Step | None:
    prefix = ("uv", "run") if (project_dir / "uv.lock").exists() else ()
    if kind == "py-lint":
        return Step("ruff check", (*prefix, "ruff", "check", *rel_files), project_dir)
    if kind == "py-type":
        if not _has_mypy_config(project_dir):
            return None
        return Step("mypy", (*prefix, "mypy", *rel_files), project_dir)
    targets = _py_test_targets(project_dir, rel_files)
    if not targets:
        return None
    argv = (*prefix, "pytest", "-q", "-m", "not integration", "-p", "no:cacheprovider", *targets)
    return Step("pytest", argv, project_dir, ok_codes=(0, _PYTEST_NO_TESTS))


def plan_py(kind: str, root: Path, files: list[str]) -> list[Step]:
    steps = (_py_step(kind, root / p, rels) for p, rels in _py_projects(root, files).items())
    return [s for s in steps if s is not None]


_PLANNERS: dict[str, Callable[[str, Path, list[str]], list[Step]]] = {
    "go-fmt": plan_go,
    "go-vet": plan_go,
    "go-lint": plan_go,
    "go-test": plan_go,
    "ts-typecheck": plan_ts,
    "ts-lint": plan_ts,
    "ts-test": plan_ts,
    "py-lint": plan_py,
    "py-type": plan_py,
    "py-test": plan_py,
}
KINDS: tuple[str, ...] = tuple(_PLANNERS)


def plan(kind: str, root: Path, files: list[str]) -> list[Step]:
    if kind not in _PLANNERS:
        raise ValueError(f"unknown gate kind {kind!r}; expected one of: {', '.join(KINDS)}")
    return _PLANNERS[kind](kind, root, files)


def _run_step(step: Step) -> bool:
    if shutil.which(step.argv[0]) is None and not Path(step.argv[0]).exists():
        print(f"✗ {step.label}: command not found: {step.argv[0]}")
        return False
    print(f"$ {' '.join(step.argv)}  (in {step.cwd})", flush=True)
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    proc = subprocess.run(
        step.argv,
        cwd=step.cwd,
        env={**env, **step.env},
        capture_output=step.fail_on_output,
        text=True,
    )
    if step.fail_on_output:
        output = (proc.stdout or "").strip()
        if output:
            print(f"✗ {step.label}: needs formatting:\n{output}")
            return False
    return proc.returncode in step.ok_codes


def run_kind(kind: str, cwd: Path, *, dry_run: bool = False, git: Git = _git) -> int:
    """Plan ``kind`` over the changed files under ``cwd``'s repo and run it; 0 = pass."""
    top = git(["rev-parse", "--show-toplevel"], cwd).strip()
    if not top:
        print("splinter gate: not inside a git repository")
        return 2
    root = Path(top)
    try:
        steps = plan(kind, root, changed_files(root, git))
    except ValueError as exc:
        print(f"splinter gate: {exc}")
        return 2
    if not steps:
        print(f"splinter gate {kind}: nothing touched in scope, skipping")
        return 0
    if dry_run:
        for step in steps:
            print(f"$ {' '.join(step.argv)}  (in {step.cwd})")
        return 0
    failed = [step.label for step in steps if not _run_step(step)]
    if failed:
        print(f"splinter gate {kind}: failed: {', '.join(failed)}")
        return 1
    return 0
