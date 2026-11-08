"""What counts as the project's own code for search and localization.

Dependency trees (``node_modules``, Go ``vendor``/``pkg/mod``, virtualenvs),
build output, caches, binaries, media, lockfiles and minified bundles are never
the project: listing them crowds the recall model's file budget and reading them
wastes the filter. :func:`is_project_path` is the single predicate every
localizer path goes through.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

#: Directory names that hold dependencies, tool caches or VCS data — never project code.
NOISE_DIRS: frozenset[str] = frozenset(
    {
        "node_modules",
        "bower_components",
        "jspm_packages",
        "vendor",
        "Pods",
        "Carthage",
        ".venv",
        "venv",
        "site-packages",
        "__pycache__",
        ".next",
        ".nuxt",
        ".svelte-kit",
        ".turbo",
        ".parcel-cache",
        ".cache",
        ".gradle",
        ".terraform",
        ".serverless",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        ".eggs",
        "htmlcov",
        ".git",
        ".hg",
        ".svn",
        ".idea",
        ".splinter",
    }
)

#: Usual build-output dir names. Only pruned when walking a tree without git: in a
#: repo, .gitignore already drops them and the same names can be real code
#: (``adapters/out``, ``internal/build``).
BUILD_DIRS: frozenset[str] = frozenset({"dist", "build", "target", "coverage"})

#: File extensions that are binary, media, archives or compiled artefacts.
BINARY_SUFFIXES: frozenset[str] = frozenset(
    {
        ".png", ".jpg", ".jpeg", ".gif", ".webp", ".avif", ".bmp", ".tif", ".tiff", ".ico",
        ".icns", ".heic", ".psd", ".ai", ".sketch", ".fig",
        ".woff", ".woff2", ".ttf", ".otf", ".eot",
        ".mp3", ".mp4", ".m4a", ".wav", ".ogg", ".oga", ".flac", ".aac", ".mov", ".avi",
        ".mkv", ".webm",
        ".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar", ".jar", ".war", ".ear",
        ".exe", ".dll", ".so", ".dylib", ".a", ".o", ".obj", ".lib", ".class", ".pyc", ".pyo",
        ".pyd", ".wasm", ".bin", ".dat", ".img", ".iso", ".dmg", ".pkg", ".deb", ".rpm",
        ".apk", ".ipa",
        ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".odt", ".ods", ".key",
        ".db", ".sqlite", ".sqlite3", ".parquet", ".pkl", ".pickle", ".npy", ".npz", ".onnx",
        ".pt", ".pth", ".safetensors", ".h5",
        ".map",
    }
)

#: Generated lockfiles and OS junk: never worth a recall slot.
JUNK_NAMES: frozenset[str] = frozenset(
    {
        "go.sum",
        "go.work.sum",
        "package-lock.json",
        "npm-shrinkwrap.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "bun.lock",
        "bun.lockb",
        "uv.lock",
        "poetry.lock",
        "Pipfile.lock",
        "Cargo.lock",
        "composer.lock",
        "Gemfile.lock",
        "mix.lock",
        "Podfile.lock",
        ".DS_Store",
        "Thumbs.db",
    }
)

_MINIFIED = (".min.js", ".min.css", ".min.mjs", ".bundle.js", ".chunk.js")
_SNIFF_BYTES = 8192


def is_project_path(path: str, repo_root: str | Path = ".") -> bool:
    """True when ``path`` is the project's own source, config or docs.

    Relative paths are taken from ``repo_root``; absolute ones must sit inside it
    (a Go module cache or a global ``node_modules`` never does).
    """
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            candidate = candidate.resolve().relative_to(Path(repo_root).resolve())
        except (OSError, ValueError):
            return False
    parts = candidate.parts
    if not parts or ".." in parts:
        return False
    dirs = parts[:-1]
    if any(part in NOISE_DIRS for part in dirs):
        return False
    if any(a == "pkg" and b == "mod" for a, b in zip(dirs, dirs[1:])):
        return False
    name = parts[-1]
    lowered = name.lower()
    if name in JUNK_NAMES or lowered.endswith(_MINIFIED):
        return False
    return Path(lowered).suffix not in BINARY_SUFFIXES


def project_paths(paths: Iterable[str], repo_root: str | Path = ".") -> list[str]:
    """``paths`` without the ones :func:`is_project_path` rejects, order kept."""
    return [p for p in paths if is_project_path(p, repo_root)]


def looks_binary(path: str | Path) -> bool:
    """True when the file's first bytes hold a NUL — binary whatever its extension."""
    try:
        with open(path, "rb") as fh:
            return b"\0" in fh.read(_SNIFF_BYTES)
    except OSError:
        return False


def walk_project_files(root: str | Path, names: Iterable[str] | None = None) -> list[str]:
    """Relative project paths under ``root`` (optionally only ``names``), noise pruned."""
    base = Path(root)
    wanted = frozenset(names) if names is not None else None
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(
            d for d in dirnames if d not in NOISE_DIRS and d not in BUILD_DIRS
        )
        rel_dir = Path(dirpath).relative_to(base)
        for name in sorted(filenames):
            if wanted is not None and name not in wanted:
                continue
            rel = str(rel_dir / name)
            if is_project_path(rel):
                found.append(rel)
    return sorted(found)
