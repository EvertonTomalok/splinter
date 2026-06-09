"""Deterministic search primitives, routed through ``rtk`` for token savings."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SearchResult:
    output: str
    tool: str
    exit_code: int


def _has_rtk() -> bool:
    return shutil.which("rtk") is not None


def grep(pattern: str, path: str = ".", *, flags: str = "", timeout: int = 30) -> SearchResult:
    if _has_rtk():
        cmd = ["rtk", "grep"]
        if flags:
            cmd.extend(flags.split())
        cmd.extend([pattern, path])
    else:
        cmd = ["grep", "-rn"]
        if flags:
            cmd.extend(flags.split())
        cmd.extend([pattern, path])

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return SearchResult(output=proc.stdout, tool="grep", exit_code=proc.returncode)


def cat(path: str, *, timeout: int = 10) -> SearchResult:
    if _has_rtk():
        cmd = ["rtk", "cat", path]
    else:
        cmd = ["cat", path]

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return SearchResult(output=proc.stdout, tool="cat", exit_code=proc.returncode)


def read_file(path: str, start: int = 0, end: int | None = None) -> SearchResult:
    p = Path(path)
    if not p.exists():
        return SearchResult(output=f"file not found: {path}", tool="read", exit_code=1)
    lines = p.read_text().splitlines()
    if end is not None:
        lines = lines[start:end]
    elif start > 0:
        lines = lines[start:]
    numbered = [f"{i + start + 1}: {line}" for i, line in enumerate(lines)]
    return SearchResult(output="\n".join(numbered), tool="read", exit_code=0)


def git_log(n: int = 10, *, timeout: int = 10) -> SearchResult:
    if _has_rtk():
        cmd = ["rtk", "git", "log", f"-{n}", "--oneline"]
    else:
        cmd = ["git", "log", f"-{n}", "--oneline"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return SearchResult(output=proc.stdout, tool="git-log", exit_code=proc.returncode)


def file_list(path: str = ".", pattern: str = "*", *, timeout: int = 10) -> SearchResult:
    p = Path(path)
    if not p.exists():
        return SearchResult(output=f"path not found: {path}", tool="file-list", exit_code=1)
    files = sorted(str(f) for f in p.rglob(pattern) if f.is_file())
    return SearchResult(output="\n".join(files), tool="file-list", exit_code=0)
