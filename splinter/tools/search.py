"""Deterministic search primitives, routed through ``rtk`` for token savings.

All functions are guaranteed not to raise — errors (including subprocess timeouts
and missing tools) are returned as a SearchResult with exit_code != 0 and an
``[unavailable: …]`` message so callers can surface the failure to the LLM without
crashing the pipeline.

Search is always scoped to the directory where splinter was invoked (CWD).
When inside a git repo, ``git grep`` / ``git ls-files`` are preferred because
they automatically respect ``.gitignore`` — skipping node_modules, vendor,
build artefacts, and other untracked trees that make raw grep impossibly slow
on large monorepos.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Timeouts tuned for large monorepos.
_GREP_TIMEOUT = 120
_CAT_TIMEOUT = 30
_GIT_TIMEOUT = 30


@dataclass(frozen=True)
class SearchResult:
    output: str
    tool: str
    exit_code: int

    @property
    def ok(self) -> bool:
        return self.exit_code == 0

    @property
    def unavailable(self) -> bool:
        """True when the result carries an [unavailable: …] error notice."""
        return self.output.startswith("[unavailable:")


def _has_rtk() -> bool:
    return shutil.which("rtk") is not None


def _is_git_repo(path: str) -> bool:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            cwd=path,
            timeout=5,
        )
        return r.returncode == 0
    except Exception:
        return False


def _error(tool: str, exc: BaseException) -> SearchResult:
    kind = type(exc).__name__
    msg = str(exc)
    return SearchResult(
        output=f"[unavailable: {tool} failed ({kind}): {msg}]",
        tool=tool,
        exit_code=1,
    )


def grep(
    pattern: str, path: str = ".", *, flags: str = "", timeout: int = _GREP_TIMEOUT
) -> SearchResult:
    """Search for pattern, respecting .gitignore when inside a git repo."""
    try:
        if _is_git_repo(path):
            # git grep only searches tracked files — skips node_modules, vendor, dist, etc.
            cmd = ["git", "grep", "-n", "--no-color"]
            if flags:
                cmd.extend(flags.split())
            cmd.append(pattern)
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=path)
            # exit 1 = no matches (not an error); exit 0 = matches found
            if proc.returncode in (0, 1):
                return SearchResult(output=proc.stdout, tool="grep", exit_code=proc.returncode)
            # git grep failed for another reason — fall through to rtk/grep

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
    except Exception as exc:
        return _error("grep", exc)


def cat(path: str, *, timeout: int = _CAT_TIMEOUT) -> SearchResult:
    if _has_rtk():
        cmd = ["rtk", "cat", path]
    else:
        cmd = ["cat", path]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return SearchResult(output=proc.stdout, tool="cat", exit_code=proc.returncode)
    except Exception as exc:
        return _error("cat", exc)


def read_file(path: str, start: int = 0, end: int | None = None) -> SearchResult:
    p = Path(path)
    if not p.exists():
        return SearchResult(output=f"file not found: {path}", tool="read", exit_code=1)
    try:
        lines = p.read_text().splitlines()
    except Exception as exc:
        return _error("read", exc)
    if end is not None:
        lines = lines[start:end]
    elif start > 0:
        lines = lines[start:]
    numbered = [f"{i + start + 1}: {line}" for i, line in enumerate(lines)]
    return SearchResult(output="\n".join(numbered), tool="read", exit_code=0)


def git_log(n: int = 10, *, timeout: int = _GIT_TIMEOUT) -> SearchResult:
    if _has_rtk():
        cmd = ["rtk", "git", "log", f"-{n}", "--oneline"]
    else:
        cmd = ["git", "log", f"-{n}", "--oneline"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return SearchResult(output=proc.stdout, tool="git-log", exit_code=proc.returncode)
    except Exception as exc:
        return _error("git-log", exc)


def file_list(path: str = ".", pattern: str = "*", *, timeout: int = _GIT_TIMEOUT) -> SearchResult:
    """List files matching pattern, respecting .gitignore when inside a git repo."""
    p = Path(path)
    if not p.exists():
        return SearchResult(output=f"path not found: {path}", tool="file-list", exit_code=1)
    try:
        if _is_git_repo(path):
            # git ls-files only lists tracked files — no node_modules / build artefacts.
            proc = subprocess.run(
                [
                    "git",
                    "ls-files",
                    "--",
                    f"*{Path(pattern).suffix}" if "*." in pattern else pattern,
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=path,
            )
            if proc.returncode == 0:
                files = sorted(proc.stdout.splitlines())
                return SearchResult(output="\n".join(files), tool="file-list", exit_code=0)
            # fall through to rglob on git failure

        files = sorted(str(f.relative_to(p)) for f in p.rglob(pattern) if f.is_file())
        return SearchResult(output="\n".join(files), tool="file-list", exit_code=0)
    except Exception as exc:
        return _error("file-list", exc)
