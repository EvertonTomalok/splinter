"""Deterministic search primitives, routed through ``rtk`` for token savings.

All functions are guaranteed not to raise — errors (including subprocess timeouts
and missing tools) are returned as a SearchResult with exit_code != 0 and an
``[unavailable: …]`` message so callers can surface the failure to the LLM without
crashing the pipeline.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Generous timeouts for large monorepos.  grep over a big TS/Go repo can take
# well over 30 s when searching unindexed files via rtk.
_GREP_TIMEOUT = 120
_CAT_TIMEOUT = 30
_GIT_LOG_TIMEOUT = 30


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


def _error(tool: str, exc: BaseException) -> SearchResult:
    """Return a graceful error SearchResult that callers can pass to the LLM."""
    kind = type(exc).__name__
    msg = str(exc).split("\n")[0][:120]
    return SearchResult(
        output=f"[unavailable: {tool} failed ({kind}): {msg}]",
        tool=tool,
        exit_code=1,
    )


def grep(pattern: str, path: str = ".", *, flags: str = "", timeout: int = _GREP_TIMEOUT) -> SearchResult:
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

    try:
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


def git_log(n: int = 10, *, timeout: int = _GIT_LOG_TIMEOUT) -> SearchResult:
    if _has_rtk():
        cmd = ["rtk", "git", "log", f"-{n}", "--oneline"]
    else:
        cmd = ["git", "log", f"-{n}", "--oneline"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return SearchResult(output=proc.stdout, tool="git-log", exit_code=proc.returncode)
    except Exception as exc:
        return _error("git-log", exc)


def file_list(path: str = ".", pattern: str = "*", *, timeout: int = _GIT_LOG_TIMEOUT) -> SearchResult:
    p = Path(path)
    if not p.exists():
        return SearchResult(output=f"path not found: {path}", tool="file-list", exit_code=1)
    try:
        files = sorted(str(f) for f in p.rglob(pattern) if f.is_file())
        return SearchResult(output="\n".join(files), tool="file-list", exit_code=0)
    except Exception as exc:
        return _error("file-list", exc)
