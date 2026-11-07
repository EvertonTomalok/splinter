"""Failure classification for engine (workhorse CLI) runs.

The ``agents`` strategy launches external engine CLIs in per-brief worktrees.
When a run ends, the orchestrator needs to know *why* it ended so that
infra/model failures (rate limit, idle stall, missing binary) reroute the same
brief to another engine without burning a workhorse attempt. This module holds
the regex evidence (limit lines, the ``AGENT_RESULT`` contract line) and
:func:`classify`, which folds a run outcome into a single
:class:`EngineFailure` by precedence ``MISSING > STALL > LIMIT > CRASH >
EMPTY`` — ``None`` means the run looks healthy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

LIMIT_TAIL_BYTES = 6000
LIMIT_PATTERN = re.compile(
    r"(?i)usage limit|rate.?limit|limit reached|credit balance|quota|"
    r"insufficient|too many requests|"
    r'"status":429| 429 '
)
RESULT_PATTERN = re.compile(r"AGENT_RESULT: ([A-Z]+) ([A-Za-z0-9-]+)")

_MAX_DETAIL_CHARS = 120


class FailureKind(StrEnum):
    """Why an engine run ended without a usable result."""

    STALL = "stall"
    LIMIT = "limit"
    CRASH = "crash"
    EMPTY = "empty"
    MISSING = "missing"


@dataclass(frozen=True)
class EngineFailure:
    """Classified engine failure: kind plus the evidence that produced it."""

    kind: FailureKind
    detail: str = ""
    exit_code: int | None = None


def _with_exit(detail: str, returncode: int | None) -> str:
    if returncode is None:
        return detail
    return f"{detail} (exit {returncode})"


def _limit_line(stream: str) -> str | None:
    tail = stream[-LIMIT_TAIL_BYTES:]
    match = LIMIT_PATTERN.search(tail)
    if match is None:
        return None
    line_start = tail.rfind("\n", 0, match.start()) + 1
    line_end = tail.find("\n", match.end())
    if line_end == -1:
        line_end = len(tail)
    return tail[line_start:line_end].strip()[:_MAX_DETAIL_CHARS]


def find_limit(stdout: str, stderr: str) -> str | None:
    """Return the matched limit/rate-limit line from the tail of stdout, else stderr.

    Only the last :data:`LIMIT_TAIL_BYTES` bytes of each stream are searched:
    limit errors surface at the end of a run, and older mentions (log excerpts,
    retry chatter) must not trigger a false LIMIT. The line containing the
    match is returned stripped to a single line of at most ~120 chars.
    """
    stdout_line = _limit_line(stdout)
    if stdout_line is not None:
        return stdout_line
    return _limit_line(stderr)


def find_result(text: str) -> tuple[str, str] | None:
    """Return the LAST ``(status, ident)`` pair reported via ``AGENT_RESULT``.

    Engines print the contract line once per attempt; retries inside the same
    session may emit several, and the last one is the final verdict. Returns
    ``None`` when no contract line is present.
    """
    matches = RESULT_PATTERN.findall(text)
    return matches[-1] if matches else None


def classify(
    *,
    returncode: int | None,
    stdout: str,
    stderr: str,
    has_result: bool,
    stalled: bool = False,
    missing: bool = False,
) -> EngineFailure | None:
    """Classify an engine run outcome into the primary infra/model failure.

    Precedence: ``MISSING > STALL > LIMIT > CRASH > EMPTY`` — the first signal
    present wins, so a stall that also exited non-zero reports as STALL.
    ``missing`` covers an absent engine binary (``FileNotFoundError`` or exit
    127, which is auto-detected); ``stalled`` covers idle-watchdog trips;
    ``has_result=False`` means no ``AGENT_RESULT`` line in the output (EMPTY).
    Returns ``None`` when the run looks healthy: exit 0 or ``None``, a result
    line, no stall/missing flags, and no limit match.
    """
    if missing or returncode == 127:
        return EngineFailure(
            FailureKind.MISSING,
            detail=_with_exit("engine binary not found", returncode),
            exit_code=returncode,
        )
    if stalled:
        return EngineFailure(
            FailureKind.STALL,
            detail=_with_exit("idle watchdog tripped", returncode),
            exit_code=returncode,
        )
    limit_line = find_limit(stdout, stderr)
    if limit_line is not None:
        return EngineFailure(FailureKind.LIMIT, detail=limit_line, exit_code=returncode)
    if returncode not in (0, None):
        return EngineFailure(
            FailureKind.CRASH,
            detail=f"non-zero exit {returncode}",
            exit_code=returncode,
        )
    if not has_result:
        return EngineFailure(
            FailureKind.EMPTY,
            detail="no AGENT_RESULT line",
            exit_code=returncode,
        )
    return None
