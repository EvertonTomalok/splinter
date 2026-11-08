"""Small, fault-tolerant file helpers for the run view."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Sequence

from splinter.memory.session import Session

_TASK_HEAD_RE = re.compile(r"^# Task (\d+)(?:/\d+| \[parallel\]):", re.MULTILINE)
_ITER_HEAD_RE = re.compile(r"^## Iteration (\d+)\s*$", re.MULTILINE)


class FileCache:
    """Read files once for a matching mtime and size."""

    def __init__(self) -> None:
        self._contents: dict[Path, tuple[tuple[int, ...], str]] = {}
        self._stories: dict[Path, tuple[str, dict[int, str]]] = {}

    def read(self, path: Path) -> str:
        try:
            stat = path.stat()
        except OSError:
            return ""
        cached = self._contents.get(path)
        key = (stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_size)
        if cached is not None and cached[0] == key:
            return cached[1]
        try:
            text = path.read_text()
        except OSError:
            return ""
        self._contents[path] = (key, text)
        return text

    def story_body(self, path: Path, story: int) -> str:
        text = self.read(path)
        cached = self._stories.get(path)
        if cached is None or cached[0] is not text:
            cached = (text, {})
            self._stories[path] = cached
        bodies = cached[1]
        if story not in bodies:
            bodies[story] = _story_body(text, story)
        return bodies[story]

    def mtime_key(self, paths: Sequence[Path]) -> tuple[int, ...]:
        keys: list[int] = []
        for path in paths:
            try:
                stat = path.stat()
            except OSError:
                keys.extend((-1, -1, -1, -1))
            else:
                keys.extend((stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_size))
        return tuple(keys)


def iteration_files(session: Session) -> dict[str, tuple[int, ...]]:
    files: dict[str, tuple[int, ...]] = {}
    for directory in ("knowledge", "runs"):
        try:
            with os.scandir(session.dir / directory) as entries:
                for entry in entries:
                    name = entry.name
                    if not name.endswith(".md") or "iter-" not in name:
                        continue
                    if directory == "knowledge" and not (
                        name.startswith("task") and "-review-iter-" in name
                    ):
                        continue
                    try:
                        stat = entry.stat()
                    except OSError:
                        continue
                    files[f"{directory}/{name}"] = (
                        stat.st_ino, stat.st_mtime_ns, stat.st_size
                    )
        except OSError:
            continue
    return files


def plan_path(session: Session, story: int) -> Path:
    numbered = session.dir / "knowledge" / f"plan-{story}.md"
    try:
        return numbered if numbered.exists() else session.dir / "knowledge" / "plan.md"
    except OSError:
        return session.dir / "knowledge" / "plan.md"


def review_note_name(story: int, n: int) -> str:
    return f"task{story - 1}-review-iter-{n}"


def run_file_path(session: Session, story: int, n: int, multi: bool) -> Path:
    name = f"task-{story}-iter-{n}.md" if multi else f"iter-{n}.md"
    return session.dir / "runs" / name


def gate_lines(loop_md: str, story: int, n: int) -> tuple[str, ...]:
    return iteration_lines(loop_md, story, n, "- gate:")


def iteration_lines(loop_md: str, story: int, n: int, prefix: str) -> tuple[str, ...]:
    """The ``prefix`` lines (``- gate:``, ``- verdict:``) of one story iteration."""
    iteration = _iteration_body(_story_body(loop_md, story), n)
    return tuple(line.strip() for line in iteration.splitlines() if line.strip().startswith(prefix))


def _story_body(loop_md: str, story: int) -> str:
    headers = list(_TASK_HEAD_RE.finditer(loop_md))
    if not headers:
        return loop_md
    for index, header in enumerate(headers):
        if int(header.group(1)) == story:
            end = headers[index + 1].start() if index + 1 < len(headers) else len(loop_md)
            return loop_md[header.end() : end]
    return ""


def _iteration_body(body: str, n: int) -> str:
    headers = list(_ITER_HEAD_RE.finditer(body))
    for index, header in enumerate(headers):
        if int(header.group(1)) == n:
            end = headers[index + 1].start() if index + 1 < len(headers) else len(body)
            return body[header.end() : end]
    return ""
