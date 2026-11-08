"""The agents run's progress, as structured files the run screen reads.

``agents-run/stories.json`` names the PRD stories and maps each brief to its
story. ``agents-run/attempts.jsonl`` holds one line when an attempt at a brief
starts and one when it ends, so the screen shows every story with its title,
the live attempt and each attempt's verdict, in sequential and parallel runs
alike. ``agents-run/checkpoint.json`` (written by the strategy) says which
briefs merged.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

from splinter.agents.refiner import Brief
from splinter.agents.runner import Task
from splinter.memory.session import Session

STORIES = "agents-run/stories.json"
ATTEMPTS = "agents-run/attempts.jsonl"
CHECKPOINT = "agents-run/checkpoint.json"

PASS = "PASS"
RETRY = "RETRY"
FAIL = "FAIL"

_MARK_BYTES = 64


@dataclass(frozen=True)
class Story:
    number: int
    id: str
    title: str


@dataclass(frozen=True)
class Attempt:
    story: int
    n: int
    brief: str
    engine: str
    tier: int
    verdict: str = ""
    note: str = ""


_Started = dict[tuple[int, int], Attempt]


def stories_of(tasks: list[Task]) -> list[Story]:
    """The PRD stories, numbered from 1, titled without a repeated id."""
    found: list[Story] = []
    for number, task in enumerate(tasks, 1):
        head = " ".join(task.description.split())[:160]
        if task.id and head.upper().startswith(task.id.upper()):
            head = head[len(task.id):].lstrip(" :·—-")
        found.append(Story(number, task.id or f"T{number}", head))
    return found


def brief_stories(briefs: list[Brief], stories: list[Story]) -> dict[str, int]:
    """Brief id → story number; a brief with no known story keeps the previous one."""
    index = {story.id.upper(): story.number for story in stories}
    found: dict[str, int] = {}
    last = 1
    for brief in briefs:
        last = index.get(brief.task_id.upper(), last) if brief.task_id else last
        found[brief.id] = last
    return found


def write_stories(session: Session, stories: list[Story], briefs: dict[str, int]) -> None:
    payload = {
        "stories": [{"number": s.number, "id": s.id, "title": s.title} for s in stories],
        "briefs": briefs,
    }
    session.write(STORIES, json.dumps(payload, ensure_ascii=False))


class AttemptLog:
    """Thread-safe writer of ``attempts.jsonl``; numbers attempts per story."""

    def __init__(self, session: Session) -> None:
        self._session = session
        self._lock = threading.Lock()
        attempts = read_attempts(session.read(ATTEMPTS))
        self._count: dict[int, int] = {}
        for attempt in attempts:
            self._count[attempt.story] = max(self._count.get(attempt.story, 0), attempt.n)
        self._open = [a for a in attempts if not a.verdict]

    def close_interrupted(self) -> None:
        """End the attempts a killed run left open, so no story shows live forever."""
        for attempt in self._open:
            self.end(attempt.story, attempt.n, RETRY, "interrupted: the run stopped")
        self._open = []

    def start(self, story: int, brief: str, engine: str, tier: int) -> int:
        with self._lock:
            n = self._count.get(story, 0) + 1
            self._count[story] = n
            self._write(
                {"event": "start", "story": story, "n": n, "brief": brief,
                 "engine": engine, "tier": tier}
            )
        return n

    def end(self, story: int, n: int, verdict: str, note: str = "") -> None:
        with self._lock:
            self._write(
                {"event": "end", "story": story, "n": n, "verdict": verdict,
                 "note": " ".join(note.split())[:300]}
            )

    def _write(self, record: dict[str, object]) -> None:
        record["ts"] = datetime.now(UTC).isoformat()
        self._session.append(ATTEMPTS, json.dumps(record, ensure_ascii=False) + "\n")


def read_attempts(text: str) -> list[Attempt]:
    """Fold start/end lines into attempts, in start order; an open one has no verdict."""
    started: _Started = {}
    for line in text.splitlines():
        _fold(started, line)
    return list(started.values())


def _fold(started: _Started, line: str) -> None:
    try:
        record = json.loads(line)
        key = (int(record["story"]), int(record["n"]))
    except (ValueError, KeyError, TypeError):
        return
    if record.get("event") == "start":
        started[key] = Attempt(
            story=key[0], n=key[1], brief=str(record.get("brief", "")),
            engine=str(record.get("engine", "")), tier=int(record.get("tier") or 0),
        )
    elif key in started:
        begun = started[key]
        started[key] = Attempt(
            begun.story, begun.n, begun.brief, begun.engine, begun.tier,
            verdict=str(record.get("verdict", "")) or RETRY,
            note=str(record.get("note", "")),
        )


class AttemptsTail:
    """Folds a growing ``attempts.jsonl``; a read takes only the new bytes off the disk.

    A line counts once its newline landed. A file that shrank, or whose bytes
    before the offset changed, is parsed from zero.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._started: _Started = {}
        self._offset = 0
        self._mark = b""

    def read(self) -> list[Attempt]:
        try:
            with self.path.open("rb") as handle:
                self._advance(handle)
        except OSError:
            self._reset()
        return list(self._started.values())

    def _reset(self) -> None:
        self._started, self._offset, self._mark = {}, 0, b""

    def _advance(self, handle: BinaryIO) -> None:
        handle.seek(self._offset - len(self._mark))
        if handle.read(len(self._mark)) != self._mark:
            self._reset()
            handle.seek(0)
        chunk = handle.read()
        end = chunk.rfind(b"\n") + 1
        if not end:
            return
        for line in chunk[:end].decode("utf-8", errors="replace").splitlines():
            _fold(self._started, line)
        self._offset += end
        self._mark = (self._mark + chunk[:end])[-_MARK_BYTES:]


def read_stories(text: str) -> tuple[list[Story], dict[str, int]]:
    try:
        data = json.loads(text)
        stories = [
            Story(int(item["number"]), str(item["id"]), str(item["title"]))
            for item in data["stories"]
        ]
        briefs = {str(k): int(v) for k, v in data["briefs"].items()}
    except (ValueError, KeyError, TypeError, AttributeError):
        return [], {}
    return stories, briefs


def read_done(text: str) -> set[str]:
    try:
        data = json.loads(text)
        return {str(item) for item in data.get("completed", [])}
    except (ValueError, AttributeError, TypeError):
        return set()


def write_done(session: Session, done: set[str]) -> None:
    session.write(CHECKPOINT, json.dumps({"completed": sorted(done)}))
