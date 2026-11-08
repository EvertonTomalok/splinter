"""Stories and iterations of an agents run, read from its progress files.

A story passes when every brief of it merged, is live while one of its
attempts is open, and fails when its last attempt ended in ``FAIL`` (a merge
conflict or a brief nobody could finish). Each attempt is one iteration.
"""

from __future__ import annotations

from splinter.agents.progress import (
    FAIL,
    PASS,
    Attempt,
    Story,
    read_attempts,
    read_done,
    read_stories,
)
from splinter.run_snapshot import TaskRow

Groups = list[tuple[int, str, list[tuple[int, str, str]]]]


def agents_groups(stories_text: str, attempts_text: str) -> Groups:
    stories, _ = read_stories(stories_text)
    by_story = _attempts_by_story(read_attempts(attempts_text))
    return [
        (
            story.number,
            _title(story),
            [(a.n, f"T{a.tier}", a.verdict or "?") for a in by_story.get(story.number, [])],
        )
        for story in stories
    ]


def agents_rows(stories_text: str, attempts_text: str, checkpoint_text: str) -> list[TaskRow]:
    stories, briefs = read_stories(stories_text)
    by_story = _attempts_by_story(read_attempts(attempts_text))
    done = read_done(checkpoint_text)
    rows: list[TaskRow] = []
    for story in stories:
        attempts = by_story.get(story.number, [])
        own = [brief for brief, number in briefs.items() if number == story.number]
        status = _status(attempts, own, done)
        last = attempts[-1] if attempts else None
        rows.append(
            TaskRow(
                number=story.number,
                title=_title(story),
                status=status,
                tier=f"T{last.tier}" if last else "",
                iters=len(attempts),
                verdict=PASS if status == "pass" else (last.verdict if last else ""),
            )
        )
    return rows


def _status(attempts: list[Attempt], own: list[str], done: set[str]) -> str:
    if own and all(brief in done for brief in own):
        return "pass"
    if any(not attempt.verdict for attempt in attempts):
        return "running"
    if attempts and attempts[-1].verdict == FAIL and attempts[-1].brief not in done:
        return "failed"
    return "pending"


def _attempts_by_story(attempts: list[Attempt]) -> dict[int, list[Attempt]]:
    found: dict[int, list[Attempt]] = {}
    for attempt in attempts:
        found.setdefault(attempt.story, []).append(attempt)
    for items in found.values():
        items.sort(key=lambda attempt: attempt.n)
    return found


def _title(story: Story) -> str:
    return f"{story.id}: {story.title}" if story.title else story.id
