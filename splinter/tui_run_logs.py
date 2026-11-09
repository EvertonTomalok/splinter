"""The run log kept per story, so a parallel run shows each story its own lines."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _Line:
    story: int | None
    markup: str


def belongs(owner: int | None, selected: int | None) -> bool:
    """A pipeline line shows everywhere; a story's line only on that story."""
    return owner is None or selected is None or owner == selected


class StoryLog:
    def __init__(self, limit: int = 5000, trim: int = 1000) -> None:
        self._limit = limit
        self._trim = trim
        self._lines: list[_Line] = []

    def add(self, story: int | None, markup: str) -> None:
        self._lines.append(_Line(story, markup))
        if len(self._lines) > self._limit:
            del self._lines[: self._trim]

    def visible(self, selected: int | None) -> list[str]:
        return [line.markup for line in self._lines if belongs(line.story, selected)]
