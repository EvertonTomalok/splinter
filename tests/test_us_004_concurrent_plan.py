"""Tests for planning (US-004): Planner.plan delegates to parse_stories."""

from __future__ import annotations

import asyncio

import pytest

from splinter.agents.planner import Planner, parse_stories
from splinter.strategies.fanout import run_bounded

MULTI_PRD = """\
### US-001: First story
**Description:** Do the first thing.
effort: small
eval_skill: skill-a
- [ ] first thing works

### US-002: Second story
**Description:** Do the second thing.
deps: [US-001]
- [ ] second thing works

### US-003: Third story
**Description:** Do the third thing.
Depends on US-001
parallelizable: true
- [ ] third thing works
"""

SINGLE_PRD = """\
### US-042: Solo story
**Description:** Only one thing.
- [ ] it works
"""

EMPTY_PRD = "Just some prose with no user stories at all."


def _task_fields(t: object) -> tuple:
    return (
        t.id,
        t.description,
        t.acceptance,
        t.effort,
        t.eval_skill,
        t.deps,
        t.parallelizable,
    )


@pytest.mark.parametrize("prd", [MULTI_PRD, SINGLE_PRD, EMPTY_PRD])
def test_concurrent_plan_matches_serial_baseline(prd: str) -> None:
    serial = parse_stories(prd)
    concurrent = Planner().plan(prd)

    assert len(concurrent) == len(serial)
    assert [t.id for t in concurrent] == [t.id for t in serial]
    assert [_task_fields(t) for t in concurrent] == [_task_fields(t) for t in serial]


def test_plan_matches_parse_stories_behavior() -> None:
    tasks = Planner().plan(MULTI_PRD)
    expected = parse_stories(MULTI_PRD)
    assert len(tasks) == len(expected)
    for t, e in zip(tasks, expected, strict=True):
        assert t.id == e.id
        assert t.description == e.description
        assert t.acceptance == e.acceptance
        assert t.effort == e.effort
        assert t.eval_skill == e.eval_skill
        assert t.deps == e.deps
        assert t.parallelizable == e.parallelizable


def test_one_failing_story_propagates_without_deadlock() -> None:
    async def scenario() -> None:
        async def _boom(m):  # type: ignore[no-untyped-def]
            raise ValueError("boom")

        items = [(lambda: _boom(MULTI_PRD))]

        with pytest.raises(ValueError, match="boom"):
            await run_bounded(items, concurrency=3)

    asyncio.run(scenario())
