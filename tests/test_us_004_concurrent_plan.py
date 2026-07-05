"""Tests for concurrent planning (US-004): Planner.plan via run_bounded."""

from __future__ import annotations

import asyncio

import pytest

from splinter.agents import planner
from splinter.agents.planner import Planner, parse_stories

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


def test_dispatches_via_run_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}
    real_run_bounded = planner.run_bounded

    async def spy(items, concurrency=None):  # type: ignore[no-untyped-def]
        calls["n_items"] = len(items)
        calls["concurrency"] = concurrency
        return await real_run_bounded(items, concurrency=concurrency)

    monkeypatch.setattr(planner, "run_bounded", spy)

    tasks = Planner(concurrency=2).plan(MULTI_PRD)

    assert len(tasks) == 3
    assert calls["n_items"] == 3
    assert calls["concurrency"] == 2


def test_one_failing_story_propagates_without_deadlock() -> None:
    async def scenario() -> None:
        p = Planner(concurrency=3)

        async def _boom(m):  # type: ignore[no-untyped-def]
            raise ValueError("boom")

        real_parse = planner._parse_one_story
        matches = list(planner._US_PATTERN.finditer(MULTI_PRD))

        async def _item(m, idx):  # type: ignore[no-untyped-def]
            if idx == 1:
                raise ValueError("boom")
            return real_parse(m)

        items = [(lambda m=m, i=i: _item(m, i)) for i, m in enumerate(matches)]

        with pytest.raises(ValueError, match="boom"):
            await planner.run_bounded(items, concurrency=p._concurrency)

    asyncio.run(scenario())
