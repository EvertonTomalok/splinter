"""Tests for bounded async fan-out primitive."""

from __future__ import annotations

import asyncio

import pytest

from splinter.strategies.fanout import run_bounded


def test_returns_input_order() -> None:
    async def scenario() -> None:
        n = 5

        def make_thunk(i: int):
            async def thunk() -> int:
                await asyncio.sleep((n - i) * 0.01)
                return i

            return thunk

        items = [make_thunk(i) for i in range(n)]
        results = await run_bounded(items, concurrency=4)
        assert results == list(range(n))

    asyncio.run(scenario())


def test_bounded_concurrency() -> None:
    async def scenario() -> None:
        current = 0
        observed_max = 0
        lock = asyncio.Lock()

        def make_thunk(i: int):
            async def thunk() -> int:
                nonlocal current, observed_max
                async with lock:
                    current += 1
                    observed_max = max(observed_max, current)
                await asyncio.sleep(0.01)
                async with lock:
                    current -= 1
                return i

            return thunk

        items = [make_thunk(i) for i in range(6)]
        results = await run_bounded(items, concurrency=2)
        assert results == list(range(6))
        assert observed_max <= 2

    asyncio.run(scenario())


def test_one_fails_cancels_others() -> None:
    async def scenario() -> None:
        cancelled = []

        async def failing() -> int:
            await asyncio.sleep(0.005)
            raise ValueError("boom")

        def make_sibling(i: int):
            async def thunk() -> int:
                try:
                    await asyncio.sleep(1)
                except asyncio.CancelledError:
                    cancelled.append(i)
                    raise
                return i

            return thunk

        items = [make_sibling(0), failing, make_sibling(2)]

        before = asyncio.all_tasks() - {asyncio.current_task()}
        with pytest.raises(ValueError, match="boom"):
            await run_bounded(items, concurrency=3)
        after = asyncio.all_tasks() - {asyncio.current_task()} - before
        assert all(task.done() for task in after)
        assert cancelled

    asyncio.run(scenario())


def test_empty_items() -> None:
    async def scenario() -> None:
        assert await run_bounded([], concurrency=4) == []

    asyncio.run(scenario())
