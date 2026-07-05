"""Bounded-concurrency async fan-out for stage item-callables."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar, cast

T = TypeVar("T")


async def run_bounded(
    items: list[Callable[[], Awaitable[T]]],
    concurrency: int,
) -> list[T]:
    """Run item thunks concurrently under a semaphore, order-stable, fail-fast."""
    if not items:
        return []

    bound = max(1, min(concurrency, len(items)))
    sem = asyncio.Semaphore(bound)

    async def _guarded(idx: int, thunk: Callable[[], Awaitable[T]]) -> tuple[int, T]:
        async with sem:
            return idx, await thunk()

    tasks = [asyncio.ensure_future(_guarded(i, thunk)) for i, thunk in enumerate(items)]

    try:
        pairs = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    results = cast("list[T]", [None] * len(items))
    for idx, value in pairs:
        results[idx] = value
    return results
