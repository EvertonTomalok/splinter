"""Bounded-concurrency async fan-out for stage item-callables."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from typing import TypeVar, cast

T = TypeVar("T")

DEFAULT_CONCURRENCY: int = 4


def _default_concurrency() -> int:
    raw = os.environ.get("SPLINTER_STAGE_CONCURRENCY")
    if raw is None:
        return DEFAULT_CONCURRENCY
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_CONCURRENCY
    return value if value > 0 else DEFAULT_CONCURRENCY


async def run_bounded(
    items: list[Callable[[], Awaitable[T]]],
    concurrency: int | None = None,
) -> list[T]:
    """Run item thunks concurrently under a semaphore, order-stable, fail-fast."""
    if not items:
        return []

    default_bound = concurrency if concurrency is not None and concurrency > 0 else None
    requested = default_bound if default_bound is not None else _default_concurrency()
    bound = max(1, min(requested, len(items)))
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
