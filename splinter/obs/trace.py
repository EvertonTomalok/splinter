"""Per-run observability: token/cost/latency accounting and a markdown summary."""

from __future__ import annotations

import time
from dataclasses import dataclass

from splinter.agents.runner import RunResult


@dataclass(frozen=True)
class RunEntry:
    model: str
    tier: int
    iteration: int
    tokens: dict[str, int]
    cost: float
    latency_s: float


class Trace:
    def __init__(self) -> None:
        self.entries: list[RunEntry] = []
        self._start = time.monotonic()

    @property
    def total_cost(self) -> float:
        return sum(e.cost for e in self.entries)

    @property
    def total_tokens(self) -> dict[str, int]:
        inp = sum(e.tokens.get("input", 0) for e in self.entries)
        out = sum(e.tokens.get("output", 0) for e in self.entries)
        return {"input": inp, "output": out}

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start

    def summary(self) -> str:
        lines = [
            "# Trace\n",
            f"- total runs: {len(self.entries)}",
            f"- total cost: ${self.total_cost:.4f}",
            f"- total tokens: {self.total_tokens}",
            f"- elapsed: {self.elapsed:.1f}s\n",
            "## Runs\n",
        ]
        for e in self.entries:
            lines.append(
                f"- iter {e.iteration}: {e.model} (T{e.tier}) "
                f"tokens={e.tokens} cost=${e.cost:.4f} {e.latency_s:.1f}s"
            )
        return "\n".join(lines) + "\n"


def log_run(trace: Trace, result: RunResult, iteration: int) -> None:
    trace.entries.append(
        RunEntry(
            model=result.model,
            tier=result.tier,
            iteration=iteration,
            tokens=result.tokens,
            cost=result.cost,
            latency_s=0.0,
        )
    )
