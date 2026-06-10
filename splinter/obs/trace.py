"""Per-run observability: token/cost/latency accounting and a markdown summary."""

from __future__ import annotations

import re
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
    task: int = 0


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

    def task_cost(self, task: int) -> float:
        return sum(e.cost for e in self.entries if e.task == task)

    def task_entries(self, task: int) -> list[RunEntry]:
        return [e for e in self.entries if e.task == task]

    def summary(self) -> str:
        lines = [
            "# Trace\n",
            f"- total runs: {len(self.entries)}",
            f"- total cost: ${self.total_cost:.4f}",
            f"- total tokens: {self.total_tokens}",
            f"- elapsed: {self.elapsed:.1f}s\n",
        ]

        tasks = sorted({e.task for e in self.entries})
        if len(tasks) > 1 or (tasks and tasks[0] > 0):
            lines.append("## Per-task\n")
            for t in tasks:
                t_entries = self.task_entries(t)
                t_cost = sum(e.cost for e in t_entries)
                t_tokens = {
                    "input": sum(e.tokens.get("input", 0) for e in t_entries),
                    "output": sum(e.tokens.get("output", 0) for e in t_entries),
                }
                lines.append(f"- task {t}: {len(t_entries)} runs, ${t_cost:.4f}, tokens={t_tokens}")
            lines.append("")

        lines.append("## Runs\n")
        for e in self.entries:
            task_prefix = f"task {e.task} " if e.task else ""
            lines.append(
                f"- {task_prefix}iter {e.iteration}: {e.model} (T{e.tier}) "
                f"tokens={e.tokens} cost=${e.cost:.4f} {e.latency_s:.1f}s"
            )
        return "\n".join(lines) + "\n"

    @classmethod
    def from_markdown(cls, md: str) -> Trace:
        trace = cls()
        run_pattern = re.compile(
            r"- (?:task (\d+) )?iter (\d+): (.+?) \(T(\d+)\) "
            r"tokens=\{([^}]*)\} cost=\$([\d.]+) ([\d.]+)s"
        )
        for m in run_pattern.finditer(md):
            task = int(m.group(1)) if m.group(1) else 0
            iteration = int(m.group(2))
            model = m.group(3)
            tier = int(m.group(4))
            tokens: dict[str, int] = {}
            for tm in re.finditer(r"'(\w+)':\s*(\d+)", m.group(5)):
                tokens[tm.group(1)] = int(tm.group(2))
            cost = float(m.group(6))
            latency = float(m.group(7))
            trace.entries.append(
                RunEntry(
                    model=model,
                    tier=tier,
                    iteration=iteration,
                    tokens=tokens,
                    cost=cost,
                    latency_s=latency,
                    task=task,
                )
            )
        return trace


def log_run(trace: Trace, result: RunResult, iteration: int, task: int = 0) -> None:
    trace.entries.append(
        RunEntry(
            model=result.model,
            tier=result.tier,
            iteration=iteration,
            tokens=result.tokens,
            cost=result.cost,
            latency_s=0.0,
            task=task,
        )
    )
