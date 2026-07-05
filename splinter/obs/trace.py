"""Per-run observability: token/cost/latency accounting and a markdown summary."""

from __future__ import annotations

import re
import threading
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
    role: str = "run"
    cost_indeterminate: bool = False
    schema_version: int = 1
    ts: str = ""


class Trace:
    def __init__(self) -> None:
        self.entries: list[RunEntry] = []
        self._start = time.monotonic()
        #: Guards ``entries`` so parallel worker threads can append while the
        #: dispatch loop reads ``total_cost`` for the budget gate — without it a
        #: read could sum a half-updated list and mis-gate the budget.
        self._lock = threading.Lock()

    def add_entry(self, entry: RunEntry) -> None:
        """Append a run entry under the lock (thread-safe for parallel tasks)."""
        with self._lock:
            self.entries.append(entry)

    @property
    def total_cost(self) -> float:
        with self._lock:
            entries = list(self.entries)
        return sum(e.cost for e in entries)

    @property
    def total_tokens(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.entries:
            for key, val in e.tokens.items():
                out[key] = out.get(key, 0) + val
        return out

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start

    @property
    def cost_by_model(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for e in self.entries:
            out[e.model] = out.get(e.model, 0.0) + e.cost
        return out

    def task_cost(self, task: int) -> float:
        return sum(e.cost for e in self.entries if e.task == task)

    def task_entries(self, task: int) -> list[RunEntry]:
        return [e for e in self.entries if e.task == task]

    def model_entries(self, model: str) -> list[RunEntry]:
        return [e for e in self.entries if e.model == model]

    def summary(self) -> str:
        by_model = self.cost_by_model
        lines = [
            "# Trace\n",
            f"- total runs: {len(self.entries)}",
            f"- total cost: ${self.total_cost:.4f}",
            f"- total tokens: {self.total_tokens}",
            f"- elapsed: {self.elapsed:.1f}s",
        ]

        for model in sorted(by_model):
            lines.append(f"- {model}: ${by_model[model]:.4f}")
        lines.append("")

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

        if self.entries:
            lines.append("## Per-model\n")
            for model in sorted(by_model):
                m_entries = self.model_entries(model)
                m_cost = by_model[model]
                m_tokens = {
                    "input": sum(e.tokens.get("input", 0) for e in m_entries),
                    "output": sum(e.tokens.get("output", 0) for e in m_entries),
                }
                lines.append(f"- {model}: {len(m_entries)} runs, ${m_cost:.4f}, tokens={m_tokens}")
            lines.append("")

        lines.append("## Runs\n")
        for e in self.entries:
            task_prefix = f"task {e.task} " if e.task else ""
            role_label = "eval" if e.role == "eval" else f"T{e.tier}"
            indet_marker = " [!cost]" if e.cost_indeterminate else ""
            ts_suffix = f" @ {e.ts}" if e.ts else ""
            lines.append(
                f"- {task_prefix}iter {e.iteration}: {e.model} "
                f"({role_label}) tokens={e.tokens} "
                f"cost=${e.cost:.4f}{indet_marker} {e.latency_s:.1f}s{ts_suffix}"
            )
        return "\n".join(lines) + "\n"

    @classmethod
    def from_markdown(cls, md: str) -> Trace:
        trace = cls()
        run_pattern = re.compile(
            r"- (?:task (\d+) )?iter (\d+): (.+?) \((?:T(\d+)|eval)\) "
            r"tokens=\{([^}]*)\} cost=\$([\d.]+)( \[!cost\])? ([\d.]+)s(?: @ (\S+))?"
        )
        for m in run_pattern.finditer(md):
            task = int(m.group(1)) if m.group(1) else 0
            iteration = int(m.group(2))
            model = m.group(3)
            if m.group(4) is not None:
                tier = int(m.group(4))
                role = "run"
            else:
                tier = 0
                role = "eval"
            tokens: dict[str, int] = {}
            for tm in re.finditer(r"'(\w+)':\s*(\d+)", m.group(5)):
                tokens[tm.group(1)] = int(tm.group(2))
            cost = float(m.group(6))
            cost_indeterminate = m.group(7) is not None
            latency = float(m.group(8))
            ts = m.group(9) or ""
            trace.entries.append(
                RunEntry(
                    model=model,
                    tier=tier,
                    iteration=iteration,
                    tokens=tokens,
                    cost=cost,
                    latency_s=latency,
                    task=task,
                    role=role,
                    cost_indeterminate=cost_indeterminate,
                    ts=ts,
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
            latency_s=result.latency_s,
            task=task,
            cost_indeterminate=result.cost_indeterminate,
            ts=result.ts,
        )
    )
