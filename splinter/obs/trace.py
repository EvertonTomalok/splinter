"""Per-run observability: token/cost/latency accounting and a markdown summary.

``Trace`` is the in-memory view; when constructed with a ``session`` every entry
is also persisted to that session's canonical ``events.jsonl`` (see
:mod:`splinter.obs.events`) — the single write path for run entries.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from splinter.agents.runner import RunResult
from splinter.memory.session import Session
from splinter.obs import events


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
    def __init__(self, session: Session | None = None) -> None:
        self.entries: list[RunEntry] = []
        self._session = session
        self._start = time.monotonic()
        #: Guards ``entries`` so parallel worker threads can append while the
        #: dispatch loop reads ``total_cost`` for the budget gate — without it a
        #: read could sum a half-updated list and mis-gate the budget.
        self._lock = threading.Lock()

    def add_entry(self, entry: RunEntry) -> None:
        """Append a run entry under the lock (thread-safe for parallel tasks) and,
        when a session is attached, persist it to ``events.jsonl`` — the only path
        that writes a run entry to disk."""
        with self._lock:
            self.entries.append(entry)
        if self._session is not None:
            ts = entry.ts or datetime.now(timezone.utc).isoformat()
            events.append_event(
                self._session,
                events.Event(
                    type="run",
                    ts=ts,
                    payload={
                        "model": entry.model,
                        "tier": entry.tier,
                        "iteration": entry.iteration,
                        "tokens": entry.tokens,
                        "cost": entry.cost,
                        "latency_s": entry.latency_s,
                        "task": entry.task,
                        "role": entry.role,
                        "cost_indeterminate": entry.cost_indeterminate,
                    },
                ),
            )

    @property
    def total_cost(self) -> float:
        with self._lock:
            entries = list(self.entries)
        return sum(e.cost for e in entries)

    @property
    def total_tokens(self) -> dict[str, int]:
        with self._lock:
            entries = list(self.entries)
        out: dict[str, int] = {}
        for e in entries:
            for key, val in e.tokens.items():
                out[key] = out.get(key, 0) + val
        return out

    @property
    def elapsed(self) -> float:
        """Wall-clock span across persisted entries when they carry real timestamps
        (e.g. after :meth:`from_jsonl`); otherwise time since this Trace was built —
        accurate for a live, non-resumed, single-process run."""
        timestamps = [e.ts for e in self.entries if e.ts]
        if len(timestamps) >= 2:
            try:
                parsed = sorted(datetime.fromisoformat(t) for t in timestamps)
                return (parsed[-1] - parsed[0]).total_seconds()
            except ValueError:
                pass
        return time.monotonic() - self._start

    @property
    def cost_by_model(self) -> dict[str, float]:
        with self._lock:
            entries = list(self.entries)
        out: dict[str, float] = {}
        for e in entries:
            out[e.model] = out.get(e.model, 0.0) + e.cost
        return out

    def task_cost(self, task: int) -> float:
        with self._lock:
            entries = list(self.entries)
        return sum(e.cost for e in entries if e.task == task)

    def task_entries(self, task: int) -> list[RunEntry]:
        with self._lock:
            entries = list(self.entries)
        return [e for e in entries if e.task == task]

    def model_entries(self, model: str) -> list[RunEntry]:
        with self._lock:
            entries = list(self.entries)
        return [e for e in entries if e.model == model]

    def summary(self) -> str:
        with self._lock:
            entries = list(self.entries)

        total_cost = sum(e.cost for e in entries)
        total_tokens: dict[str, int] = {}
        for e in entries:
            for key, val in e.tokens.items():
                total_tokens[key] = total_tokens.get(key, 0) + val

        by_model: dict[str, float] = {}
        for e in entries:
            by_model[e.model] = by_model.get(e.model, 0.0) + e.cost

        lines = [
            "# Trace\n",
            f"- total runs: {len(entries)}",
            f"- total cost: ${total_cost:.4f}",
            f"- total tokens: {total_tokens}",
            f"- elapsed: {self.elapsed:.1f}s",
        ]

        for model in sorted(by_model):
            lines.append(f"- {model}: ${by_model[model]:.4f}")
        lines.append("")

        tasks = sorted({e.task for e in entries})
        if len(tasks) > 1 or (tasks and tasks[0] > 0):
            lines.append("## Per-task\n")
            for t in tasks:
                t_entries = [e for e in entries if e.task == t]
                t_cost = sum(e.cost for e in t_entries)
                t_tokens = {
                    "input": sum(e.tokens.get("input", 0) for e in t_entries),
                    "output": sum(e.tokens.get("output", 0) for e in t_entries),
                }
                lines.append(f"- task {t}: {len(t_entries)} runs, ${t_cost:.4f}, tokens={t_tokens}")
            lines.append("")

        if entries:
            lines.append("## Per-model\n")
            for model in sorted(by_model):
                m_entries = [e for e in entries if e.model == model]
                m_cost = by_model[model]
                m_tokens = {
                    "input": sum(e.tokens.get("input", 0) for e in m_entries),
                    "output": sum(e.tokens.get("output", 0) for e in m_entries),
                }
                lines.append(f"- {model}: {len(m_entries)} runs, ${m_cost:.4f}, tokens={m_tokens}")
            lines.append("")

        lines.append("## Runs\n")
        for e in entries:
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
    def from_jsonl(cls, session: Session) -> Trace:
        """Rebuild a ``Trace`` from the session's ``events.jsonl`` and attach the
        session so subsequent ``add_entry`` calls keep persisting to it."""
        trace = cls(session=session)
        trace.entries = events.load_run_entries(session)
        return trace


def log_run(trace: Trace, result: RunResult, iteration: int, task: int = 0) -> None:
    trace.add_entry(
        RunEntry(
            model=result.model,
            tier=result.tier,
            iteration=iteration,
            tokens=result.tokens,
            cost=result.cost,
            latency_s=result.latency_s,
            task=task,
            cost_indeterminate=result.cost_indeterminate,
            ts=result.ts or datetime.now(timezone.utc).isoformat(),
        )
    )
