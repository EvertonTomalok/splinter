"""US-004: events.jsonl is the single canonical source — analyze/cost views are
computed on demand from it, never from regex-parsed markdown."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from splinter.analyze import _trace_metrics, format_run_completion
from splinter.memory.session import Session
from splinter.obs.trace import RunEntry, Trace
from splinter.pipeline import _compute_summary_cost

CASES = [
    pytest.param(
        [
            dict(
                model="sonnet",
                tier=1,
                iteration=1,
                tokens={"input": 100, "output": 50},
                cost=0.01,
                latency_s=1.0,
                task=0,
            ),
            dict(
                model="haiku",
                tier=0,
                iteration=2,
                tokens={"input": 40, "output": 10},
                cost=0.002,
                latency_s=0.5,
                task=0,
                role="eval",
            ),
        ],
        {"cost": 0.012, "runs": 2},
        id="two_entries_mixed_role",
    ),
    pytest.param(
        [
            dict(
                model="opus",
                tier=4,
                iteration=1,
                tokens={"input": 1000, "output": 500},
                cost=1.2345,
                latency_s=12.0,
                task=1,
            ),
        ],
        {"cost": 1.2345, "runs": 1},
        id="single_premium_entry",
    ),
    pytest.param(
        [
            dict(
                model="gpt",
                tier=2,
                iteration=1,
                tokens={},
                cost=0.0,
                latency_s=0.1,
                task=0,
                cost_indeterminate=True,
            ),
            dict(
                model="gpt",
                tier=2,
                iteration=2,
                tokens={"input": 5},
                cost=0.5,
                latency_s=0.2,
                task=0,
            ),
            dict(
                model="sonnet",
                tier=3,
                iteration=1,
                tokens={"input": 300},
                cost=0.3,
                latency_s=3.0,
                task=2,
            ),
        ],
        {"cost": 0.8, "runs": 3},
        id="three_entries_multi_task_indeterminate",
    ),
]


@pytest.mark.parametrize("rows, expected", CASES)
def test_events_jsonl_roundtrip_and_cost_views(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rows: list[dict[str, Any]],
    expected: dict[str, float],
) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_events_table")
    trace = Trace(session=session)
    for row in rows:
        trace.add_entry(RunEntry(**row))

    # (a) round-trips through events.jsonl — nothing lost, nothing invented.
    reloaded = Trace.from_jsonl(session)
    assert len(reloaded.entries) == len(rows)
    for entry, row in zip(reloaded.entries, rows, strict=True):
        assert entry.model == row["model"]
        assert entry.tier == row["tier"]
        assert entry.task == row["task"]
        assert entry.cost == pytest.approx(row["cost"])
        assert entry.tokens == row["tokens"]
        assert entry.cost_indeterminate == row.get("cost_indeterminate", False)
        assert entry.ts  # stamped at persist time, never blank on reload

    # (b) pipeline's cost/runs summary derives from the structured trace, not a
    # markdown re-parse — falls back to `results` only when the trace is empty,
    # which it deliberately never is here.
    cost, runs = _compute_summary_cost(reloaded, [])
    assert cost == pytest.approx(expected["cost"])
    assert runs == expected["runs"]

    # (c) analyze's on-demand metrics + one-line completion summary agree.
    metrics = _trace_metrics(session)
    assert float(metrics["cost"]) == pytest.approx(expected["cost"])
    assert metrics["runs"] == str(expected["runs"])

    completion = format_run_completion(session)
    assert completion == f"${expected['cost']:.4f} · {expected['runs']} runs"


def test_concurrent_add_entry_no_lost_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Parallel cascade workers append RunEntry concurrently — the lock in
    Trace.add_entry + events.append_event must not drop any of them."""
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_events_concurrency")
    trace = Trace(session=session)

    n_entries = 64
    per_entry_cost = 0.001

    def worker(i: int) -> None:
        trace.add_entry(
            RunEntry(
                model="m",
                tier=i % 5,
                iteration=i,
                tokens={"input": 1, "output": 1},
                cost=per_entry_cost,
                latency_s=0.0,
                task=i % 3,
            )
        )

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(worker, range(n_entries)))

    assert len(trace.entries) == n_entries

    reloaded = Trace.from_jsonl(session)
    assert len(reloaded.entries) == n_entries
    assert reloaded.total_cost == pytest.approx(per_entry_cost * n_entries)

    cost, runs = _compute_summary_cost(reloaded, [])
    assert runs == n_entries
    assert cost == pytest.approx(per_entry_cost * n_entries)
