"""Test task segmentation and iteration parsing for analyze TUI."""

from __future__ import annotations

from pathlib import Path

import pytest

from splinter.analyze import _eval_segments, _iterations, _tasks
from splinter.tui import _cap_payload


class TestTasks:
    """Test _tasks() parsing of loop.md with task headers."""

    def test_single_task_no_header(self) -> None:
        """Single task without header returns [(1, '', loop_md)]."""
        loop_md = "## Iteration 1\nTier 1\nverdict: PASS"
        result = _tasks(loop_md)
        assert len(result) == 1
        assert result[0][0] == 1
        assert result[0][1] == ""
        assert "## Iteration 1" in result[0][2]

    def test_multi_task_with_headers(self) -> None:
        """Multi-task with headers splits correctly."""
        loop_md = """# Task 1/2: First task
## Iteration 1
Tier 1
verdict: PASS

# Task 2/2: Second task
## Iteration 1
Tier 2
verdict: RETRY
"""
        result = _tasks(loop_md)
        assert len(result) == 2
        assert result[0][0] == 1
        assert result[0][1] == "First task"
        assert "## Iteration 1" in result[0][2]
        assert "Tier 1" in result[0][2]

        assert result[1][0] == 2
        assert result[1][1] == "Second task"
        assert "## Iteration 1" in result[1][2]
        assert "Tier 2" in result[1][2]

    def test_duplicate_headers_deduped(self) -> None:
        """Regression: a resumed run may re-append a task header; each task number
        must render once, not once per duplicate header."""
        loop_md = """# Task 1 [parallel]: First
## Iteration 1
verdict: PASS

# Task 2 [parallel]: Second
## Iteration 1
verdict: PASS

# Task 1 [parallel]: First
# Task 2 [parallel]: Second
"""
        result = _tasks(loop_md)
        assert [t[0] for t in result] == [1, 2]
        assert result[0][1] == "First"
        assert result[1][1] == "Second"

    def test_empty_loop_md(self) -> None:
        """Empty loop.md returns single empty task."""
        result = _tasks("")
        assert len(result) == 1
        assert result[0] == (1, "", "")

    def test_task_with_empty_title(self) -> None:
        """Task header with empty title."""
        loop_md = "# Task 1/1:\n## Iteration 1\nverdict: PASS"
        result = _tasks(loop_md)
        assert len(result) == 1
        assert result[0][1] == ""


class TestEvalSegments:
    """Test _eval_segments() detection of task boundaries."""

    def test_single_task_no_segmentation(self) -> None:
        """Single task count returns [eval_md] unsegmented."""
        eval_md = "### Iter 1:\nPASS\n### Iter 2:\nRETRY"
        result = _eval_segments(eval_md, 1)
        assert len(result) == 1
        assert result[0] == eval_md

    def test_two_tasks_reset_detection(self) -> None:
        """Reset detection splits on iter number <= previous."""
        eval_md = """### Iter 1:
PASS
### Iter 2:
RETRY
### Iter 1:
PASS
### Iter 2:
FAIL
"""
        result = _eval_segments(eval_md, 2)
        assert len(result) == 2

        assert "### Iter 1:\nPASS" in result[0]
        assert "### Iter 2:\nRETRY" in result[0]

        assert "### Iter 1:\nPASS" in result[1]
        assert "### Iter 2:\nFAIL" in result[1]

    def test_empty_eval_returns_blanks(self) -> None:
        """Empty eval.md with task_count > 1 returns empty strings."""
        result = _eval_segments("", 3)
        assert len(result) == 3
        assert all(seg == "" for seg in result)

    def test_pads_to_task_count(self) -> None:
        """Result is padded to task_count length."""
        eval_md = "### Iter 1:\nPASS"
        result = _eval_segments(eval_md, 3)
        assert len(result) == 3
        assert result[0].strip()
        assert result[1] == ""
        assert result[2] == ""


class TestCapPayload:
    """Test _cap_payload() passthrough behavior."""

    def test_short_text_unchanged(self) -> None:
        """Text under limit is returned unchanged."""
        text = "hello world"
        result = _cap_payload(text, limit=100)
        assert result == text

    def test_long_text_unchanged(self) -> None:
        """Text over limit is not truncated."""
        text = "a" * 1000
        result = _cap_payload(text, limit=200)
        assert result == text

    def test_limit_argument_ignored(self) -> None:
        """Legacy limit arg remains accepted but no truncation is applied."""
        text = "x" * 10000
        result = _cap_payload(text, limit=500)
        assert result == text


class TestIterations:
    """Test _iterations() extracts tier and verdict from task body."""

    def test_single_iteration(self) -> None:
        """Extract single iteration."""
        task_body = """## Iteration 1
tier 1
verdict: PASS
"""
        result = _iterations(task_body)
        assert len(result) == 1
        assert result[0][0] == 1
        assert result[0][1] == "T1"
        assert result[0][2] == "PASS"

    def test_multi_iteration(self) -> None:
        """Extract multiple iterations in order."""
        task_body = """## Iteration 1
tier 1
verdict: PASS

## Iteration 2
tier 2
verdict: RETRY

## Iteration 3
tier 3
verdict: FAIL
"""
        result = _iterations(task_body)
        assert len(result) == 3
        assert result[0][2] == "PASS"
        assert result[1][2] == "RETRY"
        assert result[2][2] == "FAIL"

    def test_missing_tier_or_verdict(self) -> None:
        """Missing tier/verdict defaults to ?."""
        task_body = """## Iteration 1
(no metadata)
"""
        result = _iterations(task_body)
        assert len(result) == 1
        assert result[0][1] == "T?"
        assert result[0][2] == "?"


class TestCostReconciliation:
    """_trace_metrics computes on demand from events.jsonl (no markdown parse)."""

    def test_trace_metrics_cost_matches_trace_total_cost(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
        from splinter.analyze import _trace_metrics
        from splinter.memory.session import Session
        from splinter.obs.trace import RunEntry, Trace

        session = Session("ses_cost_reconcile")
        trace = Trace(session=session)
        for entry in (
            RunEntry(
                model="m1",
                tier=1,
                iteration=1,
                tokens={"input": 100, "output": 50},
                cost=0.0100,
                latency_s=0.0,
                task=0,
            ),
            RunEntry(
                model="m1",
                tier=0,
                iteration=2,
                tokens={"input": 80, "output": 40},
                cost=0.0200,
                latency_s=0.0,
                task=0,
                role="eval",
            ),
            RunEntry(
                model="m1",
                tier=2,
                iteration=3,
                tokens={"input": 120, "output": 60},
                cost=0.0300,
                latency_s=0.0,
                task=0,
            ),
        ):
            trace.add_entry(entry)

        metrics = _trace_metrics(session)

        assert float(metrics["cost"]) == pytest.approx(trace.total_cost, abs=1e-6)
        assert trace.total_cost == pytest.approx(0.0600)

    def test_trace_metrics_excludes_pre_run_cost(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
        from splinter.analyze import _trace_metrics
        from splinter.memory.session import Session
        from splinter.obs.trace import RunEntry, Trace

        session = Session("ses_cost_pre_run")
        trace = Trace(session=session)
        trace.add_entry(
            RunEntry(model="m", tier=1, iteration=1, tokens={}, cost=0.0500, latency_s=0.0, task=0)
        )

        metrics = _trace_metrics(session)
        run_cost = float(metrics["cost"])

        assert run_cost == pytest.approx(0.0500, abs=1e-6)
        assert run_cost < 0.0501
