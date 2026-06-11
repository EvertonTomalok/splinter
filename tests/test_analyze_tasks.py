"""Test task segmentation and iteration parsing for analyze TUI."""

from __future__ import annotations

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
    """Test _cap_payload() truncation."""

    def test_short_text_unchanged(self) -> None:
        """Text under limit is returned unchanged."""
        text = "hello world"
        result = _cap_payload(text, limit=100)
        assert result == text

    def test_long_text_truncated(self) -> None:
        """Text over limit is head + marker + tail."""
        text = "a" * 1000
        result = _cap_payload(text, limit=200)
        assert len(result) <= 250
        assert "…[truncated" in result
        assert "chars]…" in result
        assert result.startswith("a")
        assert result.endswith("a")

    def test_truncation_preserves_counts(self) -> None:
        """Truncated text mentions dropped char count."""
        text = "x" * 10000
        result = _cap_payload(text, limit=500)
        assert "truncated" in result
        dropped = 10000 - result.count("x")
        assert dropped > 0


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
