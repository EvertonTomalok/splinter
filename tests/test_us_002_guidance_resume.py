from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


def test_merge_guidance_empty_guidance_no_block() -> None:
    from splinter.strategies.direct import _merge_guidance

    result = _merge_guidance("base corrections", None)
    assert result == "base corrections"

    result = _merge_guidance("base corrections", "")
    assert result == "base corrections"

    result = _merge_guidance("base corrections", "   ")
    assert result == "base corrections"


def test_merge_guidance_with_content_adds_block() -> None:
    from splinter.strategies.direct import _merge_guidance

    result = _merge_guidance("base corrections", "do X")
    assert result == "base corrections\n\n## User guidance\ndo X"
    assert result.count("## User guidance") == 1


def test_merge_guidance_empty_corrections() -> None:
    from splinter.strategies.direct import _merge_guidance

    result = _merge_guidance("", "fix the bug")
    assert result == "## User guidance\nfix the bug"
    assert result.count("## User guidance") == 1


def test_merge_guidance_idempotent_no_double_inject() -> None:
    from splinter.strategies.direct import _merge_guidance

    guidance = "fix the issue"
    base = _merge_guidance("corrections", guidance)
    assert base.count("## User guidance") == 1

    remerged = _merge_guidance(base, guidance)
    assert remerged == base
    assert remerged.count("## User guidance") == 1


def test_merge_guidance_different_guidance_appends() -> None:
    from splinter.strategies.direct import _merge_guidance

    base = _merge_guidance("corrections", "first guidance")
    assert base.count("## User guidance") == 1

    result = _merge_guidance(base, "second guidance")
    assert result.count("## User guidance") == 2


def test_merge_guidance_checkpoint_path_merges() -> None:
    from splinter.strategies.direct import _merge_guidance

    checkpoint_corrections = "previous corrections from checkpoint"
    result = _merge_guidance(checkpoint_corrections, "operator typed this")
    assert "previous corrections" in result
    assert "operator typed this" in result
    assert result.count("## User guidance") == 1


def test_merge_guidance_with_multiline_guidance() -> None:
    from splinter.strategies.direct import _merge_guidance

    guidance = "line 1\nline 2\nline 3"
    result = _merge_guidance("base", guidance)
    assert "line 1\nline 2\nline 3" in result
    assert result.count("## User guidance") == 1


def test_merge_guidance_preserves_existing_structure() -> None:
    from splinter.strategies.direct import _merge_guidance

    existing = "## Previous Section\nold content\n\n## Another Section\nmore content"
    result = _merge_guidance(existing, "new guidance")
    assert "Previous Section" in result
    assert "Another Section" in result
    assert "new guidance" in result
    assert result.count("## User guidance") == 1


def test_merge_guidance_whitespace_normalization() -> None:
    from splinter.strategies.direct import _merge_guidance

    result = _merge_guidance("  base  ", "  guidance  ")
    assert "base" in result
    assert "## User guidance" in result
    assert "guidance" in result
    assert result.count("## User guidance") == 1


def test_run_task_loop_merges_guidance_on_checkpoint_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))

    from unittest.mock import MagicMock, patch

    from splinter.agents.runner import Task
    from splinter.memory.knowledge import KnowledgeStore
    from splinter.memory.session import Session
    from splinter.models.roster import load_ladder
    from splinter.obs.trace import Trace
    from splinter.strategies.direct import DirectStrategy, RunCheckpoint

    session = Session("test_checkpoint_resume")
    session.write("knowledge/plan.md", "# Test Plan\n\ndo something")
    ladder = load_ladder()
    trace = Trace()
    knowledge = KnowledgeStore(session)

    task = Task(
        description="Test task",
        acceptance="Should work",
        eval_skill=None,
        filtered_context="",
    )

    checkpoint = RunCheckpoint(
        tier=1,
        iteration=1,
        task_index=0,
        oc_session=None,
        eval_session=None,
        corrections="previous corrections",
        eval_history=[],
        reason="test",
        gate_output="",
    )

    captured_corrections: str | None = None

    def mock_chain_handle(ctx):  # type: ignore[no-untyped-def]
        nonlocal captured_corrections
        captured_corrections = ctx.corrections

    strategy = DirectStrategy()

    with patch("splinter.strategies.direct.build_chain") as mock_build:
        mock_chain = MagicMock()
        mock_chain.handle = mock_chain_handle
        mock_build.return_value = mock_chain

        with patch.object(strategy, "_start_tier", return_value=1):
            try:
                strategy._run_task_loop(
                    task,
                    session,
                    ladder,
                    trace,
                    knowledge,
                    task_index=0,
                    effort=None,
                    budget=None,
                    max_iterations=1,
                    localization="",
                    resume=True,
                    checkpoint=checkpoint,
                    user_guidance="fix this bug",
                )
            except Exception:
                pass

    assert captured_corrections is not None
    assert "previous corrections" in captured_corrections
    assert "fix this bug" in captured_corrections
    assert captured_corrections.count("## User guidance") == 1


def test_run_task_loop_merges_guidance_on_resume_no_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))

    from unittest.mock import MagicMock, patch

    from splinter.agents.runner import Task
    from splinter.memory.knowledge import KnowledgeStore
    from splinter.memory.session import Session
    from splinter.models.roster import load_ladder
    from splinter.obs.trace import Trace
    from splinter.strategies.direct import DirectStrategy

    session = Session("test_no_checkpoint")
    session.write("knowledge/plan.md", "# Test Plan\n\ndo something")
    ladder = load_ladder()
    trace = Trace()
    knowledge = KnowledgeStore(session)

    task = Task(
        description="Test task",
        acceptance="Should work",
        eval_skill=None,
        filtered_context="",
    )

    captured_corrections: str | None = None

    def mock_chain_handle(ctx):  # type: ignore[no-untyped-def]
        nonlocal captured_corrections
        captured_corrections = ctx.corrections

    strategy = DirectStrategy()

    with patch("splinter.strategies.direct.build_chain") as mock_build:
        mock_chain = MagicMock()
        mock_chain.handle = mock_chain_handle
        mock_build.return_value = mock_chain

        with patch.object(strategy, "_start_tier", return_value=1):
            try:
                strategy._run_task_loop(
                    task,
                    session,
                    ladder,
                    trace,
                    knowledge,
                    task_index=0,
                    effort=None,
                    budget=None,
                    max_iterations=1,
                    localization="",
                    resume=True,
                    checkpoint=None,
                    user_guidance="operator guidance text",
                )
            except Exception:
                pass

    assert captured_corrections is not None
    assert "operator guidance text" in captured_corrections
    assert captured_corrections.count("## User guidance") == 1


def test_run_task_loop_no_guidance_no_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))

    from unittest.mock import MagicMock, patch

    from splinter.agents.runner import Task
    from splinter.memory.knowledge import KnowledgeStore
    from splinter.memory.session import Session
    from splinter.models.roster import load_ladder
    from splinter.obs.trace import Trace
    from splinter.strategies.direct import DirectStrategy

    session = Session("test_no_guidance")
    session.write("knowledge/plan.md", "# Test Plan\n\ndo something")
    ladder = load_ladder()
    trace = Trace()
    knowledge = KnowledgeStore(session)

    task = Task(
        description="Test task",
        acceptance="Should work",
        eval_skill=None,
        filtered_context="",
    )

    captured_corrections: str | None = None

    def mock_chain_handle(ctx):  # type: ignore[no-untyped-def]
        nonlocal captured_corrections
        captured_corrections = ctx.corrections

    strategy = DirectStrategy()

    with patch("splinter.strategies.direct.build_chain") as mock_build:
        mock_chain = MagicMock()
        mock_chain.handle = mock_chain_handle
        mock_build.return_value = mock_chain

        with patch.object(strategy, "_start_tier", return_value=1):
            try:
                strategy._run_task_loop(
                    task,
                    session,
                    ladder,
                    trace,
                    knowledge,
                    task_index=0,
                    effort=None,
                    budget=None,
                    max_iterations=1,
                    localization="",
                    resume=True,
                    checkpoint=None,
                    user_guidance=None,
                )
            except Exception:
                pass

    assert captured_corrections is not None
    assert "## User guidance" not in captured_corrections


def test_merge_guidance_exact_substring_match() -> None:
    from splinter.strategies.direct import _merge_guidance

    guidance = "fix the bug"
    base = _merge_guidance("corrections", guidance)

    remerged = _merge_guidance(base, guidance)
    assert remerged == base
    assert remerged.count("## User guidance") == 1

    block = f"## User guidance\n{guidance}"
    assert block in remerged
