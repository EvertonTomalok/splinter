"""Tests for concurrent localize(): single writer, determinism, byte-identity."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from splinter.agents import localizer
from splinter.agents.localizer import CodeAnchor, _localize_items, localize
from splinter.memory.knowledge import KnowledgeStore
from splinter.memory.session import Session
from splinter.models.roster import Ladder

_THREE_STORY_PRD = """# PRD

### US-001: First story
Do the first thing.

### US-002: Second story
Do the second thing.

### US-003: Third story
Do the third thing.
"""

_SINGLE_STORY_PRD = "# PRD\n\nJust build the one feature, no story split here.\n"


def _ladder() -> Ladder:
    return Ladder(
        tiers=[],
        effort_map={},
        eval_model="sonnet",
        eval_effort="high",
        planner_model="sonnet",
        planner_effort="high",
        localizer_recall_model="recall-model",
        localizer_recall_large_model="recall-large-model",
        localizer_precision_model="precision-model",
    )


def _session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, session_id: str) -> Session:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path / "home"))
    return Session(session_id)


def _fake_recall_phase(delays: dict[str, float] | None = None) -> Callable[..., str]:
    """Build a deterministic ``_recall_phase`` stub keyed by the US-NNN marker."""
    delays = delays or {}

    def fake(
        item_text: str,
        search_results: str,
        model: str,
        variant: str,
        timeout: int | None,
        *,
        agent: str,
        session: object,
    ) -> str:
        m = re.search(r"US-(\d+)", item_text)
        n = m.group(1) if m else "0"
        for marker, secs in delays.items():
            if marker in item_text:
                time.sleep(secs)
        return json.dumps(
            [
                {
                    "file": f"file_{n}.py",
                    "symbol": f"Symbol{n}",
                    "reason": f"reason for story {n}",
                    "confidence": 0.9,
                    "line_start": 1,
                    "line_end": 10,
                }
            ]
        )

    return fake


# ---------------------------------------------------------------------------
# _localize_items
# ---------------------------------------------------------------------------


def test_localize_items_no_headers_single_item() -> None:
    """A PRD with no US-NNN headers yields exactly one whole-PRD item."""
    assert _localize_items(_SINGLE_STORY_PRD) == [_SINGLE_STORY_PRD]


def test_localize_items_single_header_stays_single_item() -> None:
    """A single US-NNN header does not trigger a split (baseline stays N=1)."""
    prd = "# PRD\n\n### US-001: Only story\nDo it.\n"
    assert _localize_items(prd) == [prd]


def test_localize_items_multiple_headers_split_in_order() -> None:
    """2+ US-NNN headers split into one item per story, in document order."""
    items = _localize_items(_THREE_STORY_PRD)
    assert len(items) == 3
    assert "US-001" in items[0] and "US-002" not in items[0]
    assert "US-002" in items[1] and "US-003" not in items[1]
    assert "US-003" in items[2]


# ---------------------------------------------------------------------------
# localize() — concurrency, single-writer, determinism
# ---------------------------------------------------------------------------


def test_localize_byte_identical_serial_vs_concurrent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Forcing cap=1 vs a real concurrent cap yields byte-identical localization.md."""
    monkeypatch.setattr(localizer, "_recall_phase", _fake_recall_phase())
    ladder = _ladder()

    monkeypatch.setattr(localizer, "default_max_concurrency", lambda: 1)
    session_serial = _session(tmp_path, monkeypatch, "serial")
    localize(_THREE_STORY_PRD, session_serial, ladder, repo_path=str(tmp_path))
    serial_bytes = session_serial.read("knowledge/localization.md")

    monkeypatch.setattr(localizer, "default_max_concurrency", lambda: 8)
    session_concurrent = _session(tmp_path, monkeypatch, "concurrent")
    localize(_THREE_STORY_PRD, session_concurrent, ladder, repo_path=str(tmp_path))
    concurrent_bytes = session_concurrent.read("knowledge/localization.md")

    assert serial_bytes == concurrent_bytes
    assert "file_001.py" in serial_bytes
    assert "file_003.py" in serial_bytes


def test_localize_single_writer_no_double_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KnowledgeStore.write_note is called exactly once per localize() call."""
    monkeypatch.setattr(localizer, "_recall_phase", _fake_recall_phase())
    monkeypatch.setattr(localizer, "default_max_concurrency", lambda: 4)
    ladder = _ladder()
    session = _session(tmp_path, monkeypatch, "single-writer")

    calls: list[str] = []
    orig_write_note = KnowledgeStore.write_note

    def counting_write_note(self: KnowledgeStore, topic: str, md: str) -> Path:
        calls.append(topic)
        return orig_write_note(self, topic, md)

    monkeypatch.setattr(KnowledgeStore, "write_note", counting_write_note)

    localize(_THREE_STORY_PRD, session, ladder, repo_path=str(tmp_path))

    assert calls == ["localization"]


def test_localize_deterministic_order_under_jitter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """First-in-input item finishes last but still appears first in output."""
    monkeypatch.setattr(localizer, "_recall_phase", _fake_recall_phase(delays={"US-001": 0.05}))
    monkeypatch.setattr(localizer, "default_max_concurrency", lambda: 4)
    ladder = _ladder()
    session = _session(tmp_path, monkeypatch, "jitter")

    anchors = localize(_THREE_STORY_PRD, session, ladder, repo_path=str(tmp_path))

    assert [a.file for a in anchors] == ["file_001.py", "file_002.py", "file_003.py"]


def test_localize_n1_baseline_matches_original_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Single-item (no story split) run writes the exact pre-existing block format."""

    def fake_recall_phase(
        item_text: str,
        search_results: str,
        model: str,
        variant: str,
        timeout: int | None,
        *,
        agent: str,
        session: object,
    ) -> str:
        return json.dumps(
            [
                {
                    "file": "app.py",
                    "symbol": "main",
                    "reason": "entry point",
                    "confidence": 0.9,
                    "line_start": 1,
                    "line_end": 5,
                }
            ]
        )

    monkeypatch.setattr(localizer, "_recall_phase", fake_recall_phase)
    ladder = _ladder()
    session = _session(tmp_path, monkeypatch, "n1-baseline")

    anchors = localize(_SINGLE_STORY_PRD, session, ladder, repo_path=str(tmp_path))
    content = session.read("knowledge/localization.md")

    assert anchors == [
        CodeAnchor(
            file="app.py",
            symbol="main",
            reason="entry point",
            confidence=0.9,
            line_start=1,
            line_end=5,
            relevance="hot",
        )
    ]
    expected = (
        "# Localization\n\n"
        "file: app.py\n"
        "symbol: main\n"
        "line_start: 1\n"
        "line_end: 5\n"
        "reason: entry point\n"
        "confidence: 0.9\n"
        "relevance: hot\n"
        "\n"
    )
    assert content == expected
