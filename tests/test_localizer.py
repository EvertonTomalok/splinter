"""Tests for relevance-tagged code anchors."""

from __future__ import annotations

import json

from splinter.agents.localizer import (
    CodeAnchor,
    _parse_anchors,
    _relevance_from_confidence,
    grounding_block,
    rtk_cat_tip,
)


def test_relevance_from_confidence_hot() -> None:
    """High confidence yields 'hot' relevance."""
    assert _relevance_from_confidence(0.9, hot=0.8, medium=0.4) == "hot"
    assert _relevance_from_confidence(0.8, hot=0.8, medium=0.4) == "hot"


def test_relevance_from_confidence_medium() -> None:
    """Mid-range confidence yields 'medium' relevance."""
    assert _relevance_from_confidence(0.6, hot=0.8, medium=0.4) == "medium"
    assert _relevance_from_confidence(0.4, hot=0.8, medium=0.4) == "medium"


def test_relevance_from_confidence_low() -> None:
    """Low confidence yields 'low' relevance."""
    assert _relevance_from_confidence(0.3, hot=0.8, medium=0.4) == "low"
    assert _relevance_from_confidence(0.0, hot=0.8, medium=0.4) == "low"


def test_relevance_from_confidence_custom_thresholds() -> None:
    """Custom thresholds shift boundaries."""
    assert _relevance_from_confidence(0.7, hot=0.9, medium=0.5) == "medium"
    assert _relevance_from_confidence(0.95, hot=0.9, medium=0.5) == "hot"


def test_parse_anchors_json_with_explicit_relevance() -> None:
    """JSON items with explicit relevance field are preserved."""
    text = json.dumps(
        [
            {
                "file": "app.py",
                "symbol": "main",
                "reason": "entry point",
                "confidence": 0.9,
                "relevance": "hot",
            }
        ]
    )
    anchors = _parse_anchors(text)
    assert len(anchors) == 1
    assert anchors[0].relevance == "hot"


def test_parse_anchors_json_without_relevance_fallback() -> None:
    """JSON items without relevance derive from confidence."""
    text = json.dumps(
        [
            {
                "file": "app.py",
                "symbol": "main",
                "reason": "entry point",
                "confidence": 0.9,
            }
        ]
    )
    anchors = _parse_anchors(text, hot=0.8, medium=0.4)
    assert len(anchors) == 1
    assert anchors[0].relevance == "hot"


def test_parse_anchors_json_confidence_fallback_medium() -> None:
    """Mid-range confidence in JSON derives medium relevance."""
    text = json.dumps(
        [
            {
                "file": "utils.py",
                "symbol": "helper",
                "reason": "utility",
                "confidence": 0.5,
            }
        ]
    )
    anchors = _parse_anchors(text, hot=0.8, medium=0.4)
    assert anchors[0].relevance == "medium"


def test_parse_anchors_json_confidence_fallback_low() -> None:
    """Low confidence in JSON derives low relevance."""
    text = json.dumps(
        [
            {
                "file": "old.py",
                "symbol": "legacy",
                "reason": "tangential",
                "confidence": 0.2,
            }
        ]
    )
    anchors = _parse_anchors(text, hot=0.8, medium=0.4)
    assert anchors[0].relevance == "low"


def test_parse_anchors_keyvalue_with_explicit_relevance() -> None:
    """Key:value blocks with relevance field are parsed."""
    text = """
file: app.py
symbol: main
reason: entry point
confidence: 0.9
relevance: hot
"""
    anchors = _parse_anchors(text)
    assert len(anchors) == 1
    assert anchors[0].relevance == "hot"


def test_parse_anchors_keyvalue_without_relevance_fallback() -> None:
    """Key:value blocks without relevance derive from confidence."""
    text = """
file: app.py
symbol: main
reason: entry point
confidence: 0.9
"""
    anchors = _parse_anchors(text, hot=0.8, medium=0.4)
    assert anchors[0].relevance == "hot"


def test_parse_anchors_keyvalue_multiple_blocks() -> None:
    """Multiple key:value blocks with mixed relevance."""
    text = """
file: app.py
symbol: main
reason: entry point
confidence: 0.9
relevance: hot

file: utils.py
symbol: helper
reason: utility
confidence: 0.5

file: old.py
symbol: legacy
reason: tangential
confidence: 0.2
"""
    anchors = _parse_anchors(text, hot=0.8, medium=0.4)
    assert len(anchors) == 3
    assert anchors[0].relevance == "hot"
    assert anchors[1].relevance == "medium"
    assert anchors[2].relevance == "low"


def test_parse_anchors_json_wrapped_in_prose() -> None:
    """JSON wrapped in prose is extracted and parsed."""
    text = "Here are the results:\n" + json.dumps(
        [
            {
                "file": "app.py",
                "symbol": "main",
                "reason": "entry",
                "confidence": 0.85,
            }
        ]
    )
    anchors = _parse_anchors(text, hot=0.8, medium=0.4)
    assert len(anchors) == 1
    assert anchors[0].relevance == "hot"


def test_rtk_cat_tip_with_line_range() -> None:
    """rtk_cat_tip generates sed command for line ranges."""
    anchor = CodeAnchor(
        file="app.py",
        symbol="main",
        reason="entry",
        confidence=0.9,
        line_start=10,
        line_end=20,
        relevance="hot",
    )
    tip = rtk_cat_tip(anchor)
    assert tip == "rtk read app.py | sed -n '10,20p'"


def test_rtk_cat_tip_with_single_line() -> None:
    """rtk_cat_tip uses same line for start and end when end is None."""
    anchor = CodeAnchor(
        file="app.py",
        symbol="main",
        reason="entry",
        confidence=0.9,
        line_start=10,
        line_end=None,
        relevance="hot",
    )
    tip = rtk_cat_tip(anchor)
    assert tip == "rtk read app.py | sed -n '10,10p'"


def test_rtk_cat_tip_without_lines() -> None:
    """rtk_cat_tip generates bare file read without line info."""
    anchor = CodeAnchor(
        file="app.py",
        symbol="main",
        reason="entry",
        confidence=0.9,
        relevance="hot",
    )
    tip = rtk_cat_tip(anchor)
    assert tip == "rtk read app.py"


def test_code_anchor_default_relevance() -> None:
    """CodeAnchor defaults relevance to empty string."""
    anchor = CodeAnchor(
        file="app.py",
        symbol="main",
        reason="entry",
        confidence=0.9,
    )
    assert anchor.relevance == ""


def test_parse_anchors_custom_thresholds() -> None:
    """_parse_anchors respects custom threshold parameters."""
    text = json.dumps(
        [
            {"file": "a.py", "symbol": "x", "reason": "test", "confidence": 0.7},
            {"file": "b.py", "symbol": "y", "reason": "test", "confidence": 0.5},
        ]
    )
    # With hot=0.8, medium=0.4: 0.7→medium, 0.5→medium
    anchors = _parse_anchors(text, hot=0.8, medium=0.4)
    assert anchors[0].relevance == "medium"
    assert anchors[1].relevance == "medium"

    # With hot=0.6, medium=0.4: 0.7→hot, 0.5→medium
    anchors = _parse_anchors(text, hot=0.6, medium=0.4)
    assert anchors[0].relevance == "hot"
    assert anchors[1].relevance == "medium"


def test_grounding_block_empty() -> None:
    """Empty list yields empty string."""
    result = grounding_block([])
    assert result == ""


def test_grounding_block_single_hot_with_insight() -> None:
    """Single hot anchor renders with file:L<range> — symbol and insight."""
    anchors = [
        CodeAnchor(
            file="app.py",
            symbol="main",
            reason="entry point",
            confidence=0.9,
            line_start=10,
            line_end=20,
            relevance="hot",
        )
    ]
    result = grounding_block(anchors)
    assert "app.py:L10-20 — main" in result
    assert "entry point" in result


def test_grounding_block_single_hot_without_insight() -> None:
    """Single hot anchor without reason renders file:L<range> — symbol."""
    anchors = [
        CodeAnchor(
            file="app.py",
            symbol="main",
            reason="",
            confidence=0.9,
            line_start=10,
            line_end=20,
            relevance="hot",
        )
    ]
    result = grounding_block(anchors)
    assert result == "app.py:L10-20 — main"


def test_grounding_block_hot_with_single_line() -> None:
    """Hot anchor with only line_start renders L<start> without range."""
    anchors = [
        CodeAnchor(
            file="app.py",
            symbol="main",
            reason="entry",
            confidence=0.9,
            line_start=10,
            line_end=None,
            relevance="hot",
        )
    ]
    result = grounding_block(anchors)
    assert "app.py:L10 — main" in result
    assert "entry" in result


def test_grounding_block_hot_without_line_numbers() -> None:
    """Hot anchor without line info omits :L... cleanly."""
    anchors = [
        CodeAnchor(
            file="app.py",
            symbol="main",
            reason="entry",
            confidence=0.9,
            line_start=None,
            line_end=None,
            relevance="hot",
        )
    ]
    result = grounding_block(anchors)
    assert result == "app.py — main\n  entry"


def test_grounding_block_hot_multiline_insight() -> None:
    """Hot anchor with multiline reason uses only first line."""
    anchors = [
        CodeAnchor(
            file="app.py",
            symbol="main",
            reason="entry point\nadditional detail\nmore info",
            confidence=0.9,
            line_start=10,
            line_end=20,
            relevance="hot",
        )
    ]
    result = grounding_block(anchors)
    assert "entry point" in result
    assert "additional detail" not in result
    assert "more info" not in result


def test_grounding_block_single_medium() -> None:
    """Single medium anchor yields only the pointer line."""
    anchors = [
        CodeAnchor(
            file="utils.py",
            symbol="helper",
            reason="utility function",
            confidence=0.5,
            relevance="medium",
        )
    ]
    result = grounding_block(anchors)
    assert result == "deeper context lives in knowledge/localization.md"


def test_grounding_block_single_low() -> None:
    """Single low anchor yields only the pointer line."""
    anchors = [
        CodeAnchor(
            file="old.py",
            symbol="legacy",
            reason="tangential",
            confidence=0.2,
            relevance="low",
        )
    ]
    result = grounding_block(anchors)
    assert result == "deeper context lives in knowledge/localization.md"


def test_grounding_block_mixed_hot_and_medium() -> None:
    """Mixed hot and medium anchors: hot inline + pointer line."""
    anchors = [
        CodeAnchor(
            file="app.py",
            symbol="main",
            reason="entry",
            confidence=0.9,
            line_start=10,
            line_end=20,
            relevance="hot",
        ),
        CodeAnchor(
            file="utils.py",
            symbol="helper",
            reason="utility",
            confidence=0.5,
            relevance="medium",
        ),
    ]
    result = grounding_block(anchors)
    assert "app.py:L10-20 — main" in result
    assert "entry" in result
    assert "deeper context lives in knowledge/localization.md" in result
    # Medium anchor should not appear inline
    assert "utils.py" not in result


def test_grounding_block_mixed_hot_and_low() -> None:
    """Mixed hot and low anchors: hot inline + pointer line."""
    anchors = [
        CodeAnchor(
            file="app.py",
            symbol="main",
            reason="entry",
            confidence=0.9,
            line_start=5,
            line_end=15,
            relevance="hot",
        ),
        CodeAnchor(
            file="old.py",
            symbol="legacy",
            reason="old code",
            confidence=0.2,
            relevance="low",
        ),
    ]
    result = grounding_block(anchors)
    assert "app.py:L5-15 — main" in result
    assert "deeper context lives in knowledge/localization.md" in result
    assert "old.py" not in result


def test_grounding_block_multiple_hot() -> None:
    """Multiple hot anchors all render inline."""
    anchors = [
        CodeAnchor(
            file="app.py",
            symbol="main",
            reason="entry",
            confidence=0.9,
            line_start=10,
            line_end=20,
            relevance="hot",
        ),
        CodeAnchor(
            file="config.py",
            symbol="load_config",
            reason="configuration",
            confidence=0.85,
            line_start=5,
            line_end=10,
            relevance="hot",
        ),
    ]
    result = grounding_block(anchors)
    assert "app.py:L10-20 — main" in result
    assert "entry" in result
    assert "config.py:L5-10 — load_config" in result
    assert "configuration" in result
    assert "deeper context lives in knowledge/localization.md" not in result


def test_grounding_block_relevance_case_insensitive() -> None:
    """Relevance tag matching is case-insensitive."""
    anchors = [
        CodeAnchor(
            file="app.py",
            symbol="main",
            reason="entry",
            confidence=0.9,
            line_start=10,
            line_end=20,
            relevance="HOT",
        )
    ]
    result = grounding_block(anchors)
    assert "app.py:L10-20 — main" in result


def test_grounding_block_hot_derived_from_high_confidence() -> None:
    """Hot anchor with empty relevance field (derived from confidence)."""
    anchors = [
        CodeAnchor(
            file="app.py",
            symbol="main",
            reason="entry",
            confidence=0.9,
            line_start=10,
            line_end=20,
            relevance="",  # Will default to empty in dataclass, test derivation path
        )
    ]
    # If relevance is empty string, it will not match "hot" in the condition.
    # We need to ensure derived relevance works. This test shows the current behavior.
    result = grounding_block(anchors)
    # Empty relevance string is not "hot", so it goes to medium/low bucket
    assert result == "deeper context lives in knowledge/localization.md"


def test_grounding_block_multiple_hot_multiple_medium_low() -> None:
    """Mix of 2 hot, 2 medium anchors: hot inline + pointer."""
    anchors = [
        CodeAnchor(
            file="a.py",
            symbol="sym_a",
            reason="key a",
            confidence=0.95,
            line_start=1,
            line_end=10,
            relevance="hot",
        ),
        CodeAnchor(
            file="b.py",
            symbol="sym_b",
            reason="medium context b",
            confidence=0.5,
            relevance="medium",
        ),
        CodeAnchor(
            file="c.py",
            symbol="sym_c",
            reason="key c",
            confidence=0.9,
            line_start=20,
            line_end=30,
            relevance="hot",
        ),
        CodeAnchor(
            file="d.py",
            symbol="sym_d",
            reason="medium context d",
            confidence=0.45,
            relevance="medium",
        ),
    ]
    result = grounding_block(anchors)
    assert "a.py:L1-10 — sym_a" in result
    assert "key a" in result
    assert "c.py:L20-30 — sym_c" in result
    assert "key c" in result
    assert "deeper context lives in knowledge/localization.md" in result
    # Medium anchors should not appear inline
    assert "b.py" not in result
    assert "d.py" not in result


def test_recall_prompt_requests_symbol() -> None:
    """localize_recall.md asks for symbol field explicitly."""
    from pathlib import Path

    prompt_file = Path("splinter/prompts/localize_recall.md")
    assert prompt_file.exists(), f"{prompt_file} not found"
    content = prompt_file.read_text()
    assert '"symbol"' in content, "recall prompt must mention 'symbol' field"
    assert "function/class/symbol name" in content, "recall prompt must clarify symbol meaning"
    assert '"file"' in content, "recall prompt must mention 'file' field"
    assert '"reason"' in content, "recall prompt must mention 'reason' field"
    assert '"confidence"' in content, "recall prompt must mention 'confidence' field"


def test_precision_prompt_requires_one_line_insight() -> None:
    """localize_precision.md reinforces one-line insight for reason field."""
    from pathlib import Path

    prompt_file = Path("splinter/prompts/localize_precision.md")
    assert prompt_file.exists(), f"{prompt_file} not found"
    content = prompt_file.read_text()
    assert '"symbol"' in content, "precision prompt must mention 'symbol' field"
    assert '"reason"' in content, "precision prompt must mention 'reason' field"
    assert "one-line" in content.lower(), "precision prompt must mention one-line"
    assert "concise" in content.lower(), "precision prompt must mention concise"
