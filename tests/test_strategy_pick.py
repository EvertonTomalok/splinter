"""Unit tests for strategy summaries and ``splinter.strategy_pick``."""

from __future__ import annotations

import pytest

from splinter.strategies.base import Strategy
from splinter.strategies.registry import get_strategy, registered_strategies
from splinter.strategy_pick import recommend_strategy


def test_base_strategy_default_summary() -> None:
    assert Strategy.summary == ""


def test_each_registered_strategy_has_summary() -> None:
    classes = {cls.name: cls for cls in registered_strategies()}
    assert set(classes) == {"adaptive", "agents", "cascade", "direct", "sprint"}
    assert classes["direct"].summary == "one task, implement → eval → escalate fast"
    assert classes["cascade"].summary == "multi-task, dependency-ordered, checkpointed"
    assert classes["adaptive"].summary == "routes each task to cheapest tier within budget"
    assert classes["sprint"].summary == "starts flash tier, escalates only on eval failure"
    assert classes["agents"].summary == (
        "multi-story: refines briefs into waves, runs each on cheap "
        "engines in its own worktree, merges back"
    )


def test_frontmatter_strategy_wins() -> None:
    prd = "---\nfeature: demo\nstrategy: adaptive\n---\n\n### US-001: Only story\n"
    assert recommend_strategy(prd) == ("adaptive", "set in the PRD")


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("raphael", "direct"),
        ("leonardo", "cascade"),
        ("donatello", "adaptive"),
        ("michelangelo", "sprint"),
        ("splinter", "agents"),
        ("RAPHAEL", "direct"),
    ],
)
def test_frontmatter_alias_resolved_to_canonical(alias: str, canonical: str) -> None:
    prd = f"---\nstrategy: {alias}\n---\n\n### US-001: A\n### US-002: B\n"
    name, reason = recommend_strategy(prd)
    assert name == canonical
    assert name == get_strategy(alias).name
    assert reason == "set in the PRD"


@pytest.mark.parametrize("bad", ["unknown-turtle", "?", "", "123"])
def test_invalid_frontmatter_strategy_ignored(bad: str) -> None:
    prd = f"---\nstrategy: '{bad}'\n---\n\n### US-001: First\n### US-002: Second\n"
    assert recommend_strategy(prd) == ("cascade", "2 stories, dependency-ordered")


def test_malformed_frontmatter_ignored() -> None:
    prd = "---\n: [not valid yaml\n---\n\n### US-001: Only\n"
    assert recommend_strategy(prd) == ("direct", "single story")


def test_zero_stories_recommends_direct() -> None:
    prd = "# Just a plain markdown request without US headers\n"
    assert recommend_strategy(prd) == ("direct", "single story")


def test_single_story_recommends_direct() -> None:
    prd = "---\nfeature: one\n---\n\n### US-001: Single task\n**Description:** do it\n"
    assert recommend_strategy(prd) == ("direct", "single story")


def test_multiple_stories_recommends_cascade() -> None:
    prd = (
        "---\nfeature: multi\n---\n\n"
        "### US-001: First\n**Description:** a\n\n"
        "### US-002: Second\n**Description:** b\n\n"
        "### US-003: Third\n**Description:** c\n"
    )
    assert recommend_strategy(prd) == ("cascade", "3 stories, dependency-ordered")
