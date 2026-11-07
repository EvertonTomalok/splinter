"""Pure strategy recommendation from a finalized PRD."""

from __future__ import annotations

from typing import Any

import yaml

from splinter.prd_session import user_story_titles
from splinter.strategies.registry import get_strategy


def _frontmatter(prd_text: str) -> dict[str, Any]:
    stripped = prd_text.lstrip()
    if not stripped.startswith("---"):
        return {}
    parts = stripped.split("---", 2)
    if len(parts) < 3:
        return {}
    try:
        loaded = yaml.safe_load(parts[1])
    except yaml.YAMLError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def canonical_strategy_name(name: str) -> str | None:
    cleaned = name.strip()
    if not cleaned:
        return None
    try:
        return get_strategy(cleaned).name
    except ValueError:
        return None


def recommend_strategy(prd_text: str) -> tuple[str, str]:
    """Return ``(canonical_name, reason)`` for ``prd_text``."""
    fm = _frontmatter(prd_text)
    raw = fm.get("strategy")
    if isinstance(raw, str):
        canonical = canonical_strategy_name(raw)
        if canonical is not None:
            return canonical, "set in the PRD"
    n_stories = len(user_story_titles(prd_text))
    if n_stories <= 1:
        return "direct", "single story"
    return "cascade", f"{n_stories} stories, dependency-ordered"
