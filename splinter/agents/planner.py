"""Deterministic PRD parser — turns user stories into Task objects."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from pathlib import Path

import yaml

from splinter.agents.localizer import CodeAnchor
from splinter.agents.runner import Task
from splinter.scheduling import default_max_concurrency
from splinter.strategies.fanout import run_bounded


def _parse_frontmatter(text: str) -> tuple[dict, str]:  # type: ignore[type-arg]
    """Strip YAML frontmatter and return (metadata, body)."""
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            fm = yaml.safe_load(parts[1]) or {}
            body = parts[2]
            return fm, body
    return {}, text


_US_PATTERN = re.compile(
    r"###\s+(US-\d+):\s*(.+?)\n(.*?)(?=###\s+US-|\Z)",
    re.DOTALL,
)

_DEP_PATTERN = re.compile(
    r"(?:Depends on|Blocked until)\s+(US-\d+)",
    re.IGNORECASE,
)


def _parse_one_story(match: re.Match[str]) -> Task:
    """Pure transform: one ``### US-NNN`` regex match to one Task."""
    us_id = match.group(1)
    title = match.group(2).strip()
    block = match.group(3)

    desc_match = re.search(r"\*\*Description:\*\*\s*(.+)", block)
    desc = desc_match.group(1).strip() if desc_match else title

    effort_match = re.search(r"effort:\s*(\w+)", block)
    effort = effort_match.group(1) if effort_match else "normal"

    skill_match = re.search(r"eval_skill:\s*(\S+)", block)
    _raw_skill = skill_match.group(1) if skill_match else None
    skill = (
        None
        if _raw_skill is None
        or _raw_skill.lower() in ("omit", "none", "null")
        or _raw_skill.startswith("(")
        else _raw_skill
    )

    ac_lines = re.findall(r"- \[[ x]\]\s*(.+)", block)
    acceptance = "\n".join(ac_lines) if ac_lines else desc

    # Two documented forms, both honoured: the `deps: [US-001, US-002]` list
    # field and the prose `Depends on US-001` / `Blocked until US-001`. The
    # list form is what the SKILL template tells the model to emit, so it MUST
    # be parsed — dropping it makes every task look dependency-free and the DAG
    # runs them all in parallel, colliding on shared files.
    deps_list: list[str] = []
    _deps_field = re.search(r"deps:\s*\[([^\]]*)\]", block, re.IGNORECASE)
    if _deps_field:
        deps_list = re.findall(r"US-\d+", _deps_field.group(1))
    deps = list(dict.fromkeys(deps_list + _DEP_PATTERN.findall(block))) or None

    parallelizable: bool | None = None
    _par_match = re.search(r"parallelizable:\s*(true|false)", block, re.IGNORECASE)
    if _par_match:
        parallelizable = _par_match.group(1).lower() == "true"

    return Task(
        description=f"{us_id}: {desc}",
        acceptance=acceptance,
        effort=effort,
        eval_skill=skill,
        id=us_id,
        deps=deps,
        parallelizable=parallelizable,
    )


def _fallback_task(body: str) -> Task:
    return Task(
        description=body[:200].strip(),
        acceptance="implementation matches the PRD description",
    )


def parse_stories(prd_text: str) -> list[Task]:
    """Parse PRD user-story blocks into Task objects.

    Each ``### US-NNN: title`` block yields one Task with id, description,
    acceptance, effort, eval_skill, and deps populated from the block text.
    """
    _fm, body = _parse_frontmatter(prd_text)

    tasks = [_parse_one_story(m) for m in _US_PATTERN.finditer(body)]

    if not tasks:
        tasks.append(_fallback_task(body))

    return tasks


class Planner:
    """Parses PRD story blocks concurrently, one item-callable per story."""

    def __init__(self, *, concurrency: int | None = None) -> None:
        self._concurrency = concurrency or default_max_concurrency()

    async def plan_items(self, prd_text: str) -> list[Task]:
        _fm, body = _parse_frontmatter(prd_text)
        matches = list(_US_PATTERN.finditer(body))

        def _make_item(m: re.Match[str]) -> Callable[[], Awaitable[Task]]:
            async def _item() -> Task:
                return _parse_one_story(m)

            return _item

        items = [_make_item(m) for m in matches]
        tasks = await run_bounded(items, concurrency=self._concurrency)

        if not tasks:
            tasks.append(_fallback_task(body))

        return tasks

    def plan(self, prd_text: str) -> list[Task]:
        return asyncio.run(self.plan_items(prd_text))


def _tokenize(text: str) -> set[str]:
    """Lowercase keyword tokens from a text string."""
    words = re.findall(r"[A-Za-z][A-Za-z0-9_]+", text.lower())
    stop = {
        "the",
        "and",
        "for",
        "are",
        "but",
        "not",
        "you",
        "all",
        "can",
        "had",
        "her",
        "was",
        "one",
        "our",
        "out",
        "day",
        "get",
        "has",
        "him",
        "his",
        "how",
        "its",
        "may",
        "new",
        "now",
        "old",
        "see",
        "two",
        "way",
        "who",
        "boy",
        "did",
        "she",
        "use",
        "than",
        "them",
        "well",
        "were",
        "with",
        "have",
        "from",
        "they",
        "know",
        "want",
        "been",
        "good",
        "much",
        "some",
        "time",
        "very",
        "when",
        "come",
        "here",
        "just",
        "like",
        "long",
        "make",
        "many",
        "over",
        "such",
        "take",
        "will",
        "that",
        "this",
        "into",
        "also",
    }
    return {w for w in words if len(w) > 2 and w not in stop}


def assign_target_files(tasks: list[Task], anchors: list[CodeAnchor]) -> None:
    """Populate each task's ``target_files`` from localization anchors.

    Heuristic: match an anchor to a task when the anchor's reason/symbol
    keywords overlap the task's id/description tokens. Fallback (no overlap
    for a task) = all unique anchor files, deduped and order-preserved.

    Mutates tasks in place.
    """
    if not anchors:
        return

    all_files: list[str] = []
    seen_files: set[str] = set()
    for a in anchors:
        if a.file and a.file not in seen_files:
            seen_files.add(a.file)
            all_files.append(a.file)

    for task in tasks:
        task_tokens = _tokenize(f"{task.id} {task.description}")
        matched: list[str] = []
        matched_seen: set[str] = set()
        for a in anchors:
            anchor_tokens = _tokenize(f"{a.reason} {a.symbol}")
            if task_tokens & anchor_tokens:
                if a.file and a.file not in matched_seen:
                    matched_seen.add(a.file)
                    matched.append(a.file)
        task.target_files = matched if matched else list(all_files)


def plan(prd_path: str, anchors: list[CodeAnchor]) -> tuple[list[Task], str | None]:
    """Read a PRD file, parse stories, assign target files.

    Returns (tasks, strategy_name_or_None).
    """
    text = Path(prd_path).read_text()
    fm, _body = _parse_frontmatter(text)
    strategy = fm.get("strategy")
    tasks = parse_stories(text)
    assign_target_files(tasks, anchors)
    return tasks, strategy
