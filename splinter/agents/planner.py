"""Deterministic PRD parser — turns user stories into Task objects."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from splinter.agents.localizer import CodeAnchor
from splinter.agents.runner import Task


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


def parse_stories(prd_text: str) -> list[Task]:
    """Parse PRD user-story blocks into Task objects.

    Each ``### US-NNN: title`` block yields one Task with id, description,
    acceptance, effort, eval_skill, and deps populated from the block text.
    """
    _fm, body = _parse_frontmatter(prd_text)

    tasks: list[Task] = []
    for m in _US_PATTERN.finditer(body):
        us_id = m.group(1)
        title = m.group(2).strip()
        block = m.group(3)

        desc_match = re.search(r"\*\*Description:\*\*\s*(.+)", block)
        desc = desc_match.group(1).strip() if desc_match else title

        effort_match = re.search(r"effort:\s*(\w+)", block)
        effort = effort_match.group(1) if effort_match else "normal"

        skill_match = re.search(r"eval_skill:\s*(\S+)", block)
        skill = skill_match.group(1) if skill_match else None

        ac_lines = re.findall(r"- \[[ x]\]\s*(.+)", block)
        acceptance = "\n".join(ac_lines) if ac_lines else desc

        deps = _DEP_PATTERN.findall(block) or None

        tasks.append(
            Task(
                description=f"{us_id}: {desc}",
                acceptance=acceptance,
                effort=effort,
                eval_skill=skill,
                id=us_id,
                deps=deps,
            )
        )

    if not tasks:
        tasks.append(
            Task(
                description=body[:200].strip(),
                acceptance="implementation matches the PRD description",
            )
        )

    return tasks


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
