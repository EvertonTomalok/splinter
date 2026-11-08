"""Read a ``--task`` file: a YAML mapping or a markdown task (``T03-name.md``).

A markdown task is one file of a plan folder (``tasks/<plan>/README.md`` plus
``T01-….md``, ``T02-….md``). Its ``## Acceptance`` section becomes the
acceptance criteria; everything else is the description, headed by the paths of
the task file and of the plan index beside it so the planner can read both.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

__all__ = ["read_task_mapping", "split_markdown_task"]

_ACCEPTANCE_HEAD = re.compile(r"^##[ \t]+Acceptance(?:[ \t]+Criteria)?[ \t]*$", re.MULTILINE)
_NEXT_SECTION = re.compile(r"^##[ \t]+\S", re.MULTILINE)
_MARKDOWN_SUFFIXES = frozenset({".md", ".markdown"})


def read_task_mapping(path: Path) -> dict[str, Any]:
    """Return the task as a mapping with at least ``description`` and ``acceptance``.

    Raises ``OSError`` when the file cannot be read and ``ValueError`` when a YAML
    task is not a mapping or a markdown task is empty.
    """
    text = path.read_text()
    if path.suffix.lower() in _MARKDOWN_SUFFIXES:
        return _markdown_mapping(path, text)
    try:
        loaded = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"task yaml is invalid: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError("task yaml must be a mapping")
    return loaded


def split_markdown_task(text: str) -> tuple[str, str]:
    """Split a markdown task into ``(description, acceptance)``."""
    head = _ACCEPTANCE_HEAD.search(text)
    if head is None:
        return text.strip(), ""
    rest = text[head.end() :]
    following = _NEXT_SECTION.search(rest)
    acceptance = rest[: following.start()] if following else rest
    tail = rest[following.start() :] if following else ""
    description = (text[: head.start()] + tail).strip()
    return description, acceptance.strip()


def _markdown_mapping(path: Path, text: str) -> dict[str, Any]:
    description, acceptance = split_markdown_task(text)
    if not description:
        raise ValueError(f"markdown task is empty: {path}")
    return {
        "description": f"{_source_lines(path)}\n\n{description}",
        "acceptance": acceptance,
    }


def _source_lines(path: Path) -> str:
    lines = [f"Task file: `{path}`"]
    index = path.parent / "README.md"
    if index.is_file() and index.resolve() != path.resolve():
        lines.append(f"Plan index (decisions, waves, definition of done): `{index}`")
    return "\n".join(lines)
