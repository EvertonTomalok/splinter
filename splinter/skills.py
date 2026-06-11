"""Skill resolution for evaluator injection (§6.11).

Resolution precedence: **CLI ``--eval`` > story ``eval_skill`` > none** (nil means
written-criteria evaluation with no injected skill body).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedSkill:
    name: str
    description: str
    body: str
    missing: bool = field(default=False)


def _skill_paths(name: str) -> list[Path]:
    return [
        Path(f"skills/{name}/SKILL.md"),
        Path(f"splinter/skills/{name}/SKILL.md"),
    ]


def resolve_eval_skill(name: str | None) -> ResolvedSkill | None:
    if not name:
        return None

    for p in _skill_paths(name):
        if p.exists():
            text = p.read_text()
            description = ""
            body = text
            if text.startswith("---"):
                parts = text.split("---", 2)
                if len(parts) >= 3:
                    fm = yaml.safe_load(parts[1]) or {}
                    if isinstance(fm, dict):
                        description = fm.get("description", "")
                    body = parts[2].strip()
            return ResolvedSkill(name=name, description=description, body=body)

    log.warning(
        "eval skill '%s' not found (looked in skills/%s/SKILL.md and splinter/skills/%s/SKILL.md)",
        name, name, name,
    )
    return ResolvedSkill(
        name=name,
        description="",
        body="",
        missing=True,
    )
