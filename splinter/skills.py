"""Skill resolution for evaluator injection (§6.11).

Resolution precedence: **CLI ``--eval`` > story ``eval_skill`` > none** (nil means
written-criteria evaluation with no injected skill body).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ResolvedSkill:
    name: str
    description: str
    body: str


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

    raise ValueError(
        f"unknown eval skill: '{name}'. "
        f"Place a SKILL.md at skills/{name}/SKILL.md or splinter/skills/{name}/SKILL.md."
    )
