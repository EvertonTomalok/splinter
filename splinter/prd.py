from __future__ import annotations

import re
from pathlib import Path

import yaml

from splinter.memory.knowledge import KnowledgeStore
from splinter.memory.session import Session
from splinter.providers import claude_cli


def _load_prd_skill() -> str:
    skill_paths = [
        Path("skills/prd/SKILL.md"),
        Path("splinter/skills/prd/SKILL.md"),
    ]
    for p in skill_paths:
        if p.exists():
            return p.read_text()
    return ""


def run_prd(*, description: str = "", strategy: str | None = None) -> int:
    session = Session()
    skill_text = _load_prd_skill()

    if not description:
        print("describe the feature or bug:")
        description = input("> ").strip()
        if not description:
            print("error: no description provided")
            return 1

    strategy_hint = ""
    if strategy:
        strategy_hint = f"\nThe user has pre-selected the strategy: {strategy}\n"

    turn1_prompt = (
        f"{skill_text}\n\n"
        f"User request: {description}\n"
        f"{strategy_hint}\n"
        "Generate 3-5 clarifying questions with lettered options (A/B/C/D). "
        "Include a strategy question unless a strategy was pre-selected. "
        "Output ONLY the questions, no preamble."
    )

    result1 = claude_cli.run(turn1_prompt, "sonnet", effort="high")
    questions = result1.text
    session_id = result1.raw.get("_session_id", "")

    print("\n" + questions + "\n")
    print("answer with e.g. 1A,2C,3B (or type full answers):")
    answers = input("> ").strip()
    if not answers:
        print("error: no answers provided")
        return 1

    turn2_prompt = (
        f"User answers:\n{answers}\n\n"
        "Now generate the full PRD following the skill template. "
        "Include YAML frontmatter with feature, strategy, kind, created. "
        "Output the complete PRD in markdown."
    )

    resume = session_id if session_id else None
    result2 = claude_cli.run(turn2_prompt, "sonnet", effort="high", resume=resume)
    prd_text = result2.text

    if not prd_text.startswith("---"):
        fm_strategy = strategy or "cascade"
        feature = re.sub(r"[^a-zA-Z0-9-]", "-", description[:40]).strip("-").lower()
        from datetime import datetime, timezone

        frontmatter = (
            f"---\n"
            f"feature: {feature}\n"
            f"strategy: {fm_strategy}\n"
            f"kind: feature\n"
            f"created: {datetime.now(timezone.utc).isoformat()}\n"
            f"---\n\n"
        )
        prd_text = frontmatter + prd_text

    session.write("prd.md", prd_text)

    fm: dict[str, str] = {}
    if prd_text.startswith("---"):
        parts = prd_text.split("---", 2)
        if len(parts) >= 3:
            fm = yaml.safe_load(parts[1]) or {}

    idx_lines = [
        f"# Session {session.id}",
        "- prd: prd.md",
        f"- feature: {fm.get('feature', 'unknown')}",
        f"- strategy: {fm.get('strategy', 'unknown')}",
        f"- kind: {fm.get('kind', 'unknown')}",
    ]
    session.update_index("\n".join(idx_lines) + "\n")

    ks = KnowledgeStore(session)
    prd_note = f"Feature: {fm.get('feature', '')}\nStrategy: {fm.get('strategy', '')}\n"
    ks.write_note("prd-summary", prd_note)

    print(f"\nPRD saved to {session.dir / 'prd.md'}")
    print(f"Run with: uv run splinter run --prd {session.dir / 'prd.md'}")
    return 0
