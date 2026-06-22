from __future__ import annotations

import re

import yaml

from splinter.memory.knowledge import KnowledgeStore
from splinter.memory.session import Session, delete_session, new_session_id
from splinter.providers.dispatch import run_provider_session


def _abort(session: Session, message: str) -> int:
    """Print error, garbage-collect the session if nothing real was written."""
    print(message)
    _prune_prd_session(session)
    return 1


def _prune_prd_session(session: Session) -> None:
    """Delete session if it has no runnable PRD and no run trace.

    Uses prd_session_is_resumable so that sessions created by ground_localization
    (which writes knowledge/localization.md, making is_empty() return False) are
    still cleaned up when the PRD flow aborted before producing real content.
    """
    from splinter.prd_session import prd_session_is_resumable

    if session.dir.exists() and not prd_session_is_resumable(session):
        delete_session(session.id)


def _load_prd_skill() -> str:
    # Packaged template (splinter/prompts/prd.md) with .splinter override support,
    # same resolution as plan/eval.
    from splinter.configure import template_current_text

    return template_current_text("prd")


def run_prd(*, description: str = "", strategy: str | None = None, no_ground: bool = False) -> int:
    # Fresh session every run. Never reuse latest — that silently appended to
    # (or resurrected) a prior session and left empties on abort.
    session = Session(new_session_id())
    try:
        rc = _run_prd(session, description=description, strategy=strategy, no_ground=no_ground)
    except BaseException:
        # Ctrl+C at a prompt or a provider crash must not litter an empty session.
        _prune_prd_session(session)
        raise
    # Non-zero return means the PRD flow aborted; clean up if nothing runnable was produced.
    if rc != 0:
        _prune_prd_session(session)
    return rc


def _run_prd(
    session: Session, *, description: str = "", strategy: str | None = None, no_ground: bool = False
) -> int:
    from splinter.models.roster import load_ladder

    ladder = load_ladder()
    prd_model = ladder.prd_model
    prd_effort = ladder.prd_effort
    prd_timeout = ladder.prd_timeout

    skill_text = _load_prd_skill()

    if not description:
        print("describe the feature or bug:")
        description = input("> ").strip()
        if not description:
            return _abort(session, "error: no description provided")

    strategy_hint = ""
    if strategy:
        strategy_hint = f"\nThe user has pre-selected the strategy: {strategy}\n"

    grounding = ""
    if not no_ground:
        from splinter import prd_session

        grounding = prd_session.ground_localization(session, ladder, description)
    ground_section = f"## Codebase Localization (grounding)\n{grounding}\n\n" if grounding else ""

    turn1_prompt = (
        f"{skill_text}\n\n"
        f"User request: {description}\n"
        f"{ground_section}"
        f"{strategy_hint}\n"
        "Generate 3-10 clarifying questions with lettered options (A/B/C/D). "
        "Include a strategy question unless a strategy was pre-selected. "
        "Output ONLY the questions, no preamble."
    )

    result1, session_id = run_provider_session(
        turn1_prompt,
        prd_model,
        variant=prd_effort,
        output_format="json",
        timeout=prd_timeout,
        role="prd",
    )
    questions = result1.text
    session.log_llm_usage(
        prd_model,
        {
            "input": result1.tokens.get("input", 0) or 0,
            "output": result1.tokens.get("output", 0) or 0,
        },
        result1.cost,
    )

    print("\n" + questions + "\n")
    print("answer with e.g. 1A,2C,3B (or type full answers):")
    answers = input("> ").strip()
    if not answers:
        return _abort(session, "error: no answers provided")

    turn2_prompt = (
        f"{ground_section}"
        f"User answers:\n{answers}\n\n"
        "Now generate the full PRD following the skill template. "
        "Include YAML frontmatter with feature, strategy, kind, created. "
        "Output the complete PRD in markdown."
    )

    resume = session_id if session_id else None
    result2, _sid2 = run_provider_session(
        turn2_prompt,
        prd_model,
        variant=prd_effort,
        output_format="json",
        session=resume,
        timeout=prd_timeout,
        role="prd",
    )
    prd_text = result2.text
    session.log_llm_usage(
        prd_model,
        {
            "input": result2.tokens.get("input", 0) or 0,
            "output": result2.tokens.get("output", 0) or 0,
        },
        result2.cost,
    )

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
    print("\nStrategies:")
    print("  raphael   - direct:   one task, implement → eval → escalate fast")
    print("  leonardo  - cascade:  multi-task, dependency-ordered, checkpointed")
    print("  donatello - adaptive: routes each task to cheapest capable tier within budget")
    print("  michelangelo - sprint: always starts flash tier, escalates only on eval failure")
    return 0
