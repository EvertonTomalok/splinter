"""Interactive PRD refinement: the conversation that precedes ``splinter run``.

``splinter run --prd path`` no longer executes a PRD blindly. On a TTY it first
opens a TUI (:class:`splinter.tui.PrdSessionApp`) that drives this module:

1. **clarify** — opus-4.8 reads the PRD and asks lettered clarifying questions;
   the user answers in the TUI; the model rewrites the draft and asks what is
   still open. This loops until the user declares the PRD *fulfilled*.
2. **finalize** — the model emits the complete PRD (frontmatter + ``US-NNN``
   user stories) that the existing pipeline parser already understands.
3. **split/run** — the user picks a strategy and the PRD's user stories become
   tasks.

At any prompt the user can type **cowabunga**: the model stops asking and decides
everything itself (answers its own open questions, picks a strategy if needed).

opus-4.8 is the PRD model; ``high`` is already its default reasoning effort, so we
deliberately pass no ``--effort`` here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from splinter.providers import claude_cli

if TYPE_CHECKING:
    from splinter.memory.session import Session
    from splinter.models.roster import Ladder

#: opus-4.8 defaults to high effort — passing --effort would be redundant/wrong.
#: ``opus`` is the claude CLI alias; ``opus-4.8`` is NOT a valid --model id (404).
PRD_MODEL = "opus"

#: Escape hatch: hand the wheel to the model. Matched loosely (case/space-insensitive).
COWABUNGA = "cowabunga"
#: Words that mean "the PRD is good enough, stop asking".
_DONE_WORDS = {"fulfilled", "done", "ready", "go", "ship", "lgtm"}

#: PRD refinement is itself a trajectory; phases land here so ``splinter analyze``
#: can show them next to the run-loop iterations. One ``- <phase> · <detail>`` per line.
PRD_PHASE_FILE = "prd_phases.md"


def log_phase(session: "Session", phase: str, detail: str = "") -> None:
    """Append a PRD refinement phase to the session phase log (read by analyze)."""
    session.append(PRD_PHASE_FILE, f"- {phase} · {detail}" if detail else f"- {phase}")


def ground_localization(session: "Session", ladder: "Ladder", text: str) -> str:
    """Run cached localization and return a compact grounding string.

    Best-effort: returns "" on any failure so the PRD flow is never blocked.
    Cache: ``localize`` short-circuits on existing ``knowledge/localization.md``,
    so resumed/cowabunga sessions reuse cached anchors without re-grepping.
    """
    try:
        from splinter.agents.localizer import grounding_block, localize

        anchors = localize(text, session, ladder)
        return grounding_block(anchors)
    except Exception:
        return ""


def is_cowabunga(text: str) -> bool:
    return text.strip().lower() == COWABUNGA


def is_done(text: str) -> bool:
    return text.strip().lower() in _DONE_WORDS


@dataclass
class Turn:
    """One round-trip with the PRD model."""

    text: str
    session_id: str
    tokens: dict[str, int] = field(default_factory=dict)
    cost: float = 0.0


def _load_prd_skill() -> str:
    for p in (Path("skills/prd/SKILL.md"), Path("splinter/skills/prd/SKILL.md")):
        if p.exists():
            return p.read_text()
    return ""


def _ask(prompt: str, *, resume: str | None, session: object = None) -> Turn:
    # No effort: opus-4.8's default (high) is exactly what we want here.
    result = claude_cli.run(prompt, PRD_MODEL, output_format="json", resume=resume)
    sid = result.raw.get("_session_id", "") or (resume or "")
    tokens = {
        "input": result.usage.get("input_tokens", 0) or 0,
        "output": result.usage.get("output_tokens", 0) or 0,
    }
    cost, _cost_indet = claude_cli._calc_cost(PRD_MODEL, result.usage)
    if session is not None:
        try:
            session.log_llm_usage(PRD_MODEL, tokens, cost)  # type: ignore[attr-defined]
        except Exception:
            pass
    return Turn(text=result.text, session_id=sid, tokens=tokens, cost=cost)


def open_questions(
    prd_text: str, *, strategy: str | None = None, localization: str = "", session: object = None
) -> Turn:
    """First turn: read the PRD and ask clarifying questions (lettered options)."""
    skill = _load_prd_skill()
    strat_hint = (
        f"\nThe strategy is already chosen: {strategy}. Do NOT ask a strategy question.\n"
        if strategy
        else "\nDo NOT ask a strategy question. Strategy is decided later in the pipeline.\n"
    )
    ground_section = (
        f"## Codebase Localization (grounding)\n{localization}\n\n" if localization else ""
    )
    prompt = (
        f"{skill}\n\n"
        "You are refining a draft PRD with the user before implementation begins.\n"
        f"## Draft PRD / request\n{prd_text}\n"
        f"{ground_section}"
        f"{strat_hint}\n"
        "Ask 3-5 essential clarifying questions, each with lettered options (A/B/C/D). "
        "Cover only genuinely ambiguous points. Output ONLY the questions, no preamble."
    )
    return _ask(prompt, resume=None, session=session)


def _resume_preamble(prd_text: str | None, *, resume: str | None) -> str:
    """When the model conversation was lost (no ``resume`` id), re-seed it with the
    skill + the draft so a fresh conversation still knows the PRD it is refining."""
    if resume or not (prd_text and prd_text.strip()):
        return ""
    return (
        f"{_load_prd_skill()}\n\n"
        "You are RESUMING refinement of an in-progress PRD. The draft so far:\n"
        f"## Current PRD draft\n{prd_text}\n\n"
    )


def refine(
    answers: str,
    *,
    resume: str,
    prd_text: str | None = None,
    localization: str = "",
    session: object = None,
) -> Turn:
    """Incorporate the user's answers; return the updated draft + remaining questions.

    ``prd_text`` re-anchors a resumed session whose conversation id was lost.
    """
    ground_section = (
        f"## Codebase Localization (grounding)\n{localization}\n\n" if localization else ""
    )
    prompt = (
        f"{_resume_preamble(prd_text, resume=resume)}"
        f"{ground_section}"
        f"User answers:\n{answers}\n\n"
        "Incorporate these into the PRD. Then output two clearly separated parts:\n"
        "1. '## Working Draft' — the PRD so far in markdown.\n"
        "2. '## Open Questions' — any remaining lettered questions, or 'None — ready to "
        "finalize.' if the PRD is now complete.\n"
        "Keep it tight; do not re-ask anything already answered."
    )
    return _ask(prompt, resume=resume, session=session)


def finalize(
    *,
    resume: str,
    strategy: str | None,
    autodecide: bool,
    prd_text: str | None = None,
    localization: str = "",
    session: object = None,
) -> Turn:
    """Emit the complete PRD with frontmatter and ``US-NNN`` user stories."""
    decide = (
        "Resolve every open question yourself using sensible defaults — do not ask "
        "anything further.\n"
        if autodecide
        else "Use the answers gathered so far; do not ask anything further.\n"
    )
    strat_line = (
        f"Set strategy: {strategy} in the frontmatter.\n"
        if strategy
        else "Pick the best strategy yourself and set it in the frontmatter.\n"
    )
    ground_section = (
        f"## Codebase Localization (grounding)\n{localization}\n\n" if localization else ""
    )
    prompt = (
        f"{_resume_preamble(prd_text, resume=resume)}"
        f"{ground_section}"
        f"{decide}{strat_line}"
        "Now output the COMPLETE final PRD in markdown following the skill template:\n"
        "- YAML frontmatter with feature, strategy, kind, created.\n"
        "- User stories as '### US-001: Title', each with '**Description:**', an "
        "'effort:' hint (trivial|normal|hard|critical), and '**Acceptance Criteria:**' "
        "'- [ ]' checkboxes.\n"
        "Output ONLY the PRD markdown — no preamble, no trailing commentary."
    )
    return _ask(prompt, resume=resume, session=session)


def revise_final(
    instructions: str, *, resume: str, prd_text: str | None = None, session: object = None
) -> Turn:
    """Apply free-form edits to the finalized PRD and re-emit the full document."""
    prompt = (
        f"{_resume_preamble(prd_text, resume=resume)}"
        f"Apply these changes to the PRD:\n{instructions}\n\n"
        "Output the COMPLETE updated PRD markdown only — same format as before, "
        "no preamble."
    )
    return _ask(prompt, resume=resume, session=session)


def generate_prd(
    instructions: str,
    *,
    strategy: str | None = None,
    localization: str = "",
    session: object = None,
) -> Turn:
    """Generate a complete PRD directly from instructions (no Q&A).

    Used when user provides full instructions in the TUI without needing
    clarification. Returns a PRD with frontmatter + US-NNN stories.
    """
    skill = _load_prd_skill()
    strat_hint = (
        f"\nSet strategy to: {strategy} in the frontmatter.\n"
        if strategy
        else "\nPick an appropriate strategy and set it in the frontmatter.\n"
    )
    ground_section = (
        f"## Codebase Localization (grounding)\n{localization}\n\n" if localization else ""
    )
    prompt = (
        f"{skill}\n\n"
        "You are generating a complete PRD from user instructions.\n"
        f"## Instructions\n{instructions}\n"
        f"{ground_section}"
        f"{strat_hint}\n"
        "Generate the COMPLETE final PRD in markdown following the skill template:\n"
        "- YAML frontmatter with feature, strategy, kind, created.\n"
        "- User stories as '### US-001: Title', each with '**Description:**', an "
        "'effort:' hint (trivial|normal|hard|critical), and '**Acceptance Criteria:**' "
        "'- [ ]' checkboxes.\n"
        "Output ONLY the PRD markdown — no preamble, no trailing commentary."
    )
    return _ask(prompt, resume=None, session=session)


def extract_working_draft(text: str) -> str:
    """Extract the PRD content from a refine response.

    The refine prompt asks the model to output '## Working Draft' followed by
    the PRD in a fenced code block, then '## Open Questions'. Strip all that
    and return just the PRD markdown. Falls back to the full text if the
    expected structure is absent.
    """
    # Try to pull out the first fenced code block after ## Working Draft
    m = re.search(r"## Working Draft.*?```(?:markdown)?\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fallback: strip everything up to the first --- frontmatter marker
    m2 = re.search(r"^(---\n.*)", text, re.DOTALL | re.MULTILINE)
    if m2:
        return m2.group(1).strip()
    return text.strip()


def route_prd(prd_text: str) -> str:
    """Route PRD text to generate or refine flow.

    Returns 'generate' if prd_text is empty/whitespace, 'refine' otherwise.
    Used by PrdSessionApp to decide whether to ask clarifying questions
    or go straight to asking for instructions to generate from.
    """
    return "generate" if not prd_text.strip() else "refine"


def ensure_frontmatter(prd_text: str, *, description: str, strategy: str | None) -> str:
    """Guarantee parseable YAML frontmatter; the model usually supplies it already."""
    prd_text = prd_text.strip()
    if prd_text.startswith("---"):
        return prd_text + "\n"
    feature = re.sub(r"[^a-zA-Z0-9-]", "-", description[:40]).strip("-").lower() or "feature"
    fm = (
        f"---\n"
        f"feature: {feature}\n"
        f"strategy: {strategy or 'direct'}\n"
        f"kind: feature\n"
        f"created: {datetime.now(timezone.utc).isoformat()}\n"
        f"---\n\n"
    )
    return fm + prd_text + "\n"


def user_story_titles(prd_text: str) -> list[str]:
    """Pull ``US-NNN: Title`` headers from a PRD for a review summary."""
    return [
        f"{m.group(1)}: {m.group(2).strip()}"
        for m in re.finditer(r"###\s+(US-\d+):\s*(.+)", prd_text)
    ]


def prd_session_is_resumable(session: "Session") -> bool:
    """Whether a ``refining`` session holds anything worth resuming.

    True if a run already produced a trace, or the PRD has at least one
    ``US-NNN`` user story. A bare stub (e.g. ``# Test`` with no stories) gives
    the planner nothing to run, so it is junk — ``is_empty`` misses it because
    the stub text is technically non-empty.
    """
    if session.read("trace.md").strip():
        return True
    return bool(user_story_titles(session.read("prd.md")))


def prune_dead_prd_sessions(min_age_seconds: float = 60.0) -> list[str]:
    """Garbage-collect abandoned ``refining`` sessions; return the deleted ids.

    Targets sessions orphaned by a crash, force-quit, or a stub PRD that the
    content-based ``is_empty`` cleanup failed to catch. Only ``refining``
    sessions with no runnable PRD and no run are removed; anything touched in
    the last ``min_age_seconds`` is spared so a live refinement in another
    process is never nuked mid-edit.
    """
    from splinter.memory.session import (
        Session,
        _sessions_dir,
        delete_session,
        list_sessions,
    )

    now = datetime.now(timezone.utc).timestamp()
    pruned: list[str] = []
    for sid in list_sessions():
        session = Session(sid)
        if session.read_status().get("state") != "refining":
            continue
        if prd_session_is_resumable(session):
            continue
        try:
            age = now - (_sessions_dir() / sid).stat().st_mtime
        except OSError:
            continue
        if age < min_age_seconds:
            continue
        delete_session(sid)
        pruned.append(sid)
    return pruned


#: ``### US-NNN: …`` header up to the next user story (or end of document).
_STORY_BLOCK = re.compile(r"(###\s+(US-\d+)\b.*?)(?=###\s+US-\d+|\Z)", re.DOTALL)
#: An unchecked acceptance-criteria checkbox.
_OPEN_BOX = re.compile(r"-\s*\[\s*\]")


def story_id(text: str) -> str | None:
    """The ``US-NNN`` id leading a task description / story header, if any."""
    m = re.match(r"\s*(US-\d+)\b", text)
    return m.group(1) if m else None


def mark_story_done(prd_text: str, us_id: str) -> str:
    """Tick every ``- [ ]`` acceptance-criteria box inside ``us_id``'s block.

    The PRD is the durable record of progress: a completed task's checkboxes flip
    to ``- [x]`` so resume (and the human) can see what is finished. Blocks for
    other stories are left untouched.
    """

    def _tick(match: re.Match[str]) -> str:
        block, sid = match.group(1), match.group(2)
        if sid != us_id:
            return block
        return _OPEN_BOX.sub("- [x]", block)

    return _STORY_BLOCK.sub(_tick, prd_text)


def mark_all_stories_done(prd_text: str) -> str:
    """Tick every ``- [ ]`` box inside *every* ``US-NNN`` block.

    Used by the raphael single-shot run: one holistic PASS completes all stories
    at once. Checkboxes outside story blocks (if any) are left untouched.
    """

    def _tick(match: re.Match[str]) -> str:
        return _OPEN_BOX.sub("- [x]", match.group(1))

    return _STORY_BLOCK.sub(_tick, prd_text)


def completed_story_ids(prd_text: str) -> set[str]:
    """Story ids whose acceptance criteria are fully checked (≥1 box, none open).

    A story with no checkboxes at all is *not* considered complete — there is
    nothing to have ticked.
    """
    done: set[str] = set()
    for match in _STORY_BLOCK.finditer(prd_text):
        block, sid = match.group(1), match.group(2)
        has_box = "[x]" in block.lower() or _OPEN_BOX.search(block)
        if has_box and not _OPEN_BOX.search(block):
            done.add(sid)
    return done
