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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from splinter.providers import claude_cli

if TYPE_CHECKING:
    from splinter.memory.session import Session

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


def is_cowabunga(text: str) -> bool:
    return text.strip().lower() == COWABUNGA


def is_done(text: str) -> bool:
    return text.strip().lower() in _DONE_WORDS


@dataclass
class Turn:
    """One round-trip with the PRD model."""

    text: str
    session_id: str


def _load_prd_skill() -> str:
    for p in (Path("skills/prd/SKILL.md"), Path("splinter/skills/prd/SKILL.md")):
        if p.exists():
            return p.read_text()
    return ""


def _ask(prompt: str, *, resume: str | None) -> Turn:
    # No effort: opus-4.8's default (high) is exactly what we want here.
    result = claude_cli.run(prompt, PRD_MODEL, output_format="json", resume=resume)
    sid = result.raw.get("_session_id", "") or (resume or "")
    return Turn(text=result.text, session_id=sid)


def open_questions(prd_text: str, *, strategy: str | None = None) -> Turn:
    """First turn: read the PRD and ask clarifying questions (lettered options)."""
    skill = _load_prd_skill()
    strat_hint = (
        f"\nThe strategy is already chosen: {strategy}. Do NOT ask a strategy question.\n"
        if strategy
        else "\nInclude a strategy question (cascade/direct/adaptive/sprint).\n"
    )
    prompt = (
        f"{skill}\n\n"
        "You are refining a draft PRD with the user before implementation begins.\n"
        f"## Draft PRD / request\n{prd_text}\n"
        f"{strat_hint}\n"
        "Ask 3-5 essential clarifying questions, each with lettered options (A/B/C/D). "
        "Cover only genuinely ambiguous points. Output ONLY the questions, no preamble."
    )
    return _ask(prompt, resume=None)


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


def refine(answers: str, *, resume: str, prd_text: str | None = None) -> Turn:
    """Incorporate the user's answers; return the updated draft + remaining questions.

    ``prd_text`` re-anchors a resumed session whose conversation id was lost.
    """
    prompt = (
        f"{_resume_preamble(prd_text, resume=resume)}"
        f"User answers:\n{answers}\n\n"
        "Incorporate these into the PRD. Then output two clearly separated parts:\n"
        "1. '## Working Draft' — the PRD so far in markdown.\n"
        "2. '## Open Questions' — any remaining lettered questions, or 'None — ready to "
        "finalize.' if the PRD is now complete.\n"
        "Keep it tight; do not re-ask anything already answered."
    )
    return _ask(prompt, resume=resume)


def finalize(
    *, resume: str, strategy: str | None, autodecide: bool, prd_text: str | None = None
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
    prompt = (
        f"{_resume_preamble(prd_text, resume=resume)}"
        f"{decide}{strat_line}"
        "Now output the COMPLETE final PRD in markdown following the skill template:\n"
        "- YAML frontmatter with feature, strategy, kind, created.\n"
        "- User stories as '### US-001: Title', each with '**Description:**', an "
        "'effort:' hint (trivial|normal|hard|critical), and '**Acceptance Criteria:**' "
        "'- [ ]' checkboxes.\n"
        "Output ONLY the PRD markdown — no preamble, no trailing commentary."
    )
    return _ask(prompt, resume=resume)


def revise_final(instructions: str, *, resume: str, prd_text: str | None = None) -> Turn:
    """Apply free-form edits to the finalized PRD and re-emit the full document."""
    prompt = (
        f"{_resume_preamble(prd_text, resume=resume)}"
        f"Apply these changes to the PRD:\n{instructions}\n\n"
        "Output the COMPLETE updated PRD markdown only — same format as before, "
        "no preamble."
    )
    return _ask(prompt, resume=resume)


def generate_prd(instructions: str, *, strategy: str | None = None) -> Turn:
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
    prompt = (
        f"{skill}\n\n"
        "You are generating a complete PRD from user instructions.\n"
        f"## Instructions\n{instructions}\n"
        f"{strat_hint}\n"
        "Generate the COMPLETE final PRD in markdown following the skill template:\n"
        "- YAML frontmatter with feature, strategy, kind, created.\n"
        "- User stories as '### US-001: Title', each with '**Description:**', an "
        "'effort:' hint (trivial|normal|hard|critical), and '**Acceptance Criteria:**' "
        "'- [ ]' checkboxes.\n"
        "Output ONLY the PRD markdown — no preamble, no trailing commentary."
    )
    return _ask(prompt, resume=None)


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
