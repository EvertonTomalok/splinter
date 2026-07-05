"""Interactive PRD refinement: the conversation that precedes ``splinter run``.

``splinter run --prd path`` no longer executes a PRD blindly. On a TTY it first
opens a TUI (:class:`splinter.tui.PrdSessionApp`) that drives this module:

1. **clarify** — the configured PRD model reads the PRD and asks lettered clarifying questions;
   the user answers in the TUI; the model rewrites the draft and asks what is
   still open. This loops until the user declares the PRD *fulfilled*.
2. **finalize** — the model emits the complete PRD (frontmatter + ``US-NNN``
   user stories) that the existing pipeline parser already understands.
3. **split/run** — the user picks a strategy and the PRD's user stories become
   tasks.

At any prompt the user can type **cowabunga**: the model stops asking and decides
everything itself (answers its own open questions, picks a strategy if needed).

PRD model and effort come from ``.splinter/config.yaml`` (``models.prd`` and
``efforts.prd``). The default is Claude ``opus`` with ``high`` effort.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from splinter.models.roster import load_ladder
from splinter.providers.dispatch import run_provider_session

if TYPE_CHECKING:
    from splinter.memory.session import Session
    from splinter.models.roster import Ladder

#: Escape hatch: hand the wheel to the model. Matched loosely (case/space-insensitive).
COWABUNGA = "cowabunga"
#: Words that mean "the PRD is good enough, stop asking".
_DONE_WORDS = {"fulfilled", "done", "ready", "go", "ship", "lgtm"}

#: PRD refinement is itself a trajectory; phases land here so ``splinter analyze``
#: can show them next to the run-loop iterations. One ``- <phase> · <detail>`` per line.
PRD_PHASE_FILE = "prd_phases.md"
#: Numbered PRD snapshots + manifest for analyze (old vs new per refine step).
PRD_VERSION_DIR = "prd-versions"
PRD_VERSION_MANIFEST = "prd_versions.md"


def log_phase(session: "Session", phase: str, detail: str = "") -> None:
    """Append a PRD refinement phase to the session phase log (read by analyze)."""
    session.append(PRD_PHASE_FILE, f"- {phase} · {detail}" if detail else f"- {phase}")


@dataclass(frozen=True)
class PrdVersion:
    """One numbered PRD snapshot on disk."""

    num: int
    label: str
    detail: str = ""


def _prd_versions_dir(session: "Session") -> Path:
    d = session.dir / PRD_VERSION_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def prd_version_file(num: int) -> str:
    return f"{PRD_VERSION_DIR}/{num:03d}.md"


def list_prd_versions(session: "Session") -> list[PrdVersion]:
    """Parse the version manifest; falls back to scanning numbered files."""
    out: list[PrdVersion] = []
    for raw in session.read(PRD_VERSION_MANIFEST).splitlines():
        line = raw.strip()
        if not line.startswith("- "):
            continue
        body = line[2:]
        num_s, _, rest = body.partition(" · ")
        try:
            num = int(num_s)
        except ValueError:
            continue
        label, _, detail = rest.partition(" · ")
        out.append(PrdVersion(num=num, label=label.strip(), detail=detail.strip()))
    if out:
        return out
    files = sorted(_prd_versions_dir(session).glob("*.md"), key=lambda p: p.stem)
    return [PrdVersion(num=int(p.stem), label="snapshot") for p in files]


def read_prd_version(session: "Session", num: int) -> str:
    return session.read(prd_version_file(num))


def _next_prd_version_num(session: "Session") -> int:
    versions = list_prd_versions(session)
    if versions:
        return versions[-1].num + 1
    existing = sorted(_prd_versions_dir(session).glob("*.md"), key=lambda p: p.stem)
    if existing:
        return int(existing[-1].stem) + 1
    return 0


def save_prd_version(
    session: "Session", md: str, *, label: str, detail: str = ""
) -> int | None:
    """Persist a numbered PRD snapshot and append to the manifest.

    Skips when ``md`` is empty or identical to the latest saved version.
    Returns the new version number, or ``None`` when nothing was written.
    """
    text = md.strip()
    if not text:
        return None
    normalized = text + ("\n" if not text.endswith("\n") else "")
    versions = list_prd_versions(session)
    if versions:
        latest = read_prd_version(session, versions[-1].num).strip()
        if latest == normalized.strip():
            return None
    num = _next_prd_version_num(session)
    session.write(prd_version_file(num), normalized)
    manifest_line = f"- {num:03d} · {label}" + (f" · {detail}" if detail else "")
    session.append(PRD_VERSION_MANIFEST, manifest_line + "\n")
    return num


def should_accept_prd_update(previous: str, proposed: str) -> bool:
    """Reject updates that would wipe a substantive PRD with a tiny fragment."""
    prev = previous.strip()
    prop = proposed.strip()
    if not prop:
        return False
    if not prev:
        return True
    prev_stories = len(user_story_titles(prev))
    prop_stories = len(user_story_titles(prop))
    if prev_stories >= 2 and prop_stories < prev_stories:
        return False
    if len(prev) > 500 and len(prop) < max(100, len(prev) // 10):
        return False
    return True


def version_for_phase(
    versions: list[PrdVersion], phase: str, occurrence: int = 0
) -> PrdVersion | None:
    """Map the *n*th occurrence of a phase label to its saved version."""
    phase_l = phase.lower()
    seen = 0
    for ver in versions:
        if ver.label.lower() == phase_l:
            if seen == occurrence:
                return ver
            seen += 1
    return None


def render_prd_version_compare(session: "Session", version: PrdVersion) -> str:
    """Markdown for analyze: previous snapshot, current snapshot, and deltas."""
    current = read_prd_version(session, version.num).strip()
    prev_text = ""
    if version.num > 0:
        prev_text = read_prd_version(session, version.num - 1).strip()
    head = f"# PRD · {version.label}"
    if version.detail:
        head += f" — {version.detail}"
    head += f"\n\n_version {version.num:03d}_"
    parts = [head]
    prev_stories = user_story_titles(prev_text) if prev_text else []
    cur_stories = user_story_titles(current) if current else []
    if prev_text or cur_stories:
        parts.append(
            f"_Stories: {len(prev_stories)} → {len(cur_stories)} · "
            f"chars: {len(prev_text)} → {len(current)}_"
        )
    if prev_text:
        parts.append("## Previous\n\n" + prev_text)
    parts.append("## Current\n\n" + (current or "_(empty)_"))
    return "\n\n".join(parts)


def prd_version_files(session: "Session") -> list[tuple[str, str]]:
    """``(relative_path, label)`` pairs for analyze tree / expand."""
    out: list[tuple[str, str]] = []
    for v in list_prd_versions(session):
        label = f"v{v.num:03d} · {v.label}"
        if v.detail:
            label += f" · {v.detail}"
        out.append((prd_version_file(v.num), label))
    return out


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
    model: str = ""
    tokens: dict[str, int] = field(default_factory=dict)
    cost: float = 0.0


def _load_prd_skill() -> str:
    for p in (Path("skills/prd/SKILL.md"), Path("splinter/skills/prd/SKILL.md")):
        if p.exists():
            return p.read_text()
    return ""


def _ask(prompt: str, *, resume: str | None, session: object = None) -> Turn:
    ladder = load_ladder()
    prd_model = ladder.prd_model
    prd_effort = ladder.prd_effort
    prd_timeout = ladder.prd_timeout
    result, sid = run_provider_session(
        prompt,
        prd_model,
        variant=prd_effort,
        output_format="json",
        session=resume,
        timeout=prd_timeout,
        role="prd",
    )
    tokens = {
        "input": result.tokens.get("input", 0) or 0,
        "output": result.tokens.get("output", 0) or 0,
    }
    cost = result.cost
    if session is not None:
        try:
            session.log_llm_usage(prd_model, tokens, cost)  # type: ignore[attr-defined]
        except Exception:
            pass
    return Turn(text=result.text, session_id=sid or "", model=prd_model, tokens=tokens, cost=cost)


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
        "You MUST ask exactly 3-10 clarifying questions, each with lettered options (A/B/C/D). "
        "Even if the request looks complete, surface the decisions you would otherwise "
        "assume — turn each assumed default into a question whose options include that "
        "default. Never skip, never finalize, never answer the questions yourself, and "
        "never output an 'assumptions' or 'defaults' list in place of questions. "
        "Output ONLY the numbered questions with their lettered options, no preamble."
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


def _current_draft_section(current_prd: str | None) -> str:
    if not (current_prd and current_prd.strip()):
        return ""
    return (
        "## Current PRD Draft in Editor (source of truth)\n"
        f"{current_prd}\n\n"
        "Treat this editor draft as authoritative, incorporating user answers into it.\n\n"
    )


def refine(
    answers: str,
    *,
    resume: str,
    prd_text: str | None = None,
    current_prd: str | None = None,
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
        f"{_current_draft_section(current_prd)}"
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
    current_prd: str | None = None,
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
        f"{_current_draft_section(current_prd)}"
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

    Prefer the section between ``## Working Draft`` and ``## Open Questions``,
    stripping optional fenced code blocks. Falls back to frontmatter or full text.
    """
    section = re.search(
        r"## Working Draft\s*\n(.*?)(?=\n## Open Questions\b)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if section:
        body = section.group(1).strip()
        fenced = re.search(r"^```(?:markdown)?\s*\n(.*?)```", body, re.DOTALL | re.MULTILINE)
        if fenced:
            inner = fenced.group(1).strip()
            tail = body[fenced.end() :].strip()
            if tail and len(tail) > len(inner):
                return (inner + "\n\n" + tail).strip()
            return inner
        return body
    m = re.search(r"## Working Draft.*?```(?:markdown)?\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
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
    events_path = session.dir / "events.jsonl"
    if events_path.exists() and events_path.stat().st_size > 0:
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
