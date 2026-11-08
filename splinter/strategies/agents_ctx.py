"""What every brief worker of one agents run shares: settings, locks, guidance."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

from splinter.agents.acceptance import AcceptanceMode
from splinter.agents.progress import AttemptLog
from splinter.configure import CodegenStep
from splinter.engines.pool import EnginePool
from splinter.memory.orchestrator_notes import notes_for
from splinter.memory.session import Session
from splinter.scheduling import default_max_concurrency
from splinter.vcs.guard import GuardEnv
from splinter.vcs.worktree import WorktreeHandle

log = logging.getLogger("splinter.loop")


@dataclass(frozen=True)
class DirectLoopSettings:
    """What the direct loop needs when the orchestrator takes a brief over."""

    effort: str | None
    budget: float | None
    max_iterations: int
    localization: str
    eval_skill: str | None
    skip_eval: bool


@dataclass
class BriefCtx:
    """Settings of the run (fixed) and the guidance the user sends (it grows)."""

    slug: str
    run_dir: Path
    guard: GuardEnv
    pool: EnginePool
    integration: WorktreeHandle
    mode: AcceptanceMode
    stall_min: float
    max_attempts: int
    timeout: int
    direct: DirectLoopSettings
    attempts: AttemptLog
    stories: dict[str, int]
    story_count: int
    codegen: tuple[CodegenStep, ...] = ()
    concurrency: int | None = None
    worktrees: dict[str, dict[str, str]] = field(default_factory=dict)
    guidance: list[str] = field(default_factory=list)
    story_guidance: dict[int, list[str]] = field(default_factory=dict)
    session_lock: threading.Lock = field(default_factory=threading.Lock)
    git_lock: threading.Lock = field(default_factory=threading.Lock)

    def story_of(self, brief_id: str) -> int:
        return self.stories.get(brief_id, 1)

    def max_concurrency(self) -> int:
        return self.concurrency or default_max_concurrency()

    def guidance_for(self, story: int) -> list[str]:
        """What the user sent for every brief plus what they sent for ``story``."""
        with self.session_lock:
            return [*self.guidance, *self.story_guidance.get(story, [])]

    def take_guidance(self, session: Session) -> None:
        """Take every note and directive the user sent, the moment it lands.

        Receipts mark them delivered at once; every brief started from now on
        and its acceptance carry them.
        """
        note = notes_for(session, "dispatch", "agents").removeprefix("## Orchestrator notes")
        shared = session.pop_live_commands(None)
        per_story = {
            story: text
            for story in range(1, self.story_count + 1)
            if session.directive_pending(story) and (text := session.pop_live_commands(story))
        }
        with self.session_lock:
            for text in (note.strip(), shared.strip()):
                if text:
                    self.guidance.append(text)
                    log.info("orchestrator · guidance for every next brief: %s", text[:120])
            for story, text in per_story.items():
                self.story_guidance.setdefault(story, []).append(text.strip())
                log.info("orchestrator · guidance for T%d: %s", story, text[:120])

    def acceptance_notes(self, session: Session, story: int) -> str:
        notes = notes_for(session, "acceptance", "agents")
        guidance = self.guidance_for(story)
        if not guidance:
            return notes
        lines = "\n".join(f"- {text}" for text in guidance)
        return (notes + "\n\n" if notes else "") + f"## User guidance\n{lines}"
