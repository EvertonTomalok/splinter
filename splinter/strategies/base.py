"""Strategy interface and verdict value object for the orchestration loop."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from splinter.agents.runner import RunResult, Task
from splinter.enums import Decision
from splinter.memory.session import Session
from splinter.models.roster import Ladder


@dataclass(frozen=True)
class EvalVerdict:
    decision: str  # one of splinter.enums.Decision
    reason: str
    corrections: str = ""
    raw: str = ""
    #: The evaluator's own provider session id. Reused across same-runner retries
    #: so the eval LLM keeps context on this runner's attempts; reset to a fresh
    #: session only when the eval decides to change the runner (escalate).
    eval_session: str | None = None
    cost: float = 0.0
    tokens: dict[str, int] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.decision == Decision.PASS


@dataclass
class AskUserPause(Exception):
    """Eval loop needs human judgment before continuing."""

    reason: str
    corrections: str = ""
    tier: int = 0
    iteration: int = 0
    task_index: int = 0


@dataclass
class GracefulPause(Exception):
    """Pipeline paused at sub-action boundary — resumable into the in-flight stage."""

    reason: str
    corrections: str = ""
    tier: int = 0
    iteration: int = 0
    task_index: int = 0
    stage: str = ""


@dataclass
class ManualValidationPause(Exception):
    """Pipeline complete but requires manual user validation before closing."""

    def __init__(self, *, summary: str, all_passed: bool = True) -> None:
        super().__init__(summary)
        self.summary = summary
        self.all_passed = all_passed


class Strategy(ABC):
    """A turtle: orchestrates plan/run/gate/eval over a list of tasks."""

    name: str
    aliases: list[str] = []

    @abstractmethod
    def execute(
        self,
        tasks: list[Task],
        session: Session,
        ladder: Ladder,
        *,
        effort: str | None = None,
        budget: float | None = None,
        max_iterations: int = 5,
        localization: str = "",
        eval_skill: str | None = None,
        cowabunga: bool = False,
        resume: bool = False,
        claude_runner_fallback: bool = False,
        user_guidance: str | None = None,
        jump_premium: bool = False,
    ) -> list[RunResult]: ...
