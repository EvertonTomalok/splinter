"""Strategy interface and verdict value object for the orchestration loop."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

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

    @property
    def passed(self) -> bool:
        return self.decision == Decision.PASS


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
    ) -> list[RunResult]:
        ...
