"""Phase stepper widget — one line showing where the conversation stands.

The stepper is app-agnostic: it takes a list of step labels and the index of
the current one, so the PRD app can show ``Draft › Questions › Finalize › Run
setup › Run`` while the plan app shows ``Task › Questions › Plan › Run``.
"""

from __future__ import annotations

from collections.abc import Sequence

from rich.text import Text
from textual.widgets import Static

#: Steps of the PRD conversation.
PRD_STEPS: tuple[str, ...] = ("Draft", "Questions", "Finalize", "Run setup", "Run")
#: Steps of the plan conversation.
PLAN_STEPS: tuple[str, ...] = ("Task", "Questions", "Plan", "Run")


class PhaseStepper(Static):
    """A single line of steps; the current one is bold, the others dim."""

    DEFAULT_CSS = """
    PhaseStepper {
        height: 1;
        width: 100%;
        padding: 0 1;
        color: $text-muted;
    }
    """

    def __init__(
        self,
        steps: Sequence[str] = PRD_STEPS,
        current: int = 0,
        *,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._steps: tuple[str, ...] = tuple(steps)
        self._current = 0
        self.set_current(current)

    @property
    def steps(self) -> tuple[str, ...]:
        """The step labels, in order."""
        return self._steps

    @property
    def current(self) -> int:
        """Index of the current step."""
        return self._current

    @property
    def current_label(self) -> str:
        """Label of the current step."""
        if not self._steps:
            return ""
        return self._steps[self._current]

    def set_current(self, index: int) -> None:
        """Move the stepper to a step (clamped to the available steps)."""
        if not self._steps:
            self._current = 0
            return
        clamped = max(0, min(index, len(self._steps) - 1))
        if clamped != self._current:
            self._current = clamped
            self.refresh()

    def render(self) -> Text:
        text = Text()
        for index, label in enumerate(self._steps):
            if index:
                text.append(" › ", style="dim")
            if index < self._current:
                text.append("✓ ", style="dim")
                text.append(label, style="dim")
            elif index == self._current:
                text.append(label, style="bold")
            else:
                text.append(label, style="dim")
        return text
