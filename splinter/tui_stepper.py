"""Phase stepper widget — one line showing where the conversation stands.

The stepper is app-agnostic: it takes a list of step labels and the index of
the current one, so the PRD app can show ``Draft › Questions › Finalize › Run
setup › Run`` while the plan app shows ``Task › Questions › Plan › Run``.

Steps are drawn lowercase and ``/``-separated, as in the C2 design. Steps already
reached are light and clickable: a click that lands on one
posts :class:`PhaseStepper.StepClicked` and the host app decides what to do
with it (the stepper never changes phases by itself).
"""

from __future__ import annotations

from collections.abc import Sequence

from rich.cells import cell_len
from rich.text import Text
from textual.events import Click
from textual.message import Message
from textual.widgets import Static

#: Steps of the PRD conversation.
PRD_STEPS: tuple[str, ...] = ("Draft", "Questions", "Finalize", "Run setup", "Run")
#: Steps of the plan conversation.
PLAN_STEPS: tuple[str, ...] = ("Task", "Questions", "Plan", "Run")


_LIME = "#c6f432"
_TEXT = "#ecebe4"
_MUTED = "#8f8e85"
_DIM = "#7a7972"


def _short(label: str) -> str:
    """Design spelling of a step: lowercase, ``Run setup`` shown as ``setup``."""
    return label.lower().removeprefix("run ") if label != "Run" else "run"


class PhaseStepper(Static):
    """A single line of steps; the current one is lime, reached ones light, the rest grey."""

    DEFAULT_CSS = """
    PhaseStepper {
        height: 1;
        width: 100%;
        padding: 0 1;
    }
    """

    class StepClicked(Message):
        """A reached step was clicked; the app decides where it lands."""

        def __init__(self, stepper: "PhaseStepper", index: int) -> None:
            super().__init__()
            self.stepper = stepper
            self.index = index

        @property
        def control(self) -> "PhaseStepper":
            return self.stepper

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
        self._reached = 0
        self._spans: list[tuple[int, int]] = []
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

    def set_steps(self, steps: Sequence[str], current: int = 0) -> None:
        """Swap the step labels (another entry mode) and start over at ``current``."""
        self._steps = tuple(steps)
        self._current = 0
        self._reached = 0
        self.set_current(current)
        self.refresh(layout=True)

    def set_current(self, index: int) -> None:
        """Move the stepper to a step (clamped to the available steps)."""
        if not self._steps:
            self._current = 0
            return
        clamped = max(0, min(index, len(self._steps) - 1))
        self._reached = max(self._reached, clamped)
        if clamped != self._current:
            self._current = clamped
            self.refresh()

    def render(self) -> Text:
        text = Text()
        self._spans = []
        for index, label in enumerate(self._steps):
            if index:
                text.append(" / ", style=_DIM)
            start = cell_len(text.plain)
            text.append(_short(label), style=self._style(index))
            self._spans.append((start, cell_len(text.plain)))
        return text

    def _style(self, index: int) -> str:
        if index == self._current:
            return f"bold {_LIME}"
        if index <= self._reached:
            return _TEXT
        return _MUTED

    def on_click(self, event: Click) -> None:
        """Post StepClicked when the click lands on a reached step."""
        index = self._step_at(event.x - self.content_offset.x)
        if index is not None and index <= self._reached and index != self._current:
            self.post_message(self.StepClicked(self, index))

    def _step_at(self, x: int) -> int | None:
        for index, (start, end) in enumerate(self._spans):
            if start <= x < end:
                return index
        return None
