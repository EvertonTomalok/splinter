"""One-line busy status: spinner, action, model and elapsed seconds.

Model calls can take minutes; without any motion the screen looks frozen. This
widget shows a small braille spinner next to the action being taken, the model
doing it and how long it has been running.
"""

from __future__ import annotations

import time

from textual.timer import Timer
from textual.widgets import Static

#: Small braille spinner — one cell per frame.
SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
#: Seconds between frames.
FRAME_SECONDS = 0.12


class BusyStatus(Static):
    """A one-line busy indicator; hidden while idle."""

    DEFAULT_CSS = """
    BusyStatus {
        height: 1;
        width: 100%;
        padding: 0 1;
        color: $text-muted;
    }
    """

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self._action = ""
        self._model = ""
        self._started = 0.0
        self._frame = 0
        self._timer: Timer | None = None
        self._active = False
        self.display = False

    @property
    def action(self) -> str:
        """The action being taken."""
        return self._action

    @property
    def model(self) -> str:
        """The model doing the work."""
        return self._model

    @property
    def elapsed(self) -> float:
        """Seconds since the status was started."""
        if not self._active:
            return 0.0
        return time.monotonic() - self._started

    def start(self, action: str, model: str = "") -> None:
        """Show the status and start the spinner."""
        self._action = action
        self._model = model
        self._started = time.monotonic()
        self._frame = 0
        self._active = True
        self.display = True
        self._tick()
        if self.is_mounted:
            self._start_timer()

    def stop(self) -> None:
        """Hide the status and stop the spinner."""
        self._active = False
        self.display = False
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def on_mount(self) -> None:
        if self._active:
            self._start_timer()

    def _start_timer(self) -> None:
        if self._timer is None:
            self._timer = self.set_interval(FRAME_SECONDS, self._tick)

    def _tick(self) -> None:
        frame = SPINNER_FRAMES[self._frame % len(SPINNER_FRAMES)]
        self._frame += 1
        elapsed = int(time.monotonic() - self._started)
        parts = [f"{frame} {self._action}".strip(), self._model, f"{elapsed}s"]
        self.update(" · ".join(part for part in parts if part))
