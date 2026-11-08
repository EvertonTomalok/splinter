"""Motion and toasts for the C2 run screen.

Three motions run on the screen: the live heatmap cell and the live ladder rung
pulse between two colours, the stage-strip connector flows toward the current
stage and the caret blinks after the sentence while the runner streams. One
timer drives all of them — it alternates a phase every half second, so a full
pulse cycle lasts one second, like the design prototype.

Motion is off when the app runs with ``TEXTUAL_ANIMATIONS=none`` (that is
``App.animation_level == "none"``). Then no timer is created at all and the
screen keeps its still picture.

The toasts are the run's own events: a gate that passed, an escalation, a
runner that asks you and the end of the run. Every toast fires once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from textual.timer import Timer

    from splinter.run_view import RunView

#: Seconds between two motion phases (a full pulse cycle is twice that).
PHASE_SECONDS = 0.5


class MotionClock:
    """One timer that alternates the motion phase of the run screen."""

    def __init__(self, app: Any) -> None:
        self._app = app
        self._timer: Timer | None = None
        self._subscribed = False
        self.phase = False

    @property
    def enabled(self) -> bool:
        """Whether the app allows motion at all."""
        return getattr(self._app, "animation_level", "none") != "none"

    def start(self) -> None:
        """Start the pulse — unless the app turned motion off."""
        if not self.enabled:
            return
        if self._timer is None:
            self._timer = self._app.set_interval(PHASE_SECONDS, self._tick)
        self._subscribe()

    def stop(self) -> None:
        """Stop the pulse and leave the screen in its still picture."""
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _tick(self) -> None:
        """Flip the phase and repaint the parts that move."""
        self.phase = not self.phase
        self._apply()

    def _apply(self) -> None:
        """Hand the phase to the widgets that move."""
        from textual.css.query import NoMatches

        from splinter.tui_run_focus import FocusColumn
        from splinter.tui_run_rail import RunRail

        try:
            self._app.query_one("#focus", FocusColumn).set_motion(self.phase)
            self._app.query_one("#rail", RunRail).set_motion(self.phase)
            hook = getattr(self._app, "on_motion", None)
            if callable(hook):
                hook(self.phase)
        except NoMatches:
            return

    def _subscribe(self) -> None:
        """Follow the app's suspend and resume signals, when it has them."""
        if self._subscribed:
            return
        app = self._app
        subscribed = False
        for name, callback in (
            ("app_suspend_signal", self._on_suspend),
            ("app_resume_signal", self._on_resume),
        ):
            signal = getattr(app, name, None)
            if signal is None:
                continue
            try:
                signal.subscribe(app, callback)
            except Exception:
                continue
            subscribed = True
        self._subscribed = subscribed

    def _on_suspend(self, _data: object = None) -> None:
        """Stop the pulse while the app is suspended."""
        self.stop()

    def _on_resume(self, _data: object = None) -> None:
        """Start the pulse again when the app comes back."""
        self.start()


@dataclass
class ToastTracker:
    """Fires the run's toasts, each of them once."""

    seen: set[str] = field(default_factory=set)

    def events(self, prev: RunView | None, cur: RunView) -> list[str]:
        """The toasts the move from ``prev`` to ``cur`` deserves."""
        if prev is None:
            return []
        messages: list[str] = []
        for message in (
            self._gate_passed(prev, cur),
            self._escalated(prev, cur),
        ):
            if message:
                messages.append(message)
        messages.extend(self._asking(prev, cur))
        complete = self._complete(prev, cur)
        if complete:
            messages.append(complete)
        return messages

    def _gate_passed(self, prev: RunView, cur: RunView) -> str | None:
        """Toast when the run leaves the gate for the eval."""
        if prev.stage != "gate" or cur.stage != "eval":
            return None
        story = cur.live_story
        if story is None:
            return None
        return self._emit(f"gate:{story}", f"Gate passed · {_story_id(cur, story)}")

    def _escalated(self, prev: RunView, cur: RunView) -> str | None:
        """Toast when a new iteration follows an escalation."""
        story = cur.live_story
        if story is None:
            return None
        before = prev.iterations_by_story.get(story, ())
        after = cur.iterations_by_story.get(story, ())
        if len(after) <= len(before) or not before:
            return None
        previous = before[-1]
        if not previous.escalated_to:
            return None
        target = _tier(previous.escalated_to)
        return self._emit(
            f"escalate:{story}:{previous.n}",
            f"Escalated {_story_id(cur, story)} to {target}",
        )

    def _asking(self, prev: RunView, cur: RunView) -> list[str]:
        """Toast when a story starts asking the user."""
        known = {row.number: row.state for row in prev.stories}
        messages: list[str] = []
        for row in cur.stories:
            if row.state != "asked" or known.get(row.number) == "asked":
                continue
            key = f"asked:{row.number}:{row.iterations}"
            message = self._emit(key, f"{row.id} is asking you")
            if message:
                messages.append(message)
        return messages

    def _complete(self, prev: RunView, cur: RunView) -> str | None:
        """Toast when the run reaches its end."""
        if cur.stage != "done" or prev.stage == "done":
            return None
        passed = cur.counts.get("pass", 0)
        total = len(cur.stories)
        return self._emit(
            "complete", f"Run complete · {passed}/{total} · ${cur.total_cost:.4f}"
        )

    def _emit(self, key: str, message: str) -> str | None:
        """Return ``message`` the first time ``key`` is seen."""
        if key in self.seen:
            return None
        self.seen.add(key)
        return message


def _story_id(view: RunView, number: int) -> str:
    """The id of a story: ``T24`` for the story 24."""
    for row in view.stories:
        if row.number == number:
            return row.id
    return f"T{number}"


def _tier(value: str) -> str:
    """The ladder label of a tier: ``1`` becomes ``T1``."""
    text = str(value).strip()
    if not text:
        return ""
    return text if text.upper().startswith("T") else f"T{text}"
