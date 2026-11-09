"""What to do with one story: the actions a click on it offers."""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from splinter.run_view import StoryRow
from splinter.tui_run_palette import BG, EDGE, INK, LIME, MUTED, SELECTED
from splinter.tui_run_text import story_label

STORY_ACTIONS: tuple[tuple[str, str, str], ...] = (
    ("plan", "p", "Read plan"),
    ("act", "a", "Activity"),
    ("iterations", "o", "Iterations"),
    ("logs", "l", "Logs"),
    ("gate", "g", "Gate checks"),
    ("eval", "e", "Eval verdict"),
    ("diff", "d", "Diff"),
    ("files", "f", "Files"),
    ("message", "m", "Message story"),
)


INTERVENTIONS: tuple[tuple[str, str, str], ...] = (
    ("orchestrator", "x", "Send to orchestrator"),
    ("reroute", "r", "Route to next model"),
    ("tier_up", "t", "Raise tier"),
)
INTERVENTION_IDS = tuple(action for action, _key, _label in INTERVENTIONS)
ACTIVE_STATES = ("live", "fail", "paused", "asked")
_CONSEQUENCE = {
    "orchestrator": "The orchestrator finishes it in the same worktree, on the planner's tier.",
    "reroute": "It continues on the next model of this tier. That model is skipped for a while.",
    "tier_up": "It continues one tier up, with a fresh session.",
}


def _item(key: str, label: str) -> Text:
    row = Text(f"{key}  ", style=f"bold {LIME}")
    row.append(label, style=INK)
    return row


class StoryActions(ModalScreen[str | None]):
    """Dismisses with the id of the chosen action, or ``None``."""

    BINDINGS = [
        Binding("escape", "close", "Close"),
        *(
            Binding(key, f"pick('{action}')", label, show=False)
            for action, key, label in (*STORY_ACTIONS, *INTERVENTIONS)
        ),
    ]

    DEFAULT_CSS = f"""
    StoryActions {{ align: center middle; background: {BG} 70%; }}
    StoryActions #story-menu {{
        width: 56;
        height: auto;
        padding: 1 2;
        background: {BG};
        border: round {EDGE};
    }}
    StoryActions #story-menu-head {{ height: auto; margin-bottom: 1; }}
    StoryActions OptionList {{ height: auto; border: none; padding: 0; background: {BG}; }}
    StoryActions OptionList:focus {{ border: none; background-tint: {BG} 0%; }}
    StoryActions OptionList > .option-list--option-highlighted {{ background: {SELECTED}; }}
    StoryActions #story-menu-hint {{ height: 1; margin-top: 1; color: {MUTED}; }}
    """

    def __init__(self, row: StoryRow) -> None:
        super().__init__()
        self._row = row
        extra = INTERVENTIONS if row.state in ACTIVE_STATES else ()
        self._chosens = (*STORY_ACTIONS, *extra)

    def compose(self) -> ComposeResult:
        ident, title = story_label(self._row)
        head = Text(ident, style=f"bold {INK}")
        head.append(f"\n{title}", style=MUTED)
        with Vertical(id="story-menu"):
            yield Static(head, id="story-menu-head")
            yield OptionList(
                *(Option(_item(key, label), id=action) for action, key, label in self._chosens)
            )
            yield Static("enter opens · esc closes", id="story-menu-hint")

    def on_mount(self) -> None:
        self.query_one(OptionList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id)

    def action_pick(self, action: str) -> None:
        if any(known == action for known, _key, _label in self._chosens):
            self.dismiss(action)

    def action_close(self) -> None:
        self.dismiss(None)


class ConfirmIntervention(ModalScreen[bool]):
    """Interventions stop a running engine, so each one is confirmed."""

    BINDINGS = [
        Binding("y", "answer(True)", "Confirm"),
        Binding("enter", "answer(True)", "Confirm", show=False),
        Binding("n", "answer(False)", "Cancel", show=False),
        Binding("escape", "answer(False)", "Cancel"),
    ]

    DEFAULT_CSS = f"""
    ConfirmIntervention {{ align: center middle; background: {BG} 70%; }}
    ConfirmIntervention #confirm-box {{
        width: 64;
        height: auto;
        padding: 1 2;
        background: {BG};
        border: round {EDGE};
    }}
    ConfirmIntervention Static {{ height: auto; }}
    ConfirmIntervention #confirm-keys {{ margin-top: 1; color: {MUTED}; }}
    """

    def __init__(self, row: StoryRow, action: str, restarts: int = 0, running: bool = True) -> None:
        super().__init__()
        self._row = row
        self._chosen = action
        self._restarts = restarts
        self._engine_live = running

    def _body(self) -> str:
        lines = [_CONSEQUENCE.get(self._chosen, "")]
        if self._engine_live:
            lines.append("Its running engine stops now. The other stories keep going.")
        else:
            lines.append(
                "It already ended, so the run restarts. "
                f"{self._restarts} other stories restart too."
            )
        return "\n".join(line for line in lines if line)

    def compose(self) -> ComposeResult:
        label = next(
            (name for action, _key, name in INTERVENTIONS if action == self._chosen), "Retry"
        )
        head = Text(f"{label} · {story_label(self._row)[0]}?", style=f"bold {INK}")
        with Vertical(id="confirm-box"):
            yield Static(head)
            yield Static(Text(self._body(), style=MUTED))
            yield Static("y confirm · esc cancel", id="confirm-keys")

    def action_answer(self, value: bool) -> None:
        self.dismiss(value)
