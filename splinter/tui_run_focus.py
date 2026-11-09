"""Focused left-hand column for the live run screen."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from textual import events
from textual.app import ComposeResult
from textual.containers import Vertical

from splinter.run_view import IterationRow, RunView, StoryRow
from splinter.tui_blockword import BlockWord
from splinter.tui_follow import FollowLog, FollowScroll, new_bar
from splinter.tui_run_live import LiveAction, live_phrase
from splinter.tui_run_palette import DIM, INK, LIME, LIVE_LINE, MUTED, RED, Ink
from splinter.tui_run_text import cell_len, pad, story_id

_STAGES = ("plan", "run", "gate", "eval", "merge")
#: Sub-topics of each stage, in order; the first one opens when the stage is clicked.
TOPICS: dict[str, tuple[tuple[str, str], ...]] = {
    "plan": (("plan", "steps"), ("files", "files"), ("filter", "filters")),
    "run": (("logs", "logs"), ("act", "activity"), ("diff", "diff")),
    "gate": (("gate", "checks"),),
    "eval": (("eval", "verdict"),),
    "merge": (("diff", "diff"),),
    "cost": (("spend", "cost"), ("tier", "tier")),
}
#: The stage a view belongs to.
VIEW_STAGE: dict[str, str] = {
    view: stage for stage, topics in TOPICS.items() for view, _ in topics if stage != "merge"
}
_VIEW_WORD = {
    "plan": "PLAN",
    "files": "PLAN",
    "filter": "PLAN",
    "logs": "LOG",
    "gate": "GATE",
    "eval": "EVAL",
    "spend": "COST",
    "tier": "TIER",
}
_STATE_WORD = {
    "pass": ("PASS", "done"),
    "fail": ("FAIL", "fail"),
    "asked": ("ASK", "wait"),
    "paused": ("WAIT", "wait"),
    "queued": ("NEXT", "idle"),
}


LOG_HEAD = (
    f"[{MUTED}]log[/] [{DIM}]· pipeline and agents, live[/]    "
    f"[@click=app.trace][b {LIME}]L[/] [{INK}]⤢ full trace[/][/]"
)


def _live_stage(view: RunView, row: StoryRow | None) -> str:
    """The run's stage, or the story's own when several stories run at once."""
    if row is None or sum(story.state == "live" for story in view.stories) < 2:
        return view.stage
    iterations = view.iterations_by_story.get(row.number, ())
    return "eval" if iterations and not iterations[-1].verdict else "run"


class FocusColumn(Vertical):
    """Render selected story context without rebuilding the activity log."""

    DEFAULT_CSS = """
    FocusColumn #block-word { margin-bottom: 1; }
    FocusColumn #sentence { height: 1; margin-bottom: 1; }
    FocusColumn #stage-strip { height: 1; margin-bottom: 1; }
    FocusColumn #view-tabs { height: 3; border-top: solid $line; }
    FocusColumn #view-body { height: 1fr; min-height: 0; }
    FocusColumn #needs-you-slot { height: auto; }
    FocusColumn #view-scroll { height: 1fr; scrollbar-size-vertical: 1; }
    FocusColumn #view-scroll.-bottom { align-vertical: bottom; }
    FocusColumn #view-static { height: auto; }
    FocusColumn #log-pane { height: 1fr; display: none; }
    FocusColumn #log-head { height: 1; }
    FocusColumn .new-below { height: 1; dock: bottom; background: $hover; }
    """

    available_views: tuple[str, ...] = ("act",)

    def __init__(self, *, id: str | None = None, classes: str | None = None) -> None:
        super().__init__(id=id, classes=classes)
        self._keys: dict[str, object] = {}
        self._motion_phase = False
        self.current_view = "act"
        self.tier_pick = ""
        self.renders = 0
        self._last: tuple[RunView, int | None] | None = None
        self.live_action: LiveAction | None = None

    def compose(self) -> ComposeResult:
        yield BlockWord(word="", id="block-word")
        yield Ink(id="sentence")
        yield Ink(id="stage-strip")
        yield Ink(id="view-tabs")
        with Vertical(id="view-body"):
            yield Vertical(id="needs-you-slot")
            yield FollowScroll(Ink(id="view-static"), id="view-scroll")
            yield new_bar("view-scroll", "view-new")
        with Vertical(id="log-pane"):
            yield Ink(LOG_HEAD, id="log-head")
            yield FollowLog(id="log", markup=True, wrap=True, highlight=False)
            yield new_bar("log", "log-new")

    def on_resize(self, event: events.Resize) -> None:
        self.call_after_refresh(self._reapply)

    def _reapply(self) -> None:
        if self._last is not None:
            self.apply(*self._last)
        self.refresh(layout=True)

    def apply(self, view: RunView, story: int | None) -> None:
        """Update story-dependent children only when their data changed."""
        self._last = (view, story)
        row = self._row(view, story)
        iterations = view.iterations_by_story.get(row.number, ()) if row else ()
        last = iterations[-1] if iterations else None
        live = row is not None and row.state == "live"
        width = self.size.width
        self._update("word", self._word_and_role(view, row, live), self._apply_word)
        sentence = (self._sentence(view, row, last, live), self._streaming(view, row))
        self._update("sentence", sentence, self._apply_sentence)
        stage = (self._stage_index(view, row, live), self._stage_tone(row), width)
        self._update("stage", (*stage, self.current_view), self._apply_stage)
        tabs = (self.available_views, self.current_view, self.tier_pick)
        self._update("tabs", tabs, self._apply_tabs)

    def set_motion(self, phase: bool) -> None:
        """Repaint the moving parts: the caret and the stage connector."""
        self._motion_phase = phase
        if "sentence" in self._keys:
            self._apply_sentence(self._keys["sentence"])
        if "stage" in self._keys:
            self._apply_stage(self._keys["stage"])

    @staticmethod
    def _streaming(view: RunView, row: StoryRow | None) -> bool:
        """Whether the runner is streaming for the selected story."""
        return row is not None and row.state == "live" and view.stage == "run"

    def _update(self, name: str, value: object, apply: Callable[[object], None]) -> None:
        if self._keys.get(name) == value:
            return
        self._keys[name] = value
        apply(value)
        self.renders += 1

    @staticmethod
    def _row(view: RunView, story: int | None) -> StoryRow | None:
        selected = story if story is not None else view.live_story
        return next((row for row in view.stories if row.number == selected), None)

    def _word_and_role(self, view: RunView, row: StoryRow | None, live: bool) -> tuple[str, str]:
        if self.current_view in _VIEW_WORD:
            return (_VIEW_WORD[self.current_view], "live")
        if row is None or live:
            word = view.stage.upper() if view.stage not in ("done", "wait") else "WAIT"
            return (word, "wait" if word == "WAIT" else "live")
        return _STATE_WORD.get(row.state, (view.stage.upper(), "live"))

    def _sentence(
        self, view: RunView, row: StoryRow | None, last: IterationRow | None, live: bool
    ) -> str:
        if self.current_view == "spend":
            return f"Where the ${view.total_cost:.2f} went, by step and by story"
        if self.current_view == "tier":
            tier = self.tier_pick or next((r.tier for r in view.ladder if r.current), "T0")
            return f"Who runs on {tier}, and how it went"
        if row is None:
            return "Waiting for the first story"
        ident = story_id(row)
        if live:
            action = self._current_action(row, last)
            if view.stage == "run" and action is not None:
                return live_phrase(_runner(last), action, max(30, self.size.width - 4))
            return _live_sentence(view.stage, ident, last)
        live_id = next((story_id(r) for r in view.stories if r.number == view.live_story), "")
        return _state_sentence(row, ident, live_id)

    def _current_action(self, row: StoryRow, last: IterationRow | None) -> LiveAction | None:
        action = self.live_action
        if action is None or action.story != row.number:
            return None
        iteration = 1 if last is None else last.n + (1 if last.verdict else 0)
        return action if action.iteration == iteration else None

    def _apply_word(self, value: object) -> None:
        word, role = cast(tuple[str, str], value)
        block = self.query_one("#block-word", BlockWord)
        block.set_role(role)
        block.word = word

    def _apply_sentence(self, value: object) -> None:
        text, streaming = cast(tuple[str, bool], value)
        caret = f" [{LIME}]█[/]" if streaming and not self._motion_phase else ""
        if streaming and self._motion_phase:
            caret = "  "
        self.query_one("#sentence", Ink).update(f"[{INK}]{_esc(text)}[/]{caret}")

    @staticmethod
    def _stage_index(view: RunView, row: StoryRow | None, live: bool) -> int:
        if row is None or live:
            stage = _live_stage(view, row)
            return _STAGES.index(stage) if stage in _STAGES else 1
        return {"pass": 5, "queued": -1}.get(row.state, 4)

    @staticmethod
    def _stage_tone(row: StoryRow | None) -> str:
        return RED if row is not None and row.state in ("fail", "paused", "asked") else LIME

    def _apply_stage(self, value: object) -> None:
        current, tone, width, viewing = cast(tuple[int, str, int, str], value)
        parts: list[str] = []
        for index, name in enumerate(_STAGES):
            mark, style = ("■", MUTED) if index < current else ("□", DIM)
            if index == current:
                mark, style = "▶", f"b {tone}"
            if VIEW_STAGE.get(viewing) == name:
                style += " underline"
            go = f"[@click=app.stage_view('{name}')]"
            parts.append(f"{go}[{style}]{mark} {name.upper()}[/][/]")
            if index < len(_STAGES) - 1:
                flow = index == current - 1 and self._motion_phase
                parts.append(f"[{LIVE_LINE if flow else DIM}]───[/]")
        strip = " ".join(parts)
        hint = "click a stage to open it"
        if cell_len(strip) + len(hint) + 2 <= width:
            strip = pad(strip, width - len(hint)) + f"[{DIM}]{hint}[/]"
        self.query_one("#stage-strip", Ink).update(strip)

    def _apply_tabs(self, value: object) -> None:
        available, viewing, tier = cast(tuple[tuple[str, ...], str, str], value)
        topics = TOPICS.get(VIEW_STAGE.get(viewing, "run"), ())
        tabs = [(name, label) for name, label in topics if name in available]
        tabs = [(n, f"tier {tier}".rstrip() if n == "tier" else lbl) for n, lbl in tabs]
        widget = self.query_one("#view-tabs", Ink)
        widget.display = len(tabs) > 1
        if len(tabs) <= 1:
            return
        labels: list[str] = []
        marks: list[str] = []
        for name, label in tabs:
            on = name == viewing
            tone = f"b {INK}" if on else MUTED
            labels.append(f"[@click=app.set_view('{name}')][{tone}]{label}[/][/]")
            marks.append(f"[{LIME}]{'▔' * len(label)}[/]" if on else " " * len(label))
        widget.update("   ".join(labels) + "\n" + "   ".join(marks))


def _esc(text: str) -> str:
    from rich.markup import escape

    return escape(text)


def _runner(last: IterationRow | None) -> str:
    if last is None or not last.model:
        return "the runner"
    model = last.model.split("/")[-1]
    return f"{model}@{last.effort}" if last.effort and "@" not in model else model


def _live_sentence(stage: str, ident: str, last: IterationRow | None) -> str:
    iteration = f" · iteration {last.n}" if last is not None else ""
    sentences = {
        "plan": f"The planner is writing the plan for {ident}",
        "run": f"{_runner(last)} is working on {ident}{iteration}",
        "gate": "Gate is running the checks on the worktree",
        "eval": f"The judge is reviewing the diff against {ident}",
        "wait": f"{ident} waits for you",
        "done": f"{ident} is done",
    }
    return sentences.get(stage, sentences["run"])


def _state_sentence(row: StoryRow, ident: str, live_id: str) -> str:
    behind = f" behind {live_id}" if live_id else ""
    sentences = {
        "pass": f"{ident} passed and merged into the integration branch",
        "fail": f"{ident} failed after {row.iterations} iterations and needs a decision from you",
        "asked": f"{ident} is paused until you answer the runner",
        "paused": f"{ident} waits for your validation",
        "queued": f"{ident} waits in the queue{behind}",
    }
    return sentences.get(row.state, f"{ident} · {row.title}")
