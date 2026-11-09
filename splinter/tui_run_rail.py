"""Right-hand telemetry rail for the live run screen."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Input

from splinter.run_view import IterationRow, LadderRung, RunView, SpendRow, StoryRow
from splinter.tui_run_palette import Ink
from splinter.tui_run_rail_parts import (
    bars_markup,
    clock_markup,
    groups_markup,
    heatmap_markup,
    legend_markup,
    matches,
    rung_classes,
    rung_markup,
    split_markup,
    stories_head,
)
from splinter.tui_run_text import story_id


class Rung(Ink):
    """One ladder card; a click opens the tier view on it."""

    def __init__(self, tier: str) -> None:
        super().__init__(classes="rung")
        self.tier = tier

    def on_click(self, event: events.Click) -> None:
        event.stop()
        self.app.call_later(self.app.run_action, f"open_tier('{self.tier}')")


class FindInput(Input):
    """``/ find``: type an id or words of a title; ``enter`` opens, ``esc`` clears."""

    BINDINGS = [Binding("escape", "close", priority=True)]

    def action_close(self) -> None:
        rail = self.parent.parent if self.parent is not None else None
        if isinstance(rail, RunRail):
            rail.close_find()


class RunRail(Vertical):
    """Render clock, ladder, iterations and story navigation."""

    DEFAULT_CSS = """
    RunRail Ink { height: auto; width: 1fr; }
    RunRail .cap { color: $muted; }
    RunRail #clock-split { margin-bottom: 1; }
    RunRail #ladder { height: 4; margin-bottom: 1; }
    RunRail .rung {
        height: 4;
        width: 1fr;
        padding: 0 0 0 1;
        border: solid #3a3a35;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }
    RunRail .rung.-current { background: $lime; color: $focus-bg; border: solid $lime; }
    RunRail .rung.-current.-pulse { background: #a9d12a; border: solid #a9d12a; }
    RunRail .rung.-used { color: $dim; border: solid $line2; text-style: strike; }
    RunRail .rung.-ahead { color: $muted; }
    RunRail .rung.-picked { border: solid $ink; text-style: bold; }
    RunRail .rung:hover { border: solid $muted; }
    RunRail .rung.-current:hover { border: solid $lime; }
    RunRail.-short #ladder { height: 2; }
    RunRail.-short .rung { height: 2; border: none; margin-right: 1; }
    RunRail.-short .rung.-ahead { background: #141412; }
    RunRail.-short .rung:hover { border: none; background: #1d1d1a; }
    RunRail #find { height: 1; border: none; padding: 0; background: #121210; display: none; }
    RunRail #find.-open { display: block; }
    RunRail #find:focus { border: none; }
    RunRail #find > .input--placeholder { color: $dim; }
    RunRail #iter-bars { margin-bottom: 1; }
    RunRail #legend { margin-bottom: 1; }
    RunRail #groups { height: 1fr; min-height: 0; }
    """

    def __init__(self, *, id: str | None = None, classes: str | None = None) -> None:
        super().__init__(id=id, classes=classes)
        self._keys: dict[str, object] = {}
        self._motion_phase = False
        self.needs_only = False
        self.tier_pick = ""
        self.max_iterations = 5
        self.renders = 0
        self._view: RunView | None = None
        self._story: int | None = None
        self.find_text = ""

    def compose(self) -> ComposeResult:
        yield Ink(id="clock")
        yield Ink(id="clock-split")
        yield Ink(id="ladder-cap", classes="cap")
        yield Horizontal(id="ladder")
        yield Ink(id="iter-cap", classes="cap")
        yield Ink(id="iter-bars")
        yield Ink(id="stories-head")
        yield Ink(id="heatmap")
        yield Ink(legend_markup(), id="legend")
        with Vertical(id="groups"):
            yield FindInput(id="find", placeholder="find a story · id or title")
            yield VerticalScroll(Ink(id="overview"), id="overview-scroll")

    @property
    def inner_width(self) -> int:
        return max(10, self.content_size.width or self.size.width - 5)

    #: Below this many terminal rows the ladder drops its outlines to save two rows.
    SHORT_ROWS = 36

    def on_mount(self) -> None:
        self.set_class(self.app.size.height < self.SHORT_ROWS, "-short")

    def on_resize(self, event: events.Resize) -> None:
        self.set_class(self.app.size.height < self.SHORT_ROWS, "-short")
        self.call_after_refresh(self._reapply)

    def _reapply(self) -> None:
        if self._view is not None:
            self.apply(self._view, self._story)
        self.refresh(layout=True)

    def apply(self, view: RunView, story: int | None) -> None:
        """Apply independent render keys; clock is intentionally tick-only."""
        self._story = story
        row = self._row(view, story)
        iterations = view.iterations_by_story.get(row.number, ()) if row else ()
        width = self.inner_width
        self.query_one("#clock", Ink).update(clock_markup(view.elapsed_s, view.total_cost, width))
        self._update("split", (view.spend, width), self._split)
        self._update("ladder", (view.ladder, row, self.tier_pick), self._ladder)
        bars = (row, iterations, view.live_story, view.stage, width, self.max_iterations)
        self._update("iterations", bars, self._iterations)
        self._view = view
        selected = row.number if row else None
        stories = (view.stories, view.counts, view.needs_you, view.now, view.next_up)
        self._update("heatmap", (stories, selected, width, self.needs_only), self._heatmap)
        self._update("groups", (stories, selected, width, self.needs_only), self._groups)

    def open_find(self) -> None:
        box = self.query_one("#find", FindInput)
        box.add_class("-open")
        box.focus()

    def close_find(self) -> None:
        box = self.query_one("#find", FindInput)
        box.value = ""
        box.remove_class("-open")
        self._set_find("")
        self.app.set_focus(None)

    def _set_find(self, text: str) -> None:
        self.find_text = text
        if "groups" in self._keys:
            self._groups(self._keys["groups"])

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "find":
            event.stop()
            self._set_find(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "find" or self._view is None:
            return
        event.stop()
        found = matches(self._view, event.value)
        self.close_find()
        if found:
            self.app.call_later(self.app.run_action, f"select_story({found[0]})")

    def toggle_needs(self) -> None:
        self.needs_only = not self.needs_only
        self._keys.pop("heatmap", None)
        self._keys.pop("groups", None)

    def set_motion(self, phase: bool) -> None:
        """Repaint the moving parts: the live cell, the live rung and the live bar."""
        self._motion_phase = phase
        for rung in self.query(Rung):
            rung.set_class(phase and rung.has_class("-current"), "-pulse")
        for name, render in (
            ("heatmap", self._heatmap),
            ("iterations", self._iterations),
            ("groups", self._groups),
        ):
            if name in self._keys:
                render(self._keys[name])

    def _update(self, name: str, value: object, render: Callable[[object], None]) -> None:
        if self._keys.get(name) == value:
            return
        self._keys[name] = value
        render(value)
        self.renders += 1

    @staticmethod
    def _row(view: RunView, story: int | None) -> StoryRow | None:
        selected = story if story is not None else view.live_story
        return next((row for row in view.stories if row.number == selected), None)

    def _split(self, value: object) -> None:
        spend, width = cast(tuple[tuple[SpendRow, ...], int], value)
        self.query_one("#clock-split", Ink).update(split_markup(spend, width))

    def _ladder(self, value: object) -> None:
        rungs, row, picked = cast(tuple[tuple[LadderRung, ...], StoryRow | None, str], value)
        current = next((rung.tier for rung in rungs if rung.current), "")
        tier = row.tier if row is not None and row.tier else current
        ident = story_id(row)
        caption = f"model ladder · {ident} is on {tier}" if tier else "model ladder"
        self.query_one("#ladder-cap", Ink).update(caption)
        box = self.query_one("#ladder", Horizontal)
        if [rung.tier for rung in box.query(Rung)] != [rung.tier for rung in rungs]:
            box.remove_children()
            box.mount_all([Rung(rung.tier) for rung in rungs])
            self.call_after_refresh(self._fill_ladder, rungs, tier, picked)
            return
        self._fill_ladder(rungs, tier, picked)

    def _fill_ladder(self, rungs: tuple[LadderRung, ...], tier: str, picked: str) -> None:
        for widget, rung in zip(self.query(Rung), rungs, strict=False):
            widget.update(rung_markup(rung))
            for name in ("-current", "-used", "-ahead", "-picked"):
                widget.remove_class(name)
            on_rung = LadderRung(rung.tier, rung.models, rung.effort, rung.tier == tier, False)
            used = _below(rungs, rung.tier, tier)
            shown = LadderRung(rung.tier, rung.models, rung.effort, on_rung.current, used)
            widget.add_class(*rung_classes(shown, rung.tier == picked))

    def _iterations(self, value: object) -> None:
        row, iterations, _, stage, width, limit = cast(
            tuple[StoryRow | None, tuple[IterationRow, ...], int | None, str, int, int], value
        )
        if row is None:
            self.query_one("#iter-cap", Ink).update("iterations")
            self.query_one("#iter-bars", Ink).update("")
            return
        self.query_one("#iter-cap", Ink).update(f"iterations · {story_id(row)}")
        live = row.state == "live"
        markup = bars_markup(
            row.number, iterations, live, stage, width, limit, self._motion_phase
        )
        self.query_one("#iter-bars", Ink).update(markup)

    def _heatmap(self, value: object) -> None:
        view = self._view
        if view is None:
            return
        _, selected, width, _needs = cast(tuple[object, int | None, int, bool], value)
        self.query_one("#stories-head", Ink).update(stories_head(view, width, self.needs_only))
        markup = heatmap_markup(view, selected, width, self.needs_only, self._motion_phase)
        self.query_one("#heatmap", Ink).update(markup)
        self.query_one("#legend", Ink).update(legend_markup(width))

    def _groups(self, value: object) -> None:
        view = self._view
        if view is None:
            return
        _, selected, width, needs_only = cast(tuple[object, int | None, int, bool], value)
        markup = groups_markup(
            view, selected, width, needs_only, self._motion_phase, self.find_text
        )
        self.query_one("#overview", Ink).update(markup)


def _below(rungs: tuple[LadderRung, ...], tier: str, current: str) -> bool:
    order = [rung.tier for rung in rungs]
    if current not in order or tier not in order:
        return False
    return order.index(tier) < order.index(current)
