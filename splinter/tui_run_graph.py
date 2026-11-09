"""Full-screen map of the run: every story, what it waits for and what it unlocks.

Nodes sit in columns by wave, joined by routed lines. The selected node lights the
stories it waits for and the ones it unlocks; the side pane shows its plan, its
iterations and its actions at once. The screen reads the graph once a second.
"""

from __future__ import annotations

from collections.abc import Callable

from rich.markup import escape
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, VerticalScroll
from textual.geometry import Region
from textual.screen import Screen

from splinter.run_graph import GraphNode, NodeState, RunGraph, StoryBrief
from splinter.run_view import IterationRow
from splinter.tui_run_chrome import key_markup
from splinter.tui_run_graph_map import MARKS, NODE_WIDTH, map_markup, map_size, origins
from splinter.tui_run_palette import (
    AMBER,
    BG,
    DIM,
    INK,
    LIME,
    LINE,
    LINE2,
    MUTED,
    RED,
    Ink,
    verdict_label,
)
from splinter.tui_run_story_menu import INTERVENTIONS, STORY_ACTIONS

_STATE_WORDS: dict[NodeState, tuple[str, str]] = {
    "done": ("done", LIME),
    "now": ("running now", AMBER),
    "needs": ("needs you", RED),
    "ready": ("next", INK),
    "blocked": ("later", DIM),
}
_START_ORDER: tuple[NodeState, ...] = ("now", "needs", "ready", "blocked", "done")
_KEYS = (
    ("←↑↓→", "move", "screen.noop"),
    ("enter", "actions", "screen.open_actions"),
    ("n", "jump to now", "screen.jump_now"),
    ("esc", "back to the run", "screen.close"),
)


def _chips(idents: tuple[str, ...], graph: RunGraph, color: str) -> str:
    links = []
    for ident in idents:
        number = graph.number_of(ident)
        links.append(f"[@click=screen.pick({number})][{color}]{escape(ident)}[/][/]")
    return "  ".join(links) if links else f"[{DIM}]nothing[/]"


def _iteration_line(row: IterationRow) -> str:
    verdict = verdict_label(row.verdict) if row.verdict and row.verdict != "?" else "live"
    tone = {"PASS": LIME, "live": AMBER}.get(verdict, RED)
    return (
        f"[{MUTED}]{row.n:>2}[/]  [{INK}]{escape(row.tier):<3}[/] [{tone}]{escape(verdict):<6}[/]"
        f" [{MUTED}]${row.cost:.4f}  {int(row.duration_s) // 60}m{int(row.duration_s) % 60:02}s[/]"
    )


def detail_markup(node: GraphNode, brief: StoryBrief, graph: RunGraph) -> str:
    """Everything the side pane shows about one story."""
    word, tone = _STATE_WORDS[node.state]
    facts = " · ".join(
        part
        for part in (
            f"wave {node.wave}",
            node.tier,
            node.model,
            f"${node.cost:.4f}" if node.cost else "",
        )
        if part
    )
    lines = [
        f"[b {INK}]{MARKS[node.state]} {escape(node.ident)}[/]  [{tone}]{word}[/]",
        f"[{INK}]{escape(node.title)}[/]",
        f"[{MUTED}]{escape(facts)}[/]",
        "",
        f"[{LIME}]waits for[/]  {_chips(node.deps, graph, LIME)}",
        f"[{AMBER}]unlocks[/]    {_chips(node.dependents, graph, AMBER)}",
        "",
        f"[{MUTED}]iterations · {len(brief.iterations)}[/]",
        *(_iteration_line(row) for row in brief.iterations),
    ]
    if not brief.iterations:
        lines.append(f"[{DIM}]not started[/]")
    lines += ["", f"[{MUTED}]plan[/]"]
    lines += [f"[{INK}]{escape(line)}[/]" for line in brief.plan] or [f"[{DIM}]no plan yet[/]"]
    if brief.plan:
        lines.append(key_markup("p", "read the whole plan", "screen.act('plan')"))
    lines += ["", f"[{MUTED}]actions[/]"]
    lines += [
        key_markup(key, label, f"screen.act('{action}')") for action, key, label in STORY_ACTIONS
    ]
    if node.state in ("now", "needs"):
        lines += ["", f"[{MUTED}]intervene · asks first[/]"]
        lines += [
            key_markup(key, label, f"screen.act('{action}')")
            for action, key, label in INTERVENTIONS
        ]
    return "\n".join(lines)


class RunGraphScreen(Screen[None]):
    """The run as a node map, with the selected story's details beside it."""

    BINDINGS = [
        Binding("escape", "close", "Back"),
        Binding("G", "close", "Back", show=False),
        Binding("left", "move(-1,0)", "Left", show=False, priority=True),
        Binding("right", "move(1,0)", "Right", show=False, priority=True),
        Binding("up", "move(0,-1)", "Up", show=False, priority=True),
        Binding("down", "move(0,1)", "Down", show=False, priority=True),
        Binding("enter", "open_actions", "Actions", show=False),
        Binding("n", "jump_now", "Now", show=False),
        *(
            Binding(key, f"act('{action}')", label, show=False)
            for action, key, label in (*STORY_ACTIONS, *INTERVENTIONS)
        ),
    ]

    DEFAULT_CSS = f"""
    RunGraphScreen {{ background: {BG}; }}
    RunGraphScreen #graph-top {{ height: 2; padding: 0 3; border-bottom: solid {LINE}; }}
    RunGraphScreen #graph-crumbs {{ width: 1fr; height: 1; }}
    RunGraphScreen #graph-stats {{ width: auto; height: 1; }}
    RunGraphScreen #graph-main {{ height: 1fr; }}
    RunGraphScreen #graph-scroll {{
        width: 1fr;
        height: 1fr;
        padding: 1 2 0 3;
        scrollbar-size-vertical: 1;
        scrollbar-size-horizontal: 1;
        scrollbar-background: {BG};
        scrollbar-color: {LINE2};
    }}
    RunGraphScreen #graph-map {{ height: auto; text-wrap: nowrap; }}
    RunGraphScreen #graph-side {{
        width: 48;
        height: 1fr;
        padding: 1 2;
        border-left: solid {LINE};
        scrollbar-size-vertical: 1;
        scrollbar-background: {BG};
        scrollbar-color: {LINE2};
    }}
    RunGraphScreen #graph-detail {{ height: auto; }}
    RunGraphScreen #graph-keys {{ height: 2; padding: 0 3; border-top: solid {LINE}; }}
    RunGraphScreen #graph-keys Ink {{ width: auto; height: 1; margin-right: 2; }}
    """

    def __init__(
        self,
        source: Callable[[], RunGraph],
        brief: Callable[[int], StoryBrief],
        crumbs: str = "",
    ) -> None:
        super().__init__()
        self._source = source
        self._brief = brief
        self._crumbs = crumbs
        self.graph = source()
        self.selected = self._first()
        self._painted: tuple[object, ...] = ()

    def compose(self) -> ComposeResult:
        with Horizontal(id="graph-top"):
            yield Ink(self._crumbs, id="graph-crumbs")
            yield Ink(id="graph-stats")
        with Horizontal(id="graph-main"):
            with ScrollableContainer(id="graph-scroll"):
                yield Ink(id="graph-map")
            with VerticalScroll(id="graph-side"):
                yield Ink(id="graph-detail")
        with Horizontal(id="graph-keys"):
            for key, label, action in _KEYS:
                yield Ink(key_markup(key, label, action))

    def on_mount(self) -> None:
        self.query_one("#graph-scroll").can_focus = False
        self.query_one("#graph-side").can_focus = False
        self._paint()
        self._reveal()
        self.set_interval(1.0, self._tick)

    def _first(self) -> int:
        for state in _START_ORDER:
            for node in self.graph.nodes:
                if node.state == state:
                    return node.number
        return 0

    def _tick(self) -> None:
        self.graph = self._source()
        self._paint()

    def _paint(self) -> None:
        node = self.graph.node(self.selected)
        if node is None:
            return
        brief = self._brief(self.selected)
        key = (self.graph, self.selected, brief)
        if key == self._painted:
            return
        self._painted = key
        width, _height = map_size(self.graph)
        canvas = self.query_one("#graph-map", Ink)
        canvas.styles.width = width
        canvas.update(map_markup(self.graph, self.selected))
        self.query_one("#graph-detail", Ink).update(detail_markup(node, brief, self.graph))
        self.query_one("#graph-stats", Ink).update(self._stats())

    def _stats(self) -> str:
        graph = self.graph
        return (
            f"[b {INK}]{graph.count('done')}/{len(graph.nodes)}[/] [{MUTED}]done[/] "
            f"[{DIM}]·[/] [b {AMBER}]{graph.count('now')}[/] [{MUTED}]now[/] "
            f"[{DIM}]·[/] [b {RED}]{graph.count('needs')}[/] [{MUTED}]need you[/] "
            f"[{DIM}]·[/] [b {INK}]{graph.count('ready')}[/] [{MUTED}]next[/]"
        )

    def _reveal(self) -> None:
        place = origins(self.graph).get(self.selected)
        if place is None:
            return
        region = Region(place[0], place[1], NODE_WIDTH, 1)
        self.query_one("#graph-scroll").scroll_to_region(region, animate=False)

    def _select(self, number: int) -> None:
        self.selected = number
        self._paint()
        self._reveal()

    def action_noop(self) -> None:
        return

    def action_pick(self, number: int) -> None:
        self._select(number)

    def action_move(self, columns: int, rows: int) -> None:
        waves = self.graph.waves
        spot = next(
            ((c, r) for c, wave in enumerate(waves) for r, n in enumerate(wave)
             if n.number == self.selected),
            None,
        )
        if spot is None:
            return
        column = min(max(spot[0] + columns, 0), len(waves) - 1)
        row = min(max(spot[1] + rows, 0), len(waves[column]) - 1)
        self._select(waves[column][row].number)

    def action_jump_now(self) -> None:
        self._select(self._first())

    def action_open_actions(self) -> None:
        self.app.call_later(self.app.run_action, f"story_menu({self.selected})")

    def action_act(self, action: str) -> None:
        self.app.call_later(self.app.run_action, f"story_do({self.selected},'{action}')")

    def action_close(self) -> None:
        self.app.pop_screen()
