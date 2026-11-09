"""The run's stories as a graph: who waits for whom, and where each one stands."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

from splinter.agents.planner import parse_stories
from splinter.scheduling import topo_order

if TYPE_CHECKING:
    from splinter.run_view import IterationRow, StoryRow

NodeState = Literal["done", "now", "needs", "ready", "blocked"]

_SORTING_STRATEGIES = {"cascade", "adaptive"}
_FROM_ROW: dict[str, NodeState] = {
    "pass": "done",
    "live": "now",
    "fail": "needs",
    "asked": "needs",
    "paused": "needs",
}


@dataclass(frozen=True)
class PrdStory:
    number: int
    id: str
    title: str
    deps: tuple[str, ...]
    wave: int


@dataclass(frozen=True)
class GraphNode:
    number: int
    ident: str
    title: str
    state: NodeState
    wave: int
    deps: tuple[str, ...]
    dependents: tuple[str, ...]
    tier: str
    model: str
    iterations: int
    cost: float


@dataclass(frozen=True)
class StoryBrief:
    """What the graph's side pane shows without opening anything."""

    plan: tuple[str, ...]
    iterations: tuple[IterationRow, ...]


@dataclass(frozen=True)
class RunGraph:
    nodes: tuple[GraphNode, ...]

    @property
    def waves(self) -> tuple[tuple[GraphNode, ...], ...]:
        numbers = sorted({node.wave for node in self.nodes})
        return tuple(tuple(n for n in self.nodes if n.wave == wave) for wave in numbers)

    def node(self, number: int) -> GraphNode | None:
        return next((n for n in self.nodes if n.number == number), None)

    def number_of(self, ident: str) -> int | None:
        return next((n.number for n in self.nodes if n.ident == ident), None)

    def count(self, state: NodeState) -> int:
        return sum(1 for node in self.nodes if node.state == state)


def _waves(deps: dict[str, tuple[str, ...]]) -> dict[str, int]:
    found: dict[str, int] = {}

    def wave(ident: str, path: frozenset[str]) -> int:
        if ident not in found:
            inner = path | {ident}
            found[ident] = 1 + max(
                (wave(dep, inner) for dep in deps[ident] if dep not in inner), default=0
            )
        return found[ident]

    for ident in deps:
        wave(ident, frozenset())
    return found


@lru_cache(maxsize=8)
def prd_stories(prd_text: str, topological: bool) -> tuple[PrdStory, ...]:
    """The PRD's stories numbered as the run numbers them."""
    if not prd_text.strip():
        return ()
    try:
        tasks = [task for task in parse_stories(prd_text) if task.id]
    except Exception:
        return ()
    ordered = (topo_order(tasks) if topological else None) or tasks
    known = {task.id for task in ordered}
    deps = {
        str(task.id): tuple(dep for dep in (task.deps or []) if dep in known) for task in ordered
    }
    waves = _waves(deps)
    return tuple(
        PrdStory(
            number=index,
            id=str(task.id),
            title=task.description.splitlines()[0].strip() if task.description else "",
            deps=deps[str(task.id)],
            wave=waves[str(task.id)],
        )
        for index, task in enumerate(ordered, 1)
    )


def run_stories(prd_text: str, strategy: str) -> tuple[PrdStory, ...]:
    return prd_stories(prd_text, strategy in _SORTING_STRATEGIES)


def _short_title(ident: str, title: str) -> str:
    prefix = f"{ident}:"
    return title[len(prefix) :].strip() if title.startswith(prefix) else title


def build_graph(stories: Sequence[PrdStory], rows: Sequence[StoryRow]) -> RunGraph:
    by_number = {row.number: row for row in rows}
    if not stories:
        return RunGraph(tuple(_plain_node(row) for row in rows))
    states = {story.id: _state_of(by_number.get(story.number)) for story in stories}
    nodes = []
    for story in stories:
        row = by_number.get(story.number)
        state = states[story.id]
        if state == "ready" and not all(states[dep] == "done" for dep in story.deps):
            state = "blocked"
        nodes.append(
            GraphNode(
                number=story.number,
                ident=story.id,
                title=_short_title(story.id, story.title),
                state=state,
                wave=story.wave,
                deps=story.deps,
                dependents=tuple(s.id for s in stories if story.id in s.deps),
                tier=row.tier if row else "",
                model=row.model if row else "",
                iterations=row.iterations if row else 0,
                cost=row.cost if row else 0.0,
            )
        )
    return RunGraph(tuple(nodes))


def _state_of(row: StoryRow | None) -> NodeState:
    return _FROM_ROW.get(row.state, "ready") if row is not None else "ready"


def _plain_node(row: StoryRow) -> GraphNode:
    return GraphNode(
        number=row.number,
        ident=row.id,
        title=row.title,
        state=_state_of(row),
        wave=1,
        deps=(),
        dependents=(),
        tier=row.tier,
        model=row.model,
        iterations=row.iterations,
        cost=row.cost,
    )


def plan_excerpt(text: str, limit: int = 8) -> tuple[str, ...]:
    """The first lines of a plan, without its ``# Plan`` heading and blank lines."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and lines[0].lower() == "# plan":
        lines = lines[1:]
    return tuple(lines[:limit])
