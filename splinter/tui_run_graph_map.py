"""The story graph drawn on a character grid: nodes by wave, routed lines between them."""

from __future__ import annotations

from rich.markup import escape

from splinter.run_graph import GraphNode, RunGraph
from splinter.tui_run_palette import AMBER, BG, DIM, EDGE, FAIL_BG, INK, LIME, MUTED, RED

NODE_WIDTH = 8
_PITCH = 13
_TOP = 2
_NORTH, _EAST, _SOUTH, _WEST = 1, 2, 4, 8
_GLYPHS = {
    _NORTH: "│",
    _SOUTH: "│",
    _NORTH | _SOUTH: "│",
    _EAST: "─",
    _WEST: "─",
    _EAST | _WEST: "─",
    _EAST | _SOUTH: "┌",
    _SOUTH | _WEST: "┐",
    _NORTH | _EAST: "└",
    _NORTH | _WEST: "┘",
    _NORTH | _EAST | _SOUTH: "├",
    _NORTH | _SOUTH | _WEST: "┤",
    _EAST | _SOUTH | _WEST: "┬",
    _NORTH | _EAST | _WEST: "┴",
    _NORTH | _EAST | _SOUTH | _WEST: "┼",
}
_QUIET, _UNLOCKS, _WAITS = 0, 1, 2
_LINE_COLORS = {_QUIET: EDGE, _UNLOCKS: AMBER, _WAITS: LIME}
MARKS = {"done": "✓", "now": "▶", "needs": "!", "ready": "○", "blocked": "·"}
_NODE_STYLES = {
    "done": f"{LIME} on #141a08",
    "now": f"b {AMBER}",
    "needs": f"b {RED} on {FAIL_BG}",
    "ready": INK,
    "blocked": DIM,
}

_Point = tuple[int, int]
_Cells = dict[_Point, list[int]]


def origins(graph: RunGraph) -> dict[int, _Point]:
    """Where each story's node starts: ``{number: (x, y)}``."""
    return {
        node.number: (column * _PITCH, _TOP + 2 * row)
        for column, wave in enumerate(graph.waves)
        for row, node in enumerate(wave)
    }


def map_size(graph: RunGraph) -> tuple[int, int]:
    waves = graph.waves
    tallest = max((len(wave) for wave in waves), default=0)
    return max(len(waves) * _PITCH - (_PITCH - NODE_WIDTH), 0), _TOP + 2 * tallest


def _direction(start: _Point, end: _Point) -> tuple[int, int, int, int]:
    if end[0] > start[0]:
        return 1, 0, _EAST, _WEST
    if end[0] < start[0]:
        return -1, 0, _WEST, _EAST
    if end[1] > start[1]:
        return 0, 1, _SOUTH, _NORTH
    return 0, -1, _NORTH, _SOUTH


def _draw(cells: _Cells, points: list[_Point], rank: int) -> None:
    for start, end in zip(points, points[1:]):
        if start == end:
            continue
        step_x, step_y, leaving, arriving = _direction(start, end)
        x, y = start
        while (x, y) != end:
            _mark(cells, (x, y), leaving, rank)
            x, y = x + step_x, y + step_y
            _mark(cells, (x, y), arriving, rank)


def _mark(cells: _Cells, point: _Point, bit: int, rank: int) -> None:
    cell = cells.setdefault(point, [0, _QUIET])
    cell[0] |= bit
    cell[1] = max(cell[1], rank)


def _route(start: _Point, end: _Point) -> list[_Point]:
    """Out of a node's right edge, along gutters and the free rows, into the next node."""
    tail = (start[0] + NODE_WIDTH, start[1])
    first_gutter = start[0] + NODE_WIDTH + 2
    last_gutter = end[0] - 3
    tip = (end[0] - 1, end[1])
    if first_gutter == last_gutter:
        return [tail, (first_gutter, start[1]), (first_gutter, end[1]), tip]
    free_row = start[1] + 1
    return [
        tail,
        (first_gutter, start[1]),
        (first_gutter, free_row),
        (last_gutter, free_row),
        (last_gutter, end[1]),
        tip,
    ]


def _rank(edge_from: GraphNode, edge_to: GraphNode, selected: int) -> int:
    if edge_to.number == selected:
        return _WAITS
    if edge_from.number == selected:
        return _UNLOCKS
    return _QUIET


def _cells(graph: RunGraph, places: dict[int, _Point], selected: int) -> _Cells:
    cells: _Cells = {}
    by_ident = {node.ident: node for node in graph.nodes}
    edges = [(by_ident[dep], node) for node in graph.nodes for dep in node.deps if dep in by_ident]
    for edge_from, edge_to in edges:
        rank = _rank(edge_from, edge_to, selected)
        _draw(cells, _route(places[edge_from.number], places[edge_to.number]), rank)
    return cells


def _node_markup(node: GraphNode, selected: bool) -> str:
    text = f"{MARKS[node.state]} {node.ident}"[:NODE_WIDTH].ljust(NODE_WIDTH)
    style = f"b {BG} on {INK}" if selected else _NODE_STYLES[node.state]
    return f"[@click=screen.pick({node.number})][{style}]{escape(text)}[/][/]"


def _line_markup(run: str, rank: int) -> str:
    return f"[{_LINE_COLORS[rank]}]{run}[/]" if run.strip() else run


def _row_markup(
    y: int, width: int, cells: _Cells, at: dict[_Point, GraphNode], selected: int
) -> str:
    parts: list[str] = []
    run, run_rank, x = "", _QUIET, 0
    while x < width:
        node = at.get((x, y))
        if node is not None:
            parts.append(_line_markup(run, run_rank))
            parts.append(_node_markup(node, node.number == selected))
            run, x = "", x + NODE_WIDTH
            continue
        mask, rank = cells.get((x, y), (0, _QUIET))
        if rank != run_rank and mask:
            parts.append(_line_markup(run, run_rank))
            run, run_rank = "", rank
        run += _GLYPHS.get(mask, " ")
        x += 1
    parts.append(_line_markup(run.rstrip(), run_rank))
    return "".join(parts)


def map_markup(graph: RunGraph, selected: int) -> str:
    """The whole map as markup; each node is a clickable span."""
    width, height = map_size(graph)
    places = origins(graph)
    cells = _cells(graph, places, selected)
    at = {places[node.number]: node for node in graph.nodes}
    header = "".join(
        f"[{MUTED}]{f'wave {wave[0].wave}':<{_PITCH}}[/]" for wave in graph.waves
    ).rstrip()
    rows = [_row_markup(y, width, cells, at, selected) for y in range(_TOP, height)]
    return "\n".join([header, "", *rows])
