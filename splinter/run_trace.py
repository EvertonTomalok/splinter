"""The whole run as a tree of what happened: events, stories, plans, iterations, actions.

Each node carries a short label and the full text behind it (a plan, a review, the
runner's output, one tool call), so the trace screen can show the outline on the left
and the selected text on the right. Built from the files the run already writes; no
model is called and nothing is parsed twice per refresh.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from rich.markup import escape
from rich.text import Text

from splinter.memory.session import Session
from splinter.prd_session import list_prd_versions, prd_version_file
from splinter.run_graph import prd_stories
from splinter.run_view import IterationRow, RunView, StoryRow
from splinter.run_view_files import gate_lines, iteration_lines, plan_path, review_note_name
from splinter.tui_run_iteration import run_actions, run_meta
from splinter.tui_run_palette import AMBER, DIM, INK, LIME, MUTED, RED, VERDICT_COLOR, verdict_label
from splinter.tui_run_text import story_label

_STATE_TONE = {"pass": LIME, "fail": RED, "asked": RED, "paused": RED, "live": AMBER}
_CHAT_MARKUP = re.compile(r"\[(?:/|/?(?:bold|dim|cyan|green|yellow|red|magenta)[^\]]*)\]")
_SETUP_KEYS = ("strategy", "parallel", "max_concurrency", "tasks", "max_iterations", "budget")
_LOG_LINES = 20_000
_MIN_QUERY = 2
_MAX_HITS = 2_000


@dataclass
class TraceNode:
    """One row of the trace: its markup label, the text it opens, its children."""

    label: str
    detail: str = ""
    children: list[TraceNode] = field(default_factory=list)
    expanded: bool = False
    searchable: bool = True


@dataclass(frozen=True)
class TraceHit:
    """One match of a search: the node it is in and where in its text."""

    node: TraceNode
    line: int
    column: int


def search_trace(root: TraceNode, query: str) -> list[TraceHit]:
    """Every match in the tree, in reading order; repeated texts count once."""
    needle = query.strip().lower()
    if len(needle) < _MIN_QUERY:
        return []
    hits: list[TraceHit] = []
    seen: set[str] = set()
    stack = [root]
    while stack and len(hits) < _MAX_HITS:
        node = stack.pop()
        stack.extend(reversed(node.children))
        if node.searchable and node.detail and node.detail not in seen:
            seen.add(node.detail)
            hits.extend(_hits_in(node, needle))
    return hits[:_MAX_HITS]


def _hits_in(node: TraceNode, needle: str) -> list[TraceHit]:
    found: list[TraceHit] = []
    for number, line in enumerate(node.detail.split("\n")):
        lowered, column = line.lower(), 0
        while (column := lowered.find(needle, column)) >= 0:
            found.append(TraceHit(node, number, column))
            column += len(needle)
    return found


def _read(path: Path) -> str:
    try:
        return path.read_text()
    except OSError:
        return ""


def _events_node(session: Session) -> TraceNode:
    lines = [line for line in session.read("events.md").splitlines() if line.strip()]
    node = TraceNode(f"[b {MUTED}]pipeline events[/] [{DIM}]{len(lines)}[/]", "\n".join(lines))
    node.children = [
        TraceNode(f"[{MUTED}]{escape(line[:140])}[/]", line, searchable=False)
        for line in lines[-400:]
    ]
    return node


def _iteration_node(session: Session, story: int, row: IterationRow, loop: str) -> TraceNode:
    knowledge = session.dir / "knowledge"
    note = row.review_note or review_note_name(story, row.n)
    review = _read(knowledge / (note if note.endswith(".md") else f"{note}.md"))
    run_file = Path(row.run_file)
    run_path = run_file if run_file.is_absolute() else session.dir / row.run_file
    run_text = _read(run_path)
    verdict = verdict_label(row.verdict) or "live"
    tone = VERDICT_COLOR.get(row.verdict, AMBER)
    model = " ".join(part for part in (row.tier, row.model) if part)
    label = (
        f"[{INK}]iteration {row.n}[/] [{DIM}]·[/] [{MUTED}]{escape(model)}[/]"
        f" [b {tone}]{verdict}[/]"
    )
    meta = run_meta(run_text)
    if meta.get("cost"):
        label += f" [{DIM}]{escape(meta['cost'])}[/]"
    node = TraceNode(label, run_text or "no output yet")
    gate = gate_lines(loop, story, row.n) + iteration_lines(loop, story, row.n, "- verdict:")
    if review:
        node.children.append(TraceNode(f"[{tone}]review[/]", review))
    if gate:
        node.children.append(TraceNode(f"[{MUTED}]gate · verdict[/]", "\n".join(gate)))
    actions = run_actions(run_text)
    if actions:
        acts = TraceNode(f"[{MUTED}]actions[/] [{DIM}]({len(actions)})[/]", run_text)
        acts.children = [
            TraceNode(
                f"[{MUTED}]{kind:<6}[/] {escape(text[:120])}", f"{kind}  {text}", searchable=False
            )
            for kind, text in actions
        ]
        node.children.append(acts)
    if run_text:
        node.children.append(TraceNode(f"[{MUTED}]run output[/]", run_text))
    if row.escalated_to:
        node.children.append(
            TraceNode(f"[{AMBER}]↑ escalated {row.tier} → {escape(row.escalated_to)}[/]")
        )
    return node


def _story_node(session: Session, view: RunView, row: StoryRow, loop: str) -> TraceNode:
    ident, title = story_label(row)
    iterations = view.iterations_by_story.get(row.number, ())
    tone = _STATE_TONE.get(row.state, MUTED)
    label = (
        f"[b {INK}]{escape(ident)}[/] {escape(title)} [{DIM}]·[/] [b {tone}]{row.state.upper()}[/]"
        f" [{DIM}]· {len(iterations)} it · ${row.cost:.4f}[/]"
    )
    node = TraceNode(label, f"{ident} {title}\nstate: {row.state}\ntier: {row.tier}")
    node.expanded = row.state in ("fail", "asked", "paused", "live")
    plan = _read(plan_path(session, row.number))
    if plan.strip():
        node.children.append(TraceNode(f"[{LIME}]plan[/]", plan))
    files = _read(session.dir / "knowledge" / f"localization-{row.number}.md")
    if files.strip():
        node.children.append(TraceNode(f"[{MUTED}]files[/]", files))
    node.children.extend(_iteration_node(session, row.number, it, loop) for it in iterations)
    return node


def _group(label: str, tone: str, nodes: list[TraceNode], expanded: bool) -> TraceNode:
    head = f"[b {tone}]{label}[/] [{DIM}]{len(nodes)}[/]"
    summary = "\n".join(Text.from_markup(node.label).plain for node in nodes)
    return TraceNode(head, summary, nodes, expanded, searchable=False)


def _doc(name: str, text: str, tone: str = INK) -> TraceNode | None:
    return TraceNode(f"[{tone}]{escape(name)}[/]", text) if text.strip() else None


def _conversation(raw: str) -> str:
    entries = (_CHAT_MARKUP.sub("", entry).strip() for entry in raw.split("\0"))
    return "\n\n".join(entry for entry in entries if entry)


def _setup(session: Session) -> str:
    status = session.read_status()
    known = (key for key in _SETUP_KEYS if status.get(key) not in (None, ""))
    return "\n".join(f"{key}: {status[key]}" for key in known)


def _start_group(session: Session) -> TraceNode | None:
    versions = [
        _doc(f"PRD v{v.num:03d} · {v.label}", session.read(prd_version_file(v.num)))
        for v in list_prd_versions(session)
    ]
    found = [
        _doc("instructions", session.read("instructions.md")),
        _doc("conversation", _conversation(session.read("convo.md"))),
        *versions,
        _doc("final PRD", session.read("prd.md"), LIME),
        _doc("phases", session.read("prd_phases.md"), MUTED),
        _doc("run setup", _setup(session), MUTED),
    ]
    nodes = [node for node in found if node is not None]
    return _group("start", LIME, nodes, True) if nodes else None


def _filter_nodes(session: Session) -> list[TraceNode]:
    stories = prd_stories(session.read("prd.md"), False)
    nodes = []
    for number, story in enumerate(stories, 1):
        node = _doc(f"filter · {story.id}", session.read(f"knowledge/filter-{number}.md"), MUTED)
        if node is not None:
            nodes.append(node)
    return nodes


def _context_group(session: Session) -> TraceNode | None:
    found = [_doc("localize", session.read("knowledge/localization.md"), MUTED)]
    nodes = [node for node in [*found, *_filter_nodes(session)] if node is not None]
    return _group("context", MUTED, nodes, False) if nodes else None


def _log_line(raw: str) -> str:
    try:
        record = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    if not isinstance(record, dict) or record.get("type") != "log":
        return ""
    message = str((record.get("payload") or {}).get("message") or "")
    return f"{str(record.get('ts', ''))[11:19]}  {message}" if message else ""


def _log_node(session: Session) -> TraceNode | None:
    raw = _read(session.dir / "events.jsonl").splitlines()
    lines = [line for line in map(_log_line, raw) if line]
    if not lines:
        return None
    shown = lines[-_LOG_LINES:]
    return TraceNode(f"[b {MUTED}]full log[/] [{DIM}]{len(shown)} lines[/]", "\n".join(shown))


def build_trace(session: Session, view: RunView) -> TraceNode:
    """The run's tree: what needs you, what runs now, what is done, the queue, the events."""
    loop = session.read("loop.md")
    root = TraceNode(f"[b {LIME}]run[/] [{MUTED}]{escape(session.id)}[/]", expanded=True)
    root.children.extend(
        node for node in (_start_group(session), _context_group(session)) if node is not None
    )
    started = {row.number: row for row in view.stories if row.state != "queued"}
    needs = [started[n] for n in view.needs_you if n in started]
    now = [started[n] for n in view.now if n in started and started[n] not in needs]
    shown = {row.number for row in (*needs, *now)}
    done = [row for row in started.values() if row.number not in shown]
    queued = [row for row in view.stories if row.state == "queued"]
    groups = (
        ("needs you", RED, needs, True),
        ("now", AMBER, now, True),
        ("done", LIME, done, False),
    )
    for label, tone, rows, expanded in groups:
        if rows:
            nodes = [_story_node(session, view, row, loop) for row in rows]
            root.children.append(_group(label, tone, nodes, expanded))
    if queued:
        nodes = [_story_node(session, view, row, loop) for row in queued]
        root.children.append(_group("queued", MUTED, nodes, False))
    events = _events_node(session)
    if events.children:
        root.children.append(events)
    log_node = _log_node(session)
    if log_node is not None:
        root.children.append(log_node)
    return root
