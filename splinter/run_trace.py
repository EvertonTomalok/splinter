"""The whole run as a tree of what happened: events, stories, plans, iterations, actions.

Each node carries a short label and the full text behind it (a plan, a review, the
runner's output, one tool call), so the trace screen can show the outline on the left
and the selected text on the right. Built from the files the run already writes; no
model is called and nothing is parsed twice per refresh.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rich.markup import escape
from rich.text import Text

from splinter.memory.session import Session
from splinter.run_view import IterationRow, RunView, StoryRow
from splinter.run_view_files import gate_lines, iteration_lines, plan_path, review_note_name
from splinter.tui_run_iteration import run_actions, run_meta
from splinter.tui_run_palette import AMBER, DIM, INK, LIME, MUTED, RED, VERDICT_COLOR, verdict_label
from splinter.tui_run_text import story_label

_STATE_TONE = {"pass": LIME, "fail": RED, "asked": RED, "paused": RED, "live": AMBER}


@dataclass
class TraceNode:
    """One row of the trace: its markup label, the text it opens, its children."""

    label: str
    detail: str = ""
    children: list[TraceNode] = field(default_factory=list)
    expanded: bool = False


def _read(path: Path) -> str:
    try:
        return path.read_text()
    except OSError:
        return ""


def _events_node(session: Session) -> TraceNode:
    lines = [line for line in session.read("events.md").splitlines() if line.strip()]
    node = TraceNode(f"[b {MUTED}]pipeline events[/] [{DIM}]{len(lines)}[/]", "\n".join(lines))
    node.children = [TraceNode(f"[{MUTED}]{escape(line[:140])}[/]", line) for line in lines[-400:]]
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
            TraceNode(f"[{MUTED}]{kind:<6}[/] {escape(text[:120])}", f"{kind}  {text}")
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
    return TraceNode(head, summary, nodes, expanded)


def build_trace(session: Session, view: RunView) -> TraceNode:
    """The run's tree: what needs you, what runs now, what is done, the queue, the events."""
    loop = session.read("loop.md")
    root = TraceNode(f"[b {LIME}]run[/] [{MUTED}]{escape(session.id)}[/]", expanded=True)
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
        names = "\n".join(" ".join(story_label(row)) for row in queued)
        leaves = [
            TraceNode(escape(" · ".join(part for part in story_label(row) if part)), row.title)
            for row in queued
        ]
        root.children.append(
            TraceNode(f"[b {MUTED}]queued[/] [{DIM}]{len(queued)}[/]", names, leaves)
        )
    events = _events_node(session)
    if events.children:
        root.children.append(events)
    return root
