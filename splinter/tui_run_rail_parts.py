"""Pure markup builders for the right rail of the run screen."""

from __future__ import annotations

from rich.markup import escape

from splinter.run_view import IterationRow, LadderRung, RunView, SpendRow, StoryRow
from splinter.tui_run_palette import (
    AMBER,
    DIM,
    INK,
    LIME,
    LINE,
    LINE2,
    MUTED,
    RED,
    SELECTED,
    VERDICT_COLOR,
    verdict_label,
)
from splinter.tui_run_text import (
    big_rows,
    big_width,
    cell_len,
    clip,
    hms,
    money,
    pad,
    story_label,
)

CELL_COLOR = {
    "pass": LIME,
    "live": AMBER,
    "fail": RED,
    "paused": RED,
    "asked": RED,
    "queued": LINE2,
}
_NEEDS = ("fail", "paused", "asked")
#: A story cell: a block element, square in a 1:2 cell, drawn as-is by terminals
#: that lift low-contrast text (Ghostty ``minimum-contrast``).
CELL = "▄"
#: The live bar between pulses: lime one step down.
LIVE_DIM = "#5f7320"
#: The live bar is shorter than a finished one; it grows with the stage.
_LIVE_BAR = {"plan": "▂", "run": "▄", "gate": "▆", "eval": "█"}
#: Outline of a chip that is not selected.
EDGE = "#3a3a35"
CHIP_BG = "#141a08"


def clock_markup(elapsed_s: float, cost: float, width: int) -> str:
    """Elapsed and spent, side by side, in thin numerals when they fit."""
    elapsed, spent = hms(elapsed_s), money(cost)
    left_w = max(big_width(elapsed), 9)
    time_go, cost_go = "[@click=app.open_spend('time')]", "[@click=app.open_spend('cost')]"
    caps = pad(f"{time_go}[{MUTED}]elapsed ▸[/][/]", left_w + 2)
    caps += f"{cost_go}[{MUTED}]spent ▸[/][/]"
    if left_w + 2 + big_width(spent) > width:
        values = pad(f"{time_go}[b {INK}]{elapsed}[/][/]", left_w + 2)
        return f"{caps}\n{values}{cost_go}[b {LIME}]{spent}[/][/]"
    rows = [caps]
    for left, right in zip(big_rows(elapsed), big_rows(spent), strict=True):
        line = pad(f"{time_go}[{INK}]{left}[/][/]", left_w + 2)
        rows.append(f"{line}{cost_go}[{LIME}]{right}[/][/]")
    return "\n".join(rows)


def split_markup(spend: tuple[SpendRow, ...], width: int = 80) -> str:
    """``run $4.6186 · PRD+planner $0.2041 · jev $0.0052 ×6``, on one line."""
    pre = sum(row.cost for row in spend if row.label == "PRD + planner")
    jev_rows = [row for row in spend if row.label.startswith("jev")]
    jev = sum(row.cost for row in jev_rows)
    run = sum(row.cost for row in spend) - pre - jev
    calls = "".join(ch for ch in jev_rows[0].label if ch.isdigit()) if jev_rows else ""
    for pre_word, sep, times in (("PRD+planner", " · ", True), ("prd", " · ", True),
                                 ("prd", "  ", False)):
        parts = [f"run {money(run)}"]
        if pre:
            parts.append(f"{pre_word} {money(pre)}")
        if jev_rows:
            parts.append(f"jev {money(jev)}" + (f" ×{calls}" if calls and times else ""))
        line = sep.join(parts)
        if len(line) <= width:
            break
    return f"[{MUTED}]{escape(line)}[/]"


def chip(label: str, active: bool, go: str = "") -> str:
    """A one-row chip outlined by thin side bars, lime when it is the active one."""
    edge = LIME if active else EDGE
    inner = f"[b {LIME} on {CHIP_BG}] {label} [/]" if active else f"[{MUTED}] {label} [/]"
    body = f"[{edge}]▕[/]{inner}[{edge}]▏[/]"
    return f"{go}{body}[/]" if go else body


def rung_markup(rung: LadderRung) -> str:
    """Two lines of a ladder card: the tier, then its primary model."""
    model = rung.models[0].split("/")[-1] if rung.models else "—"
    return f"[b]{escape(rung.tier)}[/]\n{escape(model)}"


def rung_classes(rung: LadderRung, picked: bool) -> list[str]:
    classes = ["-current" if rung.current else "-used" if rung.used else "-ahead"]
    if picked:
        classes.append("-picked")
    return classes


def iteration_slots(iterations: tuple[IterationRow, ...], limit: int) -> int:
    return max(limit, len(iterations), 1)


def bars_markup(
    story: int,
    iterations: tuple[IterationRow, ...],
    live: bool,
    stage: str,
    width: int,
    limit: int,
    phase: bool = False,
) -> str:
    """Iteration bars (a finished one is a row and a half, the live one grows) and labels."""
    slots = iteration_slots(iterations, limit)
    size = max(3, (width - (slots - 1)) // slots)
    top: list[str] = []
    bottom: list[str] = []
    labels: list[str] = []
    for index in range(slots):
        row = iterations[index] if index < len(iterations) else None
        is_live = live and row is not None and index == len(iterations) - 1 and not row.verdict
        go = (
            f"[@click=app.open_iteration({story},{row.n})]"
            if row
            else "[@click=app.set_view('act')]"
        )
        if row is None:
            top.append(" " * size)
            bottom.append(f"[{LINE2}]{'▁' * size}[/]")
            labels.append(f"[{DIM}]{str(index + 1).center(size)}[/]")
            continue
        color = (LIVE_DIM if phase else LIME) if is_live else VERDICT_COLOR.get(row.verdict, MUTED)
        if is_live:
            top.append(" " * size)
            bottom.append(f"{go}[{color}]{_LIVE_BAR.get(stage, '▄') * size}[/][/]")
        else:
            top.append(f"{go}[{color}]{'▄' * size}[/][/]")
            bottom.append(f"{go}[{color}]{'█' * size}[/][/]")
        word = "now" if is_live else verdict_label(row.verdict) or "live"
        text = f"{row.n} {word}" if len(f"{row.n} {word}") <= size else word[:size]
        tone = LIME if is_live else VERDICT_COLOR.get(row.verdict, MUTED)
        labels.append(f"{go}[{tone}]{text.center(size)}[/][/]")
    return "\n".join(" ".join(part) for part in (top, bottom, labels))


def stories_head(view: RunView, width: int, needs_only: bool) -> str:
    done = int(view.counts.get("pass", 0))
    head = f"[{MUTED}]stories[/] [b]{done}/{len(view.stories)}[/] [{MUTED}]done[/]"
    needs = len(view.needs_you)
    if not needs:
        return head
    label = f"[{LIME if needs_only else MUTED}]f[/]  {needs} need you"
    button = chip(label, needs_only, "[@click=app.toggle_needs_filter]")
    return pad(head, width - cell_len(button)) + button


def heatmap_markup(
    view: RunView, selected: int | None, width: int, needs_only: bool, phase: bool
) -> str:
    """One cell per story; the selected cell sits on a light square."""
    spaced = -(-len(view.stories) // max(1, (width + 1) // 2)) <= 3
    per_row = max(1, (width + 1) // 2 if spaced else width)
    cells: list[str] = []
    for row in view.stories:
        color = CELL_COLOR.get(row.state, LINE2)
        if row.state == "live" and phase:
            color = "#7a5a14"
        if needs_only and row.state not in _NEEDS:
            color = LINE
        style = f"{color} on {INK}" if row.number == selected else color
        cells.append(f"[@click=app.select_story({row.number})][{style}]{CELL}[/][/]")
    joint = " " if spaced else ""
    lines = (joint.join(cells[n : n + per_row]) for n in range(0, len(cells), per_row))
    return "\n".join(lines)


def legend_markup(width: int = 80) -> str:
    """``▄ pass  ▄ running  ▄ needs you  ▄ queued``, tighter when the rail is narrow."""
    gap = "  " if width >= 40 else " "
    items = ((LIME, "pass"), (AMBER, "running"), (RED, "needs you"), (LINE2, "queued"))
    return gap.join(f"[{tone}]{CELL}[/][{DIM}] {word}[/]" for tone, word in items)


def matches(view: RunView, query: str, limit: int = 8) -> tuple[int, ...]:
    """Stories whose id, number or title holds ``query``."""
    needle = query.strip().lower()
    if not needle:
        return ()
    found = [
        row.number
        for row in view.stories
        if needle in " ".join((*story_label(row), str(row.number), row.id)).lower()
    ]
    return tuple(found[:limit])


def groups_markup(
    view: RunView,
    selected: int | None,
    width: int,
    needs_only: bool,
    phase: bool,
    query: str = "",
) -> str:
    """The needs you / now / next groups and a count for the rest."""
    lookup = {row.number: row for row in view.stories}
    now = view.now or ((view.stories[0].number,) if len(view.stories) == 1 else ())
    if query.strip():
        found = matches(view, query)
        if not found:
            return f"[{DIM}]no story matches “{escape(query.strip())}”[/]"
        rows = [lookup[number] for number in found]
        head = f"[{MUTED}]matches[/] [{DIM}]· ↵ opens the first · esc clears[/]"
        return "\n".join([head, *(_group_row(r, r.number == selected, width, phase) for r in rows)])
    sections = [("needs you", view.needs_you)]
    if not needs_only:
        sections += [("now", now), ("next", view.next_up)]
    lines: list[str] = []
    for label, numbers in sections:
        rows = [lookup[number] for number in numbers if number in lookup]
        if not rows:
            continue
        lines.append(f"[{MUTED}]{label}[/]")
        lines.extend(_group_row(row, row.number == selected, width, phase) for row in rows)
    queued = max(0, int(view.counts.get("queued", 0)) - len(view.next_up))
    passed = int(view.counts.get("pass", 0))
    rest = [f"+{queued} queued" if queued else "", f"{passed} done" if passed else ""]
    shown = [part for part in rest if part]
    if len(view.stories) > 1:
        find = f"[@click=app.find][b {LIME}]/[/] [{DIM}]find[/][/]"
        counts = f"[{DIM}]{' · '.join(shown)} · [/]" if shown else ""
        lines.append(f"{counts}{find}")
    return "\n".join(lines)


def _group_row(row: StoryRow, selected: bool, width: int, phase: bool) -> str:
    mark, tone, note = _row_mark(row, phase)
    ident, title = story_label(row)
    room = width - len(ident) - len(note) - 5
    body = f"[{tone}]{mark}[/] [b]{escape(ident)}[/] {clip(title, room)}"
    if note:
        body += f" [{DIM}]{note}[/]"
    line = pad(body, width - 2)
    if selected:
        line = f"[on {SELECTED}]{line}[/]"
    return f"[@click=app.story_menu({row.number})]{line}[/]"


def _row_mark(row: StoryRow, phase: bool) -> tuple[str, str, str]:
    if row.state == "fail":
        return "!", RED, "FAIL · review"
    if row.state == "asked":
        return "‖", RED, "paused · asks you"
    if row.state == "paused":
        return "‖", RED, "paused · validate"
    if row.state == "live":
        return "▶", DIM if phase else AMBER, ""
    if row.state == "pass":
        return "■", LIME, ""
    return "□", DIM, ""
