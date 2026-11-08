"""Story, iteration, stage and ladder projections for the run view."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Sequence

from splinter.models.roster import Ladder
from splinter.obs.trace import RunEntry
from splinter.run_snapshot import Snapshot, tier_effort, trace_span
from splinter.run_view_files import review_note_name, run_file_path

if TYPE_CHECKING:
    from splinter.run_view import IterationRow, LadderRung, StoryRow, StoryState


def entries_by_task(entries: Sequence[RunEntry]) -> dict[int, list[RunEntry]]:
    """Group the trace entries by story number (1-based)."""
    grouped: dict[int, list[RunEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.task + 1, []).append(entry)
    return grouped


def groups(
    snap: Snapshot, read_file: Callable[[Path], str] | None = None
) -> list[tuple[int, str, list[tuple[int, str, str]]]]:
    from splinter.analyze import _task_iters

    try:
        return _task_iters(snap.loop_md, session=snap.session, read_file=read_file)
    except Exception:
        return []


def story_state(
    status: str, verdict: str, number: int, current: int | None, snap: Snapshot
) -> StoryState:
    mapped: dict[str, StoryState] = {
        "pass": "pass",
        "blocked": "fail",
        "failed": "asked" if verdict == "ASK_USER" else "fail",
        "running": "live",
        "pending": "queued",
    }
    state = mapped.get(status, "queued")
    raw = str(snap.status.get("state", "")).lower()
    if number == current:
        if raw == "awaiting_user":
            return "asked"
        if raw in ("awaiting_validation", "paused"):
            return "paused"
    return state


def iteration_rows(
    story: int,
    all_groups: list[tuple[int, str, list[tuple[int, str, str]]]],
    snap: Snapshot,
    ladder: Ladder | None,
    multi: bool,
    grouped: dict[int, list[RunEntry]] | None = None,
    review_notes: frozenset[str] | None = None,
) -> tuple[IterationRow, ...]:
    from splinter.run_view import IterationRow

    raw = next((items for number, _title, items in all_groups if number == story), [])
    story_entries = (grouped if grouped is not None else entries_by_task(snap.entries)).get(
        story, []
    )
    built: list[IterationRow] = []
    for index, (n, tier, verdict) in enumerate(raw):
        level = tier_level(tier)
        entries = [entry for entry in story_entries if entry.iteration == n]
        runs = [entry for entry in entries if entry.role == "run"]
        next_tier = raw[index + 1][1] if index + 1 < len(raw) else ""
        note = review_note_name(story, n)
        note_path = snap.session.dir / "knowledge" / f"{note}.md"
        built.append(
            IterationRow(
                n=n,
                tier=f"T{level}" if level is not None else "",
                model=runs[-1].model if runs else "",
                effort=effort(level, runs, ladder),
                verdict=verdict_name(verdict),
                cost=sum(entry.cost for entry in entries),
                duration_s=sum(entry.latency_s for entry in entries),
                escalated_to=(
                    next_tier if next_tier != tier and not next_tier.endswith("?") else None
                ),
                run_file=str(run_file_path(snap.session, story, n, multi)),
                review_note=note if (
                    note in review_notes if review_notes is not None else exists(note_path)
                ) else "",
            )
        )
    return tuple(built)


def effort(level: int | None, runs: list[RunEntry], ladder: Ladder | None) -> str:
    if level is None or not runs:
        return ""
    return tier_effort(level, runs[-1].model, ladder)


def stage(
    status: dict[str, object], state: str
) -> Literal["plan", "run", "gate", "eval", "done", "wait"]:
    raw_state = state.upper()
    if raw_state in ("COMPLETED", "DONE"):
        return "done"
    if raw_state.startswith("AWAITING_") or raw_state == "PAUSED":
        return "wait"
    raw = str(status.get("stage", "")).lower()
    if raw in ("localize", "filter", "plan"):
        return "plan"
    if raw == "gate":
        return "gate"
    if raw in ("eval", "final_eval"):
        return "eval"
    return "run"


def elapsed(
    status: dict[str, object], state: str, now: datetime, entries: list[RunEntry]
) -> float:
    if state.upper() == "RUNNING":
        started = parse_time(str(status.get("started_at", "")))
        if started is not None:
            return max(0.0, (as_utc(now) - started).total_seconds())
    return trace_span(entries)


def ladder_rungs(
    ladder: Ladder | None,
    iterations: dict[int, tuple[IterationRow, ...]],
    live_story: int | None,
) -> tuple[LadderRung, ...]:
    from splinter.run_view import LadderRung

    if ladder is None:
        return ()
    current = iterations.get(live_story or 0, ())
    current_tier = current[-1].tier if current else ""
    used = {row.tier for rows in iterations.values() for row in rows if row.tier}
    return tuple(
        LadderRung(
            tier=f"T{tier.level}",
            models=tuple(tier.models),
            effort=ladder.tier_variant(tier.level) or "",
            current=f"T{tier.level}" == current_tier,
            used=f"T{tier.level}" in used,
        )
        for tier in ladder.tiers
    )


def counts(stories: tuple[StoryRow, ...]) -> dict[StoryState, int]:
    values = zero_counts()
    for story in stories:
        values[story.state] += 1
    return values


def zero_counts() -> dict[StoryState, int]:
    states: tuple[StoryState, ...] = ("pass", "fail", "paused", "live", "queued", "asked")
    return dict.fromkeys(states, 0)


def as_int(value: object) -> int:
    try:
        return int(value) if isinstance(value, (int, float, str)) else 0
    except (TypeError, ValueError):
        return 0


def tier_level(tier: str) -> int | None:
    try:
        return int(tier.removeprefix("T")) if not tier.endswith("?") else None
    except ValueError:
        return None


def verdict_name(raw: str) -> str:
    if raw == "JUMP_PREMIUM":
        return "ESCALATE"
    if raw == "ASK_USER":
        return "ASKED"
    return raw if raw in ("PASS", "RETRY", "ESCALATE", "FAIL") else ""


def exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def parse_time(value: str) -> datetime | None:
    try:
        return as_utc(datetime.fromisoformat(value))
    except ValueError:
        return None


def as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
