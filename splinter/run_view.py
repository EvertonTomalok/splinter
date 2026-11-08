"""Pure, file-backed view model for the C2 run screen."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Literal

from splinter.agents.progress import ATTEMPTS, CHECKPOINT, STORIES, Attempt, AttemptsTail
from splinter.memory.session import Session
from splinter.models.roster import Ladder, load_ladder
from splinter.obs.event_reader import RunEntryReader
from splinter.obs.trace import RunEntry
from splinter.run_snapshot import (
    Snapshot,
    TaskRow,
    current_task_no,
    task_rows,
)
from splinter.run_view_agents import agents_groups, agents_rows
from splinter.run_view_files import FileCache, iteration_files, review_note_name
from splinter.run_view_rows import (
    as_int,
    counts,
    elapsed,
    entries_by_task,
    groups,
    iteration_rows,
    ladder_rungs,
    parse_time,
    stage,
    story_state,
    zero_counts,
)

StoryState = Literal["pass", "fail", "paused", "live", "queued", "asked"]
Stage = Literal["plan", "run", "gate", "eval", "done", "wait"]


@dataclass(frozen=True)
class StoryRow:
    number: int
    id: str
    title: str
    state: StoryState
    tier: str
    model: str
    iterations: int
    cost: float
    elapsed_s: float
    wave: int


@dataclass(frozen=True)
class IterationRow:
    n: int
    tier: str
    model: str
    effort: str
    verdict: str
    cost: float
    duration_s: float
    escalated_to: str | None
    run_file: str
    review_note: str


@dataclass(frozen=True)
class SpendRow:
    label: str
    cost: float
    seconds: float
    share: float


@dataclass(frozen=True)
class LadderRung:
    tier: str
    models: tuple[str, ...]
    effort: str
    current: bool
    used: bool


@dataclass(frozen=True)
class RunView:
    stories: tuple[StoryRow, ...]
    iterations_by_story: dict[int, tuple[IterationRow, ...]]
    live_story: int | None
    stage: Stage
    spend: tuple[SpendRow, ...]
    total_cost: float
    elapsed_s: float
    ladder: tuple[LadderRung, ...]
    needs_you: tuple[int, ...]
    now: tuple[int, ...]
    next_up: tuple[int, ...]
    counts: dict[StoryState, int]


class RunViewBuilder:
    """Build views without reparsing unchanged session data."""

    def __init__(self) -> None:
        self._cache = FileCache()
        self._attempts_tail: AttemptsTail | None = None
        self._reader = RunEntryReader()
        self._entries: list[RunEntry] = []
        self._json_keys: dict[Path, tuple[int, ...]] = {}
        self._json_values: dict[Path, dict[str, object]] = {}
        self._text_keys: dict[Path, tuple[int, ...]] = {}
        self._text_values: dict[Path, str] = {}
        self._groups_key: tuple[object, ...] = ()
        self._groups: list[tuple[int, str, list[tuple[int, str, str]]]] = []
        self._rows_key: tuple[object, ...] = ()
        self._rows: list[TaskRow] = []
        self._files: dict[str, tuple[int, ...]] = {}
        self._ladder_cache_key: tuple[object, ...] = ()
        self._ladder: Ladder | None = None
        self._view_key: tuple[object, ...] = ()
        self._view: RunView | None = None
        self._story_keys: dict[int, tuple[object, ...]] = {}
        self._stories: dict[int, StoryRow] = {}
        self._iteration_keys: dict[int, tuple[object, ...]] = {}
        self._iteration_values: dict[int, tuple[IterationRow, ...]] = {}

    def build(
        self, session: Session, state: str, now: datetime, ladder: Ladder | None = None
    ) -> RunView:
        snap = self._snapshot(session, state)
        active_ladder = ladder if ladder is not None else self._load_ladder()
        try:
            self._files = iteration_files(session)
            key = (
                session.dir, state, snap.status, snap.loop_md, snap.pre_run_models,
                self._files, self._reader.revision, _ladder_key(active_ladder),
                self._agents_key(session),
            )
            if self._view is not None and key == self._view_key:
                if state.upper() != "RUNNING" or parse_time(
                    str(snap.status.get("started_at", ""))
                ) is None:
                    return self._view
                seconds = elapsed(snap.status, state, now, snap.entries)
                if seconds != self._view.elapsed_s:
                    self._view = replace(self._view, elapsed_s=seconds)
                return self._view
            self._view = self._build_view(snap, state, now, active_ladder)
            self._view_key = key
            return self._view
        except Exception:
            return _empty_view()

    def _snapshot(self, session: Session, state: str) -> Snapshot:
        try:
            entries = self._entries_for(session)
            status = self._json(session.dir / "status.json")
            pre_run = self._json(session.dir / "pre_run_usage.json").get("models", {})
            return Snapshot(
                session=session,
                state=state,
                status=status,
                loop_md=self._text(session.dir / "loop.md"),
                localization=self._text(session.dir / "knowledge/localization.md"),
                plan=self._text(session.dir / "knowledge/plan.md"),
                phases_md=self._text(session.dir / "prd_phases.md"),
                final_eval_md=self._text(session.dir / "final_eval.md"),
                entries=entries,
                pre_run_models=pre_run if isinstance(pre_run, dict) else {},
            )
        except Exception:
            return Snapshot(session, state, {}, "", "", "", "", "", [], {})

    def _text(self, path: Path) -> str:
        key = self._cache.mtime_key((path,))
        if self._text_keys.get(path) != key:
            self._text_values[path] = self._cache.read(path)
            self._text_keys[path] = key
        return self._text_values.get(path, "")

    def _entries_for(self, session: Session) -> list[RunEntry]:
        self._entries = self._reader.read(session)
        return self._entries

    def _json(self, path: Path) -> dict[str, object]:
        key = self._cache.mtime_key((path,))
        if self._json_keys.get(path) != key:
            try:
                loaded = json.loads(self._cache.read(path))
            except (TypeError, ValueError):
                loaded = {}
            self._json_values[path] = loaded if isinstance(loaded, dict) else {}
            self._json_keys[path] = key
        return self._json_values.get(path, {})

    def _load_ladder(self) -> Ladder | None:
        from splinter.models.pricing_store import STORE_PATH

        paths = (
            Path(__file__).parent / "models" / "ladder.yaml",
            Path(".splinter/config.yaml"), STORE_PATH,
        )
        key = (Path.cwd(), *self._cache.mtime_key(paths))
        if self._ladder_cache_key == key:
            return self._ladder
        try:
            self._ladder = load_ladder()
            self._ladder_cache_key = key
            return self._ladder
        except Exception:
            return None

    def _build_view(
        self, snap: Snapshot, state: str, now: datetime, ladder: Ladder | None
    ) -> RunView:
        groups = self._groups_for(snap)
        rows = self._rows_for(snap)
        total = max(
            len(rows), as_int(snap.status.get("tasks")), as_int(snap.status.get("task_total"))
        )
        indexed = {row.number: row for row in rows}
        current = current_task_no(snap, groups)
        grouped = entries_by_task(snap.entries)
        stories = tuple(
            self._story(indexed.get(number), number, current, snap, grouped)
            for number in range(1, total + 1)
        )
        iterations = self._iterations(snap, ladder, total, groups, grouped)
        live_story = next((story.number for story in stories if story.state == "live"), None)
        spend = self._spend(snap, ladder)
        return RunView(
            stories=stories,
            iterations_by_story=iterations,
            live_story=live_story,
            stage=stage(snap.status, state),
            spend=spend,
            total_cost=sum(row.cost for row in spend),
            elapsed_s=elapsed(snap.status, state, now, snap.entries),
            ladder=ladder_rungs(ladder, iterations, live_story),
            needs_you=tuple(
                story.number for story in stories if story.state in ("asked", "paused", "fail")
            ),
            now=tuple(story.number for story in stories if story.state == "live"),
            next_up=tuple(story.number for story in stories if story.state == "queued")[:5],
            counts=counts(stories),
        )

    def _story(
        self,
        task: TaskRow | None,
        number: int,
        current: int | None,
        snap: Snapshot,
        grouped: dict[int, list[RunEntry]],
    ) -> StoryRow:
        entries = grouped.get(number, [])
        key = (snap.session.dir, task, number == current, snap.status.get("state"), tuple(entries))
        if self._story_keys.get(number) == key:
            return self._stories[number]
        runs = [entry for entry in entries if entry.role == "run"]
        state = story_state(
            task.status if task else "pending",
            task.verdict if task else "",
            number,
            current,
            snap,
        )
        row = StoryRow(
            number=number,
            id=f"T{number}",
            title=task.title if task else "",
            state=state,
            tier=f"T{runs[-1].tier}" if runs else (task.tier if task else ""),
            model=runs[-1].model if runs else "",
            iterations=task.iters if task else 0,
            cost=sum(entry.cost for entry in entries),
            elapsed_s=sum(entry.latency_s for entry in entries),
            wave=0,
        )
        self._story_keys[number] = key
        self._stories[number] = row
        return row

    def _groups_for(self, snap: Snapshot) -> list[tuple[int, str, list[tuple[int, str, str]]]]:
        key = self._groups_key_for(snap.session)
        if key != self._groups_key:
            stories = self._text(snap.session.dir / STORIES)
            self._groups = (
                agents_groups(stories, self._attempts_at(snap.session.dir / ATTEMPTS))
                if stories
                else groups(snap, self._cache.read)
            )
            self._groups_key = key
        return self._groups

    def _rows_for(self, snap: Snapshot) -> list[TaskRow]:
        key = (self._groups_key, snap.status, snap.state)
        if key != self._rows_key:
            stories = self._text(snap.session.dir / STORIES)
            checkpoint = self._text(snap.session.dir / CHECKPOINT)
            try:
                self._rows = (
                    agents_rows(
                        stories, self._attempts_at(snap.session.dir / ATTEMPTS), checkpoint
                    )
                    if stories
                    else task_rows(snap, self._groups)
                )
            except Exception:
                self._rows = []
            self._rows_key = key
        return self._rows

    def _groups_key_for(self, session: Session) -> tuple[object, ...]:
        return (
            session.dir, self._text(session.dir / "loop.md"), self._files,
            self._agents_key(session),
        )

    def _agents_key(self, session: Session) -> tuple[int, ...]:
        """Changes when an agents run's progress files change, without reading them."""
        return self._cache.mtime_key(
            (session.dir / STORIES, session.dir / ATTEMPTS, session.dir / CHECKPOINT)
        )

    def _attempts_at(self, path: Path) -> list[Attempt]:
        if self._attempts_tail is None or self._attempts_tail.path != path:
            self._attempts_tail = AttemptsTail(path)
        return self._attempts_tail.read()

    def _iterations(
        self,
        snap: Snapshot,
        ladder: Ladder | None,
        total: int,
        groups: list[tuple[int, str, list[tuple[int, str, str]]]],
        grouped: dict[int, list[RunEntry]],
    ) -> dict[int, tuple[IterationRow, ...]]:
        multi = total > 1
        indexed = {number: (number, title, rows) for number, title, rows in groups}
        notes = frozenset(
            Path(name).stem for name in self._files if name.startswith("knowledge/")
        )
        ladder_key = _ladder_key(ladder)
        result: dict[int, tuple[IterationRow, ...]] = {}
        for number in range(1, total + 1):
            group = indexed.get(number, (number, "", []))
            present = tuple(
                review_note_name(number, n) in notes for n, _, _ in group[2]
            )
            key = (
                snap.session.dir, multi, tuple(group[2]), tuple(grouped.get(number, ())),
                present, ladder_key,
            )
            if self._iteration_keys.get(number) != key:
                self._iteration_values[number] = iteration_rows(
                    number, [group], snap, ladder, multi, grouped, notes
                )
                self._iteration_keys[number] = key
            result[number] = self._iteration_values[number]
        return result

    def _spend(self, snap: Snapshot, ladder: Ladder | None) -> tuple[SpendRow, ...]:
        from splinter.run_view_spend import spend_rows

        return spend_rows(snap, ladder)


def _ladder_key(ladder: Ladder | None) -> tuple[object, ...]:
    if ladder is None:
        return ()
    return (
        ladder.planner_model, ladder.planner_effort, ladder.eval_model, ladder.eval_effort,
        tuple(ladder.tier_variants.items()),
        tuple(
            (tier.level, tuple(tier.models), tuple(tier.model_variants.items()))
            for tier in ladder.tiers
        ),
    )


def build_run_view(
    session: Session, state: str, now: datetime, ladder: Ladder | None = None
) -> RunView:
    """Build one immutable screen model from the current session files."""
    return RunViewBuilder().build(session, state, now, ladder)


def _empty_view() -> RunView:
    return RunView(
        stories=(),
        iterations_by_story={},
        live_story=None,
        stage="run",
        spend=(),
        total_cost=0.0,
        elapsed_s=0.0,
        ladder=(),
        needs_you=(),
        now=(),
        next_up=(),
        counts=zero_counts(),
    )


