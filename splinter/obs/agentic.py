"""Metric API for token/cost tracking and verbatim LLM exchange capture."""

from __future__ import annotations

import json
import logging
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from typing import Any, TypeVar

from splinter.memory.session import Session, _append_lock

_T = TypeVar("_T")


def _decode(cls: type[_T], data: dict[str, Any]) -> _T:
    """Build a dataclass from JSON, ignoring unknown keys.

    Legacy records without a ``schema_version`` key decode as version 0.
    Missing required fields still raise ``TypeError`` — callers treat that
    as a genuinely malformed record, not a version mismatch.
    """
    field_names = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    kwargs = {k: v for k, v in data.items() if k in field_names}
    if "schema_version" in field_names and "schema_version" not in kwargs:
        kwargs["schema_version"] = 0
    return cls(**kwargs)


@dataclass(frozen=True)
class AgenticEvent:
    """Record of a single provider action within an agentic run."""

    task_index: int
    iteration: int
    provider: str
    model: str
    kind: str
    ts: str
    schema_version: int = 1
    extra: dict[str, Any] = field(default_factory=dict)


def append_jsonl(session: Session, event: AgenticEvent) -> None:
    """Persist event as one JSON line to task-specific JSONL file.

    File path: {session.dir}/trace/agentic-{task_index}.jsonl
    Failures are swallowed — never raises into caller.
    """
    try:
        with _append_lock(session.id):
            session._ensure_dir()
            trace_dir = session.dir / "trace"
            trace_dir.mkdir(parents=True, exist_ok=True)

            file_path = trace_dir / f"agentic-{event.task_index}.jsonl"
            with open(file_path, "a") as f:
                f.write(json.dumps(asdict(event)) + "\n")
    except Exception:
        pass


def load_agentic_events(session: Session) -> list[AgenticEvent]:
    """Load all agentic metric events from session trace directory.

    Globs trace/agentic-*.jsonl, parsing each line as JSON. Malformed or
    truncated lines are silently skipped; only valid AgenticEvent objects
    are returned. Events sorted by task_index, then by line order within
    file.
    """
    events: list[AgenticEvent] = []
    trace_dir = session.dir / "trace"

    if not trace_dir.exists():
        return events

    skipped = 0
    for file_path in sorted(trace_dir.glob("agentic-*.jsonl")):
        try:
            with open(file_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        event = _decode(AgenticEvent, data)
                        events.append(event)
                    except (json.JSONDecodeError, TypeError):
                        skipped += 1
                        continue
        except Exception:
            continue

    if skipped:
        logging.getLogger(__name__).warning("skipped %d malformed agentic events", skipped)

    return events


@dataclass(frozen=True)
class ExchangeEvent:
    """Full prompt/response record of one LLM exchange inside a stage."""

    stage: str
    task_index: int
    iteration: int
    prompt: str
    response: str
    model: str = ""
    variant: str = ""
    schema_version: int = 1


@dataclass(frozen=True)
class AgenticContext:
    """Scope carried by ContextVar for duration of one stage call."""

    session: Session
    stage: str
    task_index: int
    iteration: int


_ctx: ContextVar[AgenticContext | None] = ContextVar("agentic_ctx", default=None)


@contextmanager
def agentic_scope(
    session: Session, stage: str, task_index: int, iteration: int
) -> Generator[None, None, None]:
    """Set up context for exchange recording within a stage."""
    token = _ctx.set(
        AgenticContext(
            session=session,
            stage=stage,
            task_index=task_index,
            iteration=iteration,
        )
    )
    try:
        yield
    finally:
        _ctx.reset(token)


def record_exchange(prompt: str, response: str, *, model: str = "") -> None:
    """Record prompt+response pair within an agentic_scope."""
    ctx = _ctx.get()
    if ctx is None:
        return
    event = ExchangeEvent(
        stage=ctx.stage,
        task_index=ctx.task_index,
        iteration=ctx.iteration,
        prompt=prompt,
        response=response,
        model=model,
    )
    _persist_exchange(ctx.session, event)


def record_gate_marker() -> None:
    """Record a gate marker (empty prompt/response)."""
    record_exchange("", "")


def _now_iso() -> str:
    """Get current time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def record_action(kind: str, summary: str, *, provider: str = "claude") -> None:
    """Record a provider action (tool_use or text) within an agentic_scope.

    Args:
        kind: Action kind (e.g. "tool_use", "text")
        summary: One-line human-readable summary (e.g. "🔧 Edit /path/to/file")
        provider: Provider name (default "claude")

    If no agentic_scope is active, this is a no-op (never raises).
    """
    ctx = _ctx.get()
    if ctx is None:
        return
    event = AgenticEvent(
        task_index=ctx.task_index,
        iteration=ctx.iteration,
        provider=provider,
        model="",
        kind=kind,
        ts=_now_iso(),
        extra={"summary": summary},
    )
    append_jsonl(ctx.session, event)


def _persist_exchange(session: Session, event: ExchangeEvent) -> None:
    """Persist exchange event to agentic/task-{n}.jsonl."""
    try:
        session._ensure_dir()
        agentic_dir = session.dir / "agentic"
        agentic_dir.mkdir(exist_ok=True)
        file_path = agentic_dir / f"task-{event.task_index}.jsonl"
        with open(file_path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "stage": event.stage,
                        "task_index": event.task_index,
                        "iteration": event.iteration,
                        "prompt": event.prompt,
                        "response": event.response,
                        "model": event.model,
                        "variant": event.variant,
                        "schema_version": event.schema_version,
                    }
                )
                + "\n"
            )
    except Exception:
        pass


def read_events(session: Session, task_index: int) -> list[ExchangeEvent]:
    """Read verbatim exchange events for one task."""
    file_path = session.dir / "agentic" / f"task-{task_index}.jsonl"
    if not file_path.exists():
        return []
    events: list[ExchangeEvent] = []
    skipped = 0
    for line in file_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            events.append(_decode(ExchangeEvent, data))
        except (json.JSONDecodeError, TypeError):
            skipped += 1
            continue
    if skipped:
        logging.getLogger(__name__).warning("skipped %d malformed exchange events", skipped)
    return events


_STAGE_ORDER: tuple[str, ...] = ("localize", "filter", "plan", "run", "gate", "eval")


def _stage_index(stage: str) -> int:
    """Get sort order index for a stage name."""
    try:
        return _STAGE_ORDER.index(stage)
    except ValueError:
        return len(_STAGE_ORDER)


def _read_all_exchanges(session: Session) -> list[ExchangeEvent]:
    """Read all exchange events from agentic/ directory."""
    agentic_dir = session.dir / "agentic"
    if not agentic_dir.exists():
        return []

    exchanges: list[ExchangeEvent] = []
    skipped = 0
    for file_path in sorted(agentic_dir.glob("task-*.jsonl")):
        try:
            for line in file_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    exchange = _decode(ExchangeEvent, data)
                    exchanges.append(exchange)
                except (json.JSONDecodeError, TypeError):
                    skipped += 1
                    continue
        except Exception:
            continue

    if skipped:
        logging.getLogger(__name__).warning("skipped %d malformed exchange events", skipped)

    return exchanges


def render_agentic(session: Session) -> str:
    """Render all exchange events as readable text."""
    exchanges = _read_all_exchanges(session)
    if not exchanges:
        return "(no agentic trace)"

    by_task: dict[int, list[ExchangeEvent]] = {}
    for ex in exchanges:
        if ex.task_index not in by_task:
            by_task[ex.task_index] = []
        by_task[ex.task_index].append(ex)

    out: list[str] = []
    for task_id in sorted(by_task.keys()):
        out.append(f"===== Task {task_id} =====")
        task_exchanges = by_task[task_id]
        task_exchanges.sort(key=lambda e: (_stage_index(e.stage), e.iteration))

        for ex in task_exchanges:
            header = (
                f"{ex.stage} · task {ex.task_index} · "
                f"iter {ex.iteration} · {ex.model} · {ex.variant}"
            )
            out.append(header)
            if ex.prompt:
                out.append(f"prompt:\n{ex.prompt}")
            if ex.response:
                out.append(f"response:\n{ex.response}")

    return "\n\n".join(out)
