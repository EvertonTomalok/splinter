"""Wires PRD/task input through localize -> plan -> run -> gate -> eval -> loop."""

from __future__ import annotations

import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

from splinter.agents import planner
from splinter.agents.localizer import CodeAnchor, filter_task_context, localize
from splinter.agents.runner import Task
from splinter.memory.session import Session, new_session_id
from splinter.models.roster import load_ladder
from splinter.strategies.registry import available_strategies, get_strategy

DEFAULT_STRATEGY = "direct"

log = logging.getLogger("splinter.pipeline")

#: Substrings in an error that mark it transient (retry/continue, don't roll back).
_TRANSIENT_MARKERS = (
    "429", "500", "502", "503", "504", "529", "overloaded", "rate limit", "ratelimit",
    "timeout", "timed out", "temporarily", "try again", "connection", "econnreset",
    "unavailable", "reset by peer", "network", "socket",
)


class _SessionTraceHandler(logging.Handler):
    """Persist every ``splinter`` log record to the session's ``events.md`` so the
    Trace view is a full chronological log — each model push, tool call, gate/eval
    result, and escalate/jump/ask decision, in order."""

    def __init__(self, session: Session) -> None:
        super().__init__()
        self.session = session

    def emit(self, record: logging.LogRecord) -> None:
        try:
            ts = datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S")
            self.session.append("events.md", f"[{ts}] {record.getMessage()}")
        except Exception:  # noqa: BLE001 — tracing must never break the run
            pass


def _resolve_gate(session: Session, ladder: object) -> None:
    """Ensure the run has a gate. Precedence: already-configured (session gate.json
    or .splinter/config.yaml) → model-detected from the repo → Python defaults.

    Users can override per project via ``gate_checks`` in config.yaml, or per run
    in the PRD review phase; this just makes the planner bring one when none is set.
    """
    from splinter.agents import gate

    existing = gate.configured_gate_checks(session_dir=session.dir)
    if existing is not None:
        names = ", ".join(c.get("name", c.get("cmd", "?")) for c in existing) or "none"
        log.info("gate: using configured checks (%s)", names)
        return

    log.info("gate: detecting project checks…")
    detected = gate.detect_gate_checks(ladder)
    if detected:
        gate.save_gate_checks(session.dir, detected)
        log.info("gate: detected — %s", ", ".join(c["name"] for c in detected))
    else:
        log.warning("gate: could not detect checks — using defaults. Set `gate_checks` "
                    "in .splinter/config.yaml or specify them in the PRD review.")


def _classify_failure(exc: BaseException) -> str:
    """Transient (provider/network blip → resume continues) vs critical (bad command,
    bug → resume rolls the failing stage back and redoes it)."""
    from splinter.providers.base import ProviderGapError
    if isinstance(exc, ProviderGapError):
        return "gap"
    if isinstance(exc, (TimeoutError, subprocess.TimeoutExpired)):
        return "transient"
    msg = str(exc).lower()
    if any(m in msg for m in _TRANSIENT_MARKERS):
        return "transient"
    return "critical"


def _load_task_from_yaml(path: str) -> Task:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return Task(
        description=data.get("description", ""),
        acceptance=data.get("acceptance", ""),
        effort=data.get("effort", "normal"),
        reasoning_effort=data.get("reasoning_effort", "auto"),
        eval_skill=data.get("eval_skill"),
        suggested_tier=data.get("suggested_tier", 0),
        target_files=data.get("target_files"),
    )


def _load_tasks_from_prd(prd_path: str) -> tuple[list[Task], str | None]:
    text = Path(prd_path).read_text()
    fm, _body = planner._parse_frontmatter(text)
    strategy = fm.get("strategy")
    tasks = planner.parse_stories(text)
    return tasks, strategy


def run_pipeline(
    *,
    strategy: str | None = None,
    prd_path: str | None = None,
    task_path: str | None = None,
    effort: str | None = None,
    budget: float | None = None,
    max_iterations: int = 5,
    eval_skill: str | None = None,
    eval_model: str | None = None,
    eval_effort: str | None = None,
    cowabunga: bool = False,
    resume: bool = False,
    session: Session | None = None,
    gap_fallback_tier: int | None = None,
) -> int:
    ladder = load_ladder()
    if eval_model:
        ladder.eval_model = eval_model
    if eval_effort:
        ladder.eval_effort = eval_effort
    if session is None:
        # Fresh session per run so prior runs (especially failed ones) are kept.
        session = Session(new_session_id())

    tasks: list[Task] = []
    if task_path:
        tasks.append(_load_task_from_yaml(task_path))
    elif prd_path:
        prd_tasks, prd_strategy = _load_tasks_from_prd(prd_path)
        tasks = prd_tasks
        if strategy is None:
            strategy = prd_strategy
    else:
        print("error: provide --task or --prd")
        return 1

    strategy_name = strategy or DEFAULT_STRATEGY
    try:
        strat = get_strategy(strategy_name)
    except ValueError:
        print(
            f"error: unknown strategy '{strategy_name}'. "
            f"Available: {', '.join(available_strategies())}"
        )
        return 1

    session.set_status(
        "running",
        pid=os.getpid(),
        strategy=strategy_name,
        tasks=len(tasks),
        max_iterations=max_iterations,
        effort=effort or "",
        budget=budget if budget is not None else "",
        source=prd_path or task_path or "",
        started=datetime.now(timezone.utc).isoformat(),
        stage="localize",
    )

    idx_lines = [
        f"# Session {session.id}",
        f"- strategy: {strategy_name}",
        f"- tasks: {len(tasks)}",
    ]
    if prd_path:
        idx_lines.append(f"- prd: {prd_path}")
    session.update_index("\n".join(idx_lines) + "\n")

    # Mirror the whole "splinter" log stream into events.md so the Trace view is a
    # full chronological record of every push / result / decision.
    splog = logging.getLogger("splinter")
    trace_handler = _SessionTraceHandler(session)
    trace_handler.setLevel(logging.INFO)
    splog.addHandler(trace_handler)
    session.append(
        "events.md",
        f"=== run {'resume' if resume else 'start'} · {strategy_name} · "
        f"{datetime.now(timezone.utc).astimezone().strftime('%H:%M:%S')} ===",
    )

    try:
        log.info("session %s · strategy %s · %d task(s)%s", session.id, strategy_name,
                 len(tasks), " · 🤙 cowabunga" if cowabunga else "")

        prd_text = ""
        if prd_path:
            prd_text = Path(prd_path).read_text()
        elif tasks:
            prd_text = tasks[0].description

        localization = ""
        anchors: list[CodeAnchor] = []
        if prd_text:
            # localize() returns cached anchors on resume (no-op if file exists + parseable).
            if resume and session.has("knowledge/localization.md"):
                log.info("resume: reusing existing localization")
                localization = session.read("knowledge/localization.md")
                from splinter.agents.localizer import _parse_anchors
                anchors = _parse_anchors(localization)
                log.info("resume: re-parsed %d anchor(s)", len(anchors))
            else:
                log.info("localizing against the codebase…")
                anchors = localize(prd_text, session, ladder)
                localization = session.read("knowledge/localization.md")

        if anchors and tasks:
            planner.assign_target_files(tasks, anchors)
            session.set_status("running", stage="filter")
            log.info("filtering code context per task…")
            for i, task in enumerate(tasks):
                # Cache filter output so resume doesn't re-call the LLM.
                cache_key = f"knowledge/filter-{i + 1}.md"
                cached = session.read(cache_key)
                if resume and cached.strip():
                    task.filtered_context = cached
                    log.info("resume: reusing filtered context for task %d", i + 1)
                else:
                    task.filtered_context = filter_task_context(task, ladder)
                    if task.filtered_context:
                        session.write(cache_key, task.filtered_context)

        _resolve_gate(session, ladder)

        session.set_status("running", stage="run")
        results = strat.execute(
            tasks,
            session,
            ladder,
            effort=effort,
            budget=budget,
            max_iterations=max_iterations,
            localization=localization,
            eval_skill=eval_skill,
            cowabunga=cowabunga,
            resume=resume,
            gap_fallback_tier=gap_fallback_tier,
        )

        session.set_status("completed", stage="done")
        total = sum(r.cost for r in results)
        log.info("pipeline complete · %d run(s) · $%.4f", len(results), total)
        print(f"pipeline complete. session: {session.id}")
        print(f"  runs: {len(results)}")
        print(f"  cost: ${total:.4f}")
        return 0
    except Exception as gap_exc:
        from splinter.providers.base import ProviderGapError
        if not isinstance(gap_exc, ProviderGapError):
            raise
        session.set_status(
            "paused",
            kind=gap_exc.kind,
            resumable=gap_exc.resumable,
            provider=gap_exc.provider,
            retry_after=gap_exc.retry_after,
        )
        log.warning("run paused — %s", gap_exc)
        print(gap_exc.guidance)
        return 2
    except BaseException as exc:
        fail_class = _classify_failure(exc)
        session.set_status("failed", fail_class=fail_class)
        log.error("pipeline failed (%s): %s", fail_class, exc)
        raise
    finally:
        splog.removeHandler(trace_handler)
