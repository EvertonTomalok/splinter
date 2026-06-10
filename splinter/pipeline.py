"""Wires PRD/task input through localize -> plan -> run -> gate -> eval -> loop."""

from __future__ import annotations

import logging
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

from splinter.agents.localizer import localize
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


def _classify_failure(exc: BaseException) -> str:
    """Transient (provider/network blip → resume continues) vs critical (bad command,
    bug → resume rolls the failing stage back and redoes it)."""
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
    fm: dict[str, str] = {}
    body = text
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            fm = yaml.safe_load(parts[1]) or {}
            body = parts[2]

    strategy = fm.get("strategy")

    tasks: list[Task] = []
    us_pattern = re.compile(
        r"###\s+(US-\d+):\s*(.+?)\n(.*?)(?=###\s+US-|\Z)",
        re.DOTALL,
    )
    for m in us_pattern.finditer(body):
        us_id = m.group(1)
        title = m.group(2).strip()
        block = m.group(3)

        desc_match = re.search(r"\*\*Description:\*\*\s*(.+)", block)
        desc = desc_match.group(1).strip() if desc_match else title

        effort_match = re.search(r"effort:\s*(\w+)", block)
        effort = effort_match.group(1) if effort_match else "normal"

        skill_match = re.search(r"eval_skill:\s*(\S+)", block)
        skill = skill_match.group(1) if skill_match else None

        ac_lines = re.findall(r"- \[[ x]\]\s*(.+)", block)
        acceptance = "\n".join(ac_lines) if ac_lines else desc

        tasks.append(
            Task(
                description=f"{us_id}: {desc}",
                acceptance=acceptance,
                effort=effort,
                eval_skill=skill,
            )
        )

    if not tasks:
        tasks.append(
            Task(
                description=body[:200].strip(),
                acceptance="implementation matches the PRD description",
            )
        )

    return tasks, strategy


def run_pipeline(
    *,
    strategy: str | None = None,
    prd_path: str | None = None,
    task_path: str | None = None,
    effort: str | None = None,
    budget: float | None = None,
    max_iterations: int = 5,
    cowabunga: bool = False,
    resume: bool = False,
    session: Session | None = None,
) -> int:
    ladder = load_ladder()
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

    log.info("session %s · strategy %s · %d task(s)%s", session.id, strategy_name,
             len(tasks), " · 🤙 cowabunga" if cowabunga else "")

    try:
        prd_text = ""
        if prd_path:
            prd_text = Path(prd_path).read_text()
        elif tasks:
            prd_text = tasks[0].description

        localization = ""
        if prd_text:
            existing_loc = session.read("localization.md")
            if resume and existing_loc.strip():
                log.info("resume: reusing existing localization")
                localization = existing_loc
            else:
                log.info("localizing against the codebase…")
                localize(prd_text, session, ladder)
                localization = session.read("localization.md")

        session.set_status("running", stage="run")
        results = strat.execute(
            tasks,
            session,
            ladder,
            effort=effort,
            budget=budget,
            max_iterations=max_iterations,
            localization=localization,
            cowabunga=cowabunga,
            resume=resume,
        )
    except BaseException as exc:
        fail_class = _classify_failure(exc)
        session.set_status("failed", fail_class=fail_class)
        log.error("pipeline failed (%s): %s", fail_class, exc)
        raise

    session.set_status("completed", stage="done")
    total = sum(r.cost for r in results)
    log.info("pipeline complete · %d run(s) · $%.4f", len(results), total)
    print(f"pipeline complete. session: {session.id}")
    print(f"  runs: {len(results)}")
    total_cost = sum(r.cost for r in results)
    print(f"  cost: ${total_cost:.4f}")
    return 0
