from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from splinter.memory.session import Session


def dependency_waves(deps: Mapping[str, Sequence[str]]) -> list[list[str]]:
    if not deps:
        return []
    ids = set(deps.keys())
    all_keys = list(deps.keys())
    leveled = [k for k in all_keys if all(d in ids for d in deps[k])]
    deferred = [k for k in all_keys if any(d not in ids for d in deps[k])]
    remaining = {k: {d for d in deps[k] if d in ids} for k in leveled}
    waves: list[list[str]] = []
    pending = set(remaining)
    while pending:
        wave = [k for k in leveled if k in pending and not remaining[k]]
        if not wave:
            return [all_keys]
        waves.append(wave)
        done = set(wave)
        pending -= done
        for k in pending:
            remaining[k] -= done
    if deferred:
        waves.append(deferred)
    return waves


@dataclass(frozen=True)
class WaveFacts:
    stories: int
    waves: int
    widest: int


def wave_facts(deps: Mapping[str, Sequence[str]]) -> WaveFacts:
    waves = dependency_waves(deps)
    stories = len(deps)
    num_waves = len(waves)
    widest = max((len(w) for w in waves), default=0)
    return WaveFacts(stories=stories, waves=num_waves, widest=widest)


def facts_line(f: WaveFacts, parallel: bool) -> str:
    if f.stories == 0:
        return ""
    story_word = "story" if f.stories == 1 else "stories"
    wave_word = "dependency wave" if f.waves == 1 else "dependency waves"
    widest_story_word = "story" if f.widest == 1 else "stories"
    par_str = "parallel on" if parallel else "parallel off"
    return (
        f"{f.stories} {story_word} in {f.waves} {wave_word} · "
        f"widest wave has {f.widest} {widest_story_word} · {par_str}"
    )


def extract_task_deps(
    prd_text: str = "",
    session: Session | None = None,
) -> dict[str, list[str]]:
    if session is not None:
        try:
            from splinter.task_plan import is_plan_index, load_plan_index

            status = session.read_status()
            if status.get("plan_index") and status.get("task_path"):
                from pathlib import Path

                path = Path(str(status["task_path"]))
                if is_plan_index(path):
                    plan_index = load_plan_index(path)
                    return {t.id: list(t.deps or []) for t in plan_index.tasks}
        except Exception:
            pass

    if prd_text:
        try:
            from splinter.prd_session import user_story_titles

            titles = user_story_titles(prd_text)
            if titles:
                from splinter.agents.planner import parse_stories

                tasks = parse_stories(prd_text)
                return {t.id: list(t.deps or []) for t in tasks if t.id}
        except Exception:
            pass

        try:
            from splinter.task_plan import _deps, _task_rows

            rows = _task_rows(prd_text)
            if rows:
                ids = [r.task_id for r in rows]
                return {r.task_id: _deps(r.deps, r.task_id, ids) for r in rows}
        except Exception:
            pass

    return {}


def format_plan_steps(plan: str) -> str:
    task_plans = re.findall(r"^##\s+T[0-9]+", plan, re.MULTILINE)
    if task_plans:
        k = len(task_plans)
        return f"{k} task plan" if k == 1 else f"{k} task plans"
    steps = re.findall(r"^\s*\d+\.", plan, re.MULTILINE)
    n = len(steps)
    return f"{n} step" if n == 1 else f"{n} steps"
