"""Story order for the agents strategy: which PRD stories wait for which.

PRD stories rarely say ``deps:``; the order lives in prose ("Onda 1 …",
"Rollout", "after the contract ships"). Without it every brief lands in one
wave. :func:`with_story_deps` asks the planner once, with the story titles and
the PRD's delivery section only, for each story's direct prerequisites. Only
earlier stories count, so the result can never cycle. When the answer is
unusable each story waits for the one before it: the PRD order, which is safe.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import replace

from splinter.agents.runner import Task

log = logging.getLogger(__name__)

_DELIVERY_RE = re.compile(
    r"entrega|delivery|rollout|ondas?\b|waves?\b|fases?\b|phases?\b|milestones?|"
    r"sequ[eê]ncia|order|ordem|plano de execu",
    re.IGNORECASE,
)
_HEADING_RE = re.compile(r"^(#{2,3})\s+(.+?)\s*$")
_DELIVERY_MAX = 4000
_TITLE_MAX = 160


def delivery_section(prd: str) -> str:
    """The PRD's delivery/rollout section (heading and body), clipped; else ``""``."""
    lines = prd.splitlines()
    for index, raw in enumerate(lines):
        heading = _HEADING_RE.match(raw)
        if not heading or not _DELIVERY_RE.search(heading.group(2)):
            continue
        level = len(heading.group(1))
        body = [raw]
        for line in lines[index + 1 :]:
            nxt = _HEADING_RE.match(line)
            if nxt and len(nxt.group(1)) <= level:
                break
            body.append(line)
        text = "\n".join(body).strip()
        return text if len(text) <= _DELIVERY_MAX else text[:_DELIVERY_MAX].rstrip() + "\n…"
    return ""


def _story_line(task: Task, n: int) -> str:
    head = " ".join(task.description.split())
    head = head if len(head) <= _TITLE_MAX else head[: _TITLE_MAX - 1] + "…"
    return f"- {task.id or f't{n}'}: {head}"


def order_prompt(tasks: list[Task], prd: str) -> str:
    stories = "\n".join(_story_line(task, n) for n, task in enumerate(tasks))
    delivery = delivery_section(prd) or "(the PRD has no delivery section)"
    return (
        "Answer from the text below only. Do not run tools or open files.\n\n"
        "These user stories will be built by separate coding agents. For each story, "
        "list its direct prerequisites: earlier stories whose code, contract, schema "
        "or data it uses, or that the delivery plan puts in an earlier wave. Stories "
        "of the same wave that touch different code have no dependency. Never list a "
        "later story.\n\n"
        'Reply with JSON only: {"deps": {"US-002": ["US-001"], "US-003": []}}\n\n'
        f"Stories, in PRD order:\n{stories}\n\n"
        f"Delivery plan from the PRD:\n{delivery}\n"
    )


def _chain(tasks: list[Task]) -> list[Task]:
    return [
        replace(task, deps=[tasks[n - 1].id] if n and tasks[n - 1].id else [])
        for n, task in enumerate(tasks)
    ]


def parse_deps(raw: str, tasks: list[Task]) -> dict[str, list[str]]:
    """``{story: [earlier stories]}`` from the reply; ``ValueError`` when unusable."""
    body = re.sub(r"^```[a-z]*\s*|\s*```$", "", raw.strip())
    data = json.loads(body)
    table = data.get("deps") if isinstance(data, dict) else None
    if not isinstance(table, dict):
        raise ValueError("story order has no deps table")
    index = {task.id.upper(): n for n, task in enumerate(tasks) if task.id}
    found: dict[str, list[str]] = {}
    for key, value in table.items():
        mine = index.get(str(key).strip().upper())
        if mine is None or not isinstance(value, list):
            continue
        earlier = [
            tasks[index[dep]].id
            for dep in (str(d).strip().upper() for d in value)
            if dep in index and index[dep] < mine
        ]
        found[tasks[mine].id] = list(dict.fromkeys(earlier))
    if not found:
        raise ValueError("story order names no known story")
    return found


def with_story_deps(tasks: list[Task], prd: str, ask: Callable[[str], str]) -> list[Task]:
    """``tasks`` with ``deps`` set from the planner's story order.

    Leaves the tasks alone when any of them already has deps or there is only
    one; falls back to the PRD order (each story after the previous) when the
    answer cannot be used.
    """
    if len(tasks) < 2 or any(task.deps for task in tasks):
        return tasks
    try:
        table = parse_deps(ask(order_prompt(tasks, prd)), tasks)
    except Exception as exc:
        log.warning("refine · story order failed (%s) — stories run in PRD order", exc)
        return _chain(tasks)
    ordered = [replace(task, deps=table.get(task.id, [])) for task in tasks]
    roots = sum(1 for task in ordered if not task.deps)
    log.info("refine · story order · %d stories, %d can start first", len(tasks), roots)
    return ordered
