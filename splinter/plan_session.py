"""Plan mode: the planner conversation that precedes ``splinter run --task``.

On a TTY, ``splinter run --task x.yaml`` opens :class:`splinter.tui_plan.PlanSessionApp`,
which drives this module:

1. **clarify** — the planner model reads the task and asks lettered implementation
   questions; the user answers; the planner records decisions and what is still open.
2. **write** — the planner emits the implementation plan (same contract as the
   pipeline planner prompt). The user may revise it.
3. **run** — the plan lands in ``knowledge/plan.md`` / ``plan-1.md`` and the direct
   strategy reuses it instead of calling the planner again.

Every turn uses the **planner** step (``models.planner`` / ``efforts.planner``).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from splinter.models.roster import load_ladder
from splinter.prd_session import Turn
from splinter.providers.dispatch import run_provider_session
from splinter.templating import load_standards, render, section

if TYPE_CHECKING:
    from collections.abc import Sequence

    from splinter.agents.runner import Task
    from splinter.memory.session import Session
    from splinter.task_plan import PlanIndex

__all__ = [
    "Turn",
    "open_plan_questions",
    "parse_task_text",
    "plan_bundle",
    "refine_plan",
    "revise_plan",
    "save_plan",
    "save_task",
    "save_task_plans",
    "split_task_plans",
    "strip_plan_header",
    "task_text",
    "write_plan",
]

_TASK_HEAD = "## Task"
_ACCEPT_HEAD = "## Acceptance Criteria"
_ROLE = (
    "You are a senior engineer planning the implementation of ONE task with the user "
    "before another LLM implements it."
)


def task_text(task: "Task") -> str:
    """Render a task as editable markdown: ``## Task`` + ``## Acceptance Criteria``."""
    return (
        f"{_TASK_HEAD}\n{task.description.strip()}\n\n"
        f"{_ACCEPT_HEAD}\n{task.acceptance.strip()}\n"
    )


def parse_task_text(text: str) -> tuple[str, str]:
    """Inverse of :func:`task_text`: ``(description, acceptance)``.

    Text without a ``## Task`` header is all description.
    """
    body = text.strip()
    head = re.match(rf"^{re.escape(_TASK_HEAD)}[ \t]*\n?", body)
    if head:
        body = body[head.end() :]
    desc, sep, acceptance = body.partition(_ACCEPT_HEAD)
    if not sep:
        return desc.strip(), ""
    return desc.strip(), acceptance.strip()


def _ask(prompt: str, *, resume: str | None, session: object = None) -> Turn:
    ladder = load_ladder()
    model = ladder.planner_model
    result, sid = run_provider_session(
        prompt,
        model,
        variant=ladder.planner_effort,
        output_format="json",
        session=resume or None,
        timeout=ladder.planner_timeout,
        role="plan",
        pool=ladder.step_pool("planner"),
    )
    tokens = {
        "input": result.tokens.get("input", 0) or 0,
        "output": result.tokens.get("output", 0) or 0,
    }
    if session is not None:
        try:
            session.log_llm_usage(model, tokens, result.cost)  # type: ignore[attr-defined]
        except Exception:
            pass
    return Turn(text=result.text, session_id=sid or "", model=model, tokens=tokens,
                cost=result.cost)


def _ground(localization: str) -> str:
    return f"## Codebase Localization (grounding)\n{localization}\n\n" if localization else ""


def _resume_preamble(resume: str | None) -> str:
    """Re-seed a fresh conversation when the planner's conversation id was lost."""
    if resume:
        return ""
    return (
        f"{_ROLE}\n"
        "You are RESUMING an in-progress planning conversation whose history was lost. "
        "Rely on the task and plan below as the source of truth.\n\n"
    )


def _task_section(current_task: str) -> str:
    if not current_task.strip():
        return ""
    return (
        "## Current Task in Editor (source of truth)\n"
        f"{current_task.strip()}\n\n"
    )


def open_plan_questions(task_md: str, *, localization: str = "", session: object = None) -> Turn:
    """First turn: read the task and ask lettered implementation questions."""
    prompt = (
        f"{_ROLE}\n\n"
        f"## Task to plan\n{task_md.strip()}\n\n"
        f"{_ground(localization)}"
        "You MUST ask exactly 3-8 numbered questions, each with lettered options (A/B/C/D), "
        "about the implementation decisions you would otherwise assume: the approach, "
        "the files/modules to touch, data shapes, edge cases and error handling, tests, "
        "and what is out of scope. Turn each assumed default into a question whose options "
        "include that default. Never skip, never write the plan yet, never answer the "
        "questions yourself, and never output an 'assumptions' list in place of questions. "
        "Output ONLY the numbered questions with their lettered options, no preamble."
    )
    return _ask(prompt, resume=None, session=session)


def refine_plan(
    answers: str,
    *,
    resume: str,
    current_task: str,
    localization: str = "",
    session: object = None,
) -> Turn:
    """Incorporate the user's answers; return decisions + remaining questions."""
    prompt = (
        f"{_resume_preamble(resume)}"
        f"{_task_section(current_task)}"
        f"{_ground(localization)}"
        f"User answers:\n{answers}\n\n"
        "Incorporate these answers. Then output two clearly separated parts:\n"
        "1. '## Decisions' — bullets, one per implementation decision settled so far.\n"
        "2. '## Open Questions' — any remaining lettered questions, or 'None — ready to "
        "plan.' if nothing is open.\n"
        "Keep it tight; do not re-ask anything already answered. Do not write the plan yet."
    )
    return _ask(prompt, resume=resume, session=session)


def _plan_contract(current_task: str, localization: str) -> str:
    description, acceptance = parse_task_text(current_task)
    return render(
        "plan",
        task_section=section("Task", description),
        acceptance_section=section("Acceptance Criteria", acceptance),
        code_context_section=section("Code Context", localization),
        standards_section=section("Code Conventions", load_standards()),
    )


def _sections_rule(task_ids: Sequence[str]) -> str:
    if not task_ids:
        return ""
    order = ", ".join(task_ids)
    return (
        f"This plan covers {len(task_ids)} tasks. Write one section per task, headed exactly "
        f"`## <task id>`, in this order: {order}. Each section is a numbered list of steps for "
        "that task only, naming files and functions; a task may rely on what earlier "
        "sections build. Nothing outside the sections.\n"
    )


def write_plan(
    *,
    resume: str,
    current_task: str,
    autodecide: bool,
    localization: str = "",
    session: object = None,
    task_ids: Sequence[str] = (),
) -> Turn:
    """Emit the final implementation plan (pipeline planner contract)."""
    decide = (
        "Resolve every open question yourself using sensible defaults — do not ask "
        "anything further.\n"
        if autodecide
        else "Apply the decisions gathered in this conversation; do not ask anything further.\n"
    )
    prompt = (
        f"{_resume_preamble(resume)}"
        f"{_plan_contract(current_task, localization)}\n"
        f"{decide}"
        "Output a numbered list of steps naming the specific files and functions to change. "
        "Write direct, unambiguous instructions for another LLM to implement exactly as "
        "written.\n"
        f"{_sections_rule(task_ids)}"
        "Output ONLY the plan markdown — no preamble, no questions, no trailing commentary."
    )
    return _ask(prompt, resume=resume, session=session)


def revise_plan(
    instructions: str,
    *,
    resume: str,
    current_plan: str,
    session: object = None,
    task_ids: Sequence[str] = (),
) -> Turn:
    """Apply free-form edits to the plan and re-emit it in full."""
    prompt = (
        f"{_resume_preamble(resume)}"
        "## Current Plan in Editor (source of truth)\n"
        f"{current_plan.strip()}\n\n"
        f"Apply these changes to the plan:\n{instructions}\n\n"
        f"{_sections_rule(task_ids)}"
        "Output the COMPLETE updated plan markdown only — same numbered format, "
        "no preamble."
    )
    return _ask(prompt, resume=resume, session=session)


def strip_plan_header(plan_md: str) -> str:
    """Drop a leading ``# Plan`` heading so it is never doubled."""
    return re.sub(r"^#\s*Plan[ \t]*(\n|$)", "", plan_md.strip(), flags=re.IGNORECASE).strip()


def save_plan(session: "Session", plan_md: str) -> None:
    """Write the plan where the direct strategy looks for a reusable plan."""
    content = f"# Plan\n\n{strip_plan_header(plan_md)}\n"
    session.write("knowledge/plan.md", content)
    session.write("knowledge/plan-1.md", content)


def save_task(
    session: "Session", description: str, acceptance: str, base: dict[str, Any]
) -> Path:
    """Write the (possibly edited) task to ``<session>/task.yaml``; keep other keys."""
    data = {**base, "description": description, "acceptance": acceptance}
    return session.write("task.yaml", yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


_TASK_PLAN_HEAD = re.compile(r"^##[ \t]+([A-Z]{1,3}\d+[a-z]?)\b.*$", re.MULTILINE)
PLAN_ALL_FILE = "knowledge/plan-all.md"
PLAN_BUNDLE_FILE = "knowledge/plan-bundle.md"


def task_plan_path(task_id: str) -> str:
    """Session path of the plan written for one task of a task plan."""
    return f"knowledge/plan-{task_id}.md"


def plan_bundle(index: "PlanIndex") -> str:
    """The editable text of a task plan: README first, then every task with its criteria."""
    ids = ", ".join(task.id for task in index.tasks)
    parts = [
        f"# Task plan: {len(index.tasks)} tasks ({ids})",
        "Plan the whole set. The README is the orchestrator's guide; every task follows it.",
        "",
        f"# README (`{index.path}`)",
        index.text.strip(),
    ]
    for task in index.tasks:
        parts += ["", f"# {task.id}", task.description.strip()]
        if task.acceptance.strip():
            parts += ["", "Acceptance:", task.acceptance.strip()]
    return "\n".join(parts) + "\n"


def split_task_plans(plan_md: str, task_ids: "Sequence[str]") -> dict[str, str]:
    """Map each task id to the body of its ``## <id>`` section (first one wins)."""
    heads = list(_TASK_PLAN_HEAD.finditer(plan_md))
    plans: dict[str, str] = {}
    for position, head in enumerate(heads):
        task_id = head.group(1)
        if task_id not in task_ids or task_id in plans:
            continue
        end = heads[position + 1].start() if position + 1 < len(heads) else len(plan_md)
        body = plan_md[head.end() : end].strip()
        if body:
            plans[task_id] = body
    return plans


def save_task_plans(session: "Session", plan_md: str, task_ids: "Sequence[str]") -> list[str]:
    """Write ``plan-<id>.md`` per task section; return the ids left without a plan.

    Keyed by task id, not position: strategies reorder tasks by their deps. The
    whole plan goes to :data:`PLAN_ALL_FILE`, never to ``knowledge/plan.md``, which
    the direct strategy reuses for task 1. Tasks without a section are planned
    during the run.
    """
    body = strip_plan_header(plan_md)
    session.write(PLAN_ALL_FILE, f"# Plan\n\n{body}\n")
    plans = split_task_plans(body, task_ids)
    for task_id in task_ids:
        path = task_plan_path(task_id)
        if task_id in plans:
            session.write(path, f"# Plan\n\n{plans[task_id]}\n")
        elif session.read(path).strip():
            session.write(path, "")
    return [task_id for task_id in task_ids if task_id not in plans]
