"""Final eval gate executor — dispatches a :class:`FinalEvalEntry` to the right backend.

Dispatch table:
  command → subprocess; exit-code maps to pass/fail
  skill   → LLM call via dispatch (model/variant from entry or ladder fallback)
  cursor  → Cursor CLI wrapper (``splinter.providers.cursor``)
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from splinter.configure import FinalEvalEntry
from splinter.enums import Decision, FinalEvalKind
from splinter.strategies.base import EvalVerdict

if TYPE_CHECKING:
    from splinter.agents.runner import Task
    from splinter.models.roster import Ladder

log = logging.getLogger("splinter.final_eval")

_DEFAULT_TIMEOUT = 120
_DEFAULT_MODEL = "sonnet"
_DEFAULT_VARIANT = "high"
_OUTPUT_CAP = 2000


@dataclass(frozen=True)
class FinalEvalResult:
    name: str
    passed: bool
    output: str
    verdict: EvalVerdict | None = None
    cost: float = 0.0
    tokens: dict[str, int] = field(default_factory=dict)


def run_final_eval(
    entry: FinalEvalEntry,
    *,
    task: "Task | None" = None,
    project_dir: str = ".",
    ladder: "Ladder | None" = None,
    timeout: int | None = None,
) -> FinalEvalResult:
    """Dispatch ``entry`` to the appropriate executor and return a structured result."""
    if entry.kind == FinalEvalKind.COMMAND:
        return _run_command(entry, project_dir=project_dir, timeout=timeout)
    if entry.kind == FinalEvalKind.SKILL:
        return _run_skill(entry, task=task, ladder=ladder, timeout=timeout)
    if entry.kind == FinalEvalKind.CURSOR:
        return _run_cursor(entry, task=task, project_dir=project_dir, timeout=timeout)
    raise ValueError(f"unknown final_eval kind: {entry.kind!r}")


def run_all_final_evals(
    entries: list[FinalEvalEntry],
    *,
    task: "Task | None" = None,
    project_dir: str = ".",
    ladder: "Ladder | None" = None,
    timeout: int | None = None,
    fail_fast: bool = True,
) -> list[FinalEvalResult]:
    """Run every entry in order; stop on first failure when ``fail_fast`` is set."""
    results: list[FinalEvalResult] = []
    for entry in entries:
        result = run_final_eval(
            entry, task=task, project_dir=project_dir, ladder=ladder, timeout=timeout
        )
        results.append(result)
        if fail_fast and not result.passed:
            break
    return results


# ── kind implementations ──────────────────────────────────────────────────────

def _run_command(
    entry: FinalEvalEntry,
    *,
    project_dir: str = ".",
    timeout: int | None = None,
) -> FinalEvalResult:
    cmd = entry.cmd or ""
    try:
        proc = subprocess.run(
            cmd.split(),
            capture_output=True,
            text=True,
            timeout=timeout or _DEFAULT_TIMEOUT,
            cwd=project_dir,
        )
        passed = proc.returncode == 0
        output = (proc.stdout + proc.stderr).strip()[:_OUTPUT_CAP]
    except subprocess.TimeoutExpired:
        passed = False
        output = "timed out"
    except FileNotFoundError:
        passed = False
        output = f"command not found: {cmd}"
    verdict = EvalVerdict(
        decision=Decision.PASS if passed else Decision.RETRY,
        reason=output,
        corrections="" if passed else output,
    )
    return FinalEvalResult(
        name=entry.name, passed=passed, output=output, verdict=verdict
    )


def _run_skill(
    entry: FinalEvalEntry,
    *,
    task: "Task | None" = None,
    ladder: "Ladder | None" = None,
    timeout: int | None = None,
) -> FinalEvalResult:
    from splinter.providers.dispatch import run_provider_session
    from splinter.skills import resolve_eval_skill

    model = entry.model or (ladder.eval_model if ladder else _DEFAULT_MODEL)
    variant = str(entry.variant) if entry.variant else (
        ladder.eval_effort if ladder else _DEFAULT_VARIANT
    )

    skill = resolve_eval_skill(entry.skill or "")
    task_desc = task.description if task else ""
    task_accept = task.acceptance if task else ""

    if skill is None or skill.missing:
        skill_text = ""
    else:
        skill_text = f"{skill.body}\n\n---\n"

    prompt = (
        f"{skill_text}"
        f"Task:\n{task_desc}\n\n"
        f"Acceptance criteria:\n{task_accept}\n\n"
        "Respond with VERDICT: PASS or VERDICT: RETRY and a brief reason."
    )

    try:
        response, sid = run_provider_session(
            prompt, model, variant=variant, timeout=timeout
        )
        raw_text = response.text
        passed = "VERDICT: PASS" in raw_text.upper() or raw_text.upper().strip().startswith("PASS")
        verdict = EvalVerdict(
            decision=Decision.PASS if passed else Decision.RETRY,
            reason=raw_text,
            corrections="" if passed else raw_text,
            raw=raw_text,
            eval_session=sid,
            cost=response.cost,
            tokens=response.tokens,
        )
        return FinalEvalResult(
            name=entry.name,
            passed=passed,
            output=raw_text,
            verdict=verdict,
            cost=response.cost,
            tokens=response.tokens,
        )
    except Exception as exc:
        output = str(exc)
        log.warning("skill final_eval '%s' failed: %s", entry.name, exc)
        verdict = EvalVerdict(decision=Decision.RETRY, reason=output, corrections=output)
        return FinalEvalResult(name=entry.name, passed=False, output=output, verdict=verdict)


def _run_cursor(
    entry: FinalEvalEntry,
    *,
    task: "Task | None" = None,
    project_dir: str = ".",
    timeout: int | None = None,
) -> FinalEvalResult:
    from splinter.providers import cursor as cursor_provider

    task_desc = task.description if task else ""
    task_accept = task.acceptance if task else ""
    prompt = (
        f"Review and evaluate whether the following task has been completed correctly.\n\n"
        f"Task:\n{task_desc}\n\n"
        f"Acceptance criteria:\n{task_accept}\n\n"
        "Respond with VERDICT: PASS or VERDICT: RETRY and a brief reason."
    )

    try:
        result = cursor_provider.run(prompt, timeout=timeout, project_dir=project_dir)
        raw_text = result.text
        passed = "VERDICT: PASS" in raw_text.upper() or raw_text.upper().strip().startswith("PASS")
        verdict = EvalVerdict(
            decision=Decision.PASS if passed else Decision.RETRY,
            reason=raw_text,
            corrections="" if passed else raw_text,
            raw=raw_text,
        )
        return FinalEvalResult(
            name=entry.name, passed=passed, output=raw_text, verdict=verdict
        )
    except Exception as exc:
        output = str(exc)
        log.warning("cursor final_eval '%s' failed: %s", entry.name, exc)
        verdict = EvalVerdict(decision=Decision.RETRY, reason=output, corrections=output)
        return FinalEvalResult(name=entry.name, passed=False, output=output, verdict=verdict)
