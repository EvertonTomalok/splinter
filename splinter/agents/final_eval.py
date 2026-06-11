"""Final eval gate executor — dispatches a :class:`FinalEvalEntry` to the right backend.

Dispatch table:
  command  → subprocess; exit-code maps to pass/fail
  skill    → provider-agnostic LLM call; auto-pass/fail on VERDICT in output
  review   → provider-agnostic LLM call; returns raw output for human review
  ask_user → no LLM; pauses immediately with task context for human judgment

Provider routing (skill / review):
  entry.provider overrides; otherwise derived from entry.model via provider_for().
  "claude"   → Claude CLI  (model ids: sonnet, opus, haiku, …)
  "opencode" → OpenCode    (model ids: opencode-go/*, opencode/*, codex-*, …)
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

# Models that belong to opencode by prefix — everything else routes to claude.
# NOTE: "codex" is a separate CLI provider (not opencode). Use provider="codex" explicitly.
_OPENCODE_PREFIXES = ("opencode-go/", "opencode/", "gpt-", "o1-", "o3-")


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
    if entry.kind == FinalEvalKind.REVIEW:
        return _run_review(entry, task=task, ladder=ladder, timeout=timeout)
    if entry.kind == FinalEvalKind.ASK_USER:
        return _run_ask_user(entry, task=task)
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

def _run_ask_user(
    entry: FinalEvalEntry,
    *,
    task: "Task | None" = None,
) -> FinalEvalResult:
    """Pause immediately for user review — no LLM, no command.

    Builds a summary from the task description and acceptance criteria so the
    user has context when the TUI modal opens.
    """
    task_desc = task.description if task else ""
    task_accept = task.acceptance if task else ""
    output = (
        f"Manual review requested: {entry.name}\n\n"
        + (f"Task:\n{task_desc}\n\n" if task_desc else "")
        + (f"Acceptance criteria:\n{task_accept}" if task_accept else "")
    ).strip()
    verdict = EvalVerdict(
        decision=Decision.ASK_USER,
        reason=output,
        corrections=output,
        raw=output,
    )
    return FinalEvalResult(name=entry.name, passed=False, output=output, verdict=verdict)


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


def _resolve_model(entry: FinalEvalEntry, ladder: "Ladder | None") -> tuple[str, str]:
    """Return (model_id, variant) for this entry.

    Priority: entry.provider+model > entry.model > ladder.eval_model > default.
    When entry.provider is set but entry.model is not, picks the default model
    for that provider ("sonnet" for claude, "opencode-go/qwen3-coder" for opencode).
    """
    from splinter.models.roster import provider_for

    _OPENCODE_DEFAULT = "opencode-go/qwen3-coder"
    _CLAUDE_DEFAULT = "sonnet"

    variant = str(entry.variant) if entry.variant else (
        ladder.eval_effort if ladder else _DEFAULT_VARIANT
    )

    if entry.provider:
        if entry.model:
            return entry.model, variant
        if entry.provider == "opencode":
            return _OPENCODE_DEFAULT, variant
        if entry.provider == "codex":
            # Codex CLI provider — not yet implemented; falls back to claude default.
            # TODO: wire splinter.providers.codex_cli when available.
            log.warning("codex provider not yet implemented, falling back to claude")
            return _CLAUDE_DEFAULT, variant
        return _CLAUDE_DEFAULT, variant

    if entry.model:
        return entry.model, variant

    base = ladder.eval_model if ladder else _DEFAULT_MODEL
    # Respect explicit provider prefix on ladder eval model too.
    if any(base.startswith(p) for p in _OPENCODE_PREFIXES):
        return base, variant
    return base, variant


def _skill_prompt(entry: FinalEvalEntry, task: "Task | None", eval_mode: bool) -> str:
    """Build the LLM prompt for skill / review kinds."""
    from splinter.skills import resolve_eval_skill

    skill = resolve_eval_skill(entry.skill or "")
    skill_text = f"{skill.body}\n\n---\n" if (skill and not skill.missing) else ""
    task_desc = task.description if task else ""
    task_accept = task.acceptance if task else ""

    if eval_mode:
        return (
            f"{skill_text}"
            f"Task:\n{task_desc}\n\n"
            f"Acceptance criteria:\n{task_accept}\n\n"
            "Respond with VERDICT: PASS or VERDICT: RETRY and a brief reason."
        )
    return (
        f"{skill_text}"
        f"Review whether the following task has been completed correctly.\n\n"
        f"Task:\n{task_desc}\n\n"
        f"Acceptance criteria:\n{task_accept}\n\n"
        "Provide a detailed evaluation report. Describe what works, what doesn't, "
        "and what corrections are needed if any."
    )


def _run_skill(
    entry: FinalEvalEntry,
    *,
    task: "Task | None" = None,
    ladder: "Ladder | None" = None,
    timeout: int | None = None,
) -> FinalEvalResult:
    """Run skill via the configured provider; auto-pass/fail on VERDICT in output."""
    from splinter.providers.dispatch import run_provider_session

    model, variant = _resolve_model(entry, ladder)
    prompt = _skill_prompt(entry, task, eval_mode=True)
    log.info("skill final_eval '%s' → %s @ %s", entry.name, model, variant)

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


def _run_review(
    entry: FinalEvalEntry,
    *,
    task: "Task | None" = None,
    ladder: "Ladder | None" = None,
    timeout: int | None = None,
) -> FinalEvalResult:
    """Run skill via the configured provider; return raw output for human review.

    Never auto-passes — always raises ManualValidationPause via pipeline so the
    TUI modal opens for human judgment (approve / request changes / reject).
    """
    from splinter.providers.dispatch import run_provider_session

    model, variant = _resolve_model(entry, ladder)
    prompt = _skill_prompt(entry, task, eval_mode=False)
    log.info("review final_eval '%s' → %s @ %s", entry.name, model, variant)

    try:
        response, sid = run_provider_session(
            prompt, model, variant=variant, timeout=timeout
        )
        raw_text = response.text
        verdict = EvalVerdict(
            decision=Decision.ASK_USER,
            reason=raw_text,
            corrections=raw_text,
            raw=raw_text,
            eval_session=sid,
            cost=response.cost,
            tokens=response.tokens,
        )
        return FinalEvalResult(
            name=entry.name,
            passed=False,
            output=raw_text,
            verdict=verdict,
            cost=response.cost,
            tokens=response.tokens,
        )
    except Exception as exc:
        output = str(exc)
        log.warning("review final_eval '%s' failed: %s", entry.name, exc)
        verdict = EvalVerdict(decision=Decision.ASK_USER, reason=output, corrections=output)
        return FinalEvalResult(name=entry.name, passed=False, output=output, verdict=verdict)
