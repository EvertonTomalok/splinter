"""Resolves a task+tier into a concrete model run via a provider strategy."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from splinter.enums import Effort, Variant
from splinter.models.roster import Ladder
from splinter.obs.agentic import record_exchange
from splinter.templating import render, section

log = logging.getLogger("splinter.runner")

_TRANSIENT_GAP_KINDS = frozenset(("rate_limit", "overload"))
_MAX_GAP_RETRIES = 5


@dataclass
class Task:
    """§6.3 Task schema.

    Core fields (planner): id, description, target_files, deps, effort, eval_skill.
    Runner extras: acceptance, reasoning_effort, suggested_tier.
    filtered_context: pre-digested code context from the harness (localize → filter).
    """

    description: str
    acceptance: str
    effort: str = Effort.NORMAL
    reasoning_effort: str = Variant.AUTO
    eval_skill: str | None = None
    suggested_tier: int = 0
    target_files: list[str] | None = None
    id: str = ""
    deps: list[str] | None = None
    filtered_context: str = ""


@dataclass(frozen=True)
class RunResult:
    text: str
    model: str
    tier: int
    tokens: dict[str, int]
    cost: float
    raw: dict[str, Any]
    opencode_session: str | None = None
    cost_indeterminate: bool = False


def resolve_variant(
    task: Task,
    effort_override: str | None,
    ladder: Ladder,
    tier_level: int | None = None,
) -> str:
    """Pick the reasoning variant.

    Precedence: CLI override > task setting > per-tier config > ladder effort map.
    """
    if effort_override and effort_override != Variant.AUTO:
        return effort_override
    if task.reasoning_effort and task.reasoning_effort != Variant.AUTO:
        return task.reasoning_effort
    if tier_level is not None:
        configured = ladder.tier_variant(tier_level)
        if configured:
            return configured
    em = ladder.effort_mapping(task.effort)
    if em:
        return em.variant
    # Agentic floor — never low/minimal for code generation.
    return Variant.MEDIUM


def resolve_model(tier_level: int, ladder: Ladder) -> tuple[str, str]:
    """Map a ladder tier level to its primary model id and provider name."""
    tier = ladder.tier_by_level(tier_level)
    return tier.models[0], tier.provider


def _build_prompt(
    task: Task,
    plan: str,
    localization: str,
    corrections: str,
    *,
    is_continuation: bool = False,
) -> str:
    # code_ctx is pre-filtered by the harness (localize → filter_task_context).
    code_ctx = localization
    if corrections:
        if is_continuation:
            # Session already holds task/plan/context — send only the delta.
            return render(
                "run_fix_continue",
                corrections_section=section("Corrections from Evaluator", corrections),
            )
        return render(
            "run_fix",
            task_section=section("Original Task", task.description),
            acceptance_section=section("Acceptance Criteria", task.acceptance),
            corrections_section=section("Corrections from Evaluator", corrections),
            code_context_section=section("Code Context", code_ctx),
        )
    return render(
        "run",
        plan_section=section("Plan", plan),
        task_section=section("Task", task.description),
        acceptance_section=section("Acceptance Criteria", task.acceptance),
        code_context_section=section("Code Context", code_ctx),
    )


def run_task(
    task: Task,
    plan: str,
    tier_level: int,
    ladder: Ladder,
    *,
    effort_override: str | None = None,
    localization: str = "",
    corrections: str = "",
    opencode_session: str | None = None,
    timeout: int | None = None,
    trace: object = None,
    iteration: int = 0,
    task_index: int = 0,
) -> RunResult:
    model_id, _ = resolve_model(tier_level, ladder)
    variant = resolve_variant(task, effort_override, ladder, tier_level)
    if timeout is None:
        timeout = ladder.tier_timeout(tier_level)
    prompt = _build_prompt(
        task, plan, localization, corrections, is_continuation=opencode_session is not None
    )

    from splinter.providers.dispatch import run_provider_session

    response = None
    for attempt in range(_MAX_GAP_RETRIES + 1):
        try:
            response, _sid = run_provider_session(
                prompt, model_id, variant=variant, session=opencode_session, timeout=timeout,
                trace=trace, iteration=iteration, tier=tier_level,
                task_index=task_index, role="run",
            )
            break
        except Exception as exc:
            from splinter.providers.base import ProviderGapError

            if not isinstance(exc, ProviderGapError) or exc.kind not in _TRANSIENT_GAP_KINDS:
                raise
            if attempt >= _MAX_GAP_RETRIES:
                raise
            wait = min(5 * (2**attempt), 60)
            log.warning(
                "provider gap (%s) on attempt %d/%d — retrying in %ds",
                exc.kind,
                attempt + 1,
                _MAX_GAP_RETRIES,
                wait,
            )
            time.sleep(wait)
    assert response is not None
    record_exchange(prompt, response.text, model=model_id)

    return RunResult(
        text=response.text,
        model=model_id,
        tier=tier_level,
        tokens=response.tokens,
        cost=response.cost,
        raw=response.raw,
        opencode_session=response.session_id or opencode_session,
        cost_indeterminate=response.cost_indeterminate,
    )
