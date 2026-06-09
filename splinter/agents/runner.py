"""Resolves a task+tier into a concrete model run via a provider strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from splinter.enums import Effort, Variant
from splinter.models.roster import Ladder
from splinter.providers.registry import get_provider
from splinter.templating import render, section


@dataclass
class Task:
    description: str
    acceptance: str
    effort: str = Effort.NORMAL
    reasoning_effort: str = Variant.AUTO
    eval_skill: str | None = None
    suggested_tier: int = 0
    target_files: list[str] | None = None


@dataclass(frozen=True)
class RunResult:
    text: str
    model: str
    tier: int
    tokens: dict[str, int]
    cost: float
    raw: dict[str, Any]
    opencode_session: str | None = None


EFFORT_TO_VARIANT: dict[str, str] = {
    Effort.TRIVIAL: Variant.MINIMAL,
    Effort.NORMAL: Variant.LOW,
    Effort.HARD: Variant.HIGH,
    Effort.CRITICAL: Variant.MAX,
}


def resolve_variant(task: Task, effort_override: str | None, ladder: Ladder) -> str:
    """Pick the reasoning variant: CLI override > task setting > ladder effort map."""
    if effort_override and effort_override != Variant.AUTO:
        return effort_override
    if task.reasoning_effort and task.reasoning_effort != Variant.AUTO:
        return task.reasoning_effort
    em = ladder.effort_mapping(task.effort)
    if em:
        return em.variant
    return Variant.LOW


def resolve_model(tier_level: int, ladder: Ladder) -> tuple[str, str]:
    """Map a ladder tier level to its primary model id and provider name."""
    tier = ladder.tier_by_level(tier_level)
    return tier.models[0], tier.provider


def _build_prompt(task: Task, plan: str, localization: str, corrections: str) -> str:
    if corrections:
        return render(
            "run_fix",
            task_section=section("Original Task", task.description),
            acceptance_section=section("Acceptance Criteria", task.acceptance),
            corrections_section=section("Corrections from Evaluator", corrections),
            code_context_section=section("Code Context", localization),
        )
    return render(
        "run",
        plan_section=section("Plan", plan),
        task_section=section("Task", task.description),
        acceptance_section=section("Acceptance Criteria", task.acceptance),
        code_context_section=section("Code Context", localization),
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
    timeout: int = 600,
) -> RunResult:
    model_id, provider_name = resolve_model(tier_level, ladder)
    variant = resolve_variant(task, effort_override, ladder)
    prompt = _build_prompt(task, plan, localization, corrections)

    provider = get_provider(provider_name)
    response = provider.run(
        prompt, model_id, variant=variant, session=opencode_session, timeout=timeout
    )

    return RunResult(
        text=response.text,
        model=model_id,
        tier=tier_level,
        tokens=response.tokens,
        cost=response.cost,
        raw=response.raw,
        opencode_session=response.session_id or opencode_session,
    )
