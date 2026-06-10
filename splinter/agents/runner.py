"""Resolves a task+tier into a concrete model run via a provider strategy."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from splinter.enums import Effort, Variant
from splinter.models.roster import Ladder
from splinter.providers.registry import get_provider
from splinter.templating import render, section

log = logging.getLogger("splinter.runner")

_TRANSIENT_GAP_KINDS = frozenset(("rate_limit", "overload"))
_MAX_GAP_RETRIES = 5


@dataclass
class Task:
    """§6.3 Task schema.

    Core fields (planner): id, description, target_files, deps, effort, eval_skill.
    Runner extras: acceptance, reasoning_effort, suggested_tier.
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


@dataclass(frozen=True)
class RunResult:
    text: str
    model: str
    tier: int
    tokens: dict[str, int]
    cost: float
    raw: dict[str, Any]
    opencode_session: str | None = None


# Agentic code generation reasons poorly at low/minimal effort — the floor is
# `medium` (real-easy tasks); moderate work gets `high`, complex `xhigh` ("high+"),
# and the hardest `max`.
EFFORT_TO_VARIANT: dict[str, str] = {
    Effort.TRIVIAL: Variant.MEDIUM,
    Effort.NORMAL: Variant.HIGH,
    Effort.HARD: Variant.XHIGH,
    Effort.CRITICAL: Variant.MAX,
}


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


_MAX_FILE_CHARS = 6_000   # per file
_MAX_CTX_CHARS = 36_000  # total code context


def read_task_files(task: Task) -> str:
    """Expand task.target_files into file contents for the code-context section.

    Falls back to empty string when target_files is unset (single-task yaml run
    without prior localization); callers should then fall back to the localization
    summary string.
    """
    if not task.target_files:
        return ""
    parts: list[str] = []
    total = 0
    for i, path_str in enumerate(task.target_files):
        if total >= _MAX_CTX_CHARS:
            remaining = len(task.target_files) - i
            if remaining:
                parts.append(
                    f"*({remaining} more file(s) omitted — context cap reached)*"
                )
            break
        try:
            raw = Path(path_str).read_text()
        except OSError:
            continue
        snippet = raw[:_MAX_FILE_CHARS]
        note = " ← truncated" if len(raw) > _MAX_FILE_CHARS else ""
        parts.append(f"### {path_str}{note}\n```\n{snippet}\n```")
        total += len(snippet)
    return "\n\n".join(parts)


def _build_prompt(task: Task, plan: str, localization: str, corrections: str) -> str:
    # Use actual file contents when available; fall back to localization anchor
    # summary for single-task yaml runs that skipped the localization step.
    code_ctx = read_task_files(task) or localization
    if corrections:
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
) -> RunResult:
    model_id, provider_name = resolve_model(tier_level, ladder)
    variant = resolve_variant(task, effort_override, ladder, tier_level)
    if timeout is None:
        timeout = ladder.tier_timeout(tier_level)
    prompt = _build_prompt(task, plan, localization, corrections)

    provider = get_provider(provider_name)
    response = None
    for attempt in range(_MAX_GAP_RETRIES + 1):
        try:
            response = provider.run(
                prompt, model_id, variant=variant, session=opencode_session, timeout=timeout
            )
            break
        except Exception as exc:
            from splinter.providers.base import ProviderGapError
            if not isinstance(exc, ProviderGapError) or exc.kind not in _TRANSIENT_GAP_KINDS:
                raise
            if attempt >= _MAX_GAP_RETRIES:
                raise
            wait = min(5 * (2 ** attempt), 60)
            log.warning(
                "provider gap (%s) on attempt %d/%d — retrying in %ds",
                exc.kind, attempt + 1, _MAX_GAP_RETRIES, wait,
            )
            time.sleep(wait)
    assert response is not None

    return RunResult(
        text=response.text,
        model=model_id,
        tier=tier_level,
        tokens=response.tokens,
        cost=response.cost,
        raw=response.raw,
        opencode_session=response.session_id or opencode_session,
    )
