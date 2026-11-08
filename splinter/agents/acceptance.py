"""Per-brief acceptance check: deterministic gate plus optional light LLM eval.

The gate always runs first; the extra LLM check (``gate+eval`` / ``gate+orchestrator``)
judges only the brief's diff against its acceptance lines and speaks the simple
``VERDICT: PASS|FAIL`` protocol. Any LLM failure degrades to a failed result —
acceptance never crashes the strategy that calls it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from splinter.agents.gate import GateResult, run_gate
from splinter.enums import Decision
from splinter.providers.dispatch import run_text
from splinter.strategies.base import EvalVerdict
from splinter.templating import render

if TYPE_CHECKING:
    from collections.abc import Sequence

    from splinter.models.roster import Ladder
    from splinter.obs.trace import Trace

_DETAIL_LIMIT = 400

_VERDICT_TOKEN_RE = re.compile(r"(?i)\b(PASS|FAIL)\b")


class AcceptanceMode(StrEnum):
    GATE = "gate"
    GATE_EVAL = "gate+eval"
    GATE_ORCHESTRATOR = "gate+orchestrator"


_MODE_ALIASES: dict[str, AcceptanceMode] = {
    "gate": AcceptanceMode.GATE,
    "gate+eval": AcceptanceMode.GATE_EVAL,
    "gate_eval": AcceptanceMode.GATE_EVAL,
    "gate+orchestrator": AcceptanceMode.GATE_ORCHESTRATOR,
    "gate_orchestrator": AcceptanceMode.GATE_ORCHESTRATOR,
}


def parse_acceptance_mode(value: str | None) -> AcceptanceMode:
    """Parse an ``agents.acceptance_check`` config value into a mode.

    ``None``/blank defaults to ``gate+eval``; matching is case-insensitive and
    accepts the underscore aliases of the compound modes.
    """
    if value is None or not value.strip():
        return AcceptanceMode.GATE_EVAL
    mode = _MODE_ALIASES.get(value.strip().lower())
    if mode is None:
        raise ValueError(
            f"unknown acceptance mode {value!r} "
            "(expected gate, gate+eval or gate+orchestrator)"
        )
    return mode


@dataclass(frozen=True)
class AcceptanceResult:
    passed: bool
    mode: AcceptanceMode
    gate_passed: bool
    detail: str = ""
    verdict: EvalVerdict | None = None


def _clip(text: str, limit: int = _DETAIL_LIMIT) -> str:
    return text if len(text) <= limit else text[:limit]


def _acceptance_text(lines: Sequence[str]) -> str:
    return "\n".join(f"- {line.strip()}" for line in lines if line.strip())


def _gate_failure_text(gate: GateResult) -> str:
    parts = [f"{name}: {output or 'failed'}" for name, ok, output in gate.checks if not ok]
    return _clip("; ".join(parts) if parts else "gate failed")


def _parse_verdict(text: str) -> EvalVerdict:
    """Read the first VERDICT line: PASS → PASS, FAIL/malformed/missing → RETRY."""
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if not line.upper().strip().startswith("VERDICT:"):
            continue
        prior = [part.strip() for part in lines[:idx]]
        justification = " ".join(part for part in prior if part)
        token = _VERDICT_TOKEN_RE.search(line.split(":", 1)[1])
        if token is not None and token.group(1).upper() == "PASS":
            return EvalVerdict(
                decision=Decision.PASS,
                reason=justification or "acceptance satisfied",
                raw=text,
            )
        detail = _clip(justification or line.strip() or "VERDICT: FAIL")
        return EvalVerdict(
            decision=Decision.RETRY,
            reason=detail,
            corrections=detail,
            raw=text,
        )
    detail = "no VERDICT line in acceptance response"
    return EvalVerdict(decision=Decision.RETRY, reason=detail, corrections=detail, raw=text)


def check_acceptance(
    *,
    goal: str,
    acceptance: Sequence[str],
    mode: AcceptanceMode,
    project_dir: str = ".",
    languages: set[str] | None = None,
    ladder: Ladder | None = None,
    diff: str = "",
    trace: Trace | None = None,
    task_index: int = 0,
    orchestrator_notes: str = "",
) -> AcceptanceResult:
    """Check a brief: mechanical gate first, then (per mode) a light LLM check of
    the diff against the acceptance lines.

    ``gate+eval`` routes to the ladder's eval model at low effort; ``gate+orchestrator``
    routes to the planner model at its effort under the ``orchestrator`` role.
    """
    gate = run_gate(project_dir=project_dir, languages=languages)
    if not gate.passed:
        return AcceptanceResult(
            passed=False,
            mode=mode,
            gate_passed=False,
            detail=_gate_failure_text(gate),
        )
    if mode is AcceptanceMode.GATE:
        return AcceptanceResult(passed=True, mode=mode, gate_passed=True)
    if mode is AcceptanceMode.GATE_ORCHESTRATOR:
        role = "orchestrator"
        model = ladder.planner_model if ladder else "sonnet"
        variant = (ladder.planner_effort if ladder else "") or "low"
        timeout: int | None = ladder.planner_timeout if ladder else None
    else:
        role = "eval"
        model = ladder.eval_model if ladder else "sonnet"
        variant = "low"
        timeout = ladder.eval_timeout if ladder else None
    prompt = render(
        "acceptance_eval",
        goal=goal,
        acceptance=_acceptance_text(acceptance),
        diff=diff,
        orchestrator_notes_section=orchestrator_notes,
    )
    try:
        raw = run_text(
            prompt,
            model,
            variant=variant,
            output_format="json",
            timeout=timeout,
            trace=trace,
            task_index=task_index,
            role=role,
            pool=ladder.step_pool(
                "planner" if mode == AcceptanceMode.GATE_ORCHESTRATOR else "eval"
            ) if ladder is not None else None,
        )
    except Exception as exc:
        return AcceptanceResult(
            passed=False,
            mode=mode,
            gate_passed=True,
            detail=_clip(f"{role} acceptance check failed: {exc}"),
        )
    verdict = _parse_verdict(raw)
    return AcceptanceResult(
        passed=verdict.passed,
        mode=mode,
        gate_passed=True,
        detail="" if verdict.passed else verdict.reason,
        verdict=verdict,
    )
