"""Multi-phase development: plan → single-shot run → gate, chained on the trajectory tree.

Each phase is a user-described improvement applied after the initial PRD run.
The user picks provider/model/effort for both the planner and the runner; the
phase plan is created, executed once (no retry loop), gate-checked, and the
trajectory is updated. The loop repeats as many times as the user wants.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from splinter.agents.gate import run_gate, task_languages
from splinter.agents.runner import RunResult, Task, _build_prompt
from splinter.memory.session import Session
from splinter.models.roster import Ladder, provider_for
from splinter.obs.agentic import record_exchange
from splinter.obs.trace import RunEntry, Trace
from splinter.providers.dispatch import run_text
from splinter.providers.registry import get_provider
from splinter.templating import load_standards, render, section

log = logging.getLogger("splinter.phases")


@dataclass
class PhaseConfig:
    """Model/effort selections for a single phase.

    Provider is derived from the model id via ``provider_for()``, matching the
    convention used everywhere else in the harness.
    """

    description: str
    plan_model: str
    plan_effort: str
    run_model: str
    run_effort: str


@dataclass
class PhaseResult:
    """Outcome of a single phase run."""

    phase_number: int
    description: str
    plan: str
    run_result: RunResult
    gate_passed: bool
    gate_output: str


def _phase_plan(session: Session, description: str, cfg: PhaseConfig, phase_num: int) -> str:
    """Create an implementation plan for a phase using the selected planner model."""
    standards = load_standards()
    prompt = render(
        "plan",
        task_section=section("Phase Task", description),
        acceptance_section=section(
            "Acceptance Criteria",
            "Implement the described changes. The code must pass all existing "
            "mechanical checks (lint, typecheck, tests).",
        ),
        code_context_section=section(
            "Code Context",
            "The codebase has already been modified by previous phases. "
            "Focus on the specific changes requested.",
        ),
        standards_section=section("Code Conventions", standards),
    )
    plan_text = run_text(
        prompt,
        cfg.plan_model,
        variant=cfg.plan_effort,
        timeout=None,
        session=session,
    )
    record_exchange(prompt, plan_text, model=cfg.plan_model)
    return plan_text


def _phase_task(description: str) -> Task:
    return Task(
        description=description,
        acceptance="Implement the described changes. Must pass mechanical checks.",
        effort="normal",
        reasoning_effort="auto",
    )


def phase_count(session: Session) -> int:
    """How many phases have already been planned in this session."""
    kdir = session.dir / "knowledge"
    if not kdir.exists():
        return 0
    count = 0
    for p in sorted(kdir.glob("phase-plan-*.md")):
        if p.stat().st_size > 0:
            count += 1
    return count


def _next_phase_number(session: Session) -> int:
    return phase_count(session) + 1


def run_phase(
    cfg: PhaseConfig,
    session: Session,
    ladder: Ladder,
    *,
    trace: Trace | None = None,
) -> PhaseResult:
    """Plan and execute a single phase (one-shot: plan → run → gate, no eval loop)."""
    if trace is None:
        existing = session.read("trace.md")
        trace = Trace.from_markdown(existing) if existing.strip() else Trace()

    n = _next_phase_number(session)
    log.info("phase %d · planning with %s (effort=%s)", n, cfg.plan_model, cfg.plan_effort)
    session.append("events.md", f"[plan] phase {n} · {cfg.plan_model} · {cfg.plan_effort}")

    plan = _phase_plan(session, cfg.description, cfg, n)

    plan_file = f"knowledge/phase-plan-{n}.md"
    session.write(
        plan_file,
        f"# Phase {n} Plan\n"
        f"- planner: {cfg.plan_model} ({cfg.plan_effort})\n"
        f"- runner: {cfg.run_model} ({cfg.run_effort})\n\n"
        f"{plan}\n",
    )
    # Also write as the main plan so the runner picks it up.
    session.write("knowledge/plan.md", f"# Phase {n} Plan\n\n{plan}\n")

    task = _phase_task(cfg.description)

    log.info("phase %d · running with %s (effort=%s)", n, cfg.run_model, cfg.run_effort)
    session.append(
        "events.md",
        f"[run] phase {n} · {cfg.run_model} · {cfg.run_effort}",
    )

    run_prompt = _build_prompt(
        task, plan, localization="", corrections="", is_continuation=False
    )
    provider = get_provider(provider_for(cfg.run_model))
    response = provider.run(
        run_prompt,
        cfg.run_model,
        variant=cfg.run_effort if cfg.run_effort != "auto" else None,
        timeout=None,
    )
    record_exchange(run_prompt, response.text, model=cfg.run_model)

    result = RunResult(
        text=response.text,
        model=cfg.run_model,
        tier=0,
        tokens=response.tokens,
        cost=response.cost,
        raw=response.raw,
        opencode_session=response.session_id,
    )

    trace.entries.append(
        RunEntry(
            model=result.model,
            tier=0,
            iteration=n,
            tokens=result.tokens,
            cost=result.cost,
            latency_s=0.0,
            task=0,
            role="phase",
        )
    )
    session.write("trace.md", trace.summary())

    session.append(
        "phase_loop.md",
        f"## Phase {n} · {cfg.run_model} · ${result.cost:.4f}\n\n",
    )

    session.write(
        f"runs/phase-{n}.md",
        f"# Phase {n} — {cfg.description.splitlines()[0]}\n"
        f"- model: {result.model}\n"
        f"- tokens: {result.tokens}\n"
        f"- cost: ${result.cost:.4f}\n\n"
        f"{result.text}\n",
    )

    langs = task_languages(task)
    gate_result = run_gate(session_dir=session.dir, languages=langs)
    gate_output = ""
    if not gate_result.passed:
        gate_output = "\n\n".join(
            f"### {name}\n{out}".rstrip()
            for name, passed, out in gate_result.checks
            if not passed and out
        )

    gate_status = "PASS" if gate_result.passed else "FAIL"
    log.info("phase %d · gate %s", n, gate_status)
    session.append(
        "phase_loop.md",
        f"gate {gate_status}"
        + (f"\n\n{gate_output}" if gate_output else "")
        + "\n\n",
    )

    phase_result = PhaseResult(
        phase_number=n,
        description=cfg.description,
        plan=plan,
        run_result=result,
        gate_passed=gate_result.passed,
        gate_output=gate_output,
    )

    _write_phase_trajectory(session, phase_result)
    return phase_result


def _write_phase_trajectory(session: Session, result: PhaseResult) -> None:
    """Record the phase result in phases.md for trajectory rendering."""
    now = datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S")
    lines = [
        f"- Phase {result.phase_number} · "
        f"{'PASS' if result.gate_passed else 'FAIL'} · "
        f"{result.run_result.model} · "
        f"${result.run_result.cost:.4f} · "
        f"[{now}]",
        f"  {result.description.splitlines()[0]}",
    ]
    session.append("phases.md", "\n".join(lines))
