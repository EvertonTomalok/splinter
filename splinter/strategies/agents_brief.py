"""One brief on its effort chain: launch, classify, generate, accept, fix.

Infra failures (STALL/LIMIT/CRASH/EMPTY/MISSING) reroute the brief to the next
engine of its chain without burning an attempt; when every engine of the chain
is cooling down the brief waits for the first one instead of giving up. A
workhorse that reports ``BLOCKED`` hands the brief to the orchestrator at
once. A failed acceptance sends a fix round to the same engine session and
burns one attempt; after ``max_attempts`` the orchestrator takes over.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from splinter.agents.acceptance import AcceptanceMode, check_acceptance
from splinter.agents.brief_io import (
    engine_prompt,
    fix_prompt,
    read_report,
    worktree_diff,
    write_engine_log,
)
from splinter.agents.codegen import run_codegen
from splinter.agents.laws import brief_document, workhorse_contract
from splinter.agents.progress import FAIL, PASS, RETRY
from splinter.agents.refiner import Brief
from splinter.agents.runner import RunResult
from splinter.engines.catalog import Engine
from splinter.engines.failures import classify, find_limit, find_result
from splinter.engines.launch import EngineLaunch, run_engine
from splinter.engines.pool import PoolExhausted
from splinter.engines.routing import chain_for_effort, start_tier_for
from splinter.enums import Effort
from splinter.memory.session import Session
from splinter.models.roster import Ladder
from splinter.obs.trace import Trace, log_run
from splinter.procreg import IdleTimeoutError, ShuttingDownError, stop_requested
from splinter.strategies.agents_ctx import BriefCtx
from splinter.templating import load_standards
from splinter.vcs.guard import guarded_env
from splinter.vcs.worktree import WorktreeHandle

log = logging.getLogger("splinter.loop")

_sleep = time.sleep
_MIN_WAIT_S = 5.0
_MAX_FLOOR_S = 120.0
STOP_REASON = "the run was stopped"


@dataclass(frozen=True)
class BriefOutcome:
    """``accepted``: merge it. Otherwise the orchestrator takes over, told ``reason``."""

    accepted: bool
    result: RunResult | None = None
    engine: str = ""
    reason: str = ""


@dataclass
class _State:
    burned: int = 0
    infra_rounds: int = 0
    engine_session: str | None = None
    last_engine: str = ""
    started: float = 0.0
    sent: int = 0
    exclude: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class _Brief:
    brief: Brief
    wt: WorktreeHandle
    story: int
    tier: int
    doc: Path


def run_brief(
    brief: Brief,
    wt: WorktreeHandle,
    session: Session,
    ladder: Ladder,
    trace: Trace,
    ctx: BriefCtx,
) -> BriefOutcome:
    effort = brief.effort or Effort.NORMAL
    chain = chain_for_effort(ladder, effort)
    tier = _start_tier(ladder, effort)
    job = _Brief(brief, wt, ctx.story_of(brief.id), tier, _doc(brief, wt, ctx))
    contract = workhorse_contract(laws=load_standards(), plan_ref=str(ctx.run_dir))
    state = _State()
    while True:
        if stop_requested():
            return BriefOutcome(False, reason=STOP_REASON)
        engine = _pick(ctx, chain, state)
        if engine is None:
            state.infra_rounds += 1
            if state.infra_rounds > ctx.max_attempts:
                return BriefOutcome(False, reason="every engine of the chain kept failing")
            _wait_for_chain(ctx, chain, brief.id, state.infra_rounds)
            state.exclude.clear()
            continue
        _add_guidance(session, ctx, job)
        n = ctx.attempts.start(job.story, brief.id, engine.key, job.tier)
        with ctx.session_lock:
            session.set_status("running", stage="run", task_index=job.story - 1)
        launch = _launch(ctx, session, trace, job, engine, state, contract, n)
        if launch is None:
            continue
        outcome = _judge(session, ladder, trace, ctx, job, engine, launch, n, state)
        if outcome is not None:
            return outcome


def _pick(ctx: BriefCtx, chain: tuple[Engine, ...], state: _State) -> Engine | None:
    try:
        return ctx.pool.acquire(chain, state.exclude)
    except PoolExhausted:
        return None


def _wait_for_chain(
    ctx: BriefCtx, chain: tuple[Engine, ...], brief_id: str, infra_round: int
) -> None:
    floor = min(_MIN_WAIT_S * 2 ** (infra_round - 1), _MAX_FLOOR_S)
    wait = max(ctx.pool.wait_s(chain), floor)
    log.warning("agents: every engine for %s is cooling down — waiting %.0fs", brief_id, wait)
    _sleep(wait)


def _launch(
    ctx: BriefCtx,
    session: Session,
    trace: Trace,
    job: _Brief,
    engine: Engine,
    state: _State,
    contract: str,
    n: int,
) -> EngineLaunch | None:
    """Run the engine once; an infra failure reroutes and returns ``None``."""
    resume = (
        state.engine_session
        if engine.key == state.last_engine and engine.supports_continue
        else None
    )
    document = job.doc.read_text()
    prompt = (
        engine_prompt(job.brief.id, document, contract)
        if resume is None
        else fix_prompt(job.brief.id, document[state.sent :])
    )
    state.started = time.monotonic()
    launch, failure = _call_engine(ctx, job, engine, prompt, resume)
    if launch is None:
        write_engine_log(ctx.run_dir, job.brief.id, n, engine.key, f"infra failure: {failure}")
    else:
        write_engine_log(ctx.run_dir, job.brief.id, n, engine.key, launch.text)
        failure = _launch_failure(trace, job, engine, launch, n, state)
    if failure or launch is None:
        ctx.attempts.end(job.story, n, RETRY, failure)
        _reroute(ctx, session, job.brief.id, engine, failure)
        state.exclude.add(engine.key)
        return None
    ctx.pool.report_success(engine.key)
    kept = state.engine_session if engine.key == state.last_engine else None
    state.engine_session = launch.session_id or kept
    state.last_engine = engine.key
    state.sent = len(document)
    return launch


def _call_engine(
    ctx: BriefCtx, job: _Brief, engine: Engine, prompt: str, resume: str | None
) -> tuple[EngineLaunch | None, str]:
    """The engine's launch, or ``None`` and why it never returned one."""
    try:
        launch = run_engine(
            engine,
            prompt,
            cwd=str(job.wt.path),
            env=guarded_env(ctx.guard),
            session=resume,
            timeout=ctx.timeout,
            idle_timeout=ctx.stall_min * 60,
            effort=engine.default_effort,
        )
    except IdleTimeoutError:
        return None, "STALL: idle watchdog"
    except FileNotFoundError:
        return None, "MISSING: engine binary absent"
    except ShuttingDownError:
        raise
    except RuntimeError as exc:
        return None, f"{'LIMIT' if find_limit('', str(exc)) else 'CRASH'}: {exc}"
    return launch, ""


def _launch_failure(
    trace: Trace, job: _Brief, engine: Engine, launch: EngineLaunch, n: int, state: _State
) -> str:
    """The failure of a launch that returned; its tokens were spent, so it is traced."""
    kind = classify(
        returncode=None,
        stdout=launch.text,
        stderr="",
        has_result=find_result(launch.text) is not None,
    )
    if kind is None:
        return ""
    log_run(trace, _result(launch, job, engine, state), iteration=n, task=job.story - 1)
    return f"{kind.kind.value}: {kind.detail}"


def _result(launch: EngineLaunch, job: _Brief, engine: Engine, state: _State) -> RunResult:
    return RunResult(
        text=launch.text,
        model=launch.model,
        tier=job.tier,
        tokens=launch.tokens or {},
        cost=launch.cost,
        raw={"engine": engine.key, "brief": job.brief.id},
        opencode_session=state.engine_session,
        cost_indeterminate=launch.cost_indeterminate,
        latency_s=time.monotonic() - state.started,
    )


def _judge(
    session: Session,
    ladder: Ladder,
    trace: Trace,
    ctx: BriefCtx,
    job: _Brief,
    engine: Engine,
    launch: EngineLaunch,
    n: int,
    state: _State,
) -> BriefOutcome | None:
    """Blocked, generated, accepted — or a fix round (``None`` loops again)."""
    result = _result(launch, job, engine, state)
    log_run(trace, result, iteration=n, task=job.story - 1)
    status = (find_result(launch.text) or ("DONE", ""))[0].upper()
    if status == "BLOCKED":
        reason = _first_lines(read_report(job.wt.path)) or "the workhorse reported BLOCKED"
        ctx.attempts.end(job.story, n, RETRY, f"BLOCKED: {reason}")
        return BriefOutcome(False, result, engine.key, f"The workhorse was blocked: {reason}")
    generated = run_codegen(job.wt.path, ctx.codegen)
    if generated.ok:
        accepted = check_acceptance(
            goal=job.brief.goal,
            acceptance=job.brief.acceptance,
            mode=ctx.mode,
            project_dir=str(job.wt.path),
            ladder=ladder,
            diff_source=lambda: worktree_diff(job.wt.path),
            trace=trace,
            task_index=job.story - 1,
            orchestrator_notes=(
                ctx.acceptance_notes(session, job.story)
                if ctx.mode is not AcceptanceMode.GATE
                else ""
            ),
        )
        if accepted.passed:
            ctx.attempts.end(job.story, n, PASS, "accepted")
            return BriefOutcome(True, result, engine.key)
        detail = accepted.detail
    else:
        detail = f"code generation failed: {generated.failure}"
    state.burned += 1
    ctx.attempts.end(job.story, n, RETRY, _first_lines(detail, 1))
    if state.burned >= ctx.max_attempts:
        return BriefOutcome(False, result, engine.key, f"Acceptance kept failing: {detail}")
    _fix_round(session, ctx, job, engine.key, state.burned, detail)
    return None


def merge_failed(ctx: BriefCtx, story: int, brief_id: str, engine: str) -> None:
    n = ctx.attempts.start(story, brief_id, engine, 0)
    ctx.attempts.end(story, n, FAIL, "merge conflict: worktree kept")


def _fix_round(
    session: Session, ctx: BriefCtx, job: _Brief, engine: str, round_no: int, detail: str
) -> None:
    lines = "\n".join(f"- {line}" for line in detail.splitlines()[:10])
    with ctx.session_lock:
        job.doc.write_text(
            job.doc.read_text() + f"\n\n## Fix round {round_no}\n\nAcceptance failed:\n\n{lines}\n"
        )
        session.append(
            "events.md", f"[AGENTS] brief {job.brief.id} engine={engine} fix round {round_no}"
        )


def _add_guidance(session: Session, ctx: BriefCtx, job: _Brief) -> None:
    ctx.take_guidance(session)
    guidance = ctx.guidance_for(job.story)
    with ctx.session_lock:
        current = job.doc.read_text()
        fresh = [text for text in guidance if text not in current]
        if fresh:
            job.doc.write_text(
                current + "\n\n## User guidance\n\n" + "\n---\n".join(fresh) + "\n"
            )


def _reroute(ctx: BriefCtx, session: Session, brief_id: str, engine: Engine, reason: str) -> None:
    ctx.pool.report_infra_failure(engine.key)
    log.warning("agents: brief %s rerouting off %s — %s", brief_id, engine.key, reason)
    with ctx.session_lock:
        session.append(
            "events.md", f"[AGENTS] brief {brief_id} engine={engine.key} infra failure — {reason}"
        )


def _doc(brief: Brief, wt: WorktreeHandle, ctx: BriefCtx) -> Path:
    path = ctx.run_dir / "briefs" / f"{brief.id}.md"
    if not path.exists():
        path.write_text(
            brief_document(
                ident=brief.id,
                goal=brief.goal,
                files=brief.files,
                sketch=brief.sketch,
                acceptance=brief.acceptance,
                worktree=str(wt.path),
                branch=wt.branch,
                run_dir=str(ctx.run_dir),
            )
        )
    return path


def _start_tier(ladder: Ladder, effort: str) -> int:
    try:
        return start_tier_for(ladder, effort)
    except (AttributeError, KeyError, ValueError):
        return 0


def _first_lines(text: str, count: int = 3) -> str:
    return " ".join(line.strip() for line in text.strip().splitlines()[:count] if line.strip())
