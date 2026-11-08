"""The ``agents`` strategy — plan-with-agents ported into splinter.

Refines the PRD into waves of small briefs, then runs each one on the model
chain its effort picks (effort → ladder tier → ordered models; the engine CLI —
mcode, kimi, agy, claude, codex, opencode — comes from the model prefix) in
per-brief git worktrees forked from a dedicated integration branch. Engines are
guarded by PATH shims (they may only edit files); splinter generates code,
validates (gate + acceptance eval) and merges each brief back with ``--no-ff``.
The per-brief loop lives in :mod:`splinter.strategies.agents_brief`; when it
gives up, the orchestrator takes the brief over in the same worktree with the
brief as its plan.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from dataclasses import asdict, replace
from pathlib import Path

from splinter.agents.acceptance import AcceptanceMode, parse_acceptance_mode
from splinter.agents.brief_io import keep_report, saved_briefs_usable, slug_from
from splinter.agents.progress import (
    CHECKPOINT,
    FAIL,
    PASS,
    AttemptLog,
    brief_stories,
    read_done,
    stories_of,
    write_done,
    write_stories,
)
from splinter.agents.refiner import Brief, brief_to_task, brief_waves, refine_briefs
from splinter.agents.runner import RunResult, Task
from splinter.configure import AgentsOptions, configured_agents_options, configured_timeout
from splinter.engines.health import EngineHealth, probe
from splinter.engines.pool import EnginePool
from splinter.engines.routing import chain_for_effort
from splinter.enums import Effort
from splinter.memory.knowledge import KnowledgeStore
from splinter.memory.orchestrator_notes import notes_for
from splinter.memory.session import Session
from splinter.models.roster import Ladder
from splinter.obs.trace import Trace
from splinter.procreg import stop_requested
from splinter.scheduling import BudgetPool, DagScheduler, TaskState
from splinter.strategies.agents_brief import BriefOutcome, merge_failed, run_brief
from splinter.strategies.agents_ctx import BriefCtx, DirectLoopSettings
from splinter.strategies.direct import DirectStrategy, TaskOutcome
from splinter.strategies.registry import register
from splinter.task_plan import PLAN_INDEX_FILE
from splinter.vcs.guard import setup_guard
from splinter.vcs.worktree import (
    WorktreeHandle,
    WorktreeMergeConflict,
    commit_worktree,
    create_integration_worktree,
    create_worktree,
    merge_no_ff,
    reattach_worktree,
    teardown_worktree,
    worktree_supported,
)

log = logging.getLogger("splinter.loop")

_MIN_ENGINE_TIMEOUT_S = 2700
_PROBE_WORKERS = 4

_BriefFuture = Future[tuple[RunResult | None, bool]]


@register
class AgentsStrategy(DirectStrategy):
    """Waves of guarded workhorse briefs on a pool of engines with fallback."""

    name = "agents"
    aliases = ["splinter"]
    summary = (
        "multi-story: refines briefs into waves, runs each on cheap "
        "engines in its own worktree, merges back"
    )

    def execute(
        self,
        tasks: list[Task],
        session: Session,
        ladder: Ladder,
        *,
        effort: str | None = None,
        budget: float | None = None,
        max_iterations: int = 5,
        localization: str = "",
        eval_skill: str | None = None,
        cowabunga: bool = False,
        resume: bool = False,
        claude_runner_fallback: bool = False,
        user_guidance: str | None = None,
        jump_premium: bool = False,
        skip_planner: bool = False,
        skip_eval: bool = False,
        force_replan: bool = False,
        parallel: bool = False,
        max_concurrency: int | None = None,
    ) -> list[RunResult]:
        opts = configured_agents_options()
        trace = Trace.from_jsonl(session) if resume else Trace(session=session)
        knowledge = KnowledgeStore(session)
        run_dir = session.dir / "agents-run"
        for sub in ("briefs", "logs", "reports"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)
        direct = DirectLoopSettings(
            effort=effort,
            budget=budget,
            max_iterations=max_iterations,
            localization=localization,
            eval_skill=eval_skill,
            skip_eval=skip_eval,
        )
        if not worktree_supported():
            log.warning("agents: no git worktree support — falling back to direct loop")
            return self._direct_fallback(tasks, session, ladder, trace, knowledge, direct)

        done = read_done(session.read(CHECKPOINT)) if resume else set()
        briefs = self._briefs(tasks, session, ladder, trace, resume, done)
        slug = slug_from(tasks)
        self._write_plan_doc(run_dir, slug, brief_waves(briefs))
        self._init_state(run_dir, slug)
        stories = stories_of(tasks)
        story_map = brief_stories(briefs, stories)
        write_stories(session, stories, story_map)
        attempts = AttemptLog(session)
        attempts.close_interrupted()
        session.set_status("running", stage="run", task_total=len(tasks))

        saved = session.read_worktrees() if resume else {}
        ctx = BriefCtx(
            slug=slug,
            run_dir=run_dir,
            guard=setup_guard(run_dir),
            pool=EnginePool(self._health(opts, ladder)),
            integration=self._integration(session, slug, saved),
            mode=AcceptanceMode.GATE if skip_eval else parse_acceptance_mode(opts.acceptance_check),
            stall_min=opts.stall_min,
            max_attempts=opts.max_attempts,
            timeout=max(configured_timeout(), _MIN_ENGINE_TIMEOUT_S),
            direct=direct,
            attempts=attempts,
            stories=story_map,
            story_count=len(stories),
            codegen=opts.codegen,
            concurrency=max_concurrency if parallel else 1,
            worktrees=saved,
        )
        results = self._dispatch(briefs, session, ladder, trace, knowledge, ctx, done)
        if any(b.id not in done for b in briefs):
            raise RuntimeError("agents: incomplete briefs; integration worktree kept")
        self._finish(session, ctx.integration, slug)
        session.set_status("running", task_index=len(tasks), task_total=len(tasks))
        return results

    def _briefs(
        self,
        tasks: list[Task],
        session: Session,
        ladder: Ladder,
        trace: Trace,
        resume: bool,
        done: set[str],
    ) -> list[Brief]:
        saved = session.read("agents-run/briefs.json") if resume else ""
        if saved and saved_briefs_usable(saved, tasks, done):
            return [Brief(**item) for item in json.loads(saved)]
        session.set_status("running", stage="plan")
        briefs = refine_briefs(
            tasks,
            ladder,
            prd_text=session.read(PLAN_INDEX_FILE) or session.read("prd.md"),
            session=session,
            trace=trace,
            orchestrator_notes=notes_for(session, "refine", "agents"),
            reuse_progress=resume,
            max_workers=configured_agents_options().refine_workers,
        )
        session.write("agents-run/briefs.json", json.dumps([asdict(b) for b in briefs]))
        return briefs

    def _health(self, opts: AgentsOptions, ladder: Ladder) -> EngineHealth:
        health = EngineHealth()
        if opts.probe_on_start:
            self._probe_into(health, ladder)
        return health

    def _integration(
        self, session: Session, slug: str, saved: dict[str, dict[str, str]]
    ) -> WorktreeHandle:
        attached = reattach_worktree(f"{slug}-int") if saved.get(f"{slug}-int") else None
        integration = attached or create_integration_worktree(slug)
        session.set_worktree(f"{slug}-int", str(integration.path), integration.branch)
        return integration

    def _direct_fallback(
        self,
        tasks: list[Task],
        session: Session,
        ladder: Ladder,
        trace: Trace,
        knowledge: KnowledgeStore,
        direct: DirectLoopSettings,
    ) -> list[RunResult]:
        session.append("loop.md", "## No worktree support: direct fallback\n\n")
        results: list[RunResult] = []
        for i, task in enumerate(tasks):
            result = self._run_task_loop(
                task,
                session,
                ladder,
                trace,
                knowledge,
                task_index=i,
                effort=direct.effort,
                budget=direct.budget,
                max_iterations=direct.max_iterations,
                localization=task.filtered_context or direct.localization,
                eval_skill=direct.eval_skill,
                skip_eval=direct.skip_eval,
                outcome=TaskOutcome(),
            )
            if result is not None:
                results.append(result)
        return results

    def _dispatch(
        self,
        briefs: list[Brief],
        session: Session,
        ladder: Ladder,
        trace: Trace,
        knowledge: KnowledgeStore,
        ctx: BriefCtx,
        done: set[str],
    ) -> list[RunResult]:
        cap = ctx.max_concurrency()
        budget_pool = BudgetPool(_budget=ctx.direct.budget)
        results: list[RunResult] = []
        pending = {b.id: b for b in briefs if b.id not in done}
        scheduler = DagScheduler([brief_to_task(b) for b in pending.values()])
        futures: dict[_BriefFuture, Brief] = {}
        stop = False
        with ThreadPoolExecutor(max_workers=cap) as executor:
            while not scheduler.is_done() or futures:
                ctx.take_guidance(session)
                stop = stop or budget_pool.exhausted(trace.total_cost) or stop_requested()
                for task in [] if stop else scheduler.ready():
                    if len(futures) >= cap or budget_pool.exhausted(trace.total_cost):
                        break
                    scheduler.mark_running(task.id)
                    brief = pending[task.id]
                    future = executor.submit(
                        self._brief, brief, session, ladder, trace, knowledge, ctx
                    )
                    futures[future] = brief
                if not futures:
                    break
                finished, _ = wait(futures, timeout=5, return_when=FIRST_COMPLETED)
                for future in finished:
                    self._settle(future, futures.pop(future), session, scheduler, done, results)
        self._note_blocked(session, scheduler, ctx)
        return results

    def _settle(
        self,
        future: _BriefFuture,
        brief: Brief,
        session: Session,
        scheduler: DagScheduler,
        done: set[str],
        results: list[RunResult],
    ) -> None:
        try:
            result, merged = future.result()
        except Exception as exc:
            log.exception("agents: brief %s crashed", brief.id)
            session.append("events.md", f"[AGENTS] brief {brief.id} crashed: {exc}")
            scheduler.mark_failed(brief.id)
            return
        if result is not None:
            results.append(result)
        if merged:
            scheduler.mark_passed(brief.id)
            done.add(brief.id)
            write_done(session, done)
        else:
            log.warning("agents: brief %s did not merge — blocking dependents", brief.id)
            scheduler.mark_failed(brief.id)

    def _note_blocked(self, session: Session, scheduler: DagScheduler, ctx: BriefCtx) -> None:
        blocked = [tid for tid, st in scheduler.all_states().items() if st == TaskState.BLOCKED]
        if blocked:
            lines = "\n".join(f"- {tid}: {scheduler.blocked_reason(tid)}" for tid in blocked)
            with ctx.session_lock:
                session.append("agents-run/STATE.md", f"## Blocked briefs\n{lines}\n\n")

    def _brief(
        self,
        brief: Brief,
        session: Session,
        ladder: Ladder,
        trace: Trace,
        knowledge: KnowledgeStore,
        ctx: BriefCtx,
    ) -> tuple[RunResult | None, bool]:
        wt = self._worktree(brief, session, ctx)
        outcome = run_brief(brief, wt, session, ladder, trace, ctx)
        if not outcome.accepted:
            if stop_requested():
                log.info("agents: brief %s left for the next run — stop requested", brief.id)
                return outcome.result, False
            return self._assume(brief, outcome, session, ladder, trace, knowledge, wt, ctx)
        report = keep_report(wt.path, ctx.run_dir / "reports" / f"{brief.id}.md")
        result = outcome.result
        if report and result is not None:
            result = replace(result, text=report)
        return result, self._merge(ctx, session, brief, outcome.engine, wt)

    def _worktree(self, brief: Brief, session: Session, ctx: BriefCtx) -> WorktreeHandle:
        with ctx.git_lock:
            task_id = f"{ctx.slug}-{brief.id}"
            attached = reattach_worktree(task_id) if ctx.worktrees.get(brief.id) else None
            wt = attached or create_worktree(task_id, base_ref=ctx.integration.branch)
        session.set_worktree(brief.id, str(wt.path), wt.branch)
        return wt

    def _assume(
        self,
        brief: Brief,
        outcome: BriefOutcome,
        session: Session,
        ladder: Ladder,
        trace: Trace,
        knowledge: KnowledgeStore,
        wt: WorktreeHandle,
        ctx: BriefCtx,
    ) -> tuple[RunResult | None, bool]:
        """The orchestrator finishes the brief in its worktree, the brief as its plan."""
        story = ctx.story_of(brief.id)
        log.warning("agents: assuming brief %s — %s", brief.id, outcome.reason[:200])
        with ctx.session_lock:
            session.append(
                "events.md", f"[AGENTS] brief {brief.id} assumed by orchestrator: {outcome.reason}"
            )
        doc = ctx.run_dir / "briefs" / f"{brief.id}.md"
        plan = (doc.read_text() if doc.exists() else brief.goal) + (
            f"\n\n## Why the orchestrator took over\n\n{outcome.reason}\n"
        )
        n = ctx.attempts.start(story, brief.id, "orchestrator", 0)
        task_outcome = TaskOutcome()
        result = self._run_task_loop(
            brief_to_task(brief),
            session,
            ladder,
            trace,
            knowledge,
            task_index=story - 1,
            effort=ctx.direct.effort,
            budget=ctx.direct.budget,
            max_iterations=ctx.direct.max_iterations,
            localization=ctx.direct.localization,
            eval_skill=ctx.direct.eval_skill,
            skip_eval=ctx.direct.skip_eval,
            lock=ctx.session_lock,
            cwd=str(wt.path),
            outcome=task_outcome,
            plan_text=plan,
        )
        if not task_outcome.passed:
            ctx.attempts.end(story, n, FAIL, "the orchestrator could not finish it")
            return result, False
        ctx.attempts.end(story, n, PASS, "finished by the orchestrator")
        return result, self._merge(ctx, session, brief, "orchestrator", wt)

    def _merge(
        self,
        ctx: BriefCtx,
        session: Session,
        brief: Brief,
        engine: str,
        wt: WorktreeHandle,
    ) -> bool:
        with ctx.git_lock:
            try:
                commit_worktree(wt, f"splinter: brief {brief.id} ({engine})")
                merge_no_ff(wt, ctx.integration.path)
                teardown_worktree(wt)
            except WorktreeMergeConflict as exc:
                log.error(
                    "agents: merge conflict for %s — worktree KEPT at %s: %s",
                    brief.id, wt.path, exc,
                )
                merged = False
                line = f"- CONFLICT {brief.id} ({engine}) — worktree kept at {wt.path}\n"
            else:
                merged = True
                line = f"- merged {brief.id} ({engine}) into {ctx.integration.branch}\n"
        with ctx.session_lock:
            session.append("agents-run/STATE.md", line)
            session.append(
                "events.md", f"[AGENTS] brief {brief.id} engine={engine} merged={merged}"
            )
        if not merged:
            merge_failed(ctx, ctx.story_of(brief.id), brief.id, engine)
        return merged

    def _probe_into(self, health: EngineHealth, ladder: Ladder) -> None:
        engines = {e.key: e for effort in Effort for e in chain_for_effort(ladder, effort)}
        with ThreadPoolExecutor(max_workers=_PROBE_WORKERS) as pool:
            probes = {pool.submit(probe, engine): engine.key for engine in engines.values()}
            for future in as_completed(probes):
                key = probes[future]
                try:
                    result = future.result()
                except Exception as exc:
                    log.warning("agents: probe of %s crashed: %s", key, exc)
                    continue
                if not result.up:
                    log.warning("agents: engine %s down (%s) — breaker open", key, result.detail)
                    health.trip(key)

    def _write_plan_doc(self, run_dir: Path, slug: str, waves: list[list[Brief]]) -> None:
        lines = [f"# Plan — {slug}", "", "## Waves & Briefs", ""]
        lines += [f"- Wave {i}: {', '.join(b.id for b in wave)}" for i, wave in enumerate(waves, 1)]
        (run_dir / "README.md").write_text("\n".join([*lines, ""]))

    def _init_state(self, run_dir: Path, slug: str) -> None:
        state = run_dir / "STATE.md"
        if not state.exists():
            state.write_text(
                f"# Run state — {slug}\n\n## Merged\n\n## Verification\n\n## Orchestrator notes\n"
            )

    def _finish(self, session: Session, integration: WorktreeHandle, slug: str) -> None:
        from splinter.agents.gate import run_gate

        gate = run_gate(project_dir=str(integration.path))
        if not gate.passed:
            log.error("agents: final gate on integration branch failed — worktree KEPT")
            session.append("agents-run/STATE.md", "## Verification\n- final gate FAILED\n")
            raise RuntimeError("agents: final integration gate failed; worktree kept")
        session.append("agents-run/STATE.md", "## Verification\n- final gate passed\n")
        try:
            merge_no_ff(integration, Path.cwd(), message=f"splinter: agents run {slug}")
        except WorktreeMergeConflict as exc:
            log.error(
                "agents: integration merge conflict — worktree KEPT at %s: %s",
                integration.path, exc,
            )
            raise
        teardown_worktree(integration)
        log.info("agents: integration branch merged into the main tree")
