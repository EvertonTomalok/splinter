"""The ``agents`` strategy — plan-with-agents ported into splinter.

Refines the PRD into waves of small briefs, then runs each one on the model
chain its effort picks (effort → ladder tier → ordered models; the engine CLI —
mcode, kimi, agy, claude, codex, opencode — comes from the model prefix) in
per-brief git worktrees forked from a dedicated integration branch. Engines are
guarded by PATH shims (they may only edit files); splinter commits, validates
(gate + acceptance eval) and merges each brief back with ``--no-ff``. Infra
failures (STALL/LIMIT/CRASH/EMPTY/MISSING) trip the engine's circuit breaker
and reroute the same brief to the next model of its chain without burning attempts; after
the attempt budget the orchestrator assumes the brief via the direct loop.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path

from splinter.agents.acceptance import AcceptanceMode, check_acceptance, parse_acceptance_mode
from splinter.agents.laws import REPORT_PATH, brief_document, workhorse_contract
from splinter.agents.refiner import Brief, brief_to_task, brief_waves, refine_briefs
from splinter.agents.runner import RunResult, Task
from splinter.configure import configured_agents_options, configured_timeout
from splinter.engines.catalog import Engine
from splinter.engines.failures import classify, find_limit, find_result
from splinter.engines.health import EngineHealth, probe
from splinter.engines.launch import EngineLaunch, run_engine
from splinter.engines.pool import EnginePool, PoolExhausted
from splinter.engines.routing import chain_for_effort
from splinter.enums import Effort
from splinter.memory.knowledge import KnowledgeStore
from splinter.memory.session import Session
from splinter.models.roster import Ladder
from splinter.obs.trace import Trace, log_run
from splinter.procreg import IdleTimeoutError
from splinter.scheduling import BudgetPool, DagScheduler, TaskState, default_max_concurrency
from splinter.strategies.cascade import _append_task_header_once
from splinter.strategies.direct import DirectStrategy, TaskOutcome
from splinter.strategies.registry import register
from splinter.task_plan import PLAN_INDEX_FILE
from splinter.templating import load_standards
from splinter.vcs.guard import GuardEnv, guarded_env, setup_guard
from splinter.vcs.worktree import (
    WorktreeHandle,
    WorktreeMergeConflict,
    commit_worktree,
    create_integration_worktree,
    create_worktree,
    merge_no_ff,
    teardown_worktree,
    worktree_supported,
)

log = logging.getLogger("splinter.loop")

_FAST_INSTRUCTION = (
    "Implement exactly the brief, fast and directly. Do not explore, analyze, or overthink. "
)


def _slug_from(tasks: list[Task]) -> str:
    base = (tasks[0].id if tasks and tasks[0].id else "run").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", base).strip("-")
    return slug or "run"


def _engine_prompt(ident: str, brief_text: str, contract: str) -> str:
    report_rule = (
        "Nobody answers questions. Write .agent/report.md at the root of the worktree, "
        f"then print exactly one final line: AGENT_RESULT: DONE {ident} "
        f"(or AGENT_RESULT: BLOCKED {ident})."
    )
    return (
        "You are a fast workhorse implementer. "
        f"{_FAST_INSTRUCTION}"
        "Follow the workhorse contract below. "
        f"{report_rule}\n\n--- BRIEF ---\n{brief_text}\n\n"
        f"--- WORKHORSE CONTRACT ---\n{contract}\n"
    )


def _worktree_diff(path: Path) -> str:
    """Uncommitted change summary (status + diff) of a brief worktree."""
    status = subprocess.run(
        ["git", "status", "--short"], cwd=path, capture_output=True, text=True
    )
    diff = subprocess.run(
        ["git", "diff", "HEAD"], cwd=path, capture_output=True, text=True
    )
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=path, capture_output=True, text=True, check=True,
    )
    additions = []
    for name in untracked.stdout.split("\0"):
        if name and (path / name).is_file() and not (path / name).is_symlink():
            content = (path / name).read_text(errors="replace")
            additions.append(f"diff --git a/{name} b/{name}\nnew file\n" + content)
    return (status.stdout.strip() + "\n\n" + diff.stdout + "\n".join(additions))[:20000]


def _move_report(wt: Path, dest: Path) -> str:
    report = wt / REPORT_PATH
    if report.exists():
        try:
            text = report.read_text()
        except OSError:
            return ""
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)
        return text
    return ""


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
        mode = AcceptanceMode.GATE if skip_eval else parse_acceptance_mode(opts.acceptance_check)
        trace = Trace.from_jsonl(session) if resume else Trace(session=session)
        knowledge = KnowledgeStore(session)

        run_dir = session.dir / "agents-run"
        for sub in ("briefs", "logs", "reports"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)

        from dataclasses import asdict

        saved_briefs = session.read("agents-run/briefs.json") if resume else ""
        briefs = (
            [Brief(**item) for item in json.loads(saved_briefs)]
            if saved_briefs
            else refine_briefs(
                tasks,
                ladder,
                prd_text=session.read(PLAN_INDEX_FILE),
                session=session,
                trace=trace,
            )
        )
        session.write("agents-run/briefs.json", json.dumps([asdict(b) for b in briefs]))
        waves = brief_waves(briefs)
        slug = _slug_from(tasks)
        self._write_plan_doc(run_dir, slug, waves)
        self._init_state(session, run_dir, slug)

        guard = setup_guard(run_dir)
        health = EngineHealth()
        if opts.probe_on_start:
            self._probe_into(health, ladder)
        pool = EnginePool(health)

        results: list[RunResult] = []
        if not worktree_supported():
            log.warning("agents: no git worktree support — falling back to direct loop")
            session.append("loop.md", "## No worktree support: direct fallback\n\n")
            for i, task in enumerate(tasks):
                outcome = TaskOutcome()
                result = self._run_task_loop(
                    task,
                    session,
                    ladder,
                    trace,
                    knowledge,
                    task_index=i,
                    effort=effort,
                    budget=budget,
                    max_iterations=max_iterations,
                    localization=task.filtered_context or localization,
                    eval_skill=eval_skill,
                    skip_planner=skip_planner,
                    skip_eval=skip_eval,
                    outcome=outcome,
                )
                if result is not None:
                    results.append(result)
            return results

        saved = session.read_worktrees() if resume else {}
        record = saved.get(f"{slug}-int")
        integration = (
            WorktreeHandle(Path(record["path"]), record["branch"], f"{slug}-int")
            if record and Path(record["path"]).exists()
            else create_integration_worktree(slug)
        )
        session.set_worktree(f"{slug}-int", str(integration.path), integration.branch)
        done = self._load_checkpoint(session) if resume else set()

        ctx = _BriefCtx(
            slug=slug,
            run_dir=run_dir,
            guard=guard,
            pool=pool,
            health=health,
            integration=integration,
            mode=mode,
            stall_min=opts.stall_min,
            max_attempts=opts.max_attempts,
            timeout=max(configured_timeout(), 2700),
            effort=effort,
            budget=budget,
            max_iterations=max_iterations,
            localization=localization,
            eval_skill=eval_skill,
            skip_planner=skip_planner,
            skip_eval=skip_eval,
            concurrency=max_concurrency if parallel else 1,
            worktrees=saved,
        )

        results = self._dispatch_briefs(briefs, session, ladder, trace, knowledge, ctx, done)
        if any(b.id not in done for b in briefs):
            raise RuntimeError("agents: incomplete briefs; integration worktree kept")
        self._finish(session, run_dir, integration, slug)
        session.set_status("running", task_index=len(briefs), task_total=len(briefs))
        return results

    def _dispatch_briefs(
        self,
        briefs: list[Brief],
        session: Session,
        ladder: Ladder,
        trace: Trace,
        knowledge: KnowledgeStore,
        ctx: _BriefCtx,
        done: set[str],
    ) -> list[RunResult]:
        cap = ctx.max_concurrency()
        budget_pool = BudgetPool(_budget=ctx.budget)
        results: list[RunResult] = []
        results_lock = threading.Lock()

        pending = [b for b in briefs if b.id not in done]
        scheduler = DagScheduler([brief_to_task(b) for b in pending])
        futures: dict[Future[tuple[RunResult | None, bool]], Brief] = {}

        with ThreadPoolExecutor(max_workers=cap) as executor:
            stop_dispatch = False
            while not scheduler.is_done() or futures:
                if not stop_dispatch and budget_pool.exhausted(trace.total_cost):
                    log.info("agents: budget exhausted — draining in-flight briefs")
                    stop_dispatch = True
                if not stop_dispatch:
                    for task in scheduler.ready():
                        if len(futures) >= cap:
                            break
                        if budget_pool.exhausted(trace.total_cost):
                            stop_dispatch = True
                            break
                        brief = next(b for b in pending if b.id == task.id)
                        scheduler.mark_running(task.id)
                        task_index = briefs.index(brief)
                        futures[executor.submit(
                            self._run_brief,
                            brief,
                            task_index,
                            session,
                            ladder,
                            trace,
                            knowledge,
                            ctx,
                        )] = brief
                if not futures:
                    break
                done_futures, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done_futures:
                    brief = futures.pop(future)
                    try:
                        result, passed = future.result()
                    except Exception as exc:
                        log.error("agents: brief %s failed: %s", brief.id, exc)
                        scheduler.mark_failed(brief.id)
                        continue
                    if result is not None:
                        with results_lock:
                            results.append(result)
                    if passed:
                        scheduler.mark_passed(brief.id)
                        done.add(brief.id)
                        self._save_checkpoint(session, done)
                    else:
                        log.warning("agents: brief %s did not PASS — blocking dependents", brief.id)
                        scheduler.mark_failed(brief.id)

        blocked = [tid for tid, st in scheduler.all_states().items() if st == TaskState.BLOCKED]
        if blocked:
            reasons = [f"{tid}: {scheduler.blocked_reason(tid)}" for tid in blocked]
            with ctx.session_lock:
                session.append(
                    "loop.md",
                    "## Blocked briefs\n" + "\n".join(f"- {r}" for r in reasons) + "\n\n",
                )
        return results

    def _run_brief(
        self,
        brief: Brief,
        task_index: int,
        session: Session,
        ladder: Ladder,
        trace: Trace,
        knowledge: KnowledgeStore,
        ctx: _BriefCtx,
    ) -> tuple[RunResult | None, bool]:
        with ctx.session_lock:
            goal_line = brief.goal.splitlines()[0] if brief.goal else ""
            _append_task_header_once(
                session,
                task_index + 1,
                f"# Brief {brief.id} [agents]: {goal_line}\n\n",
            )
        with ctx.git_lock:
            record = ctx.worktrees.get(brief.id)
            wt = (
                WorktreeHandle(Path(record["path"]), record["branch"], brief.id)
                if record and Path(record["path"]).exists()
                else create_worktree(f"{ctx.slug}-{brief.id}", base_ref=ctx.integration.branch)
            )
        session.set_worktree(brief.id, str(wt.path), wt.branch)

        brief_path = ctx.run_dir / "briefs" / f"{brief.id}.md"
        if not brief_path.exists():
            brief_path.write_text(
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
        contract = workhorse_contract(laws=load_standards(), plan_ref=str(ctx.run_dir))
        env = guarded_env(ctx.guard)

        exclude: set[str] = set()
        chain = chain_for_effort(ladder, brief.effort or Effort.NORMAL)
        engine_session: str | None = None
        last_engine = ""
        attempts = 0

        while True:
            engine = self._pick_engine(ctx, chain, exclude)
            if engine is None:
                attempts += 1
                if attempts >= ctx.max_attempts:
                    return self._assume_brief(
                        brief, task_index, session, ladder, trace, knowledge, wt, ctx
                    )
                exclude.clear()
                continue
            prompt = _engine_prompt(brief.id, brief_path.read_text(), contract)
            resume_session = (
                engine_session if engine.key == last_engine and engine.supports_continue else None
            )
            started = time.monotonic()
            launch: EngineLaunch | None = None
            try:
                launch = run_engine(
                    engine,
                    prompt,
                    cwd=str(wt.path),
                    env=env,
                    session=resume_session,
                    timeout=ctx.timeout,
                    idle_timeout=ctx.stall_min * 60,
                    effort=engine.default_effort,
                )
            except IdleTimeoutError:
                self._reroute(ctx, session, brief, engine, "STALL: idle watchdog")
                exclude.add(engine.key)
                continue
            except FileNotFoundError:
                self._reroute(ctx, session, brief, engine, "MISSING: engine binary absent")
                exclude.add(engine.key)
                continue
            except RuntimeError as exc:
                limit = find_limit("", str(exc))
                kind = "LIMIT" if limit else "CRASH"
                self._reroute(ctx, session, brief, engine, f"{kind}: {exc}")
                exclude.add(engine.key)
                continue

            assert launch is not None
            failure = classify(
                returncode=0,
                stdout=launch.text,
                stderr="",
                has_result=find_result(launch.text) is not None,
            )
            if failure is not None:
                reason = f"{failure.kind.value}: {failure.detail}"
                self._reroute(ctx, session, brief, engine, reason)
                exclude.add(engine.key)
                continue
            ctx.pool.report_success(engine.key)
            engine_session = launch.session_id or engine_session
            last_engine = engine.key

            elapsed = time.monotonic() - started
            result = RunResult(
                text=launch.text,
                model=launch.model,
                tier=0,
                tokens=launch.tokens or {},
                cost=launch.cost,
                raw={"engine": engine.key, "brief": brief.id},
                opencode_session=engine_session,
                cost_indeterminate=launch.cost_indeterminate,
                latency_s=elapsed,
            )
            log_run(trace, result, iteration=attempts, task=task_index)
            acc = check_acceptance(
                goal=brief.goal,
                acceptance=brief.acceptance,
                mode=ctx.mode,
                project_dir=str(wt.path),
                ladder=ladder,
                diff=_worktree_diff(wt.path),
                trace=trace,
                task_index=task_index,
            )
            if acc.passed:
                report = _move_report(wt.path, ctx.run_dir / "reports" / f"{brief.id}.md")
                if report:
                    result = RunResult(
                        text=report,
                        model=result.model,
                        tier=0,
                        tokens=result.tokens,
                        cost=result.cost,
                        raw=result.raw,
                        opencode_session=result.opencode_session,
                        cost_indeterminate=result.cost_indeterminate,
                        latency_s=result.latency_s,
                    )
                passed = self._merge_passed(ctx, session, brief, engine.key, wt)
                with ctx.session_lock:
                    session.append(
                        "events.md",
                        f"[AGENTS] brief {brief.id} engine={engine.key} accepted "
                        f"mode={ctx.mode.value} merged={passed}",
                    )
                return result, passed

            attempts += 1
            if attempts >= ctx.max_attempts:
                return self._assume_brief(
                    brief, task_index, session, ladder, trace, knowledge, wt, ctx
                )
            with ctx.session_lock:
                brief_path.write_text(
                    brief_path.read_text()
                    + f"\n\n## Fix round {attempts}\n\nAcceptance failed:\n\n"
                    + "\n".join(f"- {line}" for line in acc.detail.splitlines()[:10])
                    + "\n"
                )
                session.append(
                    "events.md",
                    f"[AGENTS] brief {brief.id} engine={engine.key} fix round {attempts}",
                )

    def _pick_engine(
        self, ctx: _BriefCtx, chain: tuple[Engine, ...], exclude: set[str]
    ) -> Engine | None:
        try:
            return ctx.pool.acquire(chain, exclude)
        except PoolExhausted:
            return None

    def _assume_brief(
        self,
        brief: Brief,
        task_index: int,
        session: Session,
        ladder: Ladder,
        trace: Trace,
        knowledge: KnowledgeStore,
        wt: WorktreeHandle,
        ctx: _BriefCtx,
    ) -> tuple[RunResult | None, bool]:
        log.warning("agents: assuming brief %s in %s", brief.id, wt.path)
        with ctx.session_lock:
            session.append("events.md", f"[AGENTS] brief {brief.id} assumed by orchestrator")
        task = brief_to_task(brief)
        outcome = TaskOutcome()
        result = self._run_task_loop(
            task,
            session,
            ladder,
            trace,
            knowledge,
            task_index=task_index,
            effort=ctx.effort,
            budget=ctx.budget,
            max_iterations=ctx.max_iterations,
            localization=task.filtered_context or ctx.localization,
            eval_skill=ctx.eval_skill,
            skip_planner=ctx.skip_planner,
            skip_eval=ctx.skip_eval,
            lock=ctx.session_lock,
            cwd=str(wt.path),
            outcome=outcome,
        )
        if outcome.passed:
            return result, self._merge_passed(ctx, session, brief, "orchestrator", wt)
        return result, outcome.passed

    def _merge_passed(
        self,
        ctx: _BriefCtx,
        session: Session,
        brief: Brief,
        engine_key: str,
        wt: WorktreeHandle,
    ) -> bool:
        with ctx.git_lock:
            try:
                commit_worktree(wt, f"splinter: brief {brief.id} ({engine_key})")
                merge_no_ff(wt, ctx.integration.path)
                teardown_worktree(wt)
                with ctx.session_lock:
                    session.append(
                        "agents-run/STATE.md",
                        f"- merged {brief.id} ({engine_key}) into {ctx.integration.branch}\n",
                    )
                return True
            except WorktreeMergeConflict as exc:
                log.error(
                    "agents: merge conflict for %s — worktree KEPT at %s: %s",
                    brief.id,
                    wt.path,
                    exc,
                )
                with ctx.session_lock:
                    session.append(
                        "agents-run/STATE.md",
                        f"- CONFLICT {brief.id} ({engine_key}) — worktree kept at {wt.path}\n",
                    )
                return False

    def _reroute(
        self,
        ctx: _BriefCtx,
        session: Session,
        brief: Brief,
        engine: Engine,
        reason: str,
    ) -> None:
        ctx.pool.report_infra_failure(engine.key)
        log.warning("agents: brief %s rerouting off %s — %s", brief.id, engine.key, reason)
        with ctx.session_lock:
            session.append(
                "events.md",
                f"[AGENTS] brief {brief.id} engine={engine.key} infra failure — {reason}",
            )

    def _probe_into(self, health: EngineHealth, ladder: Ladder) -> None:
        seen: set[str] = set()
        engines = [e for effort in Effort for e in chain_for_effort(ladder, effort)]
        for engine in engines:
            key = engine.key
            if key in seen:
                continue
            seen.add(key)
            try:
                result = probe(engine)
            except Exception as exc:
                log.warning("agents: probe of %s crashed: %s", key, exc)
                continue
            if not result.up:
                log.warning("agents: engine %s down (%s) — breaker open", key, result.detail)
                health.trip(key)

    def _write_plan_doc(self, run_dir: Path, slug: str, waves: list[list[Brief]]) -> None:
        lines = [f"# Plan — {slug}", "", "## Waves & Briefs", ""]
        for i, wave in enumerate(waves, 1):
            ids = ", ".join(b.id for b in wave)
            lines.append(f"- Wave {i}: {ids}")
        lines.append("")
        (run_dir / "README.md").write_text("\n".join(lines))

    def _init_state(self, session: Session, run_dir: Path, slug: str) -> None:
        state = run_dir / "STATE.md"
        if not state.exists():
            state.write_text(
                f"# Run state — {slug}\n\n## Merged\n\n## Verification\n\n"
                "## Orchestrator notes\n"
            )

    @staticmethod
    def _load_checkpoint(session: Session) -> set[str]:
        raw = session.read("agents-run/checkpoint.json")
        if not raw.strip():
            return set()
        try:
            data: dict[str, list[str]] = json.loads(raw)
            return set(data.get("completed", []))
        except (json.JSONDecodeError, AttributeError):
            return set()

    @staticmethod
    def _save_checkpoint(session: Session, done_ids: set[str]) -> None:
        session.write(
            "agents-run/checkpoint.json", json.dumps({"completed": sorted(done_ids)})
        )

    def _finish(
        self,
        session: Session,
        run_dir: Path,
        integration: WorktreeHandle,
        slug: str,
    ) -> None:
        from splinter.agents.gate import run_gate

        gate = run_gate(project_dir=str(integration.path))
        if not gate.passed:
            log.error("agents: final gate on integration branch failed — worktree KEPT")
            session.append("agents-run/STATE.md", "## Verification\n- final gate FAILED\n")
            raise RuntimeError("agents: final integration gate failed; worktree kept")
        session.append("agents-run/STATE.md", "## Verification\n- final gate passed\n")
        try:
            merge_no_ff(integration, Path.cwd(), message=f"splinter: agents run {slug}")
            teardown_worktree(integration)
            log.info("agents: integration branch merged into the main tree")
        except WorktreeMergeConflict as exc:
            log.error(
                "agents: integration merge conflict — worktree KEPT at %s: %s",
                integration.path,
                exc,
            )
            raise


class _BriefCtx:
    """Per-run immutable context shared with every brief worker thread."""

    def __init__(
        self,
        *,
        slug: str,
        run_dir: Path,
        guard: GuardEnv,
        pool: EnginePool,
        health: EngineHealth,
        integration: WorktreeHandle,
        mode: AcceptanceMode,
        stall_min: float,
        max_attempts: int,
        timeout: int,
        effort: str | None,
        budget: float | None,
        max_iterations: int,
        localization: str,
        eval_skill: str | None,
        skip_planner: bool,
        skip_eval: bool,
        concurrency: int | None = None,
        worktrees: dict[str, dict[str, str]] | None = None,
    ) -> None:
        self.slug = slug
        self.run_dir = run_dir
        self.guard = guard
        self.pool = pool
        self.health = health
        self.integration = integration
        self.mode = mode
        self.stall_min = stall_min
        self.max_attempts = max_attempts
        self.timeout = timeout
        self.effort = effort
        self.budget = budget
        self.max_iterations = max_iterations
        self.localization = localization
        self.eval_skill = eval_skill
        self.skip_planner = skip_planner
        self.skip_eval = skip_eval
        self._concurrency = concurrency
        self.worktrees = worktrees or {}
        self.session_lock = threading.Lock()
        self.git_lock = threading.Lock()

    def max_concurrency(self) -> int:
        return self._concurrency or default_max_concurrency()
