"""Leonardo — the ``cascade`` multi-task dependency-ordered strategy.

Flow: topological sort on task.deps, then run each task in order with
per-task checkpoint persistence. A crash mid-run resumes at the first
un-checkpointed task. Budget exhaustion stops the cascade cleanly.

When parallel=True, independent tasks run concurrently in git worktrees via
DagScheduler + ThreadPoolExecutor, merging results back on PASS.

Inherits the full per-task Run → Gate → Eval loop from DirectStrategy.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait

from splinter.agents.evaluator import is_premium_task
from splinter.agents.runner import RunResult, Task
from splinter.memory.knowledge import KnowledgeStore
from splinter.memory.session import Session
from splinter.models.roster import Ladder
from splinter.obs.trace import Trace
from splinter.scheduling import (
    BudgetPool,
    DagScheduler,
    TaskState,
    default_max_concurrency,
    topo_order,
)
from splinter.strategies.direct import DirectStrategy, TaskOutcome
from splinter.strategies.registry import register

log = logging.getLogger("splinter.loop")


def _append_task_header_once(session: Session, task_no: int, header: str) -> None:
    """Append a ``# Task N`` header to ``loop.md`` only if that task has none yet.

    A resumed run re-dispatches every un-checkpointed task (failed, blocked, or
    mid-flight when interrupted). Without this guard each resume re-appends the
    same header, so ``analyze``/trajectory shows the same task two, three, N times
    — corrupting the run's observability. The header is written exactly once per
    task number, regardless of how many times the task is dispatched.

    Callers that share ``loop.md`` across threads must hold the session lock so the
    read-check-append is atomic.
    """
    if re.search(rf"^# Task {task_no}(?:[ \t/\[:])", session.read("loop.md"), re.MULTILINE):
        return
    session.append("loop.md", header)


@register
class CascadeStrategy(DirectStrategy):
    name = "cascade"
    aliases = ["leonardo"]

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
        ordered = self._topo_sort(tasks)

        trace = Trace.from_jsonl(session) if resume else Trace(session=session)

        knowledge = KnowledgeStore(session)
        results: list[RunResult] = []
        done = self._load_checkpoint(session) if resume else set()

        if done:
            log.info("cascade resume: %d task(s) already checkpointed", len(done))

        self._run_plan_phase(
            ordered,
            session,
            ladder,
            localization,
            trace=trace,
            skip_planner=skip_planner,
            resume=resume,
            force_replan=force_replan,
            max_concurrency=max_concurrency,
            done_ids=done,
        )

        if parallel and len(ordered) > 1:
            results = self._run_parallel_dag(
                ordered,
                session,
                ladder,
                trace,
                knowledge,
                done=done,
                effort=effort,
                budget=budget,
                max_iterations=max_iterations,
                localization=localization,
                eval_skill=eval_skill,
                cowabunga=cowabunga,
                skip_planner=skip_planner,
                skip_eval=skip_eval,
                max_concurrency=max_concurrency,
            )
        else:
            results = self._run_sequential(
                ordered,
                session,
                ladder,
                trace,
                knowledge,
                done=done,
                effort=effort,
                budget=budget,
                max_iterations=max_iterations,
                localization=localization,
                eval_skill=eval_skill,
                cowabunga=cowabunga,
                skip_planner=skip_planner,
                skip_eval=skip_eval,
                force_replan=force_replan,
            )

        session.set_status("running", task_index=len(ordered), task_total=len(ordered))
        return results

    def _run_sequential(
        self,
        ordered: list[Task],
        session: Session,
        ladder: Ladder,
        trace: Trace,
        knowledge: KnowledgeStore,
        *,
        done: set[str],
        effort: str | None,
        budget: float | None,
        max_iterations: int,
        localization: str,
        eval_skill: str | None,
        cowabunga: bool,
        skip_planner: bool,
        skip_eval: bool,
        force_replan: bool = False,
    ) -> list[RunResult]:
        results: list[RunResult] = []
        for i, task in enumerate(ordered):
            if task.id and task.id in done:
                log.info("resume: skip %s (checkpointed)", task.id)
                continue

            session.set_status(
                "running",
                stage="run",
                task_index=i,
                task_total=len(ordered),
                task=task.description.splitlines()[0],
            )
            _append_task_header_once(
                session,
                i + 1,
                f"# Task {i + 1}/{len(ordered)}: {task.description.splitlines()[0]}\n\n",
            )

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
                cowabunga=cowabunga,
                resume=False,
                skip_planner=skip_planner,
                skip_eval=skip_eval,
                force_replan=force_replan,
                outcome=outcome,
            )

            if result is not None:
                results.append(result)
                # Checkpoint only a genuine PASS — a budget stop or cowabunga
                # ASK_USER also returns a result, and checkpointing it would make
                # resume skip an unfinished task.
                if task.id and outcome.passed:
                    done.add(task.id)
                    self._save_checkpoint(session, done)

            if budget is not None and trace.total_cost >= budget:
                session.append("loop.md", f"## Budget exhausted (${trace.total_cost:.4f})\n")
                break

        return results

    def _run_parallel_dag(
        self,
        ordered: list[Task],
        session: Session,
        ladder: Ladder,
        trace: Trace,
        knowledge: KnowledgeStore,
        *,
        done: set[str],
        effort: str | None,
        budget: float | None,
        max_iterations: int,
        localization: str,
        eval_skill: str | None,
        cowabunga: bool,
        skip_planner: bool,
        skip_eval: bool,
        max_concurrency: int | None,
        start_tier_overrides: dict[str, int] | None = None,
    ) -> list[RunResult]:
        from splinter.vcs.worktree import worktree_supported

        cap = max_concurrency or default_max_concurrency()
        # trace.total_cost already sums every task's runs (dispatch logs each entry),
        # so it IS the running spend — the pool's base_cost. Nothing is added to the
        # pool per task or the cost would be counted twice (once in the trace, once
        # in the pool) and the run would abort at ~half the intended budget.
        budget_pool = BudgetPool(_budget=budget)
        session_lock = threading.Lock()
        # Serialises every git operation (worktree add/merge/remove, branch delete)
        # against the shared main-repo index — concurrent `git` from worker threads
        # races on .git/index.lock and mixes staged changes across tasks.
        git_lock = threading.Lock()
        results: list[RunResult] = []
        results_lock = threading.Lock()

        pending = [t for t in ordered if not (t.id and t.id in done)]
        # Id-less tasks can't sit in the dependency DAG (DagScheduler keys on id and
        # would silently drop them); run them sequentially after the DAG instead of
        # vanishing them. Everything with an id goes through the scheduler.
        id_pending = [t for t in pending if t.id]
        no_id_pending = [t for t in pending if not t.id]
        scheduler = DagScheduler(id_pending)

        use_worktrees = worktree_supported()
        if use_worktrees:
            log.info("cascade parallel: worktree isolation enabled (cap=%d)", cap)
        else:
            log.info(
                "cascade parallel: no worktree support, running without isolation (cap=%d)",
                cap,
            )

        futures: dict[Future[tuple[RunResult | None, bool]], Task] = {}

        with ThreadPoolExecutor(max_workers=cap) as executor:
            stop_dispatch = False
            while not scheduler.is_done() or futures:
                if not stop_dispatch and budget_pool.exhausted(trace.total_cost):
                    log.info("parallel: budget exhausted — draining in-flight, no new dispatch")
                    stop_dispatch = True

                # Re-read kowabunga on every dispatch pass (each is a scheduling
                # decision) so a mid-run toggle takes effect on the next pass. When ON,
                # premium tasks jump the queue — stuck premium work gets unblocked
                # first. When OFF, ready() order is left exactly as today (no regression).
                live_cowabunga = session.read_cowabunga()

                if not stop_dispatch:
                    ready = scheduler.ready()
                    if live_cowabunga:
                        ready = sorted(ready, key=lambda t: 0 if is_premium_task(t, ladder) else 1)
                    for task in ready:
                        if budget_pool.exhausted(trace.total_cost):
                            stop_dispatch = True
                            break
                        scheduler.mark_running(task.id)
                        task_index = ordered.index(task)
                        tier_override = (start_tier_overrides or {}).get(task.id)

                        future = executor.submit(
                            self._run_parallel_task,
                            task,
                            task_index,
                            session,
                            ladder,
                            trace,
                            knowledge,
                            session_lock,
                            git_lock,
                            use_worktrees=use_worktrees,
                            effort=effort,
                            budget=budget,
                            max_iterations=max_iterations,
                            localization=localization,
                            eval_skill=eval_skill,
                            cowabunga=live_cowabunga,
                            skip_planner=skip_planner,
                            skip_eval=skip_eval,
                            start_tier_override=tier_override,
                        )
                        futures[future] = task

                # No futures in flight and nothing new dispatched → nothing left to do.
                if not futures:
                    break

                # Drain at least one completion so in-flight results are always
                # collected and checkpointed, even after a budget stop.
                done_futures, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done_futures:
                    task = futures.pop(future)
                    try:
                        result, passed = future.result()
                    except Exception as exc:
                        log.error("parallel task %s failed: %s", task.id, exc)
                        scheduler.mark_failed(task.id)
                        continue
                    if result is not None:
                        with results_lock:
                            results.append(result)
                    if passed:
                        scheduler.mark_passed(task.id)
                        if task.id:
                            done.add(task.id)
                            self._save_checkpoint(session, done)
                    else:
                        # Not a genuine PASS (stopped at max tier, budget, ASK_USER):
                        # do NOT unblock dependents or checkpoint — they'd run against
                        # incomplete upstream work and resume would skip an unfinished task.
                        log.warning("parallel task %s did not PASS — blocking dependents", task.id)
                        scheduler.mark_failed(task.id)

        # Id-less tasks: no worktree (create_worktree needs an id), no DAG slot — run
        # them sequentially in the main repo, respecting the running budget.
        for task in no_id_pending:
            if budget is not None and trace.total_cost >= budget:
                session.append("loop.md", f"## Budget exhausted (${trace.total_cost:.4f})\n")
                break
            task_index = ordered.index(task)
            with session_lock:
                _append_task_header_once(
                    session,
                    task_index + 1,
                    f"# Task {task_index + 1} [sequential]: {task.description.splitlines()[0]}\n\n",
                )
            outcome = TaskOutcome()
            result = self._run_task_loop(
                task,
                session,
                ladder,
                trace,
                knowledge,
                task_index=task_index,
                effort=effort,
                budget=budget,
                max_iterations=max_iterations,
                localization=task.filtered_context or localization,
                eval_skill=eval_skill,
                cowabunga=cowabunga,
                resume=False,
                skip_planner=skip_planner,
                skip_eval=skip_eval,
                start_tier_override=(start_tier_overrides or {}).get(task.id),
                lock=session_lock,
                outcome=outcome,
            )
            if result is not None:
                results.append(result)

        blocked = [tid for tid, st in scheduler.all_states().items() if st == TaskState.BLOCKED]
        if blocked:
            reasons = [f"{tid}: {scheduler.blocked_reason(tid)}" for tid in blocked]
            log.warning("parallel: %d task(s) blocked — %s", len(blocked), "; ".join(reasons))
            with session_lock:
                session.append(
                    "loop.md",
                    "## Blocked tasks\n" + "\n".join(f"- {r}" for r in reasons) + "\n\n",
                )

        return results

    def _run_parallel_task(
        self,
        task: Task,
        task_index: int,
        session: Session,
        ladder: Ladder,
        trace: Trace,
        knowledge: KnowledgeStore,
        session_lock: threading.Lock,
        git_lock: threading.Lock,
        *,
        use_worktrees: bool,
        effort: str | None,
        budget: float | None,
        max_iterations: int,
        localization: str,
        eval_skill: str | None,
        cowabunga: bool,
        skip_planner: bool,
        skip_eval: bool,
        start_tier_override: int | None,
    ) -> tuple[RunResult | None, bool]:
        from splinter.vcs.worktree import (
            WorktreeMergeConflict,
            branch_has_unmerged_commits,
            commit_worktree,
            create_worktree,
            squash_merge,
            teardown_worktree,
        )

        handle = None
        with session_lock:
            _append_task_header_once(
                session,
                task_index + 1,
                f"# Task {task_index + 1} [parallel]: {task.description.splitlines()[0]}\n\n",
            )

        if use_worktrees and task.id:
            # Acquire/create the worktree under the git lock: `git worktree add` and
            # the worktrees.json read-modify-write must be atomic across threads.
            with git_lock:
                existing = session.read_worktrees().get(task.id)
                if existing:
                    from pathlib import Path

                    from splinter.vcs.worktree import WorktreeHandle

                    handle = WorktreeHandle(
                        path=Path(existing["path"]),
                        branch=existing["branch"],
                        task_id=task.id,
                    )
                    log.info("parallel: reattached worktree for %s", task.id)
                else:
                    try:
                        handle = create_worktree(task.id)
                        session.set_worktree(task.id, str(handle.path), handle.branch)
                        log.info("parallel: created worktree for %s at %s", task.id, handle.path)
                    except Exception as exc:
                        log.warning(
                            "parallel: worktree creation failed for %s: %s — continuing without",
                            task.id,
                            exc,
                        )
                        handle = None

        # The coder (and its gate/eval) must run INSIDE the worktree, else it edits
        # the shared main repo — clobbering sibling tasks and leaving the branch
        # empty for squash_merge. cwd=None falls back to the main repo (no isolation).
        cwd = str(handle.path) if handle is not None else None

        outcome = TaskOutcome()
        result = self._run_task_loop(
            task,
            session,
            ladder,
            trace,
            knowledge,
            task_index=task_index,
            effort=effort,
            budget=budget,
            max_iterations=max_iterations,
            localization=task.filtered_context or localization,
            eval_skill=eval_skill,
            cowabunga=cowabunga,
            resume=False,
            skip_planner=skip_planner,
            skip_eval=skip_eval,
            start_tier_override=start_tier_override,
            lock=session_lock,
            cwd=cwd,
            outcome=outcome,
        )

        if handle is not None:
            # Merge only a genuine PASS — never fold half-finished work back into the
            # main tree. All git ops are serialised so two tasks never touch the
            # shared index concurrently.
            #
            # A worktree is torn down ONLY after its work is safely merged into the
            # main tree. Anything else — task didn't PASS, or the merge hit an
            # unresolved conflict — keeps the worktree on disk so the user can
            # inspect and resolve it by hand. Never destroy unmerged work.
            merged = False
            if outcome.passed:
                with git_lock:
                    try:
                        commit_worktree(handle)
                        # Merge on "branch carries work", not "this run committed" —
                        # on resume the work is already committed from a prior run;
                        # keying on a fresh commit would drop it on teardown.
                        if branch_has_unmerged_commits(handle):
                            squash_merge(handle)
                            log.info("parallel: squash-merged %s", task.id)
                        else:
                            log.info("parallel: %s produced no changes — nothing to merge", task.id)
                        merged = True
                    except WorktreeMergeConflict as exc:
                        log.error(
                            "parallel: merge conflict for %s — worktree KEPT at %s for "
                            "manual resolution, not deleted: %s",
                            task.id,
                            handle.path,
                            exc,
                        )
                        raise
            else:
                log.warning(
                    "parallel: %s did not PASS — worktree KEPT at %s, not deleted",
                    task.id,
                    handle.path,
                )
            if merged:
                with git_lock:
                    try:
                        teardown_worktree(handle)
                        log.info("parallel: cleaned up worktree for %s", task.id)
                    except Exception as exc:
                        log.warning("parallel: worktree teardown failed for %s: %s", task.id, exc)

        return result, outcome.passed

    @staticmethod
    def _topo_sort(tasks: list[Task]) -> list[Task]:
        """Kahn's algorithm over task.deps. Cycle → warn + fallback to original order."""
        ordered = topo_order(tasks)
        if ordered is None:
            log.warning("cascade: dependency cycle detected — falling back to PRD order")
            return list(tasks)
        return ordered

    @staticmethod
    def _load_checkpoint(session: Session) -> set[str]:
        raw = session.read("checkpoint.json")
        if not raw.strip():
            return set()
        try:
            data: dict[str, list[str]] = json.loads(raw)
            return set(data.get("completed", []))
        except (json.JSONDecodeError, AttributeError):
            return set()

    @staticmethod
    def _save_checkpoint(session: Session, done_ids: set[str]) -> None:
        session.write("checkpoint.json", json.dumps({"completed": sorted(done_ids)}))
