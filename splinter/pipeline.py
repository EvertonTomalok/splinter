"""Wires PRD/task input through localize -> plan -> run -> gate -> eval -> loop."""

from __future__ import annotations

import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

from splinter.agents import planner
from splinter.agents.localizer import CodeAnchor, filter_task_context, localize, rtk_cat_tip
from splinter.agents.runner import RunResult, Task
from splinter.memory.session import Session, new_session_id
from splinter.models.roster import Ladder, load_ladder
from splinter.obs.agentic import agentic_scope
from splinter.obs.trace import Trace
from splinter.providers.base import ProviderGapError
from splinter.strategies.base import AskUserPause, GracefulPause, ManualValidationPause
from splinter.strategies.registry import available_strategies, get_strategy

DEFAULT_STRATEGY = "cascade"

log = logging.getLogger("splinter.pipeline")


def _warn_ladder_pricing(ladder: Ladder) -> None:
    from splinter.models.pricing import warn_missing_model_pricing

    models = [
        ladder.planner_model,
        ladder.prd_model,
        ladder.eval_model,
        ladder.localizer_recall_model,
        ladder.localizer_recall_large_model,
        ladder.localizer_precision_model,
        ladder.localizer_recall_fallback_model,
    ]
    for tier in ladder.tiers:
        models.extend(tier.models)
    for model_id in dict.fromkeys(models):
        warn_missing_model_pricing(model_id)


#: Substrings in an error that mark it transient (retry/continue, don't roll back).
_TRANSIENT_MARKERS = (
    "429",
    "500",
    "502",
    "503",
    "504",
    "529",
    "overloaded",
    "rate limit",
    "ratelimit",
    "timeout",
    "timed out",
    "temporarily",
    "try again",
    "connection",
    "econnreset",
    "unavailable",
    "reset by peer",
    "network",
    "socket",
)


class _SessionTraceHandler(logging.Handler):
    """Persist every ``splinter`` log record to the session's ``events.md`` so the
    Trace view is a full chronological log — each model push, tool call, gate/eval
    result, and escalate/jump/ask decision, in order."""

    def __init__(self, session: Session) -> None:
        super().__init__()
        self.session = session

    def emit(self, record: logging.LogRecord) -> None:
        try:
            ts = datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S")
            self.session.append("events.md", f"[{ts}] {record.getMessage()}")
        except Exception:  # noqa: BLE001 — tracing must never break the run
            pass


def _resolve_gate(session: Session, ladder: object, tasks: list[Task]) -> None:
    """Ensure the run has a gate. Precedence: already-configured (session gate.json
    or .splinter/config.yaml) → model-detected from the repo → language-specific
    defaults derived from the union of all task languages → Python defaults.

    Users can override per project via ``gate_checks`` in config.yaml, or per run
    in the PRD review phase; this just makes the planner bring one when none is set.
    """
    from splinter.agents import gate
    from splinter.agents.gate import task_languages
    from splinter.configure import gate_default_for

    existing = gate.configured_gate_checks(session_dir=session.dir)
    if existing is not None:
        names = ", ".join(c.get("name", c.get("cmd", "?")) for c in existing) or "none"
        log.info("gate: using configured checks (%s)", names)
        return

    log.info("gate: detecting project checks…")
    detected = gate.detect_gate_checks(ladder)
    if detected:
        gate.save_gate_checks(session.dir, detected)
        log.info("gate: detected — %s", ", ".join(c["name"] for c in detected))
    else:
        all_langs: set[str] = set()
        for t in tasks:
            all_langs.update(task_languages(t))
        log.info("gate: resolved languages: %s", sorted(all_langs) or ["(none)"])
        if all_langs:
            lang_checks: list[dict[str, str]] = []
            for lang in sorted(all_langs):
                lang_checks.extend(gate_default_for(lang))
            gate.save_gate_checks(session.dir, lang_checks)
            log.info("gate: using language-specific defaults for %s", sorted(all_langs))
        else:
            log.warning(
                "gate: could not detect checks — using defaults. Set `gate_checks` "
                "in .splinter/config.yaml or specify them in the PRD review."
            )


def _classify_failure(exc: BaseException) -> str:
    """Transient (provider/network blip → resume continues) vs critical (bad command,
    bug → resume rolls the failing stage back and redoes it)."""
    from splinter.providers.base import ProviderGapError

    if isinstance(exc, ProviderGapError):
        return "gap"
    if isinstance(exc, (TimeoutError, subprocess.TimeoutExpired)):
        return "transient"
    msg = str(exc).lower()
    if any(m in msg for m in _TRANSIENT_MARKERS):
        return "transient"
    return "critical"


def _run_final_eval_cli(
    *,
    session: Session,
    final_eval: str,
    eval_model: str | None,
    eval_effort: str | None,
    tasks: list[Task],
    ladder: object,
    round_index: int,
    effort_cur: str,
) -> None:
    """Run a single CLI-supplied eval skill and write results to knowledge/final-eval.md."""
    from splinter.agents.final_eval import run_final_eval
    from splinter.configure import FinalEvalEntry
    from splinter.enums import FinalEvalKind

    entry = FinalEvalEntry(
        name=final_eval,
        kind=FinalEvalKind.SKILL,
        skill=final_eval,
        model=eval_model,
    )
    task = tasks[0] if tasks else None
    result = run_final_eval(entry, task=task, ladder=ladder)  # type: ignore[arg-type]
    content = f"# Final Eval (CLI)\n\n{result.output}\n"
    session.write("knowledge/final-eval.md", content)
    verdict = "PASS" if result.passed else "FAIL"
    session.append("events.md", f"final eval (CLI): {final_eval} · {verdict}")


def _load_task_from_yaml(path: str) -> Task:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return Task(
        description=data.get("description", ""),
        acceptance=data.get("acceptance", ""),
        effort=data.get("effort", "normal"),
        reasoning_effort=data.get("reasoning_effort", "auto"),
        eval_skill=data.get("eval_skill"),
        suggested_tier=data.get("suggested_tier", 0),
        target_files=data.get("target_files"),
    )


def _load_tasks_from_prd(prd_path: str) -> tuple[list[Task], str | None]:
    text = Path(prd_path).read_text()
    fm, _body = planner._parse_frontmatter(text)
    strategy = fm.get("strategy")
    tasks = planner.parse_stories(text)
    return tasks, strategy


def _run_phase_loop_stdin(
    session: Session,
    ladder: Ladder,
    effort: str | None,
    plan_model: str | None = None,
    plan_effort: str | None = None,
    run_model: str | None = None,
    run_effort: str | None = None,
) -> int:
    """Interactive phase loop for non-TTY ``--phased`` mode (stdin/stdout)."""
    from splinter.phases import PhaseConfig, run_phase

    print("\n=== Entering phase mode ===\n")
    print("Describe what to implement next, or type 'done' to finish.")
    print()

    default_plan_model = plan_model or ladder.planner_model
    default_plan_effort = plan_effort or ladder.planner_effort
    default_run_model = run_model or (ladder.tiers[0].models[0] if ladder.tiers else "haiku")
    default_run_effort = run_effort or effort or "auto"

    if not plan_model or not run_model:
        print("Model/effort overrides: use -pm/-pe/-rm/-re flags or answer prompts below.")
        print()

    while True:
        try:
            desc = input("Phase description (or 'done'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\ndone.")
            break
        if not desc or desc.lower() in ("done", "quit", "exit"):
            break

        pm = default_plan_model
        pe = default_plan_effort
        rm = default_run_model
        re = default_run_effort

        if not plan_model:
            pm_in = input(f"  Plan model [{pm}]: ").strip()
            if pm_in:
                pm = pm_in
        if not plan_effort:
            pe_in = input(f"  Plan effort [{pe}]: ").strip()
            if pe_in:
                pe = pe_in
        if not run_model:
            rm_in = input(f"  Run model [{rm}]: ").strip()
            if rm_in:
                rm = rm_in
        if not run_effort:
            re_in = input(f"  Run effort [{re}]: ").strip()
            if re_in:
                re = re_in

        cfg = PhaseConfig(
            description=desc,
            plan_model=pm,
            plan_effort=pe,
            run_model=rm,
            run_effort=re,
        )

        try:
            result = run_phase(cfg, session, ladder)
        except Exception as exc:
            print(f"Phase failed: {exc}")
            continue

        verdict = "PASS" if result.gate_passed else "FAIL"
        print(
            f"\nPhase {result.phase_number} · {verdict} · "
            f"{result.run_result.model} · ${result.run_result.cost:.4f}\n"
        )
        if not result.gate_passed:
            print(f"Gate output:\n{result.gate_output}")
        print()

    session.set_status("completed", stage="done")
    return 0


#: trivial < normal < hard < critical — used to pick the hardest story's effort.
_EFFORT_RANK = {"trivial": 0, "normal": 1, "hard": 2, "critical": 3}


def _merge_stories_into_task(prd_text: str, stories: list[Task]) -> Task:
    """Raphael single-shot: collapse all PRD stories into ONE task.

    The description is the whole PRD body (every ``### US-NNN`` verbatim) and the
    acceptance is every story's criteria concatenated. There is no per-task
    localization — the single run gets the main localization and implements all
    stories in one session, judged holistically.
    """
    _fm, body = planner._parse_frontmatter(prd_text)
    acceptance = "\n".join(s.acceptance for s in stories if s.acceptance)
    hardest = max(
        (s.effort for s in stories),
        key=lambda e: _EFFORT_RANK.get(e, 1),
        default="normal",
    )
    return Task(
        description=body.strip(),
        acceptance=acceptance or "implementation matches the PRD",
        effort=hardest,
    )


def _compose_eval_fix_prompt(eval_output: str, user_reply: str) -> str:
    parts: list[str] = []
    if eval_output.strip():
        parts.append(f"## Final Eval Findings\n\n{eval_output.strip()}")
    if user_reply.strip():
        parts.append(f"## User Guidance\n\n{user_reply.strip()}")
    merged = "\n\n".join(parts).strip()
    return merged or "Address the latest final eval findings and return for user review."


def _build_eval_fix_task(fix_prompt: str, effort: str | None) -> Task:
    return Task(
        description=fix_prompt,
        acceptance="Apply the fixes and pass all configured final eval checks.",
        effort=effort or "normal",
    )


def _clear_round_caches(session: Session) -> None:
    """Remove stale plan/filter/per-task-localization files before a new round."""
    import re as _re

    kdir = session.dir / "knowledge"
    if not kdir.exists():
        return
    for p in kdir.iterdir():
        if not p.suffix == ".md":
            continue
        name = p.stem
        # keep localization.md (main); remove localization-N.md, plan*.md, filter-*.md
        if name == "localization":
            continue
        if _re.match(r"^(plan(-\d+)?|filter-\d+|localization-\d+)$", name):
            p.unlink(missing_ok=True)


def _load_round_history(session: Session) -> str:
    """Concatenate all round-eval-N notes into a single string."""
    import re as _re

    kdir = session.dir / "knowledge"
    if not kdir.exists():
        return ""
    notes: list[tuple[int, str, str]] = []
    for p in kdir.glob("round-eval-*.md"):
        m = _re.match(r"^round-eval-(\d+)$", p.stem)
        if m:
            notes.append((int(m.group(1)), p.stem, p.read_text()))
    if not notes:
        return ""
    parts: list[str] = []
    for _, stem, content in sorted(notes):
        parts.append(f"## {stem}\n\n{content}")
    return "\n\n".join(parts)


def _compute_summary_cost(trace: Trace, results: list[RunResult]) -> tuple[float, int]:
    """Return (cost, runs) from the persisted trace, falling back to results."""
    if trace.entries:
        return trace.total_cost, len(trace.entries)
    return sum(r.cost for r in results), len(results)


def run_pipeline(
    *,
    strategy: str | None = None,
    prd_path: str | None = None,
    task_path: str | None = None,
    effort: str | None = None,
    budget: float | None = None,
    max_iterations: int = 5,
    eval_skill: str | None = None,
    eval_model: str | None = None,
    eval_effort: str | None = None,
    cowabunga: bool = False,
    resume: bool = False,
    session: Session | None = None,
    claude_runner_fallback: bool = False,
    user_guidance: str | None = None,
    jump_premium: bool = False,
    no_ground: bool = False,
    phased: bool = False,
    phase_plan_model: str | None = None,
    phase_plan_effort: str | None = None,
    phase_run_model: str | None = None,
    phase_run_effort: str | None = None,
) -> int:
    from splinter import procreg as _procreg

    _procreg.clear_stop()

    ladder = load_ladder()
    if eval_model:
        ladder.eval_model = eval_model
    if eval_effort:
        ladder.eval_effort = eval_effort
    if claude_runner_fallback:
        from splinter.models.roster import rewrite_runners_claude

        rewrite_runners_claude(ladder)
        log.info("runtime: runner tiers rewritten to sonnet @ high")
    if session is None:
        # Fresh session per run so prior runs (especially failed ones) are kept.
        session = Session(new_session_id())

    # Read resume context now that session is guaranteed to exist.
    resume_round = 0
    resume_effort: str | None = None
    _next_planner_model: str | None = None
    _next_planner_effort: str | None = None
    _next_runner_model: str | None = None
    _next_runner_effort: str | None = None
    _next_eval_model: str | None = None
    _next_eval_effort: str | None = None
    _next_skip_planner: bool = False
    _next_skip_eval: bool = False
    _next_skip_final_eval: bool = False
    resume_from_final_eval = False
    resume_eval_findings = ""
    if resume:
        _cur = session.read_status()
        resume_round = int(_cur.get("round_index", 0))
        resume_effort = _cur.get("next_effort") or None
        _next_planner_model = _cur.get("next_planner_model") or None
        _next_planner_effort = _cur.get("next_planner_effort") or None
        _next_runner_model = _cur.get("next_runner_model") or None
        _next_runner_effort = _cur.get("next_runner_effort") or None
        _next_eval_model = _cur.get("next_eval_model") or None
        _next_eval_effort = _cur.get("next_eval_effort") or None
        _next_skip_planner = str(_cur.get("next_skip_planner", "")).lower() == "true"
        _next_skip_eval = str(_cur.get("next_skip_eval", "")).lower() == "true"
        _next_skip_final_eval = str(_cur.get("next_skip_final_eval", "")).lower() == "true"
        resume_from_final_eval = (
            str(_cur.get("stage", "")) == "final_eval"
            and str(_cur.get("state", "")) in {"awaiting_user", "awaiting_validation"}
            and resume_round > 0
        )
        if resume_from_final_eval:
            resume_eval_findings = str(_cur.get("ask_corrections", "")).strip()
    effective_effort = effort or resume_effort

    if _next_planner_model:
        ladder.planner_model = _next_planner_model
    if _next_planner_effort:
        ladder.planner_effort = _next_planner_effort
    if _next_eval_model:
        ladder.eval_model = _next_eval_model
    if _next_eval_effort:
        ladder.eval_effort = _next_eval_effort
    if _next_runner_model:
        from splinter.models.roster import rewrite_runner_tiers

        rewrite_runner_tiers(
            ladder,
            model=_next_runner_model,
            variant=_next_runner_effort or "high",
        )
        _eff = _next_runner_effort or "high"
        log.info("runtime: runner tiers rewritten to %s @ %s", _next_runner_model, _eff)
    _warn_ladder_pricing(ladder)
    if any(
        [
            _next_planner_model,
            _next_planner_effort,
            _next_runner_model,
            _next_runner_effort,
            _next_eval_model,
            _next_eval_effort,
            _next_skip_planner,
            _next_skip_eval,
            _next_skip_final_eval,
        ]
    ):
        session.clear_next_config()

    tasks: list[Task] = []
    if task_path:
        tasks.append(_load_task_from_yaml(task_path))
    elif prd_path:
        prd_tasks, prd_strategy = _load_tasks_from_prd(prd_path)
        tasks = prd_tasks
        if strategy is None:
            strategy = prd_strategy
    else:
        print("error: provide --task or --prd")
        return 1

    strategy_name = strategy or DEFAULT_STRATEGY
    eval_fix_prompt = ""
    effective_user_guidance = user_guidance
    if resume_from_final_eval:
        eval_fix_prompt = _compose_eval_fix_prompt(resume_eval_findings, user_guidance or "")
        tasks = [_build_eval_fix_task(eval_fix_prompt, effective_effort)]
        strategy_name = "direct"
        if _next_skip_planner:
            effective_user_guidance = eval_fix_prompt
        else:
            effective_user_guidance = None
        session.write(
            f"knowledge/eval-fix-input-{resume_round}.md",
            f"# Eval Fix Input — Round {resume_round}\n\n{eval_fix_prompt}\n",
        )
        log.info(
            "final-eval resume round %d: forcing direct single-task eval-fix flow",
            resume_round,
        )
    try:
        strat = get_strategy(strategy_name)
    except ValueError:
        print(
            f"error: unknown strategy '{strategy_name}'. "
            f"Available: {', '.join(available_strategies())}"
        )
        return 1

    # Raphael (direct) is single-shot: merge every PRD story into one task and skip
    # the per-task filter/localization phases below. Other strategies are untouched.
    single_shot = getattr(strat, "name", "") == "direct"
    if single_shot and prd_path and len(tasks) > 1:
        n_stories = len(tasks)
        tasks = [_merge_stories_into_task(Path(prd_path).read_text(), tasks)]
        log.info("raphael single-shot: merged %d stories into one task", n_stories)

    session.set_status(
        "running",
        pid=os.getpid(),
        strategy=strategy_name,
        tasks=len(tasks),
        max_iterations=max_iterations,
        effort=effort or "",
        budget=budget if budget is not None else "",
        source=prd_path or task_path or "",
        started=datetime.now(timezone.utc).isoformat(),
        stage="localize",
        cowabunga=cowabunga,
    )

    idx_lines = [
        f"# Session {session.id}",
        f"- strategy: {strategy_name}",
        f"- tasks: {len(tasks)}",
    ]
    if prd_path:
        idx_lines.append(f"- prd: {prd_path}")
    session.update_index("\n".join(idx_lines) + "\n")

    # Mirror the whole "splinter" log stream into events.md so the Trace view is a
    # full chronological record of every push / result / decision.
    splog = logging.getLogger("splinter")
    trace_handler = _SessionTraceHandler(session)
    trace_handler.setLevel(logging.INFO)
    splog.addHandler(trace_handler)
    session.append(
        "events.md",
        f"=== run {'resume' if resume else 'start'} · {strategy_name} · "
        f"{datetime.now(timezone.utc).astimezone().strftime('%H:%M:%S')} ===",
    )

    try:
        log.info(
            "session %s · strategy %s · %d task(s)%s",
            session.id,
            strategy_name,
            len(tasks),
            " · 🤙 cowabunga" if cowabunga else "",
        )

        prd_text = ""
        if prd_path:
            prd_text = Path(prd_path).read_text()
        elif tasks:
            prd_text = tasks[0].description

        localization = ""
        anchors: list[CodeAnchor] = []
        if prd_text and not resume_from_final_eval:
            if resume and resume_round > 0:
                _clear_round_caches(session)
            if resume and resume_round == 0 and session.has("knowledge/localization.md"):
                log.info("resume: reusing existing localization")
                localization = session.read("knowledge/localization.md")
                from splinter.agents.localizer import _parse_anchors

                anchors = _parse_anchors(
                    localization,
                    hot=ladder.localizer_relevance_hot,
                    medium=ladder.localizer_relevance_medium,
                )
                log.info("resume: re-parsed %d anchor(s)", len(anchors))
            else:
                log.info("localizing against the codebase…")
                with agentic_scope(session, "locate", 0, 0):
                    anchors = localize(prd_text, session, ladder)
                localization = session.read("knowledge/localization.md")

        if anchors and tasks and not single_shot:
            planner.assign_target_files(tasks, anchors)
            session.set_status("running", stage="filter")
            log.info("filtering code context per task…")
            for i, task in enumerate(tasks):
                # Cache filter output so resume doesn't re-call the LLM.
                cache_key = f"knowledge/filter-{i + 1}.md"
                cached = session.read(cache_key)
                if resume and cached.strip():
                    task.filtered_context = cached
                    log.info("resume: reusing filtered context for task %d", i + 1)
                else:
                    task.filtered_context = filter_task_context(task, ladder, session=session)
                    if task.filtered_context:
                        session.write(cache_key, task.filtered_context)

            # Write per-task localization file: relevant anchors + rtk tips.
            for i, task in enumerate(tasks):
                loc_key = f"knowledge/localization-{i + 1}.md"
                if resume and session.read(loc_key).strip():
                    log.info("resume: reusing per-task localization for task %d", i + 1)
                    continue
                task_files = set(task.target_files or [])
                task_anchors = [a for a in anchors if a.file in task_files]
                if not task_anchors:
                    continue
                loc_lines = [f"# Localization — Task {i + 1}\n"]
                for a in task_anchors:
                    loc_part = f"L{a.line_start}-L{a.line_end}" if a.line_start else ""
                    label = (
                        f"{a.file}:{loc_part} — {a.symbol}"
                        if loc_part
                        else f"{a.file} — {a.symbol}"
                    )
                    rtk_tip = rtk_cat_tip(a)
                    loc_lines.append(
                        f"### {label}\n"
                        f"file: {a.file}\n"
                        f"symbol: {a.symbol}\n"
                        + (
                            f"line_start: {a.line_start}\nline_end: {a.line_end}\n"
                            if a.line_start
                            else ""
                        )
                        + f"rtk: {rtk_tip}\n"
                        f"reason: {a.reason}\n"
                    )
                session.write(loc_key, "\n".join(loc_lines))

        _resolve_gate(session, ladder, tasks)

        round_history = _load_round_history(session)
        if round_history:
            session.write("knowledge/previous_rounds.md", round_history)

        _runner_model = ladder.tiers[0].models[0] if ladder.tiers else "none"
        _first_tier_level = ladder.tiers[0].level if ladder.tiers else 0
        _runner_variant = ladder.tier_variants.get(_first_tier_level, "?") if ladder.tiers else "?"
        session.append(
            "events.md",
            f"round {resume_round} config · "
            f"planner={ladder.planner_model}@{ladder.planner_effort} · "
            f"runner={_runner_model}@{_runner_variant} · "
            f"eval={ladder.eval_model}@{ladder.eval_effort}",
        )

        session.set_status("running", stage="run")
        results = strat.execute(
            tasks,
            session,
            ladder,
            effort=effective_effort,
            budget=budget,
            max_iterations=max_iterations,
            localization=localization,
            eval_skill=eval_skill,
            cowabunga=cowabunga,
            resume=resume,
            claude_runner_fallback=claude_runner_fallback,
            user_guidance=effective_user_guidance,
            jump_premium=jump_premium,
            skip_planner=_next_skip_planner,
            skip_eval=_next_skip_eval,
        )

        from splinter.agents.final_eval import run_all_final_evals
        from splinter.configure import load_config, load_final_eval

        _session_fe_path = session.dir / "final_eval.yaml"
        if _session_fe_path.exists():
            _fe_config = yaml.safe_load(_session_fe_path.read_text()) or {}
            final_eval_entries = load_final_eval(_fe_config)
            log.info("final eval: loaded from session dir (%d entries)", len(final_eval_entries))
        else:
            final_eval_entries = load_final_eval(load_config())
        if final_eval_entries and not _next_skip_final_eval:
            session.set_status("running", stage="final_eval")
            log.info("running %d final eval(s)…", len(final_eval_entries))
            task_for_eval = tasks[0] if tasks else None
            fe_results = run_all_final_evals(
                final_eval_entries,
                task=task_for_eval,
                project_dir=str(Path.cwd()),
                ladder=ladder,
            )
            fe_summary = "\n".join(
                f"- {r.name}: {'PASS' if r.passed else 'FAIL'} — {r.output}" for r in fe_results
            )
            fe_verbatim = "\n\n---\n\n".join(r.output for r in fe_results)
            session.write("final_eval.md", fe_verbatim + "\n")
            round_content = f"# Final Eval — Round {resume_round + 1}\n\n{fe_verbatim}\n"
            session.write(f"round-final-eval-{resume_round}.md", round_content)
            session.write(f"knowledge/final-eval-{resume_round}.md", round_content)
            rd = session.round_dir(resume_round)
            (rd / "final-eval.md").write_text(round_content)
            log.info("final eval results:\n%s", fe_summary)
            all_passed = all(r.passed for r in fe_results)
            if not all_passed:
                failed = [r.name for r in fe_results if not r.passed]
                log.warning("final eval FAILED: %s", ", ".join(failed))
                fe_fail_text = "\n".join(r.output for r in fe_results if not r.passed)
                round_eval_content = f"# Round {resume_round} Eval\n\n{fe_fail_text}\n"
                session.write(f"knowledge/round-eval-{resume_round}.md", round_eval_content)
                (rd / "round-eval.md").write_text(round_eval_content)
                raise ManualValidationPause(summary=fe_summary, all_passed=False)
            session.set_status(
                "running",
                stage="final_eval",
                final_eval_passed=True,
                final_eval_summary=fe_summary,
            )

        session.set_status("completed", stage="done")
        trace_md = session.read("trace.md")
        trace = Trace.from_markdown(trace_md)
        total, runs = _compute_summary_cost(trace, results)
        log.info("pipeline complete · %d run(s) · $%.4f", runs, total)
        print(f"pipeline complete. session: {session.id}")
        print(f"  runs: {runs}")
        print(f"  cost: ${total:.4f}")

        if phased:
            return _run_phase_loop_stdin(
                session,
                ladder,
                effort,
                plan_model=phase_plan_model,
                plan_effort=phase_plan_effort,
                run_model=phase_run_model,
                run_effort=phase_run_effort,
            )

        return 0
    except ManualValidationPause as val_exc:
        from splinter.models.roster import bump_effort

        cur_effort = effective_effort or "normal"
        session.set_status(
            "awaiting_user",
            stage="final_eval",
            final_eval_summary=val_exc.summary,
            final_eval_passed=val_exc.all_passed,
            round_index=resume_round + 1,
            next_effort=bump_effort(cur_effort),
            ask_corrections=val_exc.summary,
            next_skip_planner="true",
            next_skip_eval="true",
        )
        log.info("run paused — awaiting manual validation")
        print(f"run complete — awaiting manual validation.\n{val_exc.summary}")
        print(f"  validate: splinter resume {session.id}")
        return 3
    except GracefulPause as gp:
        session.set_status(
            "paused",
            ask_reason=gp.reason,
            ask_corrections=gp.corrections,
            ask_tier=gp.tier,
            ask_iteration=gp.iteration,
            task_index=gp.task_index,
            stage=gp.stage,
        )
        log.warning("run paused gracefully at stage '%s' — iter %d", gp.stage, gp.iteration)
        print(f"run paused — graceful stop at stage '{gp.stage}'.\n  {gp.reason}")
        print(f"  resume: splinter resume {session.id}")
        return 3
    except AskUserPause as ask_exc:
        session.set_status(
            "awaiting_user",
            ask_reason=ask_exc.reason,
            ask_corrections=ask_exc.corrections,
            ask_tier=ask_exc.tier,
            ask_iteration=ask_exc.iteration,
            task_index=ask_exc.task_index,
            stage="run",
            round_index=0,
            next_effort="",
            final_eval_summary="",
            final_eval_passed="",
            cowabunga=cowabunga,
        )
        log.warning("run paused — needs your input: %s", ask_exc.reason)
        print(f"run paused — needs your input.\n  {ask_exc.reason}")
        print(f"  resume: splinter resume {session.id}")
        return 3
    except ProviderGapError as gap_exc:
        session.set_status(
            "paused",
            kind=gap_exc.kind,
            resumable=gap_exc.resumable,
            provider=gap_exc.provider,
            retry_after=gap_exc.retry_after,
        )
        log.warning("run paused — %s", gap_exc)
        print(gap_exc.guidance)
        return 2
    except BaseException as exc:
        fail_class = _classify_failure(exc)
        err_msg = f"{type(exc).__name__}: {str(exc)}"
        session.set_status("failed", fail_class=fail_class, error=err_msg)
        log.error("pipeline failed (%s): %s", fail_class, exc)
        raise
    finally:
        splog.removeHandler(trace_handler)
