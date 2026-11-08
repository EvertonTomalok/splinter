"""Typer CLI entrypoint for the splinter harness."""

from __future__ import annotations

from enum import Enum
from typing import Annotated

import typer

app = typer.Typer(
    name="splinter",
    help="Multiagent orchestration harness: expensive model plans, cheap models execute.",
    no_args_is_help=True,
    add_completion=False,
)


class ExpandStep(str, Enum):
    plan = "plan"
    loop = "loop"
    eval = "eval"
    final_eval = "final_eval"
    localization = "localization"
    trace = "trace"
    agentic = "agentic"
    all = "all"


@app.command()
def setup() -> None:
    """Verify environment and providers."""
    from splinter.setup import run_setup

    raise typer.Exit(run_setup())


@app.command()
def engines(
    probe: Annotated[
        bool,
        typer.Option(
            "--probe",
            help="Live-probe installed engines (spends real quota, ~2 min per engine)",
        ),
    ] = False,
    only: Annotated[
        str, typer.Option("--only", help="Comma-separated engine keys to probe")
    ] = "",
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON instead of tables")] = False,
) -> None:
    """List the workhorse engine catalog, optionally probing engine health."""
    import json as _json

    from splinter.engines.catalog import CATALOG, engine_by_key
    from splinter.engines.health import is_reviewer, probe_report
    from splinter.engines.health import probe as probe_engine

    if not probe:
        if as_json:
            rows = [
                {
                    "key": e.key,
                    "label": e.label,
                    "provider": e.provider,
                    "model": e.model,
                    "role": e.role,
                }
                for e in CATALOG
            ]
            typer.echo(_json.dumps({"engines": rows}, indent=2))
        else:
            typer.echo("Engine catalog (splinter run --strategy agents):\n")
            typer.echo(engine_table())
        raise typer.Exit(0)

    keys = [k.strip() for k in only.split(",") if k.strip()] if only else [e.key for e in CATALOG]
    unknown = [k for k in keys if engine_by_key(k) is None]
    if unknown:
        typer.echo(f"unknown engine(s): {', '.join(unknown)}", err=True)
        raise typer.Exit(2)
    from splinter.engines.health import EngineProbe

    probes: list[EngineProbe] = []
    sonnet_up: bool | None = None
    for key in keys:
        engine = engine_by_key(key)
        assert engine is not None
        if is_reviewer(key):
            continue
        result = probe_engine(engine)
        probes.append(result)
        if key == "sonnet":
            sonnet_up = result.up
    if any(is_reviewer(k) for k in keys):
        for key in keys:
            if not is_reviewer(key):
                continue
            probes.append(
                EngineProbe(
                    key=key,
                    up=bool(sonnet_up),
                    binary="claude-credit",
                    detail="inferred from sonnet",
                )
            )
    if as_json:
        typer.echo(
            _json.dumps(
                {
                    "engines": [
                        {"key": p.key, "up": p.up, "binary": p.binary, "detail": p.detail}
                        for p in probes
                    ]
                },
                indent=2,
            )
        )
    else:
        typer.echo(probe_report(probes))
    raise typer.Exit(0)


def engine_table() -> str:
    from splinter.engines.catalog import CATALOG

    lines = ["| Key | Engine | Provider | Role |", "|---|---|---|---|"]
    for e in CATALOG:
        lines.append(f"| `{e.key}` | {e.label} | {e.provider} | {e.role} |")
    return "\n".join(lines)


@app.command()
def prd(
    description: Annotated[str, typer.Argument(help="Feature/bug description")] = "",
    strategy: Annotated[str | None, typer.Option(help="Pre-select strategy")] = None,
    no_ground: Annotated[
        bool,
        typer.Option("--no-ground", help="Skip codebase grounding before PRD Q&A"),
    ] = False,
) -> None:
    """Generate a PRD interactively."""
    import os
    import sys

    tty = sys.stdin.isatty() and sys.stdout.isatty()
    if not tty or os.environ.get("SPLINTER_NO_TUI"):
        from splinter.prd import run_prd

        raise typer.Exit(run_prd(description=description, strategy=strategy, no_ground=no_ground))

    from splinter.tui import run_prd_interactive

    run_kwargs = {
        "description": description,
        "strategy": strategy,
        "prd_path": None,
        "task_path": None,
        "effort": None,
        "budget": None,
        "max_iterations": 5,
        "cowabunga": False,
        "no_ground": no_ground,
        "phased": False,
    }
    raise typer.Exit(run_prd_interactive(run_kwargs))


@app.command()
def run(
    strategy: Annotated[str | None, typer.Option(help="Strategy name or turtle alias")] = None,
    prd: Annotated[str | None, typer.Option(help="Path to prd.md")] = None,
    task: Annotated[
        str | None,
        typer.Option(help="Path to task.yaml or a markdown task (tasks/<plan>/T01-….md)"),
    ] = None,
    effort: Annotated[str | None, typer.Option(help="Override reasoning effort")] = None,
    budget: Annotated[float | None, typer.Option(help="Max cost in dollars")] = None,
    max_iterations: Annotated[int, typer.Option(help="Max loop iterations")] = 5,
    eval: Annotated[
        str | None, typer.Option("--eval", help="Override eval skill (skill name)")
    ] = None,
    eval_model: Annotated[
        str | None, typer.Option("--eval-model", help="Override evaluator model")
    ] = None,
    eval_effort: Annotated[
        str | None,
        typer.Option("--eval-effort", help="Override evaluator reasoning effort"),
    ] = None,
    cowabunga: Annotated[
        bool,
        typer.Option(
            "--cowabunga",
            help=(
                "Full autonomy: skip the PRD Q&A / task plan mode and never wake the "
                "human on ASK_USER"
            ),
        ),
    ] = False,
    no_ground: Annotated[
        bool,
        typer.Option("--no-ground", help="Skip codebase grounding before PRD Q&A"),
    ] = False,
    quiet: Annotated[bool, typer.Option(help="Plain log output instead of the live TUI")] = False,
    use_cc_only: Annotated[
        bool,
        typer.Option(
            "--use-cc-only",
            help="Swap to Claude-only runners (haiku/sonnet/opus) before running",
        ),
    ] = False,
    phased: Annotated[
        bool,
        typer.Option(
            "--phased",
            help="Enter multi-phase mode: after the run, keep asking for next changes.",
        ),
    ] = False,
    phase_plan_model: Annotated[
        str | None,
        typer.Option(
            "-pm",
            "--plan-model",
            help="Model for phase planning (e.g. opus, sonnet, deepseek-v4-pro)",
        ),
    ] = None,
    phase_plan_effort: Annotated[
        str | None,
        typer.Option(
            "-pe",
            "--plan-effort",
            help="Reasoning effort for phase planner (low, medium, high, max, auto)",
        ),
    ] = None,
    phase_run_model: Annotated[
        str | None,
        typer.Option(
            "-rm",
            "--run-model",
            help="Model for phase execution (e.g. haiku, deepseek-v4-flash-free)",
        ),
    ] = None,
    phase_run_effort: Annotated[
        str | None,
        typer.Option(
            "-re",
            "--run-effort",
            help="Reasoning effort for phase runner (low, medium, high, max, auto)",
        ),
    ] = None,
    parallel: Annotated[
        bool | None,
        typer.Option(
            "--parallel/--no-parallel",
            help=(
                "Run independent tasks concurrently in git worktrees. "
                "Omit to be asked on the PRD accept screen (interactive runs)"
            ),
        ),
    ] = None,
    max_concurrency: Annotated[
        int | None,
        typer.Option(
            "--max-concurrency",
            help="Max parallel tasks (default: cpu_count - 1)",
        ),
    ] = None,
) -> None:
    """Run a task or PRD through a strategy."""
    import os
    import sys

    if use_cc_only:
        from splinter.configure import _swap_config

        rc = _swap_config("config.claude.yaml")
        if rc != 0:
            raise typer.Exit(rc)

    # Tri-state: None = not specified (interactive PRD run asks on the accept
    # screen), True = --parallel, False = --no-parallel. Only an explicit True
    # is gated here against worktree support / single-shot strategies.
    _effective_parallel = parallel
    if parallel:
        from splinter.vcs.worktree import worktree_supported

        if not worktree_supported():
            _effective_parallel = False
            import logging as _logging

            _logging.getLogger("splinter").warning(
                "--parallel ignored: git worktree not available in this directory"
            )
        elif strategy == "direct" or strategy in ("raphael",):
            _effective_parallel = False
            import logging as _logging

            _logging.getLogger("splinter").info(
                "--parallel ignored for direct strategy (single-task)"
            )

    run_kwargs = {
        "strategy": strategy,
        "prd_path": prd,
        "task_path": task,
        "effort": effort,
        "budget": budget,
        "max_iterations": max_iterations,
        "eval_skill": eval,
        "eval_model": eval_model,
        "eval_effort": eval_effort,
        "cowabunga": cowabunga,
        "no_ground": no_ground,
        "phased": phased,
        "phase_plan_model": phase_plan_model,
        "phase_plan_effort": phase_plan_effort,
        "phase_run_model": phase_run_model,
        "phase_run_effort": phase_run_effort,
        "parallel": _effective_parallel,
        "max_concurrency": max_concurrency,
    }

    tty = sys.stdin.isatty() and sys.stdout.isatty()
    if quiet or not tty or os.environ.get("SPLINTER_NO_TUI"):
        import logging

        from splinter.pipeline import run_pipeline

        # No accept screen to ask on — an unspecified flag defaults to sequential.
        if run_kwargs["parallel"] is None:
            run_kwargs["parallel"] = False
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        raise typer.Exit(run_pipeline(**run_kwargs))  # type: ignore[arg-type]

    # A PRD on a TTY gets refined interactively before it runs. --cowabunga is
    # honoured inside the session: it decides everything and runs without asking.
    if prd and not task:
        from splinter.tui import run_prd_interactive

        raise typer.Exit(run_prd_interactive(run_kwargs))

    # A task on a TTY opens plan mode: the planner asks, plans, the user edits, then
    # it runs. --cowabunga skips the plan screen entirely.
    if task and not cowabunga:
        from splinter.tui_plan import run_plan_interactive

        raise typer.Exit(run_plan_interactive(run_kwargs))

    from splinter.tui import run_with_tui

    raise typer.Exit(run_with_tui(run_kwargs))


@app.command()
def resume(
    session: Annotated[
        str | None,
        typer.Argument(help="Session id to resume (default: latest refining session)"),
    ] = None,
    reset: Annotated[
        bool,
        typer.Option("--reset", help="Re-run a failed run from the head (fresh localize + plan)."),
    ] = False,
    use_cc_only: Annotated[
        bool,
        typer.Option(
            "--use-cc-only",
            help="Swap to Claude-only runners (haiku/sonnet/opus) before resuming",
        ),
    ] = False,
) -> None:
    """Resume a session: PRD refinement, or a failed/interrupted run.

    Transient failures continue from where they stopped; critical failures roll the
    failing stage back and redo it. ``--reset`` re-runs from the head.
    """
    if use_cc_only:
        from splinter.configure import _swap_config

        rc = _swap_config("config.claude.yaml")
        if rc != 0:
            raise typer.Exit(rc)

    from splinter.tui import resume_session

    raise typer.Exit(resume_session(session, reset=reset))


@app.command()
def analyze(
    session: Annotated[str | None, typer.Option(help="Session id")] = None,
    watch: Annotated[bool, typer.Option(help="Live-refresh until the run finishes")] = False,
    expand: Annotated[
        ExpandStep | None, typer.Option(help="One-shot: print a step's full markdown")
    ] = None,
    no_interactive: Annotated[
        bool, typer.Option("--no-interactive", help="Static overview instead of the TUI")
    ] = False,
) -> None:
    """Inspect a session (interactive TUI on a TTY)."""
    from splinter.analyze import run_analyze

    raise typer.Exit(
        run_analyze(
            session_id=session,
            expand=expand.value if expand else None,
            watch=watch,
            interactive=False if no_interactive else None,
        )
    )


@app.command()
def configure(
    gate_checks: Annotated[str | None, typer.Option(help="Comma-separated gate commands")] = None,
    timeout: Annotated[
        int | None, typer.Option(help="Per-model-call timeout in seconds (default 3600)")
    ] = None,
    init_prompts: Annotated[
        bool, typer.Option(help="Scaffold editable prompt templates into ./.splinter/prompts/")
    ] = False,
    force: Annotated[
        bool, typer.Option(help="Overwrite existing prompt templates with --init-prompts")
    ] = False,
    no_interactive: Annotated[
        bool, typer.Option("--no-interactive", help="Skip the model-selection TUI")
    ] = False,
    use_default: Annotated[
        bool,
        typer.Option(
            "--use-default",
            help="Restore config.yaml from config.opencode.yaml",
        ),
    ] = False,
    use_cc_only: Annotated[
        bool,
        typer.Option(
            "--use-cc-only",
            help="Activate config.claude.yaml (Claude-only runners)",
        ),
    ] = False,
    sync_prices: Annotated[
        bool, typer.Option("--sync-prices", help="Fetch and save live model pricing")
    ] = False,
    agents_acceptance: Annotated[
        str | None,
        typer.Option(
            "--agents-acceptance",
            help="Brief acceptance check: gate | gate+eval | gate+orchestrator",
        ),
    ] = None,
    agents_stall_min: Annotated[
        float | None,
        typer.Option("--agents-stall-min", help="Idle watchdog minutes before STALL"),
    ] = None,
    agents_max_attempts: Annotated[
        int | None,
        typer.Option("--agents-max-attempts", help="Burned attempts before assuming a brief"),
    ] = None,
    agents_probe_on_start: Annotated[
        bool | None,
        typer.Option(
            "--agents-probe-on-start/--no-agents-probe-on-start",
            help="Live-probe the effort-chain models before wave 1 (spends quota)",
        ),
    ] = None,
    decisions_provider: Annotated[
        str | None,
        typer.Option("--decisions-provider", help="Decision-model provider (openrouter)"),
    ] = None,
    decisions_model: Annotated[
        str | None,
        typer.Option(
            "--decisions-model",
            help="Decision model id (default typesafe/jev-1.14)",
        ),
    ] = None,
    jev_recall: Annotated[
        bool | None,
        typer.Option(
            "--jev-recall/--no-jev-recall",
            help="Decision model ranks/filters recall anchors",
        ),
    ] = None,
    jev_recall_large: Annotated[
        bool | None,
        typer.Option(
            "--jev-recall-large/--no-jev-recall-large",
            help="Decision model ranks recall-large anchors (big search outputs)",
        ),
    ] = None,
    jev_filter: Annotated[
        bool | None,
        typer.Option(
            "--jev-filter/--no-jev-filter",
            help="Decision model ranks target files before the filter summary",
        ),
    ] = None,
    decisions_threshold: Annotated[
        float | None,
        typer.Option("--decisions-threshold", help="Relevance cutoff for decision answers"),
    ] = None,
) -> None:
    """Pick per-step models in a TUI (default), then write config.yaml."""
    from splinter.configure import run_configure

    raise typer.Exit(
        run_configure(
            gate_checks=gate_checks,
            timeout=timeout,
            init_prompts=init_prompts,
            force=force,
            interactive=False if no_interactive else None,
            use_default=use_default,
            use_cc_only=use_cc_only,
            sync_prices_flag=sync_prices,
            agents_acceptance=agents_acceptance,
            agents_stall_min=agents_stall_min,
            agents_max_attempts=agents_max_attempts,
            agents_probe_on_start=agents_probe_on_start,
            decisions_provider=decisions_provider,
            decisions_model=decisions_model,
            jev_recall=jev_recall,
            jev_filter=jev_filter,
            jev_recall_large=jev_recall_large,
            decisions_threshold=decisions_threshold,
        )
    )


@app.command()
def gate(
    kind: Annotated[
        str,
        typer.Argument(
            help="go-fmt | go-vet | go-lint | go-test | ts-typecheck | ts-lint | ts-test | "
            "py-lint | py-type | py-test"
        ),
    ],
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print the commands instead of running them.")
    ] = False,
) -> None:
    """Run one gate kind over only the files the change touched (for gate_checks)."""
    from pathlib import Path

    from splinter.agents.gate_scope import run_kind

    raise typer.Exit(run_kind(kind, Path.cwd(), dry_run=dry_run))


def main(argv: list[str] | None = None) -> int:
    """Console-script entrypoint; returns a process exit code."""
    try:
        # standalone_mode=False makes Click return the value instead of sys.exit.
        result = app(args=argv, standalone_mode=False)
    except typer.Exit as exc:
        return int(exc.exit_code)
    except KeyboardInterrupt:
        print("\naborted.")
        return 130
    except SystemExit as exc:  # raised by Click on --help / usage errors
        return int(exc.code or 0)
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
