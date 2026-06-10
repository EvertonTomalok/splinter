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
    localization = "localization"
    trace = "trace"
    all = "all"


@app.command()
def setup() -> None:
    """Verify environment and providers."""
    from splinter.setup import run_setup

    raise typer.Exit(run_setup())


@app.command()
def prd(
    description: Annotated[str, typer.Argument(help="Feature/bug description")] = "",
    strategy: Annotated[str | None, typer.Option(help="Pre-select strategy")] = None,
) -> None:
    """Generate a PRD interactively."""
    from splinter.prd import run_prd

    raise typer.Exit(run_prd(description=description, strategy=strategy))


@app.command()
def run(
    strategy: Annotated[str | None, typer.Option(help="Strategy name or turtle alias")] = None,
    prd: Annotated[str | None, typer.Option(help="Path to prd.md")] = None,
    task: Annotated[str | None, typer.Option(help="Path to task.yaml")] = None,
    effort: Annotated[str | None, typer.Option(help="Override reasoning effort")] = None,
    budget: Annotated[float | None, typer.Option(help="Max cost in dollars")] = None,
    max_iterations: Annotated[int, typer.Option(help="Max loop iterations")] = 5,
    cowabunga: Annotated[
        bool,
        typer.Option(
            "--cowabunga",
            help="Full autonomy: skip the PRD Q&A and never wake the human on ASK_USER",
        ),
    ] = False,
    quiet: Annotated[
        bool, typer.Option(help="Plain log output instead of the live TUI")
    ] = False,
) -> None:
    """Run a task or PRD through a strategy."""
    import os
    import sys

    run_kwargs = {
        "strategy": strategy,
        "prd_path": prd,
        "task_path": task,
        "effort": effort,
        "budget": budget,
        "max_iterations": max_iterations,
        "cowabunga": cowabunga,
    }

    tty = sys.stdin.isatty() and sys.stdout.isatty()
    if quiet or not tty or os.environ.get("SPLINTER_NO_TUI"):
        import logging

        from splinter.pipeline import run_pipeline

        logging.basicConfig(level=logging.INFO, format="%(message)s")
        raise typer.Exit(run_pipeline(**run_kwargs))  # type: ignore[arg-type]

    # A PRD on a TTY gets refined interactively before it runs. --cowabunga is
    # honoured inside the session: it decides everything and runs without asking.
    if prd and not task:
        from splinter.tui import run_prd_interactive

        raise typer.Exit(run_prd_interactive(run_kwargs))

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
) -> None:
    """Resume a session: PRD refinement, or a failed/interrupted run.

    Transient failures continue from where they stopped; critical failures roll the
    failing stage back and redo it. ``--reset`` re-runs from the head.
    """
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
    gate_checks: Annotated[
        str | None, typer.Option(help="Comma-separated gate commands")
    ] = None,
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
        )
    )


def main(argv: list[str] | None = None) -> int:
    """Console-script entrypoint; returns a process exit code."""
    try:
        # standalone_mode=False makes Click return the value instead of sys.exit.
        result = app(args=argv, standalone_mode=False)
    except typer.Exit as exc:
        return exc.exit_code
    except KeyboardInterrupt:
        print("\naborted.")
        return 130
    except SystemExit as exc:  # raised by Click on --help / usage errors
        return int(exc.code or 0)
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
