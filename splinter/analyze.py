"""``splinter analyze`` — inspect session memory off disk.

Three ways to use it:

* **interactive TUI** (default on a TTY): a Textual app with a live trajectory
  tree and a detail pane — see :mod:`splinter.tui`. ``q``/``Ctrl-C`` quits.
* ``--watch``: a plain live-refresh of the overview until the run finishes.
* ``--expand <step>`` / non-TTY: one-shot print (good for ``watch -n2`` or CI).

This module keeps the pure parsing + string renderers; the TUI imports them.
"""

from __future__ import annotations

import os
import re
import sys
import time

from rich.console import Console

from splinter.memory.session import Session

# Renders the markup in render_overview() for the non-TUI (print) code paths.
_console = Console()

EXPANDABLE = ("plan", "loop", "eval", "localization", "trace", "knowledge", "all")

_EXPAND_FILES = {
    "plan": "knowledge/plan.md",
    "loop": "loop.md",
    "eval": "eval.md",
    "localization": "knowledge/localization.md",
    "trace": "trace.md",
}

_CLEAR = "\033[2J\033[H"


# --- parsing helpers -------------------------------------------------------


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by another user
    return True


def _run_state(session: Session) -> str:
    status = session.read_status()
    state = str(status.get("state", ""))
    if state == "running":
        return "RUNNING" if _pid_alive(status.get("pid")) else "INTERRUPTED"
    if state == "awaiting_user":
        return "AWAITING_USER"
    if state:
        return state.upper()
    return "DONE" if session.read("trace.md") else "UNKNOWN"


def _iterations(loop_md: str) -> list[tuple[int, str, str]]:
    """Parse loop.md into (iteration, tier, verdict) tuples in order."""
    out: list[tuple[int, str, str]] = []
    blocks = re.split(r"^## Iteration (\d+)\s*$", loop_md, flags=re.MULTILINE)
    for i in range(1, len(blocks), 2):
        body = blocks[i + 1]
        tier_match = re.search(r"tier (\d+)", body)
        tier = f"T{tier_match.group(1)}" if tier_match else "T?"
        verdict_match = re.search(r"verdict:\s*(\w+)", body)
        verdict = verdict_match.group(1) if verdict_match else "?"
        out.append((int(blocks[i]), tier, verdict))
    return out


def _prd_phases(phases_md: str) -> list[tuple[str, str]]:
    """Parse prd_phases.md into ordered (phase, detail) pairs."""
    out: list[tuple[str, str]] = []
    for raw in phases_md.splitlines():
        line = raw.strip()
        if not line.startswith("- "):
            continue
        name, _, detail = line[2:].partition(" · ")
        out.append((name.strip(), detail.strip()))
    return out


def _loop_block(loop_md: str, n: int) -> str:
    blocks = re.split(r"^## Iteration (\d+)\s*$", loop_md, flags=re.MULTILINE)
    for i in range(1, len(blocks), 2):
        if int(blocks[i]) == n:
            return f"## Iteration {n}\n{blocks[i + 1].strip()}"
    return ""


def _eval_block(eval_md: str, n: int) -> str:
    parts = re.split(r"^### Iter (\d+):", eval_md, flags=re.MULTILINE)
    for i in range(1, len(parts), 2):
        if int(parts[i]) == n:
            return f"### Iter {n}:{parts[i + 1].rstrip()}"
    return ""


def _trace_metrics(trace_md: str) -> dict[str, str]:
    metrics: dict[str, str] = {}
    for key, pattern in (
        ("cost", r"total cost: \$([\d.]+)"),
        ("runs", r"total runs: (\d+)"),
        ("tokens", r"total tokens: (\{.*?\})"),
        ("elapsed", r"elapsed: ([\d.]+s)"),
    ):
        m = re.search(pattern, trace_md)
        if m:
            metrics[key] = m.group(1)
    return metrics


def _plan_files(session: Session) -> list[tuple[str, str]]:
    kdir = session.dir / "knowledge"
    if not kdir.exists():
        return []
    plans = sorted(kdir.glob("plan-*.md"))
    result: list[tuple[str, str]] = []
    for p in plans:
        label = p.stem.replace("plan-", "plan-")
        result.append((f"knowledge/{p.name}", label))
    return result


def _knowledge_notes(session: Session) -> list[tuple[str, str]]:
    kdir = session.dir / "knowledge"
    if not kdir.exists():
        return []
    notes = sorted(kdir.glob("*.md"))
    return [(f"knowledge/{p.name}", p.stem) for p in notes]


# --- renderers (return strings; pure, testable) ----------------------------


_OVERVIEW_EMOJI = {
    "RUNNING": "🟡", "COMPLETED": "🟢", "DONE": "🟢",
    "FAILED": "🔴", "INTERRUPTED": "🟠", "UNKNOWN": "⚪",
}

_VERDICT_COLOR = {
    "PASS": "green", "RETRY": "yellow", "ESCALATE": "yellow",
    "JUMP_PREMIUM": "magenta", "ASK_USER": "cyan", "FAIL": "red",
}


def _verdict_tag(verdict: str) -> str:
    color = _VERDICT_COLOR.get(verdict, "white")
    return f"[{color}]{verdict}[/]"


def format_run_completion(session: Session) -> str:
    """One-line summary for a finished run (tasks, cost, runs)."""
    status = session.read_status()
    metrics = _trace_metrics(session.read("trace.md"))
    parts: list[str] = []
    task_total = status.get("task_total") or status.get("tasks")
    task_index = status.get("task_index")
    if task_total:
        done = task_index if task_index is not None else task_total
        parts.append(f"{done}/{task_total} tasks")
    cost = metrics.get("cost")
    if cost:
        parts.append(f"${cost}")
    runs = metrics.get("runs")
    if runs:
        parts.append(f"{runs} runs")
    return " · ".join(parts) if parts else "done"


def render_overview(session: Session, state: str) -> str:
    import os

    status = session.read_status()
    localization = session.read("knowledge/localization.md")
    plan = session.read("knowledge/plan.md")
    loop = session.read("loop.md")
    trace = session.read("trace.md")

    state_color = {
        "RUNNING": "yellow", "COMPLETED": "green", "DONE": "green",
        "FAILED": "red", "INTERRUPTED": "dark_orange", "AWAITING_USER": "magenta",
    }.get(state, "white")
    emoji = _OVERVIEW_EMOJI.get(state, "⚪")
    completed = state in ("COMPLETED", "DONE")

    lines = [
        f"[bold]splinter[/] · [cyan]{session.id}[/]",
        f"{emoji} [bold {state_color}]{state}[/]",
    ]
    if completed:
        lines.append(f"[bold green]✅ All tasks complete[/] — {format_run_completion(session)}")
    lines.append("")
    lines.extend([
        f"[dim]strategy[/] [b]{status.get('strategy', '?')}[/]  "
        f"[dim]·[/]  [b]{status.get('tasks', '?')}[/] [dim]tasks[/]",
    ])

    metrics = _trace_metrics(trace)
    if metrics:
        line = (f"[green]💰 ${metrics.get('cost', '0')}[/]  [dim]·[/]  "
                f"{metrics.get('runs', '0')} [dim]runs[/]")
        if "tokens" in metrics:
            line += f"  [dim]·[/]  [dim]tokens[/] {metrics['tokens']}"
        lines.append(line)

    if status.get("source"):
        lines.append(f"[dim]📄 {os.path.basename(str(status['source']))}[/]")

    iters = _iterations(loop)
    max_iters = status.get("max_iterations", "?")
    current_stage = status.get("stage", "")
    running = state == "RUNNING"
    passed = any(v == "PASS" for _, _, v in iters)

    anchors = len([ln for ln in localization.splitlines() if ln.strip().startswith("- ")])
    plan_steps = len(re.findall(r"^\s*\d+\.", plan, re.MULTILINE))
    all_plans = _plan_files(session)

    def step(done: bool, current: bool, name: str, detail: str = "") -> str:
        if current and running:
            icon, color = "▶", "yellow"
        elif done:
            icon, color = "✓", "green"
        else:
            icon, color = "○", "grey50"
        detail_md = f"  [dim]{detail}[/]" if detail else ""
        return f"  [{color}]{icon}[/] [{color}]{name:<9}[/]{detail_md}"

    last_verdict = iters[-1][2] if iters else ""
    has_eval = any(v in ("PASS", "RETRY", "ESCALATE") for _, _, v in iters)

    lines.append("")
    lines.append("[bold]STEPS[/]")
    lines.append(step(bool(localization), current_stage == "localize", "localize",
                      f"{anchors} anchors" if localization else ""))
    plan_detail = ""
    if plan:
        if len(all_plans) > 1:
            plan_detail = f"{len(all_plans)} plans · {plan_steps} steps"
        else:
            plan_detail = f"{plan_steps} steps"
    lines.append(step(bool(plan), current_stage == "plan", "plan", plan_detail))
    if iters:
        n, tier, _ = iters[-1]
        lines.append(step(
            bool(iters),
            running and current_stage == "run" and not passed and not completed,
            "run",
            f"iter {n}/{max_iters} · {tier}" if not completed else "done",
        ))
        eval_detail = f"last: {last_verdict}" if last_verdict else ""
        lines.append(step(
            passed or completed,
            running and has_eval and not passed and not completed,
            "eval",
            eval_detail if not completed else "PASS",
        ))
    else:
        lines.append(step(False, current_stage == "run", "run"))
        lines.append(step(False, False, "eval"))

    phases = _prd_phases(session.read("prd_phases.md"))
    if phases or iters:
        chain = [f"[dim]{phase}[/]" for phase, _ in phases]
        chain += [f"{tier}·{_verdict_tag(verdict)}" for _, tier, verdict in iters]
        lines.append("")
        lines.append("[bold]TRAJECTORY[/]")
        lines.append("  " + " [dim]→[/] ".join(chain))

    return "\n".join(lines)


def render_trajectory(session: Session) -> str:
    phases = _prd_phases(session.read("prd_phases.md"))
    iters = _iterations(session.read("loop.md"))
    if not phases and not iters:
        return "no iterations yet."
    lines = ["Trajectory:"]
    for i, (phase, detail) in enumerate(phases, 1):
        lines.append(f"  P{i}. {phase}" + (f" · {detail}" if detail else ""))
    for n, tier, verdict in iters:
        lines.append(f"  {n}. {tier} · {verdict}")
    if iters:
        lines.append("\nexpand one with: iter <n>")
    return "\n".join(lines)


def render_iteration(session: Session, n: int) -> str:
    loop = _loop_block(session.read("loop.md"), n)
    run_out = session.read(f"runs/iter-{n}.md")
    ev = _eval_block(session.read("eval.md"), n)
    if not (loop or run_out or ev):
        return f"no iteration {n}."

    chunks: list[str] = []
    if loop:
        chunks.append(f"--- summary ---\n{loop}")
    if run_out:
        chunks.append(f"--- runner output ---\n{run_out.strip()}")
    if ev:
        chunks.append(f"--- eval ---\n{ev}")
    return "\n\n".join(chunks)


def render_expand(session: Session, what: str) -> str:
    if what == "knowledge":
        notes = _knowledge_notes(session)
        if not notes:
            return "===== knowledge =====\n(empty)"
        out: list[str] = []
        for filename, label in notes:
            content = session.read(filename)
            out.append(f"===== {label} =====\n{content.strip() if content else '(empty)'}")
        return "\n\n".join(out)

    if what == "plan":
        plans = _plan_files(session)
        if plans:
            out = []
            for filename, label in plans:
                content = session.read(filename)
                out.append(f"===== {label} =====\n{content.strip() if content else '(empty)'}")
            return "\n\n".join(out)
        content = session.read(_EXPAND_FILES["plan"])
        return f"===== plan =====\n{content.strip() if content else '(empty)'}"

    if what == "all":
        targets = list(_EXPAND_FILES)
        out = []
        for name in targets:
            if name == "plan":
                plans = _plan_files(session)
                if plans:
                    for filename, label in plans:
                        content = session.read(filename)
                        out.append(
                            f"===== {label} =====\n"
                            f"{content.strip() if content else '(empty)'}"
                        )
                    continue
            content = session.read(_EXPAND_FILES[name])
            out.append(f"===== {name} =====\n{content.strip() if content else '(empty)'}")
        notes = _knowledge_notes(session)
        extra = [n for n in notes if n[1] not in ("plan", "localization")
                 and not n[1].startswith("plan-")]
        if extra:
            for filename, label in extra:
                content = session.read(filename)
                out.append(
                    f"===== {label} =====\n{content.strip() if content else '(empty)'}"
                )
        return "\n\n".join(out)

    targets = [what]
    out = []
    for name in targets:
        content = session.read(_EXPAND_FILES[name])
        out.append(f"===== {name} =====\n{content.strip() if content else '(empty)'}")
    return "\n\n".join(out)


# --- live watch ------------------------------------------------------------


def watch_loop(session: Session, interval: float = 2.0) -> None:
    """Re-render the overview every ``interval`` seconds until the run ends."""
    try:
        while True:
            state = _run_state(session)
            sys.stdout.write(_CLEAR)
            _console.print(render_overview(session, state))
            if state != "RUNNING":
                print("\n(run finished)")
                return
            print(f"\n(watching every {interval:g}s — Ctrl-C to stop)")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n(stopped watching)")


# --- entrypoint ------------------------------------------------------------


def run_analyze(
    *,
    session_id: str | None = None,
    expand: str | None = None,
    watch: bool = False,
    interactive: bool | None = None,
) -> int:
    if interactive is None:
        interactive = sys.stdin.isatty() and sys.stdout.isatty()

    # No session, interactive, plain view: open the TUI session browser.
    if session_id is None and interactive and not watch and not expand:
        from splinter.tui import run_session_browser

        return run_session_browser()

    session = Session(session_id)

    if not session.dir.exists() or not session.read_index():
        print(f"no session found: {session.id}")
        return 1

    if watch:
        watch_loop(session)
        return 0

    if expand:
        _console.print(render_overview(session, _run_state(session)))
        print()
        print(render_expand(session, expand))
        return 0

    if interactive:
        from splinter.tui import run_tui

        run_tui(session)
        return 0

    _console.print(render_overview(session, _run_state(session)))
    return 0
