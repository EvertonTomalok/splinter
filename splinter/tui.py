"""Textual TUIs for splinter.

* :class:`AnalyzeApp` — ``splinter analyze`` inspector: a tree of steps + the
  escalation trajectory on the left, a markdown detail pane on the right.
* :class:`RunApp` — ``splinter run`` dashboard: a live overview on the left and a
  real-time log pane on the right streaming what the pipeline is doing, while the
  pipeline executes on a worker thread.

``q`` or ``Ctrl-C`` quits either app (and, for a run, aborts it).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import DataTable, Footer, Header, Markdown, RichLog, Static, Tree
from textual.widgets.tree import TreeNode
from textual.worker import Worker, WorkerState

from splinter.analyze import (
    _iterations,
    _loop_block,
    _run_state,
    _trace_metrics,
    render_overview,
)
from splinter.memory.session import Session, delete_session, list_sessions

REFRESH_SECONDS = 2.0

_STATE_EMOJI = {
    "RUNNING": "🟡",
    "COMPLETED": "🟢",
    "FAILED": "🔴",
    "INTERRUPTED": "🟠",
    "DONE": "🟢",
    "UNKNOWN": "⚪",
}


def _overview_md(session: Session, state: str) -> str:
    status = session.read_status()
    metrics = _trace_metrics(session.read("trace.md"))
    iters = _iterations(session.read("loop.md"))
    anchors = [
        ln for ln in session.read("localization.md").splitlines()
        if ln.strip().startswith("- ")
    ]

    lines = [
        f"# {session.id}",
        f"{_STATE_EMOJI.get(state, '⚪')} **{state}** · "
        f"strategy `{status.get('strategy', '?')}` · tasks {status.get('tasks', '?')}",
        "",
    ]
    if metrics:
        lines.append(
            f"💰 **${metrics.get('cost', '0')}** · "
            f"runs {metrics.get('runs', '0')} · tokens `{metrics.get('tokens', '{}')}`"
        )
        lines.append("")

    lines.append("## Steps")
    lines.append(f"- localize — {len(anchors)} anchors")
    lines.append(f"- plan — {'✓' if session.read('plan.md') else 'pending'}")
    if iters:
        n, tier, verdict = iters[-1]
        lines.append(f"- run/eval — iter {n}/{status.get('max_iterations', '?')} "
                     f"· {tier} · last **{verdict}**")
    else:
        lines.append("- run/eval — pending")

    if iters:
        lines.append("")
        lines.append("## Trajectory")
        lines.append(" → ".join(f"`{tier}·{verdict}`" for _, tier, verdict in iters))
    return "\n".join(lines)


def _iteration_md(session: Session, n: int) -> str:
    summary = _loop_block(session.read("loop.md"), n)
    run_out = session.read(f"runs/iter-{n}.md").strip()

    eval_md = session.read("eval.md")
    parts = re.split(r"^### Iter (\d+):", eval_md, flags=re.MULTILINE)
    eval_block = ""
    for i in range(1, len(parts), 2):
        if int(parts[i]) == n:
            eval_block = parts[i + 1].strip()
            break

    md = [f"# Iteration {n}"]
    if summary:
        md.append("## Summary")
        md.append(summary)
    if run_out:
        md.append("## Runner output")
        md.append(f"```\n{run_out}\n```")
    if eval_block:
        md.append("## Eval verdict")
        md.append(eval_block)
    if len(md) == 1:
        md.append("_no data for this iteration yet._")
    return "\n\n".join(md)


def _file_md(session: Session, label: str, filename: str) -> str:
    content = session.read(filename).strip()
    if not content:
        return f"# {label}\n\n_empty_"
    return f"# {label}\n\n{content}"


class AnalyzeApp(App[None]):
    """Live session inspector."""

    CSS = """
    Tree { width: 38%; border-right: solid $primary; }
    #detail { padding: 0 1; }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("ctrl+c", "quit", "Quit"),
        ("r", "reload", "Refresh"),
    ]

    def __init__(self, session: Session) -> None:
        super().__init__()
        self.session = session
        self._traj_node: TreeNode[Any] | None = None
        self._timer: Any = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            yield Tree("session", id="nav")
            with VerticalScroll():
                yield Markdown(id="detail")
        yield Footer()

    def on_mount(self) -> None:
        self._build_tree()
        self.action_reload()
        self._timer = self.set_interval(REFRESH_SECONDS, self.action_reload)

    # --- tree ---
    def _build_tree(self) -> None:
        tree = self.query_one("#nav", Tree)
        tree.root.expand()

        overview = tree.root.add_leaf("📊 Overview", data={"kind": "overview"})
        overview.allow_expand = False

        steps = tree.root.add("🧩 Steps", expand=True)
        steps.add_leaf("localize", data={"kind": "file", "label": "Localization",
                                         "file": "localization.md"})
        steps.add_leaf("plan", data={"kind": "file", "label": "Plan", "file": "plan.md"})
        steps.add_leaf("trace", data={"kind": "file", "label": "Trace", "file": "trace.md"})

        self._traj_node = tree.root.add("📈 Trajectory", expand=True)
        self._refresh_trajectory()

    def _refresh_trajectory(self) -> None:
        if self._traj_node is None:
            return
        self._traj_node.remove_children()
        for n, tier, verdict in _iterations(self.session.read("loop.md")):
            self._traj_node.add_leaf(
                f"#{n} · {tier} · {verdict}", data={"kind": "iter", "n": n}
            )

    # --- detail ---
    def _detail(self) -> Markdown:
        return self.query_one("#detail", Markdown)

    def _show_overview(self) -> None:
        self._detail().update(_overview_md(self.session, _run_state(self.session)))

    def _render_data(self, data: dict[str, Any] | None) -> None:
        if not data:
            self._show_overview()
            return
        kind = data.get("kind")
        if kind == "iter":
            self._detail().update(_iteration_md(self.session, data["n"]))
        elif kind == "file":
            self._detail().update(_file_md(self.session, data["label"], data["file"]))
        else:
            self._show_overview()

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted[Any]) -> None:
        self._render_data(event.node.data)

    # --- actions ---
    def action_reload(self) -> None:
        state = _run_state(self.session)
        emoji = _STATE_EMOJI.get(state, "⚪")
        self.title = f"splinter analyze · {self.session.id}"
        self.sub_title = f"{emoji} {state}"
        self._refresh_trajectory()

        node = self.query_one("#nav", Tree).cursor_node
        self._render_data(node.data if node is not None else None)

        if state != "RUNNING" and self._timer is not None:
            # Run finished — stop polling.
            self._timer.stop()
            self._timer = None


def run_tui(session: Session) -> None:
    AnalyzeApp(session).run()


# --- session browser -------------------------------------------------------


class SessionPicker(App[str | None]):
    """Pick a session: ↑/↓ navigate, Enter open, d delete, q quit."""

    CSS = "DataTable { height: 1fr; }"

    BINDINGS = [
        ("enter", "open", "Open"),
        ("d", "delete", "Delete"),
        ("q", "quit", "Quit"),
        ("escape", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(cursor_type="row")
        yield Footer()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        # Enter on a row — DataTable emits this; open the session.
        self.exit(event.row_key.value)

    def on_mount(self) -> None:
        self.title = "splinter · sessions"
        table = self.query_one(DataTable)
        table.add_columns("session", "state", "cost")
        self._reload()
        table.focus()

    def _reload(self) -> None:
        table = self.query_one(DataTable)
        table.clear()
        sessions = list_sessions()
        if not sessions:
            self.sub_title = "no sessions"
            return
        self.sub_title = f"{len(sessions)} session(s)"
        for sid in sessions:
            session = Session(sid)
            metrics = _trace_metrics(session.read("trace.md"))
            table.add_row(sid, _run_state(session), f"${metrics.get('cost', '0')}", key=sid)

    def _current_id(self) -> str | None:
        table = self.query_one(DataTable)
        if table.row_count == 0:
            return None
        row = table.coordinate_to_cell_key(table.cursor_coordinate).row_key
        return row.value

    def action_open(self) -> None:
        self.exit(self._current_id())

    def action_delete(self) -> None:
        sid = self._current_id()
        if sid:
            delete_session(sid)
            self._reload()


def run_session_browser() -> int:
    """Loop: pick a session in the TUI, view it, return to the picker on quit."""
    while True:
        sid = SessionPicker().run()
        if not sid:
            return 0
        AnalyzeApp(Session(sid)).run()


# --- run dashboard ---------------------------------------------------------


class _TextualLogHandler(logging.Handler):
    """Forwards ``splinter`` log records to a RichLog on the app thread."""

    def __init__(self, app: RunApp) -> None:
        super().__init__()
        self.app = app

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        try:
            self.app.call_from_thread(self.app.write_log, msg, record.levelno)
        except Exception:
            pass  # app shutting down


class RunApp(App[int]):
    """Live dashboard for ``splinter run``: overview + streaming activity log."""

    CSS = """
    #overview { width: 42%; border-right: solid $primary; padding: 0 1; }
    RichLog { padding: 0 1; }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("ctrl+c", "quit", "Quit"),
    ]

    def __init__(self, session: Session, run_kwargs: dict[str, Any]) -> None:
        super().__init__()
        self.session = session
        self.run_kwargs = run_kwargs
        self.rc = 0
        self.error = ""
        self._timer: Any = None
        self._handler: logging.Handler | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            yield Static(id="overview")
            yield RichLog(id="log", markup=True, wrap=True, highlight=True)
        yield Footer()

    def on_mount(self) -> None:
        self._refresh()
        self._timer = self.set_interval(0.5, self._refresh)

        self._handler = _TextualLogHandler(self)
        self._handler.setFormatter(logging.Formatter("%(message)s"))
        splog = logging.getLogger("splinter")
        splog.setLevel(logging.INFO)
        splog.addHandler(self._handler)

        self.run_worker(self._work, thread=True, name="pipeline", exclusive=True)

    def on_unmount(self) -> None:
        if self._handler is not None:
            logging.getLogger("splinter").removeHandler(self._handler)

    async def action_quit(self) -> None:
        # Kill any running provider subprocess so the worker thread can unblock.
        from splinter import procreg

        procreg.terminate_all()
        self.exit(self.rc)

    def _work(self) -> None:
        from splinter.pipeline import run_pipeline

        try:
            self.rc = run_pipeline(**self.run_kwargs)
        except BaseException as exc:  # noqa: BLE001 — surface any failure in the log
            self.rc = 1
            self.error = str(exc)
            try:
                self.call_from_thread(self.write_log, f"ERROR: {exc}", logging.ERROR)
            except Exception:
                pass

    def write_log(self, msg: str, level: int = logging.INFO) -> None:
        color = {logging.ERROR: "red", logging.WARNING: "yellow"}.get(level)
        self.query_one("#log", RichLog).write(f"[{color}]{msg}[/]" if color else msg)

    def _refresh(self) -> None:
        state = _run_state(self.session)
        self.title = f"splinter run · {self.session.id}"
        self.sub_title = f"{_STATE_EMOJI.get(state, '⚪')} {state}"
        self.query_one("#overview", Static).update(render_overview(self.session, state))

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.name != "pipeline":
            return
        if event.state not in (WorkerState.SUCCESS, WorkerState.ERROR):
            return

        self._refresh()
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

        if self.rc == 0:
            self.write_log("— finished — press q to quit —")
        else:
            # On failure, finish the TUI automatically (after a brief glimpse).
            self.write_log(f"— run failed (rc={self.rc}) — closing —", logging.ERROR)
            from splinter import procreg

            procreg.terminate_all()
            self.set_timer(1.5, lambda: self.exit(self.rc))


def run_with_tui(run_kwargs: dict[str, Any]) -> int:
    """Create a fresh session, run the pipeline on a worker thread under RunApp."""
    from splinter.memory.session import new_session_id

    session = Session(new_session_id())
    app = RunApp(session, {**run_kwargs, "session": session})
    app.run()
    if app.rc != 0:
        print(f"run failed (session {session.id}){': ' + app.error if app.error else ''}")
    else:
        print(f"run complete. session: {session.id}")
    return app.rc
