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
from collections.abc import Iterable
from typing import Any

from rich.markup import escape
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult, SystemCommand
from textual.command import DiscoveryHit, Hit, Hits
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.system_commands import SystemCommandsProvider
from textual.timer import Timer
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Markdown,
    RichLog,
    Rule,
    Select,
    Static,
    TextArea,
    Tree,
)
from textual.widgets.tree import TreeNode
from textual.worker import Worker, WorkerState

from splinter.analyze import (
    _iterations,
    _knowledge_notes,
    _loop_block,
    _plan_files,
    _prd_phases,
    _run_state,
    _trace_metrics,
    format_run_completion,
    render_overview,
)
from splinter.memory.session import Session, delete_session, list_sessions

REFRESH_SECONDS = 2.0
AUTO_REFRESH_SECONDS = 1.0

_STATE_EMOJI = {
    "RUNNING": "🟡",
    "COMPLETED": "🟢",
    "FAILED": "🔴",
    "INTERRUPTED": "🟠",
    "AWAITING_USER": "🟣",
    "PAUSED": "🟠",
    "DONE": "🟢",
    "UNKNOWN": "⚪",
}


def _overview_md(session: Session, state: str) -> str:
    status = session.read_status()
    metrics = _trace_metrics(session.read("trace.md"))
    iters = _iterations(session.read("loop.md"))
    from splinter.agents.localizer import _count_anchors

    anchors_count = _count_anchors(session.read("knowledge/localization.md"))

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
    lines.append(f"- localize — {anchors_count} anchors")
    all_plans = _plan_files(session)
    if len(all_plans) > 1:
        lines.append(f"- plan — {len(all_plans)} plans")
        for filename, label in all_plans:
            content = session.read(filename)
            has_plan = "✓" if content.strip() else "pending"
            lines.append(f"  - {label} — {has_plan}")
    else:
        lines.append(f"- plan — {'✓' if session.read('knowledge/plan.md') else 'pending'}")
    if iters:
        n, tier, verdict = iters[-1]
        lines.append(
            f"- run/eval — iter {n}/{status.get('max_iterations', '?')} "
            f"· {tier} · last **{verdict}**"
        )
    else:
        lines.append("- run/eval — pending")

    phases = _prd_phases(session.read("prd_phases.md"))
    if phases or iters:
        lines.append("")
        lines.append("## Trajectory")
        steps = [f"`{phase}`" for phase, _ in phases]
        steps += [f"`{tier}·{verdict}`" for _, tier, verdict in iters]
        lines.append(" → ".join(steps))
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
        if filename == "trace.md":
            # trace.md is only written once an iteration finishes; show live status
            # instead of a bare "empty" while the run is still working.
            loop = session.read("loop.md").strip()
            if loop:
                return (
                    f"# {label}\n\n_no trace summary yet — run in progress_\n\n"
                    f"## Loop so far\n\n{loop}"
                )
            return f"# {label}\n\n_run in progress — no iterations finished yet._"
        return f"# {label}\n\n_empty_"
    return f"# {label}\n\n{content}"


def _trace_md(session: Session) -> str:
    """Full chronological event log (events.md) + headline metrics."""
    metrics = _trace_metrics(session.read("trace.md"))
    events = session.read("events.md").strip()
    parts = ["# Trace"]
    if metrics:
        parts.append(
            f"💰 **${metrics.get('cost', '0')}** · runs {metrics.get('runs', '0')} · "
            f"tokens `{metrics.get('tokens', '{}')}`"
        )
    body = events if events else "run in progress — no events yet"
    # Fenced so emoji/brackets in the stream render verbatim (no markdown mangling).
    parts.append(f"## Events\n\n```\n{body}\n```")
    return "\n\n".join(parts)


def _prd_phase_md(session: Session, phase: str, detail: str) -> str:
    """Detail for a trajectory phase — routed to the artifact that phase produced,
    not the PRD for every node."""
    phase_l = phase.lower()
    head = f"# {phase}" + (f" — {detail}" if detail else "")

    if phase_l == "run":
        metrics = _trace_metrics(session.read("trace.md"))
        iters = _iterations(session.read("loop.md"))
        parts = [head]
        if metrics:
            parts.append(
                f"💰 **${metrics.get('cost', '0')}** · runs {metrics.get('runs', '0')} · "
                f"tokens `{metrics.get('tokens', '{}')}`"
            )
        if iters:
            parts.append(
                "## Iterations\n" + "\n".join(f"- #{n} · {tier} · {v}" for n, tier, v in iters)
            )
        # Live tail of the event log so an in-flight run shows the model working
        # (it logs tool calls / text before any iteration is finalized in loop.md).
        tail = session.read("events.md").strip().splitlines()[-25:]
        if tail:
            parts.append("## Live\n\n```\n" + "\n".join(tail) + "\n```")
        elif not iters:
            parts.append("_starting run…_")
        return "\n\n".join(parts)

    if phase_l == "strategy":
        from splinter import prd_session

        titles = prd_session.user_story_titles(session.read("prd.md"))
        chosen = detail or str(session.read_status().get("strategy", "?"))
        parts = [head, f"**Strategy:** `{chosen}`"]
        if titles:
            parts.append("## Tasks\n" + "\n".join(f"- {escape(t)}" for t in titles))
        return "\n\n".join(parts)

    # clarify / refine / finalize — PRD lifecycle phases: show the PRD as it stands.
    prd = session.read("prd.md").strip()
    body = prd if prd else "_PRD draft not captured yet._"
    return f"# PRD · {phase}{f' — {detail}' if detail else ''}\n\n{body}"


_PALETTE_CSS = """
    CommandPalette { align-horizontal: right; }
    CommandPalette > Vertical { width: 55%; }
"""

_MAXIMIZE_CSS = """
    App.--maximized {
        #nav { display: none; }
        #overview { display: none; }
        #draftpane { display: none; }
        #run-left { display: none; }
    }
"""


def _find_shortcuts_cmd(screen: Any, app: Any) -> SystemCommand:
    if screen.query("HelpPanel"):
        return SystemCommand(
            "Find Shortcuts",
            "Hide the keys and widget help panel",
            app.action_hide_help_panel,
        )
    return SystemCommand(
        "Find Shortcuts",
        "Show keys and shortcuts for the focused widget",
        app.action_show_help_panel,
    )


class _OrderedCommandsProvider(SystemCommandsProvider):
    """Preserves get_system_commands insertion order (no alphabetical sort)."""

    async def discover(self) -> Hits:
        for cmd in self.app.get_system_commands(self.screen):
            if cmd.discover:
                yield DiscoveryHit(cmd.title, cmd.callback, help=cmd.help)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for cmd in self.app.get_system_commands(self.screen):
            if (match := matcher.match(cmd.title)) > 0:
                yield Hit(match, matcher.highlight(cmd.title), cmd.callback, help=cmd.help)


class AnalyzeApp(App[None]):
    """Live session inspector."""

    CSS = (
        """
    Tree { width: 38%; border-right: solid $primary; }
    #detail { padding: 0 1; }
    """
        + _PALETTE_CSS
        + _MAXIMIZE_CSS
    )

    COMMANDS = {_OrderedCommandsProvider}

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("ctrl+c", "quit", "Quit"),
        ("r", "reload", "Refresh"),
        ("R", "toggle_auto", "Auto-refresh"),
    ]

    _maximized: reactive[bool] = reactive(False)
    _auto: reactive[bool] = reactive(False)

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
        self._do_reload()
        # Auto-refresh starts on while the run is live; Shift+R toggles it,
        # and `r` does a one-shot manual refresh whenever auto is off.
        self._auto = _run_state(self.session) == "RUNNING"

    # --- tree ---
    def _build_tree(self) -> None:
        tree = self.query_one("#nav", Tree)
        tree.root.expand()

        overview = tree.root.add_leaf("📊 Overview", data={"kind": "overview"})
        overview.allow_expand = False

        steps = tree.root.add("🧩 Steps", expand=True)
        if self.session.read("prd.md"):
            steps.add_leaf("prd", data={"kind": "file", "label": "PRD", "file": "prd.md"})
        steps.add_leaf(
            "localize",
            data={"kind": "file", "label": "Localization", "file": "knowledge/localization.md"},
        )
        plans = _plan_files(self.session)
        if plans:
            for filename, label in plans:
                steps.add_leaf(label, data={"kind": "file", "label": label, "file": filename})
        else:
            steps.add_leaf(
                "plan", data={"kind": "file", "label": "Plan", "file": "knowledge/plan.md"}
            )
        steps.add_leaf("trace", data={"kind": "trace"})

        notes = _knowledge_notes(self.session)
        extra = [
            (fn, lbl)
            for fn, lbl in notes
            if lbl not in ("plan", "localization") and not lbl.startswith("plan-")
        ]
        if extra:
            kn = tree.root.add("📝 Knowledge", expand=False)
            for filename, label in extra:
                kn.add_leaf(label, data={"kind": "file", "label": label, "file": filename})

        self._traj_node = tree.root.add("📈 Trajectory", expand=True)
        self._refresh_trajectory()

    def _refresh_trajectory(self) -> None:
        if self._traj_node is None:
            return
        self._traj_node.remove_children()
        for phase, detail in _prd_phases(self.session.read("prd_phases.md")):
            label = f"📝 {phase}" + (f" · {detail}" if detail else "")
            self._traj_node.add_leaf(
                label, data={"kind": "prd_phase", "phase": phase, "detail": detail}
            )
        for n, tier, verdict in _iterations(self.session.read("loop.md")):
            self._traj_node.add_leaf(f"#{n} · {tier} · {verdict}", data={"kind": "iter", "n": n})

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
        elif kind == "prd_phase":
            self._detail().update(_prd_phase_md(self.session, data["phase"], data["detail"]))
        elif kind == "trace":
            self._detail().update(_trace_md(self.session))
        elif kind == "file":
            self._detail().update(_file_md(self.session, data["label"], data["file"]))
        else:
            self._show_overview()

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted[Any]) -> None:
        self._render_data(event.node.data)

    # --- actions ---
    def action_reload(self) -> None:
        # `r` — one-shot manual refresh, only while auto-refresh is off
        # (when auto is on, the interval timer already does the polling).
        if self._auto:
            return
        self._do_reload()

    def action_toggle_auto(self) -> None:
        """Shift+R — flip the 1s auto-refresh on/off."""
        self._auto = not self._auto

    def _do_reload(self) -> None:
        state = _run_state(self.session)
        self.title = f"splinter analyze · {self.session.id}"
        self._refresh_trajectory()

        node = self.query_one("#nav", Tree).cursor_node
        self._render_data(node.data if node is not None else None)

        # Run finished — stop auto-polling (watcher tears the timer down).
        if state != "RUNNING" and self._auto:
            self._auto = False
        else:
            self._update_subtitle(state)

    def _update_subtitle(self, state: str) -> None:
        emoji = _STATE_EMOJI.get(state, "⚪")
        self.sub_title = f"{emoji} {state} · auto-refresh {'on' if self._auto else 'off'}"

    def watch__auto(self, val: bool) -> None:
        if val and self._timer is None:
            self._timer = self.set_interval(AUTO_REFRESH_SECONDS, self._do_reload)
        elif not val and self._timer is not None:
            self._timer.stop()
            self._timer = None
        self._update_subtitle(_run_state(self.session))

    def get_system_commands(self, screen: Any) -> Iterable[SystemCommand]:
        yield _find_shortcuts_cmd(screen, self)
        yield SystemCommand("Theme", "Change the current theme", self.action_change_theme)
        if self._maximized:
            yield SystemCommand("Minimize", "Restore default layout", self.action_toggle_maximize)
        else:
            yield SystemCommand("Maximize", "Maximize right panel", self.action_toggle_maximize)
        yield SystemCommand(
            "Screenshot",
            "Save an SVG screenshot of the current screen",
            lambda: self.set_timer(0.1, self.deliver_screenshot),
        )
        yield SystemCommand("Quit", "Quit the application", self.action_quit)

    def action_toggle_maximize(self) -> None:
        self._maximized = not self._maximized

    def watch__maximized(self, val: bool) -> None:
        self.set_class(val, "--maximized")


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

    def __init__(self, app: Any) -> None:
        super().__init__()
        self.app = app

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        try:
            self.app.call_from_thread(self.app.write_log, msg, record.levelno)
        except Exception:
            pass  # app shutting down


class _GapModal(ModalScreen[str]):
    """Shown when the pipeline pauses due to a provider gap (rc=2).

    Countdown sleeps then retries; or the user can switch to Claude or exit.
    """

    DEFAULT_CSS = """
    _GapModal {
        align: center middle;
        background: $background 60%;
    }
    _GapModal > Vertical#gap-dialog {
        width: 70;
        height: auto;
        border: round $warning;
        background: $surface;
        padding: 1 3;
    }
    _GapModal #gap-header {
        height: auto;
        margin-bottom: 1;
    }
    _GapModal #gap-title {
        text-style: bold;
        color: $warning;
        width: 1fr;
    }
    _GapModal #gap-provider-label {
        color: $text-muted;
        text-align: right;
        width: auto;
    }
    _GapModal #gap-body {
        color: $text;
        margin-bottom: 1;
    }
    _GapModal #gap-countdown {
        text-align: center;
        text-style: bold;
        color: $warning;
        background: $warning 15%;
        height: 1;
        margin-bottom: 1;
        display: none;
    }
    _GapModal.counting #gap-countdown {
        display: block;
    }
    _GapModal #gap-picker {
        height: 3;
        align: center middle;
        margin-bottom: 1;
        display: none;
    }
    _GapModal.picking #gap-picker {
        display: block;
    }
    _GapModal.picking #gap-actions {
        display: none;
    }
    _GapModal #gap-picker-label {
        height: 3;
        content-align: center middle;
        margin-right: 1;
        color: $text-muted;
    }
    _GapModal #gap-picker-unit {
        height: 3;
        content-align: center middle;
        margin: 0 1;
        color: $text-muted;
    }
    _GapModal #gap-duration {
        width: 12;
    }
    _GapModal #start {
        width: auto;
        min-width: 12;
        margin-left: 1;
    }
    _GapModal #gap-actions {
        height: 3;
        align-horizontal: center;
    }
    _GapModal Button {
        width: 1fr;
        height: 3;
        margin: 0 1;
        border: none;
        text-style: bold;
    }
    _GapModal Button:focus {
        text-style: bold underline;
    }
    _GapModal Button.-active {
        border: none;
    }
    """

    BINDINGS = [
        ("s", "press('sleep')", "Sleep & retry"),
        ("c", "press('claude')", "Use Claude"),
        ("e", "press('exit')", "Exit"),
        ("escape", "exit_modal", "Cancel"),
    ]

    countdown: reactive[int] = reactive(0)

    def __init__(
        self,
        kind: str = "",
        provider: str = "",
        retry_after: int | None = None,
    ) -> None:
        super().__init__()
        self._kind = kind
        self._provider = provider
        self._sleep_secs = retry_after or 60
        self._tick_timer: Any = None

    def compose(self) -> ComposeResult:
        kind_label = self._kind.replace("_", " ").title() if self._kind else "Provider Gap"
        is_billing = self._kind == "insufficient_balance"
        body = (
            "Provider is out of balance. Top up, then retry — or switch this run to "
            "Claude to keep going."
            if is_billing
            else "Provider is unavailable after repeated retries. Wait and retry, "
            "switch to Claude, or stop the run."
        )
        with Vertical(id="gap-dialog"):
            with Horizontal(id="gap-header"):
                yield Static(f"⏸  Run Paused · {kind_label}", id="gap-title")
                if self._provider:
                    yield Static(f"via {self._provider}", id="gap-provider-label")
            yield Rule()
            yield Static(body, id="gap-body")
            yield Label("", id="gap-countdown")
            with Horizontal(id="gap-picker"):
                yield Label("Sleep for", id="gap-picker-label")
                yield Input(
                    value=str(self._sleep_secs),
                    type="integer",
                    id="gap-duration",
                )
                yield Label("seconds", id="gap-picker-unit")
                yield Button("Start", id="start", variant="success")
            with Horizontal(id="gap-actions"):
                yield Button("  Sleep & Retry  (s)", id="sleep", variant="warning")
                yield Button("  Use Claude  (c)", id="claude", variant="primary")
                yield Button("  Exit  (e)", id="exit", variant="error")

    def on_mount(self) -> None:
        self.query_one("#claude", Button).focus()

    def watch_countdown(self, value: int) -> None:
        label = self.query_one("#gap-countdown", Label)
        label.update(f"⏳  Retrying in {value}s — press Esc to cancel" if value > 0 else "")

    def action_press(self, button_id: str) -> None:
        try:
            self.query_one(f"#{button_id}", Button).press()
        except Exception:
            if button_id == "exit":
                self.action_exit_modal()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "sleep":
            self._open_picker()
        elif bid == "start":
            self._start_countdown()
        else:
            self._cancel_timer()
            self.dismiss(bid or "exit")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "gap-duration":
            self._start_countdown()

    def _open_picker(self) -> None:
        self.add_class("picking")
        inp = self.query_one("#gap-duration", Input)
        inp.value = str(self._sleep_secs)
        inp.focus()

    def _start_countdown(self) -> None:
        raw = self.query_one("#gap-duration", Input).value.strip()
        try:
            secs = int(raw)
        except ValueError:
            secs = self._sleep_secs
        secs = max(1, secs)
        self._sleep_secs = secs
        self.remove_class("picking")
        self.add_class("counting")
        self.countdown = secs
        self._tick_timer = self.set_interval(1.0, self._tick)

    def _tick(self) -> None:
        self.countdown -= 1
        if self.countdown <= 0:
            self._cancel_timer()
            self.dismiss("retry")

    def _cancel_timer(self) -> None:
        if self._tick_timer is not None:
            self._tick_timer.stop()
            self._tick_timer = None

    def action_exit_modal(self) -> None:
        # Esc cancels the active countdown or duration picker before exiting.
        if self.has_class("counting"):
            self._cancel_timer()
            self.remove_class("counting")
            self.countdown = 0
            self.query_one("#sleep", Button).focus()
            return
        if self.has_class("picking"):
            self.remove_class("picking")
            self.query_one("#sleep", Button).focus()
            return
        self._cancel_timer()
        self.dismiss("exit")


class _AskUserModal(ModalScreen[tuple[str, str] | None]):
    """Shown when the eval loop needs human judgment (ASK_USER / max-tier escalate)."""

    DEFAULT_CSS = """
    _AskUserModal {
        align: center middle;
        background: $background 60%;
    }
    _AskUserModal > Vertical#ask-dialog {
        width: 80;
        height: auto;
        max-height: 90%;
        border: round $primary;
        background: $surface;
        padding: 1 2;
    }
    _AskUserModal #ask-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }
    _AskUserModal #ask-reason {
        color: $text-muted;
        margin-bottom: 1;
    }
    _AskUserModal #ask-response {
        height: 8;
        margin-bottom: 1;
    }
    _AskUserModal #ask-actions {
        height: 3;
        align-horizontal: center;
    }
    _AskUserModal Button {
        width: 1fr;
        height: 3;
        margin: 0 1;
        border: none;
        text-style: bold;
    }
    """

    BINDINGS = [
        ("a", "submit_answer", "Answer"),
        ("p", "jump_premium", "Jump Premium"),
        ("c", "action_cowabunga", "Cowabunga"),
        ("e", "exit_modal", "Exit"),
        ("escape", "exit_modal", "Cancel"),
    ]

    def __init__(self, reason: str = "", corrections: str = "") -> None:
        super().__init__()
        self._reason = reason
        self._corrections = corrections

    def compose(self) -> ComposeResult:
        with Vertical(id="ask-dialog"):
            yield Static("❓  Run Paused · Your input needed", id="ask-title")
            yield Rule()
            yield Static(
                self._reason or "The evaluator needs guidance to continue.",
                id="ask-reason",
            )
            yield Label("Your answer (optional context for the runner):")
            yield TextArea(self._corrections, id="ask-response")
            with Horizontal(id="ask-actions"):
                yield Button("  Answer  (a)", id="answer", variant="success")
                yield Button("  Jump Premium  (p)", id="jump_premium", variant="primary")
                yield Button("  Cowabunga  (c)", id="cowabunga", variant="warning")
                yield Button("  Exit  (e)", id="exit", variant="error")

    def on_mount(self) -> None:
        self.query_one("#answer", Button).focus()

    def action_submit_answer(self) -> None:
        self.query_one("#answer", Button).press()

    def action_jump_premium(self) -> None:
        self.query_one("#jump_premium", Button).press()

    def action_cowabunga(self) -> None:
        self.query_one("#cowabunga", Button).press()

    def action_exit_modal(self) -> None:
        self.query_one("#exit", Button).press()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or "exit"
        if bid == "answer":
            text = self.query_one("#ask-response", TextArea).text.strip()
            self.dismiss(("answer", text))
        elif bid == "jump_premium":
            text = self.query_one("#ask-response", TextArea).text.strip()
            self.dismiss(("jump_premium", text))
        elif bid == "cowabunga":
            self.dismiss(("cowabunga", ""))
        else:
            self.dismiss(None)


class RunApp(App[int]):
    """Live dashboard for ``splinter run``: overview + streaming activity log."""

    CSS = (
        """
    #run-left { width: 42%; border-right: solid $primary; }
    #overview { height: auto; border-bottom: solid $primary; padding: 0 1; }
    #run-editor-pane { height: 1fr; }
    RichLog {
        padding: 0 1;
    }
    """
        + _PALETTE_CSS
        + _MAXIMIZE_CSS
    )

    COMMANDS = {_OrderedCommandsProvider}

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("ctrl+c", "quit", "Quit"),
        ("shift+p", "pause", "Pause"),
        ("ctrl+s", "save_prd", "Save PRD"),
    ]

    _maximized: reactive[bool] = reactive(False)

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
            with Vertical(id="run-left"):
                yield Static(id="overview")
                yield EditorPane(self.session.read("prd.md"), id="run-editor-pane")
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
        logging.getLogger("splinter.live").setLevel(logging.INFO)

        if self.session.read_status().get("state") == "awaiting_user":
            self.call_after_refresh(self._show_ask_user_modal)
        else:
            self.run_worker(self._work, thread=True, name="pipeline", exclusive=True)

    def _show_ask_user_modal(self) -> None:
        st = self.session.read_status()
        reason = str(st.get("ask_reason", ""))
        corrections = str(st.get("ask_corrections", ""))
        cp = self.session.read("run_checkpoint.json").strip()
        if cp:
            try:
                import json

                data = json.loads(cp)
                gate = str(data.get("gate_output", "")).strip()
                corr = str(data.get("corrections", "")).strip()
                parts = [p for p in (corr, gate) if p]
                if parts:
                    corrections = "\n\n".join(parts)
            except (json.JSONDecodeError, TypeError):
                pass

        def _on_choice(result: tuple[str, str] | None) -> None:
            if self._timer is None:
                self._timer = self.set_interval(0.5, self._refresh)
            if result is None:
                self.exit(3)
                return
            action, text = result
            if action == "answer":
                self.write_log("— continuing with your answer —", logging.WARNING)
                self._run_pipeline_worker(
                    resume=True, user_guidance=text or None, jump_premium=False, cowabunga=False
                )
            elif action == "jump_premium":
                self.write_log("— jumping to premium tier —", logging.WARNING)
                self._run_pipeline_worker(
                    resume=True,
                    user_guidance=text or None,
                    jump_premium=True,
                    cowabunga=False,
                )
            elif action == "cowabunga":
                self.write_log("— cowabunga — proceeding autonomously —", logging.WARNING)
                self._run_pipeline_worker(
                    resume=True, user_guidance=None, jump_premium=False, cowabunga=True
                )
            else:
                self.exit(3)

        self.push_screen(_AskUserModal(reason, corrections), callback=_on_choice)

    def _run_pipeline_worker(
        self,
        *,
        resume: bool = False,
        user_guidance: str | None = None,
        jump_premium: bool = False,
        cowabunga: bool = False,
        claude_runner_fallback: bool = False,
    ) -> None:
        def _run() -> None:
            from splinter.pipeline import run_pipeline

            kwargs = {
                **self.run_kwargs,
                "resume": resume,
                "user_guidance": user_guidance,
                "jump_premium": jump_premium,
                "cowabunga": cowabunga or bool(self.run_kwargs.get("cowabunga")),
                "claude_runner_fallback": claude_runner_fallback,
            }
            try:
                self.rc = run_pipeline(**kwargs)
            except BaseException as exc:  # noqa: BLE001
                self.rc = 1
                self.error = str(exc)
                try:
                    self.call_from_thread(self.write_log, f"ERROR: {exc}", logging.ERROR)
                except Exception:
                    pass

        self.run_worker(_run, thread=True, name="pipeline", exclusive=True)

    def on_unmount(self) -> None:
        if self._handler is not None:
            logging.getLogger("splinter").removeHandler(self._handler)

    async def action_quit(self) -> None:
        # Kill any running provider subprocess so the worker thread can unblock.
        from splinter import procreg

        procreg.terminate_all()
        self.exit(self.rc)

    async def action_pause(self) -> None:
        """Shift+P — kill current subprocess and pause the run."""
        from splinter import procreg

        procreg.terminate_all()
        self.session.set_status("paused", reason="user_pause")
        self.write_log(
            "— paused by user (Shift+P) — resume with: splinter resume —", logging.WARNING
        )
        self.rc = 2
        self.exit(2)

    def action_save_prd(self) -> None:
        """Ctrl+S — persist live PRD edits back to the session (picked up on the
        next task / resume; tasks already in flight keep their loaded copy)."""
        content = self.query_one("#run-editor-pane", EditorPane).get_content()
        self.session.write("prd.md", content)
        self.write_log("— PRD saved —", logging.WARNING)

    def _work(self) -> None:
        self._run_pipeline_worker()

    def write_log(self, msg: str, level: int = logging.INFO) -> None:
        # Streamed model text/tool args are arbitrary — escape so stray `[` markup
        # (e.g. "fix [bug]") doesn't raise MarkupError when the RichLog renders.
        safe = escape(msg)
        color = {logging.ERROR: "red", logging.WARNING: "yellow"}.get(level)
        self.query_one("#log", RichLog).write(f"[{color}]{safe}[/]" if color else safe)

    def _write_run_complete(self) -> None:
        summary = format_run_completion(self.session)
        log = self.query_one("#log", RichLog)
        log.write(f"[bold green]✅ RUN COMPLETE[/] — {escape(summary)}")
        log.write("[dim]press q to quit · uv run splinter analyze to inspect[/]")
        self.sub_title = f"🟢 COMPLETE · {summary}"

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
            self._write_run_complete()
        elif self.rc == 2:
            self.write_log("— run PAUSED (provider gap) —", logging.WARNING)
            st = self.session.read_status()
            kind = str(st.get("kind", ""))
            provider = str(st.get("provider", ""))
            retry_after = st.get("retry_after")
            self.call_after_refresh(self._show_gap_modal, kind, provider, retry_after)
        elif self.rc == 3:
            self.write_log("— run PAUSED (needs your input) —", logging.WARNING)
            self.call_after_refresh(self._show_ask_user_modal)
        else:
            # On failure, finish the TUI automatically (after a brief glimpse).
            self.write_log(f"— run failed (rc={self.rc}) — closing —", logging.ERROR)
            from splinter import procreg

            procreg.terminate_all()
            self.set_timer(1.5, lambda: self.exit(self.rc))

    def _show_gap_modal(self, kind: str, provider: str = "", retry_after: object = None) -> None:
        try:
            ra: int | None = (
                int(retry_after) if isinstance(retry_after, (int, str, float)) else None
            )
        except (TypeError, ValueError):
            ra = None

        def _on_choice(choice: str | None) -> None:
            if self._timer is None:
                self._timer = self.set_interval(0.5, self._refresh)
            if choice == "claude":
                self.write_log("— switching to Claude (sonnet @ high) —", logging.WARNING)
                self._run_pipeline_worker(resume=True, claude_runner_fallback=True)
            elif choice == "retry":
                self.write_log("— sleep done, retrying… —", logging.WARNING)
                self._run_pipeline_worker(resume=True)
            else:
                self.exit(2)

        self.push_screen(_GapModal(kind, provider, ra), callback=_on_choice)

    def get_system_commands(self, screen: Any) -> Iterable[SystemCommand]:
        yield _find_shortcuts_cmd(screen, self)
        yield SystemCommand("Theme", "Change the current theme", self.action_change_theme)
        if self._maximized:
            yield SystemCommand("Minimize", "Restore default layout", self.action_toggle_maximize)
        else:
            yield SystemCommand("Maximize", "Maximize right panel", self.action_toggle_maximize)
        yield SystemCommand(
            "Screenshot",
            "Save an SVG screenshot of the current screen",
            lambda: self.set_timer(0.1, self.deliver_screenshot),
        )
        yield SystemCommand("Quit", "Quit the application", self.action_quit)

    def action_toggle_maximize(self) -> None:
        self._maximized = not self._maximized

    def watch__maximized(self, val: bool) -> None:
        self.set_class(val, "--maximized")


def run_with_tui(run_kwargs: dict[str, Any], session: Session | None = None) -> int:
    """Run the pipeline on a worker thread under RunApp.

    A ``session`` may be supplied (e.g. the one the PRD conversation just wrote
    into) so the PRD draft and its execution live in the same session dir;
    otherwise a fresh session is created.
    """
    from splinter.memory.session import new_session_id

    if session is None:
        session = Session(new_session_id())
    app = RunApp(session, {**run_kwargs, "session": session})
    app.run()
    if app.rc != 0:
        print(f"run failed (session {session.id}){': ' + app.error if app.error else ''}")
    else:
        print(f"run complete. session: {session.id}")
    return app.rc


# --- configure -------------------------------------------------------------


class ConfigureApp(App[bool]):
    """Pick a model per pipeline step, then write config.yaml on save."""

    CSS = """
    #rows { padding: 0 1; }
    .step {
        height: auto;
        min-height: 3;
        border-left: thick $primary;
        padding-left: 1;
        margin-bottom: 1;
    }
    .step.run { border-left: thick $success; }
    .step-info { width: 44; height: 100%; align: left middle; }
    .step-name { text-style: bold; height: 1; }
    .step-desc { color: $text-muted; height: auto; }
    .model-sel { width: 1fr; height: 3; }
    .effort-sel { width: 14; height: 3; margin-left: 1; }
    .timeout-inp { width: 14; height: 3; margin-left: 1; }
    Select > SelectCurrent { height: 3; }
    """

    BINDINGS = [
        ("s", "save", "Save"),
        ("q", "quit", "Quit"),
        ("ctrl+c", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        from splinter.configure import available_models, current_model_selections

        self.saved = False
        self.saved_path = ""
        self._models = available_models()
        current = current_model_selections()
        self._cur_models = current["models"]
        self._cur_efforts = current["efforts"]
        self._cur_timeouts = current["timeouts"]

    @staticmethod
    def _select(
        options: list[tuple[str, str]], current: object, choices: list[str], **kwargs: Any
    ) -> Select[str]:
        # Pass value= only for a real selection; this Textual rejects value=BLANK.
        if current in choices:
            return Select(options, value=str(current), **kwargs)
        return Select(options, **kwargs)

    def _row(
        self,
        sid: str,
        name: str,
        desc: str,
        model: object,
        effort: object,
        timeout: object = None,
        *,
        run: bool = False,
    ) -> Horizontal:
        from splinter.configure import EFFORT_CHOICES

        model_opts = [(m, m) for m in self._models]
        effort_opts = [(e, e) for e in EFFORT_CHOICES]
        info = Vertical(
            Label(name, classes="step-name"),
            classes="step-info",
        )
        model_sel = self._select(
            model_opts, model, self._models, id=sid, tooltip=desc, classes="model-sel"
        )
        effort_sel = self._select(
            effort_opts,
            effort,
            EFFORT_CHOICES,
            id=f"{sid}__eff",
            prompt="effort",
            tooltip="reasoning effort",
            classes="effort-sel",
        )
        timeout_inp = Input(
            value=str(timeout) if timeout else "",
            id=f"{sid}__to",
            type="integer",
            placeholder="3600",
            tooltip="per-call timeout (seconds)",
            classes="timeout-inp",
        )
        return Horizontal(
            info,
            model_sel,
            effort_sel,
            timeout_inp,
            classes="step run" if run else "step",
        )

    def compose(self) -> ComposeResult:
        from splinter.configure import MODEL_STEPS, TIER_STEPS

        rows: list[Horizontal] = [
            self._row(
                key,
                label,
                desc,
                self._cur_models.get(key),
                self._cur_efforts.get(key),
                self._cur_timeouts.get(key),
            )
            for key, label, desc in MODEL_STEPS
        ]
        tier_models = self._cur_models.get("tiers", [])
        tier_efforts = self._cur_efforts.get("tiers", [])
        tier_timeouts = self._cur_timeouts.get("tiers", [])
        for i, (label, desc) in enumerate(TIER_STEPS):
            rows.append(
                self._row(
                    f"tier_{i}",
                    label,
                    desc,
                    tier_models[i] if i < len(tier_models) else None,
                    tier_efforts[i] if i < len(tier_efforts) else None,
                    tier_timeouts[i] if i < len(tier_timeouts) else None,
                    run=True,
                )
            )

        yield Header()
        yield VerticalScroll(*rows, id="rows")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "splinter · configure"
        self.sub_title = "model · effort · timeout per step — s: save · q: cancel"

    def action_save(self) -> None:
        from splinter.configure import MODEL_STEPS, TIER_STEPS, write_model_config

        def sel_value(sid: str) -> str:
            value = self.query_one(f"#{sid}", Select).value
            # A blank Select yields the BLANK sentinel (not a str) — treat as "".
            return value if isinstance(value, str) else ""

        def to_value(sid: str) -> int | None:
            raw = self.query_one(f"#{sid}", Input).value.strip()
            return int(raw) if raw.isdigit() and int(raw) > 0 else None

        models: dict[str, Any] = {}
        efforts: dict[str, Any] = {}
        timeouts: dict[str, Any] = {}
        for key, _, _ in MODEL_STEPS:
            if sel_value(key):
                models[key] = sel_value(key)
            efforts[key] = sel_value(f"{key}__eff")
            timeouts[key] = to_value(f"{key}__to")

        tier_models: list[str] = []
        tier_efforts: list[str] = []
        tier_timeouts: list[int | None] = []
        for i in range(len(TIER_STEPS)):
            model = sel_value(f"tier_{i}") or self._cur_models["tiers"][i]
            tier_models.append(model)
            tier_efforts.append(sel_value(f"tier_{i}__eff"))
            tier_timeouts.append(to_value(f"tier_{i}__to"))
        models["tiers"] = tier_models
        efforts["tiers"] = tier_efforts
        timeouts["tiers"] = tier_timeouts

        self.saved_path = str(write_model_config(models, efforts, timeouts=timeouts))
        self.saved = True
        self.exit(True)


def run_configure_tui() -> int:
    app = ConfigureApp()
    app.run()
    if app.saved:
        print(f"config written to {app.saved_path}")
    else:
        print("configure cancelled — nothing written.")
    return 0


# --- interactive PRD session -----------------------------------------------


def _fm_block(prd_text: str) -> tuple[dict[str, Any], str]:
    """Split a PRD into (frontmatter dict, body)."""
    import yaml

    if prd_text.startswith("---"):
        parts = prd_text.split("---", 2)
        if len(parts) >= 3:
            try:
                fm = yaml.safe_load(parts[1]) or {}
            except yaml.YAMLError:
                fm = {}
            return (fm if isinstance(fm, dict) else {}), parts[2]
    return {}, prd_text


def _set_fm_strategy(prd_text: str, strategy: str) -> str:
    """Force the ``strategy:`` field in the PRD frontmatter to ``strategy``."""
    fm, body = _fm_block(prd_text)
    if not fm:
        return prd_text
    fm["strategy"] = strategy
    import yaml

    return f"---\n{yaml.safe_dump(fm, sort_keys=False).strip()}\n---{body}"


class ConfirmQuit(ModalScreen[bool]):
    """Are-you-sure dialog before abandoning a PRD session."""

    CSS = """
    ConfirmQuit { align: center middle; }
    #box {
        width: 80; max-width: 90%; height: auto; padding: 1 2;
        border: thick $warning; background: $surface;
    }
    #box Static { width: 100%; height: auto; }
    #cmd { color: $text; background: $boost; padding: 0 1; margin: 1 0; }
    #qbuttons { height: 3; align-horizontal: center; margin-top: 1; }
    #qbuttons Button { margin: 0 1; }
    """

    BINDINGS = [("escape", "stay", "Stay")]

    def __init__(self, session_id: str) -> None:
        super().__init__()
        self.session_id = session_id

    def compose(self) -> ComposeResult:
        with Vertical(id="box"):
            yield Static("Leave this PRD session?")
            yield Static("The draft is saved — resume later with:")
            yield Static(f"uv run splinter resume {self.session_id}", id="cmd")
            with Horizontal(id="qbuttons"):
                yield Button("Leave", id="leave", variant="error")
                yield Button("Stay", id="stay", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "leave")

    def action_stay(self) -> None:
        self.dismiss(False)


class ComposerTextArea(TextArea):
    """PRD composer box: Enter submits; Shift+Enter inserts a newline."""

    _NEWLINE_KEYS = ("shift+enter",)

    async def _on_key(self, event: events.Key) -> None:
        if event.key in self._NEWLINE_KEYS:
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            assert isinstance(self.app, PrdSessionApp)
            self.app.action_send()
            return
        await super()._on_key(event)


class EditorPane(Static):
    """Shared editor pane: TextArea (editable) + Markdown (preview)."""

    CSS = """
    EditorPane {
        width: 100%;
        height: 100%;
    }
    EditorPane > Horizontal {
        width: 100%;
        height: 100%;
    }
    EditorPane #editor {
        width: 50%;
        border-right: solid $primary;
    }
    EditorPane #preview {
        width: 50%;
        padding: 0 1;
    }
    """

    def __init__(self, initial_content: str = "", id: str | None = None) -> None:
        super().__init__(id=id)
        self._initial_content = initial_content
        self._update_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield TextArea(id="editor")
            yield Markdown(id="preview")

    def on_mount(self) -> None:
        editor = self.query_one("#editor", TextArea)
        editor.text = self._initial_content
        self._refresh_preview()
        self._update_timer = self.set_interval(0.5, self._refresh_preview)

    def on_unmount(self) -> None:
        if self._update_timer:
            self._update_timer.stop()

    def _refresh_preview(self) -> None:
        try:
            editor = self.query_one("#editor", TextArea)
            preview = self.query_one("#preview", Markdown)
            content = editor.text or "_(empty)_"
            preview.update(content)
        except Exception:
            pass

    def get_content(self) -> str:
        try:
            return self.query_one("#editor", TextArea).text
        except Exception:
            return ""

    def set_content(self, content: str) -> None:
        try:
            editor = self.query_one("#editor", TextArea)
            editor.text = content
            self._refresh_preview()
        except Exception:
            pass


class PrdSessionApp(App[int | None]):
    """Refine a PRD with the user, pick a strategy, then hand off to the runner.

    Left pane: empty editable instructions. Right pane: PRD preview + conversation.
    Phases: ``generate`` (from instructions) → ``strategy`` (pick a turtle) → ``review``
    (eyeball the user stories) → exit with the run kwargs.
    """

    CSS = (
        """
    #draftpane {
        width: 50%;
        border-right: solid $primary;
    }
    #instructions {
        height: 1fr;
        border: round $primary;
    }
    #chatpane {
        width: 1fr;
    }
    #convo {
        height: 1fr;
        padding: 0 1;
    }
    #preview {
        height: 1fr;
        padding: 0 1;
        display: none;
    }
    #preview.-active {
        display: block;
    }
    #composer {
        dock: bottom;
        height: auto;
    }
    #entry {
        height: 8;
        border: round $primary;
    }
    #actions {
        height: 4;
        padding: 0 1;
    }
    #actions Button {
        height: 4;
        margin: 0 1 0 0;
    }
    """
        + _PALETTE_CSS
        + _MAXIMIZE_CSS
    )

    COMMANDS = {_OrderedCommandsProvider}

    BINDINGS = [
        ("ctrl+c", "abort", "Abort"),
        ("escape", "abort", "Abort"),
        ("ctrl+s", "send", "Send"),
    ]

    _maximized: reactive[bool] = reactive(False)

    def __init__(self, session: Session, run_kwargs: dict[str, Any]) -> None:
        super().__init__()
        self.session = session
        self.run_kwargs = run_kwargs
        self.cowabunga = bool(run_kwargs.get("cowabunga"))
        self.resuming = bool(run_kwargs.get("resume"))
        self.trusted = False
        self.phase = "init"
        self.claude_session = ""
        self.final_prd = ""
        self.strategy: str | None = run_kwargs.get("strategy")
        self._busy = False
        self._initial_prd = ""
        self._desc = run_kwargs.get("description", "")
        self._convo_lines: list[str] = []
        self._generating = False

    def _save_state(self) -> None:
        """Persist enough to resume this refinement: conversation id, phase, strategy."""
        self.session.set_status(
            "refining",
            source="prd",
            phase=self.phase,
            claude_session=self.claude_session,
            strategy=self.strategy or "?",
        )

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical(id="draftpane"):
                entry = ComposerTextArea(id="instructions", soft_wrap=True)
                entry.border_title = "Instructions"
                entry.border_subtitle = "↵ generate · ⇧↵ newline"
                yield entry
            with Vertical(id="chatpane"):
                yield RichLog(id="convo", markup=True, wrap=True)
                yield Markdown(id="preview")
                with Vertical(id="composer"):
                    entry_reply = ComposerTextArea(id="entry", soft_wrap=True)
                    entry_reply.border_subtitle = "↵ send · ⇧↵ newline"
                    yield entry_reply
                    with Horizontal(id="actions"):
                        yield Button("Send (⌃S)", id="send", variant="primary")
                        yield Button("Generate PRD", id="generate", variant="success")
        yield Footer()

    def on_mount(self) -> None:
        from pathlib import Path

        from splinter import prd_session

        self.title = "splinter · PRD"
        self.sub_title = "🤙 cowabunga" if self.cowabunga else "refining"
        path = self.run_kwargs.get("prd_path")
        try:
            self._initial_prd = Path(path).read_text() if path else ""
        except OSError as exc:
            self._fail(f"cannot read PRD: {exc}")
            return
        if not self._initial_prd.strip():
            self._initial_prd = self.session.read("prd.md")
        fm, _ = _fm_block(self._initial_prd)
        if not self._desc:
            self._desc = str(fm.get("feature", "")) or self._first_line(self._initial_prd)
        self._set_preview(self._initial_prd)

        if self.resuming:
            self._resume()
            return

        route = prd_session.route_prd(self._initial_prd)
        if route == "generate":
            self._generating = True
            self._say("[dim]Write instructions on the left, then click 'Generate PRD'.[/]")
            self._set_busy(False, "instructions")
            self._focus_instructions()
        elif self._initial_prd.strip() and not self.cowabunga:
            self.trusted = True
            self._enter_trusted()
        else:
            if self.cowabunga:
                self._set_busy(True, "cowabunga — the model is deciding everything…")
                self._say("[magenta]🤙 cowabunga — no questions, finalizing the PRD myself.[/]")
                self._spawn(self._finalize_worker, autodecide=True)
            else:
                self._set_busy(True, "reading the PRD, drafting questions…")
                self._say("[dim]reading the PRD, drafting clarifying questions…[/]")
                self._spawn(self._questions_worker)

    @staticmethod
    def _first_line(text: str) -> str:
        for ln in text.splitlines():
            if ln.strip() and not ln.strip().startswith("---"):
                return ln.strip()
        return "feature"

    def _resume(self) -> None:
        """Re-enter a saved refinement at its last phase, reusing the conversation."""
        from splinter.strategies.registry import available_strategies

        status = self.session.read_status()
        self.claude_session = str(status.get("claude_session", "") or "")
        saved = status.get("strategy")
        if saved and saved != "?":
            self.strategy = str(saved)
        self.final_prd = self.session.read("prd.md")
        phase = str(status.get("phase") or "chat")

        self._replay_convo()
        self._say(f"[magenta]⟳ resumed {self.session.id} at phase '{phase}'.[/]")
        if not self.claude_session:
            self._say(
                "[yellow]No saved conversation id — prior context may be lost; "
                "answers still apply to the current draft.[/]"
            )

        if phase == "trust":
            self.phase = "trust"
            self.trusted = True
            self._enter_trusted()
        elif phase == "review":
            self.phase = "review"
            self._show_stories()
            self._say(
                "[green]Type 'accept' to run, 'cowabunga' to run as-is, or describe changes.[/]"
            )
            self._render_actions("review")
            self._set_busy(False, "accept / edit / gate: <cmds> / changes / cowabunga")
        elif phase == "strategy":
            self.phase = "strategy"
            self._say(
                "Pick a strategy "
                f"({', '.join(available_strategies())}), or 'cowabunga' to let me decide."
            )
            self._render_actions("strategy")
            self._set_busy(False, "strategy name / cowabunga")
        else:  # clarify / refine both live in the chat phase
            self.phase = "chat"
            self._say("[green]Continue: answer, 'fulfilled' to finalize, or 'cowabunga'.[/]")
            self._render_actions("chat")
            self._set_busy(False, "your answers / fulfilled / cowabunga")
        self._save_state()

    def _enter_trusted(self) -> None:
        """Load non-empty PRD as-is into editable left pane; no generation step."""
        from splinter import prd_session

        self.phase = "trust"
        self.final_prd = self._initial_prd

        def _complete_mount() -> None:
            prd_session.log_phase(self.session, "trust")
            self._say("[green]PRD loaded. Edit on the left if needed, then Send PRD.[/]")
            self._render_actions("trust")
            self._set_busy(False, "edit / send PRD / cowabunga")
            self._save_state()

        async def _mount_edit() -> None:
            draftpane = self.query_one("#draftpane", Vertical)
            await draftpane.remove_children()
            edit = TextArea(id="draft-edit", soft_wrap=True, text=self._initial_prd)
            await draftpane.mount(edit)
            self.call_from_thread(_complete_mount)

        self.run_worker(_mount_edit(), name="mount-edit")

    def _read_trusted_draft(self) -> str:
        """Read edited text from draft-edit TextArea."""
        try:
            return self.query_one("#draft-edit", TextArea).text
        except Exception:
            return self._initial_prd

    def _to_strategy_phase(self) -> None:
        """Transition to strategy selection phase (shared by multiple paths)."""
        from splinter.strategies.registry import available_strategies

        self._say("[green]✅ PRD finalized.[/]")
        self._say(
            "Pick a strategy "
            f"({', '.join(available_strategies())}), or 'cowabunga' to let me decide & run."
        )
        self.phase = "strategy"
        self._save_state()
        self._render_actions("strategy")
        self._set_busy(False, "strategy name / cowabunga")

    def _accept_trusted(self) -> None:
        """Accept edited PRD and proceed to strategy selection."""
        from splinter import prd_session

        text = self._read_trusted_draft()
        self.final_prd = prd_session.ensure_frontmatter(
            text, description=self._desc, strategy=self.strategy
        )
        self._set_preview(self.final_prd)
        prd_session.log_phase(self.session, "trust-accept")
        n_stories = len(prd_session.user_story_titles(self.final_prd))
        if n_stories:
            self._show_stories()
        self._to_strategy_phase()

    def _on_trust(self, text: str) -> None:
        """Handle composer input in trust phase: cowabunga triggers run."""
        from splinter import prd_session

        if prd_session.is_cowabunga(text):
            self.final_prd = self._read_trusted_draft()
            self.final_prd = prd_session.ensure_frontmatter(
                self.final_prd, description=self._desc, strategy=self.strategy
            )
            self._begin_run(autopick=True)

    # --- ui helpers ---
    def _say(self, msg: str, *, persist: bool = True) -> None:
        self.query_one("#convo", RichLog).write(msg)
        if persist:
            self._convo_lines.append(msg)
            # NUL-delimit records: a message may contain newlines, and splitting on
            # them would break a markup tag across lines (MarkupError on replay).
            self.session.write("convo.md", "\x00".join(self._convo_lines))

    def _replay_convo(self) -> None:
        """Reprint the saved conversation so a resumed session shows what was asked."""
        prior = self.session.read("convo.md")
        if not prior.strip():
            return
        # New sessions are NUL-delimited; tolerate the old newline format too.
        records = prior.split("\x00") if "\x00" in prior else prior.splitlines()
        records = [r for r in records if r]
        convo = self.query_one("#convo", RichLog)
        for rec in records:
            # RichLog defers markup parsing to render time, so a try/except around
            # write() won't catch bad markup — validate now and escape if it fails.
            try:
                Text.from_markup(rec)
            except Exception:  # noqa: BLE001 — bad markup in an old log shouldn't abort resume
                rec = escape(rec)
            convo.write(rec)
        self._convo_lines = records
        convo.write("[dim]─── resumed ───[/]")

    def _set_busy(self, busy: bool, placeholder: str = "") -> None:
        self._busy = busy
        entry = self.query_one("#entry", TextArea)
        entry.disabled = busy
        for btn in self.query("#actions Button"):
            btn.disabled = busy
        if placeholder:
            if self._generating:
                self.query_one("#instructions", TextArea).border_title = placeholder
            else:
                entry.border_title = placeholder
        if not busy:
            if self._generating:
                self._focus_instructions()
            else:
                entry.focus()

    def _render_actions(self, phase: str) -> None:
        """Swap the action buttons to match the phase (chat / strategy / review / trust)."""
        from splinter.strategies.registry import registered_strategies

        btns = [Button("Send (⌃S)", id="send", variant="primary")]
        if phase == "strategy":
            for cls in registered_strategies():
                alias = cls.aliases[0] if cls.aliases else cls.name
                label = Text.from_markup(f"[b]{alias}[/]\n[dim]{cls.name}[/]")
                btns.append(Button(label, id=f"strat-{cls.name}", variant="success"))
            btns.append(Button("Cowabunga", id="cowabunga", variant="warning"))
        elif phase == "review":
            btns.append(Button("Accept", id="accept", variant="success"))
            btns.append(Button("Edit", id="edit", variant="primary"))
            btns.append(Button("Cowabunga", id="cowabunga", variant="warning"))
        elif phase == "trust":
            btns.append(Button("Send PRD", id="accept", variant="success"))
            btns.append(Button("Cowabunga", id="cowabunga", variant="warning"))
        else:  # chat
            btns.append(Button("Fulfilled", id="fulfilled", variant="success"))
            btns.append(Button("Cowabunga", id="cowabunga", variant="warning"))
        bar = self.query_one("#actions", Horizontal)

        async def _swap() -> None:
            # await the removal before mounting, else the old ids collide with the new.
            await bar.remove_children()
            await bar.mount(*btns)

        self.run_worker(_swap(), name="actions")

    def _set_preview(self, md: str) -> None:
        preview = self.query_one("#preview", Markdown)
        if md.strip():
            preview.update(md)
            preview.add_class("-active")
        else:
            preview.update("_(empty)_")
            preview.add_class("-active")
        if md.strip():
            self.session.write("prd.md", md)

    def _focus_instructions(self) -> None:
        self.query_one("#instructions", TextArea).focus()

    def _fail(self, msg: str) -> None:
        self._say(f"[red]ERROR: {escape(msg)}[/]")
        self.set_timer(2.0, lambda: self.exit(None))

    def _spawn(self, fn: Any, **kw: Any) -> None:
        from functools import partial

        self.run_worker(partial(fn, **kw), thread=True, name="prd")

    def _seed(self) -> str | None:
        """Draft to re-anchor the model when the saved conversation id was lost.

        Returns ``None`` while a conversation is live (server keeps the context);
        only kicks in for a resumed session with no ``claude_session`` id.
        """
        if self.claude_session:
            return None
        return self.final_prd or self._initial_prd or None

    # --- workers (run off the UI thread) ---
    def _questions_worker(self) -> None:
        from splinter import prd_session

        try:
            turn = prd_session.open_questions(self._initial_prd, strategy=self.strategy)
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(self._fail, f"PRD model: {exc}")
            return
        self.call_from_thread(self._after_questions, turn.text, turn.session_id)

    def _refine_worker(self, answers: str) -> None:
        from splinter import prd_session

        try:
            turn = prd_session.refine(answers, resume=self.claude_session, prd_text=self._seed())
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(self._fail, f"PRD model: {exc}")
            return
        self.call_from_thread(self._after_refine, turn.text, turn.session_id)

    def _finalize_worker(self, autodecide: bool) -> None:
        from splinter import prd_session

        try:
            turn = prd_session.finalize(
                resume=self.claude_session,
                strategy=self.strategy,
                autodecide=autodecide,
                prd_text=self._seed(),
            )
            prd = prd_session.ensure_frontmatter(
                turn.text, description=self._desc, strategy=self.strategy
            )
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(self._fail, f"PRD model: {exc}")
            return
        self.call_from_thread(self._after_finalize, prd, turn.session_id)

    def _revise_worker(self, instructions: str) -> None:
        from splinter import prd_session

        try:
            turn = prd_session.revise_final(
                instructions, resume=self.claude_session, prd_text=self._seed()
            )
            prd = prd_session.ensure_frontmatter(
                turn.text, description=self._desc, strategy=self.strategy
            )
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(self._fail, f"PRD model: {exc}")
            return
        self.call_from_thread(self._after_revise, prd, turn.session_id)

    def _generate_worker(self, instructions: str) -> None:
        from splinter import prd_session

        try:
            turn = prd_session.generate_prd(instructions, strategy=self.strategy)
            prd = prd_session.ensure_frontmatter(
                turn.text, description=self._desc or "feature", strategy=self.strategy
            )
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(self._fail, f"PRD model: {exc}")
            return
        self.call_from_thread(self._after_generate, prd, turn.session_id)

    # --- worker callbacks (back on the UI thread) ---
    def _after_questions(self, questions: str, sid: str) -> None:
        from splinter import prd_session

        self.claude_session = sid
        prd_session.log_phase(self.session, "clarify")
        self.phase = "chat"
        self._save_state()
        self._set_preview(self._initial_prd)
        self._say(escape(questions))
        self._say(
            "[green]Answer (e.g. 1A,2C), or type 'fulfilled' to finalize, "
            "or 'cowabunga' to let me decide.[/]"
        )
        self.phase = "chat"
        self._render_actions("chat")
        self._set_busy(False, "your answers / fulfilled / cowabunga")

    def _after_refine(self, draft: str, sid: str) -> None:
        from splinter import prd_session

        self.claude_session = sid
        prd_session.log_phase(self.session, "refine")
        self.phase = "chat"
        self._save_state()
        self._set_preview(draft)
        self._say("[green]Updated. Answer remaining questions, 'fulfilled', or 'cowabunga'.[/]")
        self._render_actions("chat")
        self._set_busy(False, "your answers / fulfilled / cowabunga")

    def _after_finalize(self, prd: str, sid: str) -> None:
        from splinter import prd_session

        self.claude_session = sid
        self.final_prd = prd
        n_stories = len(prd_session.user_story_titles(prd))
        prd_session.log_phase(self.session, "finalize", f"{n_stories} stories")
        self._set_preview(prd)
        if self.cowabunga:
            self._begin_run(autopick=True)
            return
        self._to_strategy_phase()

    def _after_revise(self, prd: str, sid: str) -> None:
        from splinter import prd_session

        self.claude_session = sid
        self.final_prd = prd
        prd_session.log_phase(self.session, "revise")
        self.phase = "review"
        self._save_state()
        self._set_preview(prd)
        self._show_stories()
        self._say("[green]Revised. Type 'accept' to run, or describe more changes.[/]")
        self._render_actions("review")
        self._set_busy(False, "accept / edit / gate: <cmds> / changes / cowabunga")

    def _after_generate(self, prd: str, sid: str) -> None:
        from splinter import prd_session

        self.claude_session = sid
        self.final_prd = prd
        self._generating = False
        n_stories = len(prd_session.user_story_titles(prd))
        prd_session.log_phase(self.session, "generate", f"{n_stories} stories")
        self._set_preview(prd)
        self._to_strategy_phase()

    def _show_stories(self) -> None:
        from splinter import prd_session

        titles = prd_session.user_story_titles(self.final_prd)
        if titles:
            self._say("[bold]Tasks:[/]")
            for t in titles:
                self._say(f"  • {escape(t)}")
        else:
            self._say("[yellow]No US-NNN stories found — the PRD runs as a single task.[/]")

    # --- input dispatch ---
    def _on_generate(self) -> None:
        """Generate button — read instructions and generate PRD."""
        if self._busy:
            return
        instructions = self.query_one("#instructions", TextArea).text.strip()
        if not instructions:
            self._say("[yellow]Please write instructions first.[/]")
            return
        self._set_busy(True, "generating PRD from instructions…")
        self._say("[cyan]Generating PRD from instructions…[/]")
        self._spawn(self._generate_worker, instructions=instructions)

    def action_send(self) -> None:
        """⌃S / Send button — submit whatever is in the text box."""
        entry = self.query_one("#entry", TextArea)
        self._submit(entry.text)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "send":
            self.action_send()
        elif bid == "generate":
            self._on_generate()
        elif bid == "accept":
            if self.phase == "trust":
                self._accept_trusted()
            else:
                self._submit("accept")
        elif bid == "edit":
            self._on_edit()
        elif bid in ("fulfilled", "cowabunga", "run"):
            self._submit(bid)
        elif bid.startswith("strat-"):
            self._submit(bid[len("strat-") :])

    def _submit(self, raw: str) -> None:
        if self._busy:
            return
        text = raw.strip()
        self.query_one("#entry", TextArea).text = ""
        if not text:
            return
        self._say(f"[cyan]> {escape(text)}[/]")
        handler = {
            "chat": self._on_chat,
            "strategy": self._on_strategy,
            "review": self._on_review,
            "trust": self._on_trust,
        }.get(self.phase)
        if handler:
            handler(text)

    def _on_chat(self, text: str) -> None:
        from splinter import prd_session

        if prd_session.is_cowabunga(text):
            self._set_busy(True, "cowabunga — finalizing…")
            self._spawn(self._finalize_worker, autodecide=True)
        elif prd_session.is_done(text):
            self._set_busy(True, "finalizing the PRD…")
            self._spawn(self._finalize_worker, autodecide=False)
        else:
            self._set_busy(True, "incorporating your answers…")
            self._spawn(self._refine_worker, answers=text)

    def _on_strategy(self, text: str) -> None:
        from splinter import prd_session
        from splinter.strategies.registry import available_strategies

        if prd_session.is_cowabunga(text):
            self._begin_run(autopick=True)
            return
        if text.lower() not in available_strategies():
            self._say(f"[yellow]Unknown strategy. Pick: {', '.join(available_strategies())}[/]")
            return
        self.strategy = text.lower()
        prd_session.log_phase(self.session, "strategy", self.strategy)
        self.final_prd = _set_fm_strategy(self.final_prd, self.strategy)
        self.phase = "review"
        self._save_state()
        self._set_preview(self.final_prd)
        self._show_stories()
        self._say("[green]Type 'accept' to run, 'cowabunga' to run as-is, or describe changes.[/]")
        self._say(
            "[dim]Gate auto-detected at run; set it yourself with "
            "`gate: <cmd1>; <cmd2>` (or `gate: none`).[/]"
        )
        self._render_actions("review")
        self._set_busy(False, "accept / edit / gate: <cmds> / changes / cowabunga")

    def _on_review(self, text: str) -> None:
        from splinter import prd_session

        # Let the user set the mechanical gate for this run, e.g.
        #   gate: npm run lint; npm test
        # Stored per-session; the planner only auto-detects when none is set.
        if text.lower().startswith("gate:"):
            from splinter.agents import gate

            checks = gate.parse_gate_spec(text.split(":", 1)[1])
            gate.save_gate_checks(self.session.dir, checks)
            if checks:
                self._say("[green]Gate set:[/] " + ", ".join(c["cmd"] for c in checks))
            else:
                self._say("[yellow]Gate disabled — no mechanical checks this run.[/]")
            self._set_busy(False, "accept / edit / gate: <cmds> / cowabunga")
            return

        if prd_session.is_cowabunga(text):
            self._begin_run()
            return
        if text.lower() in {"accept", "run", "yes", "go", "y"}:
            self._begin_run()
            return
        if text.lower() == "edit":
            self._on_edit()
            return
        self._set_busy(True, "applying your changes…")
        self._spawn(self._revise_worker, instructions=text)

    def _on_edit(self) -> None:
        """Edit button — return to revising instructions while preserving final_prd."""
        self._set_busy(False, "describe changes / accept / cowabunga")
        self._say("[green]Edit mode — describe changes; PRD kept.[/]")
        self.phase = "review"
        self._save_state()
        self.query_one("#entry", TextArea).focus()

    # --- finish ---
    def _begin_run(self, autopick: bool = False) -> None:
        from splinter import prd_session

        if autopick or not self.strategy:
            fm, _ = _fm_block(self.final_prd)
            self.strategy = self.strategy or str(fm.get("strategy") or "") or "cascade"
            self.final_prd = _set_fm_strategy(self.final_prd, self.strategy)
        prd_session.log_phase(self.session, "run", self.strategy or "cascade")
        self.session.write("prd.md", self.final_prd)
        self.session.update_index(
            f"# Session {self.session.id}\n- prd: prd.md\n- strategy: {self.strategy}\n"
        )
        self._say(f"[green]▶ running with strategy '{self.strategy}'…[/]")
        self.phase = "run"
        self.exit(0)


    def action_abort(self) -> None:
        # Confirm first — the draft is recoverable, but a stray ESC shouldn't nuke the run.
        if isinstance(self.screen, ConfirmQuit):
            return  # dialog already up

        def _decide(leave: bool | None) -> None:
            if leave:
                from splinter import procreg

                self._say(
                    "[yellow]Leaving — resume later with:[/] "
                    f"[bold]uv run splinter resume {self.session.id}[/]"
                )
                procreg.terminate_all()
                self.exit(None)

        self.push_screen(ConfirmQuit(self.session.id), _decide)

    def get_system_commands(self, screen: Any) -> Iterable[SystemCommand]:
        yield _find_shortcuts_cmd(screen, self)
        yield SystemCommand("Theme", "Change the current theme", self.action_change_theme)
        if self._maximized:
            yield SystemCommand("Minimize", "Restore default layout", self.action_toggle_maximize)
        else:
            yield SystemCommand("Maximize", "Maximize right panel", self.action_toggle_maximize)
        yield SystemCommand(
            "Screenshot",
            "Save an SVG screenshot of the current screen",
            lambda: self.set_timer(0.1, self.deliver_screenshot),
        )
        yield SystemCommand("Quit", "Quit the application", self.action_quit)

    def action_toggle_maximize(self) -> None:
        self._maximized = not self._maximized

    def watch__maximized(self, val: bool) -> None:
        self.set_class(val, "--maximized")


def _prd_run_kwargs(prd_path: str, session: Session, run_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Build canonical run_kwargs from finalized PRD and strategy."""
    fm, _ = _fm_block(session.read("prd.md"))
    strategy = str(fm.get("strategy") or "") or "cascade"
    return {
        **run_kwargs,
        "strategy": strategy,
        "prd_path": prd_path,
        "task_path": None,
    }


def run_prd_interactive(run_kwargs: dict[str, Any]) -> int:
    """Refine the PRD in a TUI, then execute it in-app and return exit code."""
    from splinter.memory.session import new_session_id

    session = Session(new_session_id())
    result = PrdSessionApp(session, run_kwargs).run()
    if result is None:
        if session.dir.exists():
            print(f"PRD session aborted — resume later: uv run splinter resume {session.id}")
        return 0
    if isinstance(result, int) and result == 0:
        status = session.read_status()
        if status.get("phase") == "run":
            prd_path = str(session.dir / "prd.md")
            final_run_kwargs = _prd_run_kwargs(prd_path, session, run_kwargs)
            return run_with_tui(final_run_kwargs, session=session)
    if isinstance(result, int):
        return result
    return 0


def _resume_prd(session: Session, status: dict[str, Any]) -> int:
    """Re-enter a PRD refinement; on finalize, run it in-app and return exit code."""
    saved_strategy = status.get("strategy")
    run_kwargs: dict[str, Any] = {
        "strategy": saved_strategy if saved_strategy and saved_strategy != "?" else None,
        "prd_path": str(session.dir / "prd.md"),
        "task_path": None,
        "effort": None,
        "budget": None,
        "max_iterations": 5,
        "cowabunga": False,
        "resume": True,
    }
    result = PrdSessionApp(session, run_kwargs).run()
    if result is None:
        print(f"PRD session aborted — resume later: uv run splinter resume {session.id}")
        return 0
    if isinstance(result, int) and result == 0:
        new_status = session.read_status()
        if new_status.get("phase") == "run":
            prd_path = str(session.dir / "prd.md")
            final_run_kwargs = _prd_run_kwargs(prd_path, session, run_kwargs)
            return run_with_tui(final_run_kwargs, session=session)
    if isinstance(result, int):
        return result
    return 0


#: Which artifact a given stage produces — dropped to redo that stage on rollback.
_STAGE_ARTIFACT = {"localize": "knowledge/localization.md", "run": "knowledge/plan.md"}


def _resume_run(session: Session, status: dict[str, Any], *, reset: bool = False) -> int:
    """Re-enter a failed/interrupted pipeline run.

    - ``reset``: ignore all artifacts, re-run from the head (fresh localize + plan).
    - critical failure: roll the failing stage back (drop its artifact, redo it).
    - transient failure (provider/network blip): keep everything, continue.
    """
    saved_strategy = status.get("strategy")
    source = str(status.get("source") or "")
    prd_path: str | None = None
    task_path: str | None = None
    if source.endswith((".yaml", ".yml")):
        task_path = source
    elif session.read("prd.md").strip():
        prd_path = str(session.dir / "prd.md")
    elif source:
        prd_path = source
    else:
        print(f"session {session.id}: no PRD or task input recorded — cannot resume run.")
        return 1

    def _num(key: str) -> Any:
        val = status.get(key)
        return val if val not in (None, "") else None

    fail_class = str(status.get("fail_class") or "")
    stage = str(status.get("stage") or "")
    if reset:
        print(f"resetting run {session.id} — re-running from the head.")
    elif fail_class == "critical":
        artifact = _STAGE_ARTIFACT.get(stage)
        if artifact:
            path = session.dir / artifact
            path.unlink(missing_ok=True)
            print(f"critical failure at stage '{stage}' — rolling back, redoing it.")
        else:
            print(f"critical failure at stage '{stage}' — redoing run.")
    else:
        print(f"resuming run {session.id} (reusing localization + plan)…")

    run_kwargs: dict[str, Any] = {
        "strategy": saved_strategy if saved_strategy and saved_strategy != "?" else None,
        "prd_path": prd_path,
        "task_path": task_path,
        "effort": _num("effort"),
        "budget": _num("budget"),
        "max_iterations": int(status.get("max_iterations") or 5),
        "cowabunga": False,
        "resume": not reset,
    }
    return run_with_tui(run_kwargs, session=session)


def resume_session(session_id: str | None, *, reset: bool = False) -> int:
    """Resume any session: PRD refinement, or a failed/interrupted pipeline run.

    ``reset`` forces a run to re-run from the head (fresh localize + plan).
    """
    from splinter.analyze import _run_state
    from splinter.memory.session import list_sessions

    resumable_run = {"FAILED", "INTERRUPTED", "PAUSED", "AWAITING_USER"}
    sessions = list_sessions()

    sid = session_id
    if sid is None:
        for cand in sessions:
            if Session(cand).read_status().get("state") == "refining":
                sid = cand
                break
        if sid is None:
            for cand in sessions:
                if _run_state(Session(cand)) in resumable_run:
                    sid = cand
                    break
        if sid is None:
            print("no resumable session found (none refining, failed, or interrupted).")
            return 1

    if sid not in sessions:
        print(f"no such session: {sid}")
        return 1

    session = Session(sid)
    status = session.read_status()
    if status.get("state") == "refining":
        return _resume_prd(session, status)

    state = _run_state(session)
    if state == "RUNNING":
        print(f"session {sid} is still running — not resumable while its process is alive.")
        return 1
    if state in ("COMPLETED", "DONE"):
        summary = format_run_completion(session)
        print(f"session {sid} finished ({summary}). Opening analyze.")
        AnalyzeApp(session).run()
        return 0
    if state in resumable_run:
        return _resume_run(session, status, reset=reset)

    print(f"session {sid} is not resumable (state: {state}).")
    return 1
