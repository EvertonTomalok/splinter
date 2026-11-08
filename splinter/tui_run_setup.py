"""Run setup modal for PRD strategy, parallel, gate, and final-eval configuration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, OptionList, Static
from textual.widgets.option_list import Option

from splinter.agents import gate
from splinter.prd_session import user_story_titles
from splinter.strategies.registry import registered_strategies
from splinter.strategy_pick import recommend_strategy
from splinter.task_waves import extract_task_deps, facts_line, wave_facts

if TYPE_CHECKING:
    from splinter.memory.session import Session


@dataclass(frozen=True)
class RunSetup:
    strategy: str
    parallel: bool | None = None


def worktree_supported() -> bool:
    from splinter.vcs import worktree

    return worktree.worktree_supported()


def should_ask_parallel(
    prd_text: str,
    strategy: str | None,
    parallel_override: bool | None = None,
) -> bool:
    if parallel_override is not None:
        return False
    if (strategy or "cascade").lower() in ("direct", "raphael"):
        return False
    if len(user_story_titles(prd_text)) < 2:
        return False
    return worktree_supported()


def format_stories_preview(prd_text: str, max_items: int = 12) -> str:
    titles = user_story_titles(prd_text)
    if not titles:
        return "No user stories (runs as a single task)"
    lines: list[str] = []
    for item in titles[:max_items]:
        sid, sep, title = item.partition(": ")
        lines.append(f"{sid} · {title}" if sep else item)
    extra = len(titles) - max_items
    if extra > 0:
        lines.append(f"+{extra} more")
    return "\n".join(lines)


def _session_gate_checks(session_dir: Path | None) -> list[dict[str, str]] | None:
    if session_dir is None:
        return None
    path = Path(session_dir) / gate.GATE_FILE
    if not path.exists():
        return None
    return gate.configured_gate_checks(".", session_dir)


def gate_summary(session_dir: Path | None) -> str:
    checks = _session_gate_checks(session_dir)
    if checks is None:
        return "auto-detect"
    if not checks:
        return "none"
    return "; ".join(c["cmd"] for c in checks if c.get("cmd"))


def initial_gate_input(session_dir: Path | None) -> str:
    checks = _session_gate_checks(session_dir)
    if checks is None:
        return ""
    if not checks:
        return "none"
    return "; ".join(c["cmd"] for c in checks if c.get("cmd"))


def apply_gate_spec(session_dir: Path, spec: str) -> list[dict[str, str]] | None:
    cleaned = spec.strip()
    if not cleaned:
        (Path(session_dir) / gate.GATE_FILE).unlink(missing_ok=True)
        return None
    Path(session_dir).mkdir(parents=True, exist_ok=True)
    checks = gate.parse_gate_spec(cleaned, "unknown")
    gate.save_gate_checks(session_dir, checks)
    return checks


def save_final_eval_choice(session_dir: Path, result: dict[str, str | None]) -> str:
    Path(session_dir).mkdir(parents=True, exist_ok=True)
    fe_path = Path(session_dir) / "final_eval.yaml"
    entry: dict[str, str | None] = {"name": result["name"], "kind": result["kind"]}
    for key in ("cmd", "skill", "provider", "model", "effort"):
        if result.get(key):
            entry[key] = result[key]
    fe_path.write_text(yaml.dump({"final_eval": [entry]}, default_flow_style=False))
    kind = result.get("kind") or ""
    return {
        "ask_user": "User Review (manual review after run)",
        "skill": f"Run Skill: {result.get('skill', '')}",
        "command": f"Run Command: {result.get('cmd', '')}",
    }.get(kind, kind)


def final_eval_summary(session_dir: Path | None) -> str:
    from splinter.configure import load_config, load_final_eval

    try:
        if session_dir is not None:
            fe_path = Path(session_dir) / "final_eval.yaml"
            if fe_path.exists():
                cfg = yaml.safe_load(fe_path.read_text()) or {}
                entries = load_final_eval(cfg)
            else:
                entries = load_final_eval(load_config())
        else:
            entries = load_final_eval(load_config())
    except Exception:
        entries = []
    if not entries:
        return "none"
    return ", ".join(e.kind for e in entries)


class GateSettingsModal(ModalScreen[str | None]):
    DEFAULT_CSS = """
    GateSettingsModal {
        align: center middle;
        background: $background 60%;
    }
    GateSettingsModal > Vertical#gate-dialog {
        width: 76;
        max-width: 95%;
        height: auto;
        border: round $primary;
        background: $surface;
        padding: 1 2;
    }
    GateSettingsModal #gate-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }
    GateSettingsModal #gate-hint {
        color: $text-muted;
        margin-bottom: 1;
    }
    GateSettingsModal #gate-actions {
        height: 3;
        align-horizontal: right;
        margin-top: 1;
    }
    GateSettingsModal Button {
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "save", "Save"),
    ]

    def __init__(self, session_dir: Path | None = None, initial: str = "") -> None:
        super().__init__()
        self._session_dir = session_dir
        self._initial = initial

    def compose(self) -> ComposeResult:
        with Vertical(id="gate-dialog"):
            yield Static("Gate settings", id="gate-title")
            yield Static(
                "Shell commands separated by ; (empty = auto-detect, none = disabled)",
                id="gate-hint",
            )
            yield Input(
                value=self._initial,
                placeholder="cmd1; cmd2  (empty = auto-detect, none = disabled)",
                id="gate-input",
            )
            with Horizontal(id="gate-actions"):
                yield Button("Save", id="gate-save", variant="primary")
                yield Button("Cancel", id="gate-cancel", variant="default")

    def on_mount(self) -> None:
        self.query_one("#gate-input", Input).focus()

    def _commit(self) -> None:
        val = self.query_one("#gate-input", Input).value
        if self._session_dir is not None:
            apply_gate_spec(self._session_dir, val)
        self.dismiss(val)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self._commit()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "gate-save":
            self._commit()
        else:
            self.dismiss(None)

    def action_save(self) -> None:
        self._commit()

    def action_cancel(self) -> None:
        self.dismiss(None)


class RunSetupModal(ModalScreen[RunSetup | None]):
    DEFAULT_CSS = """
    RunSetupModal {
        align: center middle;
        background: $background 60%;
    }
    RunSetupModal > Vertical#run-setup-dialog {
        width: 136;
        max-width: 96%;
        height: auto;
        max-height: 85%;
        border: round $primary;
        background: $surface;
        padding: 1 2;
    }
    RunSetupModal #run-setup-title {
        text-style: bold;
        color: $primary;
    }
    RunSetupModal #run-setup-facts {
        color: $text-muted;
        margin-bottom: 1;
    }
    RunSetupModal #setup-sweep {
        height: 1;
        display: none;
        color: $accent;
        margin-bottom: 1;
    }
    RunSetupModal #run-setup-scroll {
        height: auto;
        max-height: 1fr;
    }
    RunSetupModal #run-setup-stories {
        color: $text-muted;
        margin-bottom: 1;
    }
    RunSetupModal #run-setup-strat-label {
        text-style: bold;
        margin-top: 1;
    }
    RunSetupModal #run-setup-strategy {
        height: auto;
        max-height: 8;
        margin-bottom: 1;
    }
    RunSetupModal #run-setup-parallel {
        height: auto;
        margin-bottom: 1;
    }
    RunSetupModal #run-setup-summary {
        color: $text-muted;
    }
    RunSetupModal #run-setup-cfg-buttons {
        height: 3;
        margin-bottom: 1;
    }
    RunSetupModal #run-setup-cfg-buttons Button {
        margin-right: 1;
    }
    RunSetupModal #run-setup-actions {
        height: 3;
        align-horizontal: left;
    }
    RunSetupModal #run-setup-actions Button {
        margin-right: 1;
    }
    RunSetupModal #run-setup-footer {
        color: $text-muted;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("enter", "run", "Run"),
        Binding("escape", "back", "Back"),
        Binding("p", "toggle_parallel", "Parallel", show=False),
    ]

    def __init__(
        self,
        prd_text: str = "",
        session: Session | None = None,
        *,
        parallel: bool | None = None,
        on_note: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__()
        self._prd_text = prd_text
        self._session = session
        self._parallel_override = parallel
        self._on_note = on_note
        self._strategies = registered_strategies()
        self._recommended, self._reason = recommend_strategy(prd_text)
        self._deps = extract_task_deps(prd_text, session)
        self._facts = wave_facts(self._deps)
        self._sweep_timer: Any = None
        self._sweep_pos = 0

    @property
    def _session_dir(self) -> Path | None:
        return self._session.dir if self._session is not None else None

    def _current_parallel(self) -> bool:
        if self._parallel_override is not None:
            return self._parallel_override
        try:
            cb = self.query_one("#run-setup-parallel", Checkbox)
            if cb.display:
                return bool(cb.value)
        except Exception:
            pass
        return False

    def _facts_line(self) -> str:
        return facts_line(self._facts, self._current_parallel())

    def _strategy_tag(self, name: str) -> str:
        if self._reason == "set in the PRD" and name == self._recommended:
            return "★ set in the PRD"
        if name == "agents" and self._facts.widest > 3:
            return f"fits {self._facts.widest}-wide waves"
        if name == self._recommended:
            return "recommended"
        return ""

    def _strategy_options(self) -> list[Option]:
        options: list[Option] = []
        aliases = [cls.aliases[0] if cls.aliases else cls.name for cls in self._strategies]
        alias_w = max((len(a) for a in aliases), default=0)
        name_w = max((len(cls.name) for cls in self._strategies), default=0)
        for alias, cls in zip(aliases, self._strategies):
            label = Text.assemble(
                (alias.ljust(alias_w), "bold"),
                "  ",
                (cls.name.ljust(name_w), "dim"),
                "  — ",
                cls.summary,
            )
            tag = self._strategy_tag(cls.name)
            if tag:
                label.append(f"  {tag}", style="bold green" if "★" in tag else "bold")
            options.append(Option(label, id=cls.name))
        return options

    def _summary_line(self) -> str:
        sdir = self._session_dir
        return f"Gate: {gate_summary(sdir)} · Final eval: {final_eval_summary(sdir)}"

    def compose(self) -> ComposeResult:
        dialog = Vertical(id="run-setup-dialog")
        dialog.border_subtitle = "enter Run · esc Back"
        with dialog:
            yield Static("Run it how?", id="run-setup-title")
            yield Static(self._facts_line(), id="run-setup-facts")
            yield Static(id="setup-sweep")
            with VerticalScroll(id="run-setup-scroll"):
                yield Static(format_stories_preview(self._prd_text), id="run-setup-stories")
                yield Static("Strategy", id="run-setup-strat-label")
                yield OptionList(
                    *self._strategy_options(),
                    id="run-setup-strategy",
                    classes="strategy-list",
                )
                yield Checkbox(
                    "Run stories in parallel (git worktrees)",
                    id="run-setup-parallel",
                )
                yield Static(self._summary_line(), id="run-setup-summary")
                with Horizontal(id="run-setup-cfg-buttons"):
                    yield Button("Settings…", id="run-setup-settings")
                    yield Button("Final eval…", id="run-setup-final-eval")
            with Horizontal(id="run-setup-actions"):
                yield Button("Run", id="run-setup-run", variant="primary")
                yield Button("Back to chat", id="run-setup-back", variant="default")
                yield Button("Cowabunga", id="run-setup-cowabunga", variant="warning")

    def query_one(self, selector: Any = None, expect_type: Any = None) -> Any:
        if selector == "#strategy-list":
            return super().query_one("#run-setup-strategy", expect_type)
        return super().query_one(selector, expect_type)

    def on_mount(self) -> None:
        ol = self.query_one("#run-setup-strategy", OptionList)
        names = [cls.name for cls in self._strategies]
        if self._recommended in names:
            ol.highlighted = names.index(self._recommended)
        self._update_parallel_visibility()
        self._update_facts()
        ol.focus()

    def _highlighted_strategy(self) -> str:
        ol = self.query_one("#run-setup-strategy", OptionList)
        idx = ol.highlighted
        if idx is not None and 0 <= idx < len(self._strategies):
            return str(self._strategies[idx].name)
        return self._recommended

    def _update_parallel_visibility(self) -> None:
        cb = self.query_one("#run-setup-parallel", Checkbox)
        cb.display = should_ask_parallel(
            self._prd_text,
            self._highlighted_strategy(),
            self._parallel_override,
        )
        self._update_facts()

    def _update_facts(self) -> None:
        try:
            facts_widget = self.query_one("#run-setup-facts", Static)
            text = self._facts_line()
            facts_widget.update(text)
            facts_widget.display = bool(text)
        except Exception:
            pass

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        if event.checkbox.id == "run-setup-parallel":
            self._update_facts()

    def action_toggle_parallel(self) -> None:
        cb = self.query_one("#run-setup-parallel", Checkbox)
        if cb.display:
            cb.value = not cb.value
            self._update_facts()

    def start_scan(self) -> None:
        sweep = self.query_one("#setup-sweep", Static)
        sweep.display = True
        if getattr(self.app, "animation_level", None) == "none":
            sweep.update("scanning…")
            return
        self._sweep_pos = 0
        if self._sweep_timer is None:
            self._sweep_timer = self.set_interval(0.05, self._tick_sweep)

    def stop_scan(self) -> None:
        if self._sweep_timer is not None:
            self._sweep_timer.stop()
            self._sweep_timer = None
        sweep = self.query_one("#setup-sweep", Static)
        sweep.display = False

    def _tick_sweep(self) -> None:
        if getattr(self.app, "animation_level", None) == "none":
            self.query_one("#setup-sweep", Static).update("scanning…")
            return
        width = 30
        self._sweep_pos = (self._sweep_pos + 1) % width
        chars = [" "] * width
        for i in range(3):
            chars[(self._sweep_pos + i) % width] = "━"
        self.query_one("#setup-sweep", Static).update("".join(chars))

    def _refresh_summary(self) -> None:
        self.query_one("#run-setup-summary", Static).update(self._summary_line())

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option_list.id == "run-setup-strategy":
            self._update_parallel_visibility()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "run-setup-strategy":
            event.stop()
            self.action_run()

    def _open_gate_settings(self) -> None:
        sdir = self._session_dir
        modal = GateSettingsModal(sdir, initial=initial_gate_input(sdir))

        def _after(val: str | None) -> None:
            if val is not None:
                self._refresh_summary()

        self.app.push_screen(modal, _after)

    def _open_final_eval(self) -> None:
        from splinter.tui import _FinalEvalModal

        sdir = self._session_dir

        def _after(result: dict[str, str | None] | None) -> None:
            if result is None or sdir is None:
                return
            label = save_final_eval_choice(sdir, result)
            if self._on_note is not None:
                self._on_note(f"[green]Final eval set:[/] {label}")
            self._refresh_summary()

        self.app.push_screen(_FinalEvalModal(), _after)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "run-setup-run":
            self.action_run()
        elif bid == "run-setup-back":
            self.action_back()
        elif bid == "run-setup-cowabunga":
            parallel = self._parallel_override if self._parallel_override is not None else False
            self.dismiss(RunSetup(strategy=self._recommended, parallel=parallel))
        elif bid == "run-setup-settings":
            self._open_gate_settings()
        elif bid == "run-setup-final-eval":
            self._open_final_eval()

    def action_run(self) -> None:
        strategy = self._highlighted_strategy()
        if self._parallel_override is not None:
            parallel = self._parallel_override
        elif should_ask_parallel(self._prd_text, strategy, self._parallel_override):
            parallel = bool(self.query_one("#run-setup-parallel", Checkbox).value)
        else:
            parallel = False
        self.dismiss(RunSetup(strategy=strategy, parallel=parallel))

    def action_back(self) -> None:
        self.dismiss(None)
