"""Plan mode TUI for ``splinter run --task``: ask, plan, edit, then run.

Same layout as :class:`splinter.tui.PrdSessionApp`. The **planner** model asks
implementation questions about the task, the user answers, the planner writes
the plan, the user edits it, and Run hands the session to ``RunApp``. The direct
strategy then reuses ``knowledge/plan.md`` instead of planning again.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

from rich.markup import escape
from rich.text import Text
from textual.app import App, ComposeResult, SystemCommand
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Button, Footer, RichLog, Static, TextArea

from splinter import plan_session
from splinter.agents.runner import Task
from splinter.memory.session import Session
from splinter.prd_session import Turn, is_cowabunga
from splinter.task_file import read_task_mapping
from splinter.task_plan import PLAN_INDEX_FILE, PlanIndex, is_plan_index, load_plan_index
from splinter.task_waves import format_plan_steps
from splinter.tui import (
    _MAXIMIZE_CSS,
    _PALETTE_CSS,
    ComposerTextArea,
    ConfirmQuit,
    _find_shortcuts_cmd,
    _OrderedCommandsProvider,
)
from splinter.tui_questions import (
    Question,
    QuestionForm,
    extract_open_questions,
    parse_questions,
)
from splinter.tui_questions_one import QuestionIndex
from splinter.tui_run_palette import Ink
from splinter.tui_shell import SHELL_CSS, ShellBar, crumb_markup, use_splinter_theme
from splinter.tui_status import BusyStatus
from splinter.tui_stepper import PLAN_STEPS, PhaseStepper
from splinter.tui_waiting import WaitingRoom

DEFAULT_STRATEGY = "raphael"
_PLAN_WORDS = {"fulfilled", "done", "plan", "write plan"}
_RUN_WORDS = {"accept", "run", "go"}
_HINTS = {
    "chat": "anything else the planner should know",
    "review": "or tell the planner what to change · ^s sends",
}
_PHASE_STEPS = {"init": 0, "clarify": 1, "chat": 1, "review": 2, "run": 3}
#: The stepper step the questions live on (clickable to go back to them).
_QUESTIONS_STEP = 1
_COWABUNGA_TOOLTIPS = {
    "chat": "Let the planner answer the open questions, write the plan and run it",
    "review": "Run this plan now",
}


class PlanSessionApp(App[int | None]):
    """Plan one task with the planner model before it runs.

    Phases: ``clarify`` (planner asks) → ``chat`` (answers until Write plan) →
    ``review`` (edit the plan on the right) → ``run`` (save + exit 0).
    """

    CSS = (
        """
    PlanSessionApp.-waiting #session-body { display: none; }
    PlanSessionApp.-waiting Footer { display: none; }
    #chatpane { width: 1fr; padding: 1 4 0 4; }
    #draftpane {
        width: 40%;
        max-width: 64;
        padding: 1 2 0 2;
        border-left: solid #1d1d1a;
    }
    #doc-head { height: 1; margin-bottom: 1; }
    #task-edit, #plan-edit {
        height: 1fr;
        border: none;
        padding: 0;
        background: #0a0a09;
    }
    #task-edit:focus, #plan-edit:focus { border: none; background-tint: #0a0a09 0%; }
    #answered-head { height: 2; margin-top: 1; border-top: solid #1d1d1a; color: #8f8e85; }
    #question-index { height: auto; max-height: 8; }
    #doc-cost { height: 1; color: #7a7972; }
    #plan-ready { height: auto; margin-top: 1; }
    #plan-ready-title { text-style: bold; color: #ecebe4; }
    #plan-ready-lead { color: #8f8e85; margin-bottom: 1; }
    #plan-ready-steps { color: #8f8e85; margin-bottom: 1; }
    #plan-ready-keys { height: 1; margin-top: 1; }
    #plan-ready-keys Button { height: 1; min-width: 0; margin: 0 3 0 0; }
    PlanSessionApp.-ready #convo { display: none; }
    PlanSessionApp.-ready #questions-host { display: none; }
    PlanSessionApp.-ready #actions { display: none; }
    PlanSessionApp.-ready Footer { display: none; }
    PlanSessionApp.-ready #plan-ready { height: 1fr; }
    PlanSessionApp.-ready #plan-ready-gap { height: 1fr; }
    PlanSessionApp.-asking #convo { display: none; }
    PlanSessionApp.-asking #question-form { height: 1fr; max-height: 100%; }
    #plan-ready-gap { height: 0; }
    #convo {
        height: 1fr; padding: 0 1;
        overflow-x: hidden;
        scrollbar-size-horizontal: 0; scrollbar-size-vertical: 1;
    }
    #questions-host { height: auto; }
    #busy { height: 1; }
    #composer {
        dock: bottom;
        height: auto;
        border-top: solid #1d1d1a;
        padding: 1 0 0 0;
    }
    #entry {
        height: 3;
        border: none;
        border-left: wide #7a7972;
        padding: 0 1;
    }
    #entry:focus { border: none; border-left: wide #c6f432; }
    #actions { height: 1; margin: 1 0; padding: 0; align: left middle; }
    """
        + SHELL_CSS
        + _PALETTE_CSS
        + _MAXIMIZE_CSS
    )

    COMMANDS = {_OrderedCommandsProvider}

    BINDINGS = [
        Binding("ctrl+c", "abort", "Abort", show=False),
        Binding("escape", "plan_back", "Back"),
        Binding("ctrl+s", "send", "Send", key_display="ctrl+s"),
        Binding("r", "run_plan", "Run"),
        Binding("e", "edit_plan", "Edit plan", show=False),
        Binding("f2", "configure", "Configure", show=False),
    ]

    FATAL_EXIT_DELAY = 2.0

    _maximized: reactive[bool] = reactive(False)

    def __init__(self, session: Session, run_kwargs: dict[str, Any]) -> None:
        super().__init__()
        self.session = session
        self.run_kwargs = run_kwargs
        self.resuming = bool(run_kwargs.get("resume"))
        self.no_ground = bool(run_kwargs.get("no_ground"))
        self.strategy = str(run_kwargs.get("strategy") or DEFAULT_STRATEGY)
        self.phase = "init"
        self.claude_session = ""
        self._task_base: dict[str, Any] = {}
        self._task_name = Path(str(run_kwargs.get("task_path") or "task.yaml")).name
        self._localization: str | None = None
        self._convo_lines: list[str] = []
        self._busy = False
        self._auto_run = False
        self._failed = False
        self._last_job: tuple[Callable[..., None], dict[str, Any]] | None = None
        self._started_at = datetime.now(timezone.utc).isoformat()
        self._questions: list[Question] = []
        self._last_questions: list[Question] = []
        self._last_answers: dict[int, str] = {}
        self._model_name = ""
        self._plan_index: PlanIndex | None = None

    # --- layout ---
    def compose(self) -> ComposeResult:
        stepper = PhaseStepper(PLAN_STEPS, id="stepper")
        yield ShellBar(crumb_markup("task", self._task_name), stepper)
        with Horizontal(id="session-body"):
            with Vertical(id="chatpane"):
                plan_ready = Vertical(id="plan-ready")
                plan_ready.display = False
                with plan_ready:
                    yield Static("Plan ready.", id="plan-ready-title")
                    yield Static("", id="plan-ready-lead")
                    yield Static("", id="plan-ready-steps")
                    yield Static("", id="plan-ready-gap")
                    with Horizontal(id="plan-ready-keys"):
                        yield Button(
                            f"[b #c6f432]↵[/] [b]run with {self.strategy}[/]", id="plan-run"
                        )
                        yield Button("[b #c6f432]e[/] edit plan", id="plan-edit-key")
                        yield Button("[b #c6f432]esc[/] back to questions", id="plan-back")
                yield RichLog(id="convo", markup=True, wrap=True)
                with Vertical(id="questions-host"):
                    yield QuestionForm(id="question-form")
                with Vertical(id="composer"):
                    yield BusyStatus(id="busy")
                    entry = ComposerTextArea(id="entry", soft_wrap=True)
                    entry.placeholder = "anything else the planner should know"
                    yield entry
                    with Horizontal(id="actions"):
                        yield from self._action_buttons("chat")
            with Vertical(id="draftpane"):
                yield Ink(self._doc_head(), id="doc-head")
                yield TextArea(id="task-edit", soft_wrap=True, language="markdown")
                plan_edit = TextArea(id="plan-edit", soft_wrap=True, language="markdown")
                plan_edit.display = False
                yield plan_edit
                yield Static("answered", id="answered-head")
                yield QuestionIndex(id="question-index")
                yield Static("", id="doc-cost")
        yield WaitingRoom()
        yield Footer()

    def _doc_head(self) -> str:
        return (
            "[#8f8e85]task + plan[/] [#7a7972]· e to edit · click a question to jump[/]"
        )

    def action_mode(self, name: str) -> None:
        """The ``--prd`` chip: fixed here, the task is already chosen."""
        if name != "task":
            self.notify("This session plans a task; start a new one for a PRD.")

    def on_mount(self) -> None:
        self.title = "splinter · plan"
        use_splinter_theme(self)
        if self.resuming:
            self._resume()
            return
        path = Path(str(self.run_kwargs.get("task_path") or ""))
        if is_plan_index(path):
            self._open_plan_index(path)
            return
        try:
            loaded = read_task_mapping(path)
        except (OSError, ValueError) as exc:
            self._fatal(f"cannot read task: {exc}")
            return
        self._task_base = loaded
        task = Task(
            description=str(loaded.get("description", "")),
            acceptance=str(loaded.get("acceptance", "")),
        )
        self._load_task_editor(plan_session.task_text(task))
        self._set_phase("clarify")
        self._save_state()
        self._ask_questions()

    def _open_plan_index(self, path: Path) -> None:
        try:
            self._plan_index = load_plan_index(path)
        except (OSError, ValueError) as exc:
            self._fatal(f"cannot read task plan: {exc}")
            return
        if not self.run_kwargs.get("strategy"):
            self.strategy = "cascade"
        self._task_name = f"{path.name} · {len(self._plan_index.tasks)} tasks"
        self._load_task_editor(plan_session.plan_bundle(self._plan_index))
        self._set_phase("clarify")
        self._save_state()
        self._ask_questions()

    def _task_ids(self) -> tuple[str, ...]:
        return tuple(t.id for t in self._plan_index.tasks) if self._plan_index else ()

    def _save_plan(self, plan: str) -> None:
        if self._plan_index is None:
            plan_session.save_plan(self.session, plan)
            return
        missing = plan_session.save_task_plans(self.session, plan, self._task_ids())
        if missing:
            self._say(
                f"[yellow]No plan section for {', '.join(missing)}: "
                "the planner plans them during the run.[/]"
            )

    def _load_task_editor(self, text: str) -> None:
        editor = self.query_one("#task-edit", TextArea)
        editor.text = text
        self.query_one(ShellBar).set_crumb(crumb_markup("task", self._task_name))

    def _ask_questions(self) -> None:
        self._say("[dim]planner reading the task, drafting implementation questions…[/]")
        self._set_busy(True, "planner reading the task…")
        self._spawn(self._questions_worker, task_md=self._task_md())

    # --- resume ---
    def _resume(self) -> None:
        status = self.session.read_status()
        self.claude_session = str(status.get("claude_session") or "")
        saved = status.get("strategy")
        if saved and saved != "?" and not self.run_kwargs.get("strategy"):
            self.strategy = str(saved)
        self._task_name = str(status.get("task_name") or self._task_name)
        if status.get("plan_index"):
            self._resume_plan_index(Path(str(status.get("task_path") or "")))
        else:
            self._task_base = self._read_saved_task(status)
            task = Task(
                description=str(self._task_base.get("description", "")),
                acceptance=str(self._task_base.get("acceptance", "")),
            )
            self._load_task_editor(plan_session.task_text(task))
        self._replay_convo()
        phase = str(status.get("phase") or "chat")
        plan = self._saved_plan()
        self._say(f"[magenta]⟳ resumed {self.session.id} at phase '{phase}'.[/]")
        if phase in ("review", "run") and plan:
            self._enter_review(plan, "[green]Plan restored. Edit it on the right, then run.[/]")
        elif phase == "clarify" and not self.claude_session:
            self._set_phase("clarify")
            self._ask_questions()
        else:
            self._enter_chat()

    def _resume_plan_index(self, path: Path) -> None:
        try:
            self._plan_index = load_plan_index(path)
        except (OSError, ValueError) as exc:
            self._fatal(f"cannot read task plan: {exc}")
            return
        saved = self.session.read(plan_session.PLAN_BUNDLE_FILE)
        self._load_task_editor(saved or plan_session.plan_bundle(self._plan_index))

    def _read_saved_task(self, status: dict[str, Any]) -> dict[str, Any]:
        for raw in (self.session.dir / "task.yaml", status.get("task_path")):
            if not raw:
                continue
            try:
                return read_task_mapping(Path(str(raw)))
            except (OSError, ValueError):
                continue
        return {}

    def _saved_plan(self) -> str:
        name = plan_session.PLAN_ALL_FILE if self._plan_index else "knowledge/plan.md"
        text = self.session.read(name)
        return plan_session.strip_plan_header(text) if text.strip() else ""

    def _replay_convo(self) -> None:
        prior = self.session.read("convo.md")
        records = [r for r in prior.split("\x00") if r] if prior.strip() else []
        convo = self.query_one("#convo", RichLog)
        for rec in records:
            try:
                Text.from_markup(rec)
            except Exception:  # noqa: BLE001 — bad markup in an old log must not abort resume
                rec = escape(rec)
            convo.write(rec)
        self._convo_lines = records
        if records:
            convo.write("[dim]─── resumed ───[/]")

    # --- state ---
    def _set_phase(self, phase: str) -> None:
        self.phase = phase
        try:
            stepper = self.query_one("#stepper", PhaseStepper)
        except Exception:
            self.sub_title = phase
            return
        stepper.set_current(_PHASE_STEPS.get(phase, 0))
        self.sub_title = stepper.current_label
        if phase != "chat":
            self._set_questions([])

    def _task_md(self) -> str:
        return str(self.query_one("#task-edit", TextArea).text)

    def _plan_md(self) -> str:
        return str(self.query_one("#plan-edit", TextArea).text)

    def _persist_task(self) -> Path:
        if self._plan_index is not None:
            self.session.write(PLAN_INDEX_FILE, self._plan_index.text)
            return self.session.write(plan_session.PLAN_BUNDLE_FILE, self._task_md())
        description, acceptance = plan_session.parse_task_text(self._task_md())
        return plan_session.save_task(self.session, description, acceptance, self._task_base)

    def _save_state(self) -> None:
        task_path = self._persist_task()
        self.session.set_status(
            "refining",
            source="task",
            phase=self.phase,
            claude_session=self.claude_session,
            task_path=str(self._plan_index.path if self._plan_index else task_path),
            plan_index=self._plan_index is not None,
            task_name=self._task_name,
            strategy=self.strategy,
            started_at=self._started_at,
        )

    # --- ui helpers ---
    def _say(self, msg: str, *, persist: bool = True) -> None:
        self.query_one("#convo", RichLog).write(msg)
        if persist:
            self._convo_lines.append(msg)
            self.session.write("convo.md", "\x00".join(self._convo_lines))

    def _set_busy(self, busy: bool, placeholder: str = "") -> None:
        self._busy = busy
        self.set_class(busy, "-waiting")
        waiting = self.query_one(WaitingRoom)
        if busy:
            waiting.start(placeholder.rstrip("…").strip(), self._busy_model())
        else:
            waiting.stop()
        entry = self.query_one("#entry", TextArea)
        entry.disabled = busy
        for btn in self.query("#actions Button"):
            btn.disabled = busy
        for selector in (
            "#question-form RadioSet",
            "#question-form Button",
            "#question-form Input",
        ):
            for widget in self.query(selector):
                widget.disabled = busy
        try:
            status = self.query_one("#busy", BusyStatus)
        except Exception:
            status = None
        if status is not None:
            if busy:
                status.start(placeholder.rstrip("…").strip(), self._busy_model())
            else:
                status.stop()
        if placeholder and not busy:
            entry.placeholder = placeholder
        if not busy:
            entry.focus()

    async def action_configure(self) -> None:
        from splinter.tui_session_config import edit_session_config

        previous_model = self._busy_model()
        if await edit_session_config(self, busy=self._busy):
            self._model_name = ""
            if self._busy_model() != previous_model:
                self.claude_session = ""
            self._save_state()

    def _busy_model(self) -> str:
        if not self._model_name:
            try:
                from splinter.models.roster import load_ladder

                self._model_name = str(load_ladder().planner_model)
            except Exception:
                self._model_name = ""
        return self._model_name

    def _set_questions(self, questions: list[Question]) -> None:
        self._questions = list(questions)
        self.set_class(bool(questions), "-asking")
        if questions:
            self._last_questions = list(questions)
        if not self.is_running:
            return
        try:
            form = self.query_one("#question-form", QuestionForm)
        except Exception:
            return

        count = len(self._questions)
        form.meta = f"planner · {self._busy_model() or 'planner'} · {count} questions"

        async def _swap() -> None:
            await form.set_questions(self._questions)
            form.display = bool(self._questions)

        self.run_worker(_swap, name="questions")  # type: ignore[arg-type]
        self._refresh_question_index()

    def _refresh_question_index(self) -> None:
        """Point the right-pane question list at the current questions."""
        try:
            index = self.query_one("#question-index", QuestionIndex)
        except Exception:
            return
        index.refresh_from(self._questions, self._last_answers)

    def on_question_form_answers_changed(self, event: QuestionForm.AnswersChanged) -> None:
        self._last_answers = dict(event.answers)
        self._refresh_question_index()

    def on_question_form_completed(self, event: QuestionForm.Completed) -> None:
        """Auto-advance ended: focus Write plan; the form never submits alone."""
        try:
            self.query_one("#write-plan", Button).focus()
        except Exception:
            pass

    def action_jump_question(self, n: int | str) -> None:
        """QuestionIndex row click: jump to that question in the form."""
        try:
            self.query_one("#question-form", QuestionForm).jump_to(int(n))
        except Exception:
            return

    def on_phase_stepper_step_clicked(self, event: PhaseStepper.StepClicked) -> None:
        if self._busy or event.index != _QUESTIONS_STEP or not self._last_questions:
            return
        self._back_to_questions()

    def _back_to_questions(self) -> None:
        """Go back to the questions step, re-filled with the saved answers."""
        self._hide_plan_ready()
        self._set_phase("chat")

        async def _swap() -> None:
            form = self.query_one("#question-form", QuestionForm)
            await form.set_questions(self._last_questions)
            form.restore_answers(self._last_answers)

        self.run_worker(_swap, name="questions")  # type: ignore[arg-type]
        self._render_actions("chat")
        self._set_busy(False, _HINTS["chat"])

    def _compile_answers(self, text: str) -> str:
        stripped = text.strip()
        words = _PLAN_WORDS | _RUN_WORDS | {"cowabunga"}
        if not self._questions or self.phase != "chat" or stripped.lower() in words:
            return stripped
        try:
            form = self.query_one("#question-form", QuestionForm)
        except Exception:
            return stripped
        return form.compile_answer(stripped)

    def _action_buttons(self, phase: str) -> list[Button]:
        review = phase == "review"
        btns = [Button("Send", id="send", variant="default" if review else "primary")]
        if review:
            btns.append(Button("Run", id="run", variant="primary"))
        else:
            btns.append(Button("Write plan", id="write-plan", variant="success"))
        btns.append(
            Button(
                "Cowabunga",
                id="cowabunga",
                variant="warning",
                tooltip=_COWABUNGA_TOOLTIPS.get(phase, _COWABUNGA_TOOLTIPS["chat"]),
            )
        )
        if self._failed:
            btns.append(Button("Retry", id="retry", variant="error"))
        return btns

    def _render_actions(self, phase: str) -> None:
        btns = self._action_buttons(phase)
        bar = self.query_one("#actions", Horizontal)

        async def _swap() -> None:
            await bar.remove_children()
            await bar.mount(*btns)

        self.run_worker(_swap, name="actions")  # type: ignore[arg-type]

    def _fatal(self, msg: str) -> None:
        self._say(f"[red]ERROR: {escape(msg)}[/]", persist=False)
        self.set_timer(self.FATAL_EXIT_DELAY, lambda: self.exit(None))

    def _fail(self, msg: str) -> None:
        """A model error keeps the screen open: show it, offer Retry."""
        self._say(f"[red]ERROR: {escape(msg)}[/]")
        self._failed = True
        self._render_actions("review" if self.phase == "review" else "chat")
        self._set_busy(False, "retry, or keep typing")

    def _spawn(self, fn: Callable[..., None], **kw: Any) -> None:
        self._failed = False
        self._last_job = (fn, kw)
        waiting = self.query_one(WaitingRoom)
        self.run_worker(partial(waiting.run_job, fn, **kw), thread=True, name="plan")

    def _retry(self) -> None:
        if self._busy or self._last_job is None:
            return
        fn, kw = self._last_job
        self._say("[dim]retrying…[/]")
        self._set_busy(True, "retrying…")
        self._spawn(fn, **kw)

    # --- workers (off the UI thread) ---
    def _grounding(self, text: str) -> str:
        if self.no_ground:
            return ""
        if self._localization is None:
            from splinter import prd_session
            from splinter.models.roster import load_ladder

            self._localization = prd_session.ground_localization(
                self.session, load_ladder(), text
            )
        return self._localization

    def _run_turn(self, after: Callable[[Turn], None], fn: Callable[..., Turn], **kw: Any) -> None:
        try:
            turn = fn(session=self.session, **kw)
        except Exception as exc:  # noqa: BLE001 — surfaced in the UI with Retry
            self.call_from_thread(self._fail, f"planner: {exc}")
            return
        self.call_from_thread(after, turn)

    def _questions_worker(self, task_md: str) -> None:
        self._run_turn(
            self._after_questions,
            plan_session.open_plan_questions,
            task_md=task_md,
            localization=self._grounding(task_md),
        )

    def _refine_worker(self, answers: str, task_md: str) -> None:
        self._run_turn(
            self._after_refine,
            plan_session.refine_plan,
            answers=answers,
            resume=self.claude_session,
            current_task=task_md,
            localization=self._grounding(task_md),
        )

    def _write_worker(self, task_md: str, autodecide: bool) -> None:
        self._run_turn(
            self._after_plan,
            plan_session.write_plan,
            resume=self.claude_session,
            current_task=task_md,
            autodecide=autodecide,
            localization=self._grounding(task_md),
            task_ids=self._task_ids(),
        )

    def _revise_worker(self, instructions: str, plan_md: str) -> None:
        self._run_turn(
            self._after_revise,
            plan_session.revise_plan,
            instructions=instructions,
            resume=self.claude_session,
            current_plan=plan_md,
            task_ids=self._task_ids(),
        )

    # --- worker callbacks (UI thread) ---
    def _take_session(self, turn: Turn) -> None:
        self.claude_session = turn.session_id or self.claude_session

    def _after_questions(self, turn: Turn) -> None:
        self._take_session(turn)
        self._say(escape(turn.text))
        self._enter_chat(parse_questions(turn.text))

    def _after_refine(self, turn: Turn) -> None:
        self._take_session(turn)
        self._say(escape(turn.text))
        self._enter_chat(extract_open_questions(turn.text))

    def _after_plan(self, turn: Turn) -> None:
        self._take_session(turn)
        plan = plan_session.strip_plan_header(turn.text)
        self._save_plan(plan)
        if self._auto_run:
            self._show_plan(plan)
            self._run()
            return
        self._enter_review(plan, "[green]Plan written. Edit it on the right, then run.[/]")

    def _after_revise(self, turn: Turn) -> None:
        self._take_session(turn)
        plan = plan_session.strip_plan_header(turn.text)
        self._save_plan(plan)
        self._enter_review(plan, "[green]Plan revised. Run, or describe more changes.[/]")

    def _enter_chat(self, questions: list[Question] | None = None) -> None:
        self._set_phase("chat")
        self._save_state()
        if questions is not None:
            self._set_questions(questions)
        self._render_actions("chat")
        self._set_busy(False, _HINTS["chat"])

    def _show_plan(self, plan: str) -> None:
        plan_edit = self.query_one("#plan-edit", TextArea)
        plan_edit.text = plan
        self.query_one("#task-edit", TextArea).display = False
        plan_edit.display = True
        self.set_class(True, "-ready")
        try:
            plan_ready = self.query_one("#plan-ready", Vertical)
            steps = format_plan_steps(plan)
            answered = ", written with your answers" if self._last_answers else ""
            self.query_one("#plan-ready-lead", Static).update(
                f"{steps} on the right{answered}. Edit it there, or run it as is."
            )
            self.query_one("#entry", TextArea).placeholder = (
                "or tell the planner what to change · ^s sends"
            )
            self.query_one("#plan-ready-steps", Static).update(_step_lines(plan))
            plan_ready.display = True
        except Exception:
            pass

    def _hide_plan_ready(self) -> None:
        self.set_class(False, "-ready")
        self.query_one("#entry", TextArea).placeholder = "anything else the planner should know"
        self.query_one("#plan-ready", Vertical).display = False

    def _enter_review(self, plan: str, hint: str) -> None:
        self._show_plan(plan)
        self._set_phase("review")
        self._save_state()
        self._say(hint)
        self._render_actions("review")
        self._set_busy(False, _HINTS["review"])
        try:
            self.query_one("#plan-run", Button).focus()
        except Exception:
            return

    # --- input dispatch ---
    def action_edit_plan(self) -> None:
        if self.phase == "review":
            self.query_one("#plan-edit", TextArea).focus()

    def action_plan_back(self) -> None:
        """``esc``: back to the questions from the plan, else leave."""
        if self.phase == "review" and self._last_questions and not self._busy:
            self._back_to_questions()
            return
        self.action_abort()

    def action_run_plan(self) -> None:
        if self.phase == "review":
            self._run()

    def action_send(self) -> None:
        self._submit(self._compile_answers(self.query_one("#entry", TextArea).text))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "send":
            self.action_send()
        elif bid == "retry":
            self._retry()
        elif bid == "write-plan":
            self._submit("plan")
        elif bid in ("run", "cowabunga"):
            self._submit(bid)
        elif bid == "plan-run":
            self.action_run_plan()
        elif bid == "plan-edit-key":
            self.action_edit_plan()
        elif bid == "plan-back":
            self.action_plan_back()

    def _submit(self, raw: str) -> None:
        if self._busy:
            return
        text = raw.strip()
        self.query_one("#entry", TextArea).text = ""
        if not text:
            return
        self._say(f"[cyan]> {escape(text)}[/]")
        if self.phase == "review":
            self._on_review(text)
        else:
            self._on_chat(text)

    def _on_chat(self, text: str) -> None:
        task_md = self._task_md()
        self._auto_run = is_cowabunga(text)
        if self._auto_run:
            self._set_busy(True, "cowabunga — writing the plan, then running…")
            self._spawn(self._write_worker, task_md=task_md, autodecide=True)
        elif text.lower() in _PLAN_WORDS:
            self._set_busy(True, "planner writing the plan…")
            self._spawn(self._write_worker, task_md=task_md, autodecide=False)
        else:
            self._set_busy(True, "planner incorporating your answers…")
            self._spawn(self._refine_worker, answers=text, task_md=task_md)

    def _on_review(self, text: str) -> None:
        if is_cowabunga(text) or text.lower() in _RUN_WORDS:
            self._run()
            return
        self._set_busy(True, "planner revising the plan…")
        self._spawn(self._revise_worker, instructions=text, plan_md=self._plan_md())

    # --- finish ---
    def _run(self) -> None:
        plan = self._plan_md()
        if not plan.strip():
            self._say("[yellow]The plan is empty — write or revise it before running.[/]")
            return
        self._save_plan(plan)
        self._set_phase("run")
        self._save_state()
        self._say(f"[green]▶ running with strategy '{self.strategy}'…[/]")
        self.exit(0)

    def action_abort(self) -> None:
        if isinstance(self.screen, ConfirmQuit):
            return

        def _decide(leave: bool | None) -> None:
            if leave:
                from splinter import procreg

                procreg.terminate_all()
                self.exit(None)

        self.push_screen(ConfirmQuit(self.session.id, "plan"), _decide)

    def get_system_commands(self, screen: Any) -> Iterable[SystemCommand]:
        yield _find_shortcuts_cmd(screen, self)
        yield SystemCommand("Theme", "Change the current theme", self.action_change_theme)
        label = "Minimize" if self._maximized else "Maximize"
        yield SystemCommand(label, "Toggle the right panel size", self.action_toggle_maximize)
        yield SystemCommand("Quit", "Quit the application", self.action_quit)

    def action_toggle_maximize(self) -> None:
        self._maximized = not self._maximized

    def watch__maximized(self, val: bool) -> None:
        self.set_class(val, "--maximized")


def _step_lines(plan: str) -> str:
    """The numbered steps of the plan, lime numbers, muted text."""
    rows = []
    for line in plan.splitlines():
        number, dot, rest = line.strip().partition(". ")
        if dot and number.isdigit():
            rows.append(f"[#c6f432]{number}[/]  [#ecebe4]{escape(rest.strip()[:90])}[/]")
    return "\n".join(rows[:12])


def _handoff(session: Session, run_kwargs: dict[str, Any], result: int | None) -> int:
    """After the plan screen: run in-app on Run, else print how to resume."""
    from splinter.tui import run_with_tui

    status = session.read_status()
    if result == 0 and status.get("phase") == "run":
        task_path = status.get("task_path") if status.get("plan_index") else None
        final = {
            **run_kwargs,
            "task_path": str(task_path or session.dir / "task.yaml"),
            "prd_path": None,
            "strategy": str(status.get("strategy") or DEFAULT_STRATEGY),
            "resume": False,
        }
        return run_with_tui(final, session=session)
    if session.dir.exists():
        print(f"plan session aborted — resume later: uv run splinter resume {session.id}")
    return 0


def run_plan_interactive(run_kwargs: dict[str, Any]) -> int:
    """Plan the task in a TUI, then execute it in the same session."""
    from splinter.memory.session import new_session_id

    session = Session(new_session_id())
    result = PlanSessionApp(session, run_kwargs).run()
    return _handoff(session, run_kwargs, result)


def _resume_plan(session: Session, status: dict[str, Any]) -> int:
    """Re-enter a plan-mode session at its saved phase."""
    saved = status.get("strategy")
    run_kwargs: dict[str, Any] = {
        "strategy": saved if saved and saved != "?" else None,
        "task_path": str(status.get("task_path") or session.dir / "task.yaml"),
        "prd_path": None,
        "effort": None,
        "budget": None,
        "max_iterations": 5,
        "cowabunga": False,
        "resume": True,
    }
    result = PlanSessionApp(session, run_kwargs).run()
    return _handoff(session, run_kwargs, result)
