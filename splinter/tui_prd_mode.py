"""The two entry modes of the draft screen: ``--prd`` (writer, questions, setup) and
``--task`` (straight to the run, no questions).

The header chips switch between them while nothing has been generated yet. Each mode
names its stage word, its steps, the question it asks and what ``^s`` does.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from splinter.memory.session import Session
from splinter.tui_stepper import PRD_STEPS

#: Steps of a task: pick it, then run it (``^t`` adds questions and a plan in between).
TASK_STEPS: tuple[str, ...] = ("Task", "Run")

_LIME = "#c6f432"
_INK = "#ecebe4"
_MUTED = "#8f8e85"


@dataclass(frozen=True)
class DraftMode:
    """Copy and steps of one entry mode."""

    name: str
    word: str
    steps: tuple[str, ...]
    question: str
    hint: str
    placeholder: str
    action: str

    def sentence(self) -> str:
        return f"[b {_INK}]{self.question}[/] [{_LIME}]▌[/]\n[{_MUTED}]{self.hint}[/]"

    def keys(self) -> str:
        middle = (
            f"[@click=app.plan_task][b {_LIME}]^t[/] [{_INK}]plan it first[/][/]    "
            f"[@click=app.pick('file')][b {_LIME}]^o[/] [{_MUTED}]file[/][/]    "
            f"[@click=app.pick('folder')][b {_LIME}]^l[/] [{_MUTED}]folder[/][/]    "
            f"[b {_LIME}]tab[/] [{_MUTED}]complete[/]    "
            if self.name == "task"
            else f"[b {_LIME}]↵[/] [{_MUTED}]newline[/]    "
        )
        return (
            f"[@click=app.send][b {_LIME}]^s[/] [{_INK}]{self.action}[/][/]    "
            f"{middle}"
            f"[@click=app.abort][b {_LIME}]esc[/] [{_MUTED}]abort[/][/]"
        )


MODES: dict[str, DraftMode] = {
    "prd": DraftMode(
        name="prd",
        word="DRAFT",
        steps=PRD_STEPS,
        question="What should splinter build?",
        hint="Goal, constraints, files you know. The writer asks only what it can't decide.",
        placeholder="e.g. add Retry-After handling to the payments client",
        action="generate PRD",
    ),
    "task": DraftMode(
        name="task",
        word="TASK",
        steps=TASK_STEPS,
        question="What should splinter do?",
        hint="A task file, a plan folder or a README, or the task in one line. "
        "^s runs it now; ^t lets the planner ask and write the plan first.",
        placeholder="e.g. tasks/c2-focus-tui/README.md",
        action="run it",
    ),
}


def task_source(session: Session, text: str) -> Path:
    """The task to run: an existing path as typed, else the text saved as ``task.md``."""
    typed = text.strip()
    if "\n" not in typed:
        candidate = Path(typed).expanduser()
        if candidate.exists():
            return candidate.resolve()
    session.write("task.md", f"# Task\n\n{typed}\n")
    return session.dir / "task.md"
