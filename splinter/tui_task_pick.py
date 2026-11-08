"""Where the task comes from: a path typed on one line, the project tree, or the OS picker.

The ``--task`` draft screen has no free-form editor. The task is a file (yaml or
markdown), a plan folder or a README that links task files, or one line of text
(saved as ``task.md``). :class:`TaskPicker` holds a one-line path input with ``tab``
completion and the project tree under it: moving through the tree fills the path,
``enter`` on a file takes it. ``^o`` and ``^l`` open the system's own file and folder
choosers when there is one (macOS ``osascript``).
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import DirectoryTree, Input

from splinter.tui_run_palette import Ink

#: Files a task can be.
TASK_SUFFIXES = {".md", ".markdown", ".yaml", ".yml", ".txt"}
_SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache", ".ruff_cache",
    ".pytest_cache", ".splinter", "dist", "build", ".idea", ".vscode",
}
_PROMPT = "Pick the task for splinter"


def relative(path: Path, root: Path) -> str:
    """``path`` relative to ``root`` when it is inside it, else absolute; folders end in ``/``."""
    try:
        shown = str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        shown = str(path)
    if path.is_dir() and not shown.endswith("/"):
        shown = "./" if shown == "." else f"{shown}/"
    return shown


def complete(text: str, root: Path) -> str:
    """Complete ``text`` as a path under ``root`` to the longest common prefix."""
    typed = os.path.expanduser(text)
    base = Path(typed) if os.path.isabs(typed) else root / typed
    folder, stem = (base, "") if typed.endswith("/") or not typed else (base.parent, base.name)
    try:
        names = sorted(
            entry.name + ("/" if entry.is_dir() else "")
            for entry in folder.iterdir()
            if entry.name.startswith(stem) and not entry.name.startswith(".")
        )
    except OSError:
        return text
    if not names:
        return text
    common = os.path.commonprefix(names)
    head = text[: len(text) - len(stem)] if stem else text
    return head + common


def system_pick(folder: bool) -> str:
    """The path the OS chooser returns, or ``""`` (cancelled or no chooser here)."""
    if sys.platform != "darwin":
        return ""
    kind = "choose folder" if folder else "choose file"
    script = f'POSIX path of ({kind} with prompt "{_PROMPT}")'
    try:
        done = subprocess.run(
            ["osascript", "-e", script], capture_output=True, text=True, timeout=600
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


class TaskTree(DirectoryTree):
    """The project tree, folders and task-shaped files only."""

    ICON_NODE = "▸ "
    ICON_NODE_EXPANDED = "▾ "

    def filter_paths(self, paths: Iterable[Path]) -> Iterable[Path]:
        return [
            path
            for path in paths
            if not path.name.startswith(".")
            and path.name not in _SKIP_DIRS
            and (path.is_dir() or path.suffix.lower() in TASK_SUFFIXES)
        ]


def looks_like_path(text: str) -> bool:
    """One token that reads as a path (``tasks/x.yaml``, ``~/plan/``, ``a.md``)."""
    typed = text.strip()
    if not typed or any(ch.isspace() for ch in typed):
        return False
    return "/" in typed or typed.startswith(("~", ".")) or Path(typed).suffix in TASK_SUFFIXES


class PathInput(Input):
    """One line: the task path, or the task itself in a few words."""

    BINDINGS = [
        Binding("tab", "complete", "Complete", show=False, priority=True),
        Binding("down", "to_tree", "Tree", show=False),
    ]

    def action_complete(self) -> None:
        """Complete the path; with nothing left to complete, move on to the tree."""
        picker = self.parent
        if not isinstance(picker, TaskPicker):
            return
        done = complete(self.value, picker.root)
        if done == self.value:
            picker.focus_tree()
            return
        self.value = done
        self.cursor_position = len(done)

    def action_to_tree(self) -> None:
        picker = self.parent
        if isinstance(picker, TaskPicker):
            picker.focus_tree()


class TaskPicker(Vertical):
    """Path input on top, the project tree under it."""

    DEFAULT_CSS = """
    TaskPicker { height: 1fr; }
    TaskPicker #task-path {
        height: 1;
        border: none;
        border-left: wide #7a7972;
        padding: 0 1;
        background: transparent;
        margin-bottom: 1;
    }
    TaskPicker #task-path:focus { border: none; border-left: wide #c6f432; }
    TaskPicker #task-path > .input--placeholder { color: #7a7972; }
    TaskPicker #task-tree-head { height: 1; color: #8f8e85; }
    TaskPicker #task-tree {
        height: 1fr;
        background: #0a0a09;
        padding: 0 0 0 1;
        border-left: solid #1d1d1a;
        scrollbar-size-vertical: 1;
    }
    TaskPicker #task-tree:focus { background-tint: #0a0a09 0%; }
    TaskPicker #task-tree > .tree--cursor { background: #1c2409; color: #ecebe4; }
    TaskPicker #task-tree > .tree--guides { color: #2a2a26; }
    TaskPicker #task-tree > .directory-tree--folder { color: #ecebe4; text-style: bold; }
    TaskPicker #task-tree > .directory-tree--file { color: #8f8e85; }
    TaskPicker #task-tree > .directory-tree--extension { color: #7a7972; }
    """

    class Chosen(Message):
        """A file was taken from the tree or the system chooser."""

        def __init__(self, path: str) -> None:
            super().__init__()
            self.path = path

    def __init__(self, root: Path | None = None, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self.root = (root or Path.cwd()).resolve()

    def compose(self) -> ComposeResult:
        yield PathInput(
            id="task-path",
            placeholder="path to a task, a plan folder or a README · or the task in a line",
        )
        yield Ink(
            "[#8f8e85]or pick it[/] [#7a7972]· ↓ from the path · ↑↓ move · → open folder"
            " · ↵ take[/]",
            id="task-tree-head",
        )
        yield TaskTree(self.root, id="task-tree")

    @property
    def value(self) -> str:
        return self.query_one("#task-path", PathInput).value.strip()

    def focus_tree(self) -> None:
        self.query_one("#task-tree", TaskTree).focus()

    def focus_path(self) -> None:
        self.query_one("#task-path", PathInput).focus()

    def set_path(self, path: str) -> None:
        box = self.query_one("#task-path", PathInput)
        box.value = path
        box.cursor_position = len(path)

    @on(DirectoryTree.FileSelected, "#task-tree")
    def _take_file(self, event: DirectoryTree.FileSelected) -> None:
        event.stop()
        shown = relative(event.path, self.root)
        self.set_path(shown)
        self.post_message(self.Chosen(shown))

    @on(DirectoryTree.NodeHighlighted, "#task-tree")
    def _follow(self, event: DirectoryTree.NodeHighlighted[object]) -> None:
        if not self.query_one("#task-tree", TaskTree).has_focus:
            return
        entry = event.node.data
        path = getattr(entry, "path", None)
        if isinstance(path, Path):
            self.set_path(relative(path, self.root))

    def take_system_pick(self, picked: str) -> None:
        if picked:
            shown = relative(Path(picked), self.root)
            self.set_path(shown)
            self.post_message(self.Chosen(shown))
