"""Activity-log lines of the run screen: ``14:55:09  think  round the wait up``."""

from __future__ import annotations

import logging
import re

from rich.markup import escape

from splinter.tui_run_palette import AMBER, DIM, INK, LIME, MUTED, ORCH_BG, RED

KIND_COLOR = {
    "read": DIM,
    "edit": LIME,
    "think": MUTED,
    "run": MUTED,
    "gate": RED,
    "eval": AMBER,
    "plan": MUTED,
    "done": LIME,
    "you": LIME,
    "orch": AMBER,
}
_READ_TOOLS = {"read", "view", "glob", "grep", "ls", "list", "search", "find", "fetch", "webfetch"}
_EDIT_TOOLS = {"edit", "write", "multiedit", "patch", "apply_patch", "create", "str_replace"}
_RUN_TOOLS = {"bash", "shell", "exec", "run", "command"}
_TOOL_RE = re.compile(r"^🔧\s*(\S+)\s*(.*)$")
_KEYWORDS = (
    ("👤", "you"),
    ("orch →", "orch"),
    ("RUN COMPLETE", "done"),
    ("✅", "done"),
    ("gate", "gate"),
    ("eval", "eval"),
    ("judge", "eval"),
    ("plan", "plan"),
)
KIND_WIDTH = 5


def classify(message: str) -> tuple[str, str]:
    """Split a log message into its kind and the text shown after it."""
    text = message.strip()
    tool = _TOOL_RE.match(text)
    if tool:
        name, detail = tool.group(1), tool.group(2)
        return _tool_kind(name), detail or name
    if text.startswith("💬"):
        return "think", text.removeprefix("💬").strip()
    lowered = text.lower()
    for needle, kind in _KEYWORDS:
        if needle.lower() in lowered:
            return kind, text
    return "", text


def _tool_kind(name: str) -> str:
    bare = name.lower().split(".")[-1]
    if bare in _READ_TOOLS:
        return "read"
    if bare in _EDIT_TOOLS:
        return "edit"
    if bare in _RUN_TOOLS:
        return "run"
    return bare[:KIND_WIDTH]


def format_line(message: str, level: int, prefix: str) -> str:
    """Markup for one log line; ``prefix`` is the stamp (and task tag)."""
    kind, text = classify(message)
    tone = {logging.ERROR: RED, logging.WARNING: AMBER}.get(level, INK)
    if kind == "you":
        tone = LIME
    label = f"[{KIND_COLOR.get(kind, MUTED)}]{kind:<{KIND_WIDTH}}[/]"
    line = f"{prefix}{label}  [{tone}]{escape(text)}[/]"
    return f"[on {ORCH_BG}]{line}[/]" if kind == "orch" else line


def stamp_prefix(stamp: str, task: str = "") -> str:
    tag = f"[{DIM}][T{task}][/] " if task else ""
    return f"[{DIM}]{stamp}[/]  {tag}"
