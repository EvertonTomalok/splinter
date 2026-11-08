"""Files and text around one brief: the engine prompt, its diff, report and logs."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from splinter.agents.laws import REPORT_PATH
from splinter.agents.refiner import sanitize_brief_id
from splinter.agents.runner import Task

DIFF_LIMIT = 60_000
NEW_FILE_LIMIT = 6_000

_FAST_INSTRUCTION = (
    "Implement exactly the brief, fast and directly. Do not explore, analyze, or overthink. "
)


def slug_from(tasks: list[Task]) -> str:
    base = (tasks[0].id if tasks and tasks[0].id else "run").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", base).strip("-")
    return slug or "run"


def engine_prompt(ident: str, brief_text: str, contract: str) -> str:
    report_rule = (
        "Nobody answers questions. Write .agent/report.md at the root of the worktree, "
        f"then print exactly one final line: AGENT_RESULT: DONE {ident} "
        f"(or AGENT_RESULT: BLOCKED {ident} when the brief cannot be done by editing files, "
        "with the reason in the report)."
    )
    return (
        "You are a fast workhorse implementer. "
        f"{_FAST_INSTRUCTION}"
        "Follow the workhorse contract below. "
        f"{report_rule}\n\n--- BRIEF ---\n{brief_text}\n\n"
        f"--- WORKHORSE CONTRACT ---\n{contract}\n"
    )


def _git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=path, capture_output=True, text=True, check=False
    ).stdout


def worktree_diff(path: Path) -> str:
    """Every uncommitted change of a brief worktree, file list first.

    The list names each changed and new file (``--untracked-files=all``), so a
    judge sees which files exist even when the text below is cut. Each new
    file shows at most ``NEW_FILE_LIMIT`` characters.
    """
    status = _git(path, "status", "--short", "--untracked-files=all").strip()
    diff = _git(path, "diff", "HEAD")
    untracked = _git(path, "ls-files", "--others", "--exclude-standard", "-z")
    additions: list[str] = []
    for name in untracked.split("\0"):
        file = path / name
        if not name or not file.is_file() or file.is_symlink():
            continue
        content = file.read_text(errors="replace")
        if len(content) > NEW_FILE_LIMIT:
            content = content[:NEW_FILE_LIMIT] + f"\n… ({len(content)} characters in all)"
        additions.append(f"diff --git a/{name} b/{name}\nnew file\n{content}")
    body = f"{status}\n\n{diff}\n" + "\n".join(additions)
    if len(body) <= DIFF_LIMIT:
        return body
    return body[:DIFF_LIMIT] + "\n… (diff cut; the file list above is complete)"


def read_report(worktree: Path) -> str:
    try:
        return (worktree / REPORT_PATH).read_text()
    except OSError:
        return ""


def keep_report(worktree: Path, dest: Path) -> str:
    """Copy the workhorse report out of the worktree before it is torn down."""
    text = read_report(worktree)
    if text:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)
    return text


def write_engine_log(run_dir: Path, brief_id: str, attempt: int, engine: str, text: str) -> None:
    log = run_dir / "logs" / f"{brief_id}-{attempt}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(f"engine: {engine}\n\n{text}")


def saved_briefs_usable(raw: str, tasks: list[Task], done: set[str]) -> bool:
    """Whether a resume may reuse ``briefs.json`` instead of refining again.

    A run killed before refine finished leaves the 1:1 fallback behind (one
    brief per story, the story text as goal). Reusing that skips the plan, so
    it is refined again unless a brief already finished.
    """
    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        return False
    if not isinstance(items, list) or not items:
        return False
    if done:
        return True
    copied = len(items) == len(tasks) and all(
        isinstance(item, dict)
        and item.get("id") == sanitize_brief_id(task.id or f"t{n}")
        and item.get("goal") == task.description
        for n, (item, task) in enumerate(zip(items, tasks))
    )
    return not copied
