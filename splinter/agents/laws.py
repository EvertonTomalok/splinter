"""Workhorse contract, AGENT_RESULT protocol parsing and brief document rendering.

Port of the agent-engine COMMON.md (workhorse contract) and BRIEF.md templates:
the brief is the authority for the task, the contract for the protocol.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

REPORT_PATH = ".agent/report.md"

RESULT_LINE_PATTERN: re.Pattern[str] = re.compile(r"AGENT_RESULT: ([A-Z]+) ([A-Za-z0-9-]+)")


@dataclass(frozen=True)
class AgentResult:
    status: str
    ident: str


def parse_agent_result(text: str) -> AgentResult | None:
    """Parse the last ``AGENT_RESULT: <STATUS> <ident>`` line in text, if any."""
    matches = RESULT_LINE_PATTERN.findall(text)
    if not matches:
        return None
    status, ident = matches[-1]
    return AgentResult(status=status, ident=ident)


_CONTRACT_HEAD = (
    "# Workhorse contract (every agent reads this after its brief)\n"
    "\n"
    "You are a fast implementer. The plan, the design and the review belong to the "
    "orchestrator. Your job is\n"
    "the brief, nothing else, done quickly.\n"
    "\n"
    "1. Read the brief, then only the files it names and what they directly call. Skip "
    "exploring the\n"
    "   repository, reading docs the brief does not cite, or redesigning anything.\n"
    "2. Edit only the files the brief lists. A file it does not list stays untouched; if "
    "the work truly needs\n"
    "   one, write that in the report and stop.\n"
    "3. Follow the project rules below and the code around you: same naming, comment "
    "density and idioms.\n"
    "   Everything you write is English except text a person reads on a screen.\n"
    "4. Run no build, test, lint, generator, package manager, container or git write: the "
    "orchestrator\n"
    '   verifies and commits after you, and the run guard refuses them anyway ("BLOCKED by '
    'the run guard":\n'
    "   take it as the answer and keep editing). Reading commands (`ls`, `cat`, `rg`, "
    "`git diff`,\n"
    "   `git status`) are fine.\n"
    "5. Workhorse mode: execute directly without thinking or over-analysis. Implement the "
    "narrow sketch fast.\n"
    "   If returned to this session with adjustment notes, apply the fixes immediately "
    "and finish.\n"
    "6. Finish in minutes. When the acceptance line holds from your diff, stop.\n"
    "7. Last two actions, always: write `.agent/report.md` at the root of your worktree "
    "(files changed, what\n"
    "   each change does, anything left open, any deviation from the sketch), then print "
    "exactly one line:\n"
    "   `AGENT_RESULT: DONE <id>` or `AGENT_RESULT: BLOCKED <id>`.\n"
)

_LAWS_HEADER = "\n## The laws of this project (read them, they bind every change)\n\n"

_PLAN_LINE = (
    "- The plan this run implements: {{plan}}/README.md "
    "(read it; the brief is the authority for the task, the plan for decisions)."
)


def workhorse_contract(*, laws: str = "", plan_ref: str = "") -> str:
    """Render the COMMON.md workhorse contract.

    Non-empty ``laws`` keeps "The laws of this project" (COMMON.md's ``{{laws}}``
    placeholder); empty laws drop the section header but keep the plan reference
    line (``{{plan}}`` → ``plan_ref``).
    """
    parts = [_CONTRACT_HEAD]
    if laws:
        parts.append(_LAWS_HEADER + laws.rstrip("\n") + "\n")
    parts.append("\n" + _PLAN_LINE.replace("{{plan}}", plan_ref) + "\n")
    return "".join(parts)


def _context_line(ident: str, *, worktree: str, branch: str, run_dir: str) -> str:
    parts: list[str] = []
    if worktree:
        piece = f"Worktree: `{worktree}`"
        if branch:
            piece += f" (branch `{branch}-{ident}`)"
        parts.append(piece + ".")
    elif branch:
        parts.append(f"Branch: `{branch}-{ident}`.")
    if run_dir:
        parts.append(f"Run folder: `{run_dir}`.")
    if not parts:
        return ""
    return " ".join(parts)


def brief_document(
    *,
    ident: str,
    goal: str = "",
    files: Sequence[str] = (),
    sketch: str = "",
    acceptance: Sequence[str] = (),
    worktree: str = "",
    branch: str = "",
    run_dir: str = "",
) -> str:
    """Render the BRIEF.md document handed to one workhorse agent."""
    lines: list[str] = []
    goal_text = goal.strip()
    goal_line = goal_text.splitlines()[0] if goal_text else ""
    lines.append(f"# Brief {ident} — {goal_line}" if goal_line else f"# Brief {ident}")
    lines.append("")
    context = _context_line(ident, worktree=worktree, branch=branch, run_dir=run_dir)
    if context:
        lines.append(context)
        lines.append("")
    lines.append("## Goal")
    lines.append("")
    if goal_text:
        lines.append(goal_text)
        lines.append("")
    lines.append("## Files you may touch")
    lines.append("")
    for path in files:
        lines.append(f"- `{path}`")
    if files:
        lines.append("")
    lines.append("Everything else stays as it is.")
    lines.append("")
    lines.append("## Change sketch")
    lines.append("")
    if sketch:
        lines.append(sketch.rstrip("\n"))
        lines.append("")
    lines.append("## Acceptance")
    lines.append("")
    for item in acceptance:
        lines.append(f"- [ ] {item}")
    if acceptance:
        lines.append("")
    lines.append("## Report")
    lines.append("")
    lines.append(
        f"Write `{REPORT_PATH}` in your worktree, then print "
        f"`AGENT_RESULT: DONE {ident}` (or `BLOCKED {ident}`)."
    )
    return "\n".join(lines) + "\n"
