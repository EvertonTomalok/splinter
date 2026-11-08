"""What one iteration did and why it ended the way it did, as typed log lines.

An opened iteration on the activity view reads three files the run already writes:
the review note (decision, reason, corrections), the gate and verdict lines of
``loop.md``, and the runner's output (model, cost, and every 💬 thought and 🔧 tool
call). They render as one column of ``kind  text`` rows, the same grammar as the live
log, so a failed iteration says what broke, what the judge asked for, and what the
runner tried.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from rich.text import Text

from splinter.tui_run_log import KIND_COLOR, classify
from splinter.tui_run_palette import AMBER, DIM, INK, LIME, LINE, MUTED, RED, VERDICT_COLOR

#: Actions shown per iteration; older ones fold into one ``… N earlier`` row.
MAX_ACTIONS = 14
_KIND_W = 9
_TOKEN = re.compile(r"'?(input|output)'?\s*:\s*(\d+)")
_WORKTREE = re.compile(r"^.*?/\.splinter/worktrees/[^/]+/")
_META = re.compile(r"^-\s*(model|cost|tokens):\s*(.+)$", re.IGNORECASE)


def _row(kind: str, text: str, tone: str, width: int, kind_tone: str = "") -> Text:
    line = Text("  │ ", style=LINE)
    line.append(f"{kind:<{_KIND_W}}", style=kind_tone or KIND_COLOR.get(kind, MUTED))
    room = max(10, width - _KIND_W - 6)
    shown = text if len(text) <= room else text[: room - 1] + "…"
    line.append(shown, style=tone)
    return line


def _tokens(raw: str) -> str:
    """``{'input': 2399, 'output': 45327}`` → ``2.4k in · 45.3k out``."""
    found = {key: int(value) for key, value in _TOKEN.findall(raw)}
    if not found:
        return raw
    parts = [f"{_k(found[key])} {short}" for key, short in (("input", "in"), ("output", "out"))]
    return " · ".join(part for part, key in zip(parts, ("input", "output")) if key in found)


def _k(value: int) -> str:
    return f"{value / 1000:.1f}k" if value >= 1000 else str(value)


def _short_path(text: str) -> str:
    return " ".join(_WORKTREE.sub("", part) for part in text.split(" "))


def run_meta(run_text: str) -> dict[str, str]:
    """``model``, ``cost`` and ``tokens`` from the runner output header."""
    meta: dict[str, str] = {}
    for line in run_text.splitlines()[:12]:
        match = _META.match(line.strip())
        if match:
            meta[match.group(1).lower()] = match.group(2).strip()
    return meta


def run_actions(run_text: str) -> list[tuple[str, str]]:
    """``(kind, text)`` for every 💬 thought and 🔧 tool call of the run."""
    actions: list[tuple[str, str]] = []
    for line in run_text.splitlines():
        body = line.strip().removeprefix("- ").strip()
        if not body.startswith(("💬", "🔧")):
            continue
        kind, text = classify(body)
        actions.append((kind or "note", _short_path(text.strip())))
    return actions


def _verdict_rows(decision: str, reason: str, width: int) -> list[Text]:
    word = decision.split()[0].upper() if decision else ""
    tone = VERDICT_COLOR.get(word, MUTED)
    rows = [_row("verdict", f"{word or '—'}  {reason}".strip(), INK, width, tone)]
    return rows


def _gate_rows(gate: Sequence[str], width: int) -> list[Text]:
    rows: list[Text] = []
    for line in gate:
        body = line.removeprefix("- gate:").strip()
        verdict, _, detail = body.partition(" ")
        if verdict.upper() == "PASS":
            rows.append(_row("gate", "✓ all checks passed", LIME, width, LIME))
            continue
        failed = detail.removeprefix("failed:").strip() or "failed"
        rows.append(_row("gate", f"✗ {failed}", RED, width, RED))
    return rows


def _fix_rows(corrections: str, width: int) -> list[Text]:
    lines = [line.strip().lstrip("-* ").strip() for line in corrections.splitlines()]
    lines = [line for line in lines if line and line.lower() != "none"]
    return [_row("fix", line, AMBER, width, AMBER) for line in lines[:6]]


def _action_rows(actions: list[tuple[str, str]], width: int) -> list[Text]:
    rows: list[Text] = []
    hidden = len(actions) - MAX_ACTIONS
    if hidden > 0:
        rows.append(_row("", f"… {hidden} earlier actions", DIM, width))
    for kind, text in actions[-MAX_ACTIONS:]:
        rows.append(_row(kind, text, INK if kind == "edit" else MUTED, width))
    return rows


def iteration_detail(
    run_text: str,
    review: dict[str, str],
    gate: Sequence[str],
    verdict_line: str,
    width: int = 100,
) -> Text:
    """The opened iteration: verdict and reason, gate, fixes asked, cost, then actions."""
    decision = review.get("decision", "")
    reason = review.get("reason", "")
    if not decision and verdict_line:
        decision, _, reason = verdict_line.removeprefix("- verdict:").strip().partition("—")
    rows: list[Text] = []
    if decision or reason:
        rows.extend(_verdict_rows(decision.strip(), reason.strip(), width))
    rows.extend(_gate_rows(gate, width))
    rows.extend(_fix_rows(review.get("corrections", ""), width))
    meta = run_meta(run_text)
    if meta:
        parts = [meta.get("model", ""), meta.get("cost", ""), _tokens(meta.get("tokens", ""))]
        rows.append(_row("model", " · ".join(part for part in parts if part), MUTED, width))
    rows.extend(_action_rows(run_actions(run_text), width))
    if not rows:
        rows.append(_row("", "no output yet", DIM, width))
    return Text("\n").join(rows)
