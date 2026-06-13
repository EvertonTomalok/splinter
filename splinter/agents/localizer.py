"""Two-phase LLM-driven localizer: deterministic search first, then LLM filter."""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass

from splinter.memory.knowledge import KnowledgeStore
from splinter.memory.session import Session
from splinter.models.roster import Ladder
from splinter.obs.agentic import record_exchange
from splinter.providers.base import ProviderGapError
from splinter.providers.dispatch import run_text
from splinter.tools import search as search_tools

log = logging.getLogger("splinter.localizer")

#: Search-tool output above this size routes recall to the big-context model.
LARGE_CONTEXT_CHARS = 20_000


@dataclass(frozen=True)
class CodeAnchor:
    file: str
    symbol: str
    reason: str
    confidence: float
    line_start: int | None = None
    line_end: int | None = None
    relevance: str = ""


def _rtk_available() -> bool:
    """Check if rtk command is available in PATH."""
    return shutil.which("rtk") is not None


def rtk_cat_tip(anchor: CodeAnchor) -> str:
    """Generate a command-line tip to read the anchor's code."""
    if anchor.line_start:
        end = anchor.line_end or anchor.line_start
        return f"rtk read {anchor.file} | sed -n '{anchor.line_start},{end}p'"
    return f"rtk read {anchor.file}"


def _relevance_from_confidence(conf: float, *, hot: float, medium: float) -> str:
    """Derive relevance tag from confidence value using thresholds."""
    if conf >= hot:
        return "hot"
    elif conf >= medium:
        return "medium"
    else:
        return "low"


def _count_anchors(text: str) -> int:
    """Count anchors in a localization.md by counting 'file:' block headers."""
    return sum(1 for ln in text.splitlines() if ln.strip().startswith("file:"))


def _strip_frontmatter(text: str) -> str:
    """Remove YAML frontmatter (--- ... ---) from PRD text."""
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            return parts[2].strip()
    return text.strip()


def _parse_anchors(text: str, *, hot: float = 0.8, medium: float = 0.4) -> list[CodeAnchor]:
    anchors: list[CodeAnchor] = []

    def _items_from_json(raw: str) -> list[CodeAnchor]:
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
        if not isinstance(data, list):
            return []
        result = []
        for item in data:
            if not isinstance(item, dict):
                continue
            ls = item.get("line_start")
            le = item.get("line_end")
            conf = float(item.get("confidence", 0.5))
            rel = item.get("relevance")
            if not rel:
                rel = _relevance_from_confidence(conf, hot=hot, medium=medium)
            result.append(
                CodeAnchor(
                    file=item.get("file", ""),
                    symbol=item.get("symbol", ""),
                    reason=item.get("reason", ""),
                    confidence=conf,
                    line_start=int(ls) if ls is not None else None,
                    line_end=int(le) if le is not None else None,
                    relevance=rel,
                )
            )
        return result

    # Try bare JSON first (ideal output).
    anchors = _items_from_json(text)
    if anchors:
        return anchors

    # LLMs often wrap JSON in prose — extract the first [...] block and retry.
    m = re.search(r"\[.*?\]", text, re.DOTALL)
    if m:
        anchors = _items_from_json(m.group(0))
        if anchors:
            return anchors

    # Fall back to the structured key:value block format written by localize().
    # Split on blank lines to get individual anchor blocks.
    blocks = re.split(r"\n\s*\n", text)
    for block in blocks:
        file_m = re.search(r"^file:\s*(.+)$", block, re.MULTILINE)
        if not file_m:
            continue
        symbol_m = re.search(r"^symbol:\s*(.*)$", block, re.MULTILINE)
        reason_m = re.search(r"^reason:\s*(.+)$", block, re.MULTILINE)
        conf_m = re.search(r"^confidence:\s*([\d.]+)$", block, re.MULTILINE)
        ls_m = re.search(r"^line_start:\s*(\d+)$", block, re.MULTILINE)
        le_m = re.search(r"^line_end:\s*(\d+)$", block, re.MULTILINE)
        rel_m = re.search(r"^relevance:\s*(.+)$", block, re.MULTILINE)
        conf = float(conf_m.group(1)) if conf_m else 0.5
        if rel_m:
            rel = rel_m.group(1).strip()
        else:
            rel = _relevance_from_confidence(conf, hot=hot, medium=medium)
        anchors.append(
            CodeAnchor(
                file=file_m.group(1).strip(),
                symbol=symbol_m.group(1).strip() if symbol_m else "",
                reason=reason_m.group(1).strip() if reason_m else "",
                confidence=conf,
                line_start=int(ls_m.group(1)) if ls_m else None,
                line_end=int(le_m.group(1)) if le_m else None,
                relevance=rel,
            )
        )
    return anchors


_META_FILE_NAMES = ("AGENTS.md", "CLAUDE.md")
_META_SKILL_GLOBS = ("skills/*/SKILL.md", "splinter/skills/*/SKILL.md")
_META_FILE_MAX_CHARS = 800  # per meta file snippet shown to recall LLM


def _find_meta_files(repo_path: str = ".") -> list[str]:
    """Return paths to AGENTS.md, CLAUDE.md, and skill SKILL.md files under repo_path."""
    from pathlib import Path as _Path

    root = _Path(repo_path)
    found: list[str] = []
    seen: set[str] = set()

    def _add(p: "_Path") -> None:
        try:
            rel = str(p.relative_to(root))
        except ValueError:
            rel = str(p)
        if rel not in seen:
            seen.add(rel)
            found.append(rel)

    for name in _META_FILE_NAMES:
        for p in sorted(root.rglob(name)):
            if p.is_file():
                _add(p)

    for pattern in _META_SKILL_GLOBS:
        for p in sorted(root.glob(pattern)):
            if p.is_file():
                _add(p)

    return found


_LS_MAX_LINES = 3000  # file listing cap passed to recall model


def _run_search_tools(prd_text: str, repo_path: str = ".") -> str:
    """Gather fast, deterministic signals: meta files + full tracked file listing.

    No grep here — the recall LLM reads the file listing and picks relevant
    paths directly.  Grep would require a prior LLM call to scope it, which is
    exactly what the recall model does.
    """
    from pathlib import Path as _Path

    lines: list[str] = []
    lines.append("# Search Tool Results\n")

    # Always include meta/agent files — AGENTS.md, CLAUDE.md, skill SKILL.md
    meta_files = _find_meta_files(repo_path)
    if meta_files:
        lines.append("## Meta / Agent / Skill Files")
        lines.append("These describe project conventions, agent behaviour, and available skills.")
        for mf in meta_files:
            try:
                content = (_Path(repo_path) / mf).read_text()
            except OSError:
                try:
                    content = _Path(mf).read_text()
                except OSError:
                    continue
            snippet = content[:_META_FILE_MAX_CHARS]
            note = " ← truncated" if len(content) > _META_FILE_MAX_CHARS else ""
            lines.append(f"### {mf}{note}\n```\n{snippet}\n```")
        lines.append("")

    # Full tracked file listing — git ls-files respects .gitignore so node_modules,
    # vendor, dist, __pycache__ etc. are already excluded automatically.
    fl_result = search_tools.file_list(repo_path)
    if fl_result.unavailable:
        lines.append(f"## File Listing\n{fl_result.output}\n")
    else:
        listing_lines = fl_result.output.splitlines()
        truncated = len(listing_lines) > _LS_MAX_LINES
        listing = "\n".join(listing_lines[:_LS_MAX_LINES])
        if truncated:
            extra = len(listing_lines) - _LS_MAX_LINES
            listing += f"\n... ({extra} more files — use file name patterns to infer)"
        lines.append("## Tracked Files (git ls-files)")
        lines.append(listing)
        lines.append("")

    return "\n".join(lines)


def _recall_phase(
    prd_text: str,
    search_results: str,
    model: str,
    variant: str = "minimal",
    timeout: int | None = None,
    agent: str = "build",
    session: object = None,
) -> str:
    """Cheap model filters the raw search results to a candidate list."""
    text = _strip_frontmatter(prd_text)
    prompt = (
        "I want to implement this feature — look at all tracked files in the repo and "
        "find the relevant sources. For each relevant file, give me a quick description "
        "of why it matters for this feature.\n\n"
        f"## Feature\n{text}\n\n"
        f"## All Tracked Files\n{search_results}\n\n"
        "Return a JSON array. Each item MUST have:\n"
        '  "file": path string,\n'
        '  "symbol": function/class/symbol name or "" for file-level,\n'
        '  "reason": one sentence — why this file is relevant to the feature,\n'
        '  "confidence": 0.0–1.0,\n'
        '  "line_start": null,\n'
        '  "line_end": null\n'
        "Use file names and directory structure to infer relevance. "
        "Be thorough — include implementation files, tests, and config. "
        "Always include AGENTS.md, CLAUDE.md, and skills/*/SKILL.md if present — "
        "they define how the runner must behave. "
        "Output ONLY the JSON array."
    )
    text = run_text(
        prompt,
        model,
        variant=variant,
        output_format="text",
        timeout=timeout,
        agent=agent,
        session=session,
    )
    record_exchange(prompt, text, model=model)
    return text


_FILTER_MAX_FILE_CHARS = 4_000  # per file in the filter context
_FILTER_MAX_TOTAL_CHARS = 40_000  # total across all candidate files


_CANDIDATE_FILE_RE = re.compile(
    r"[\w./\-]+\.(?:py|ts|tsx|js|jsx|go|rs|java|rb|sh|yaml|yml|toml|json|md)"
)


def _extract_candidate_files(recall_output: str) -> list[str]:
    """Pull unique file paths mentioned in the recall output (order-preserved)."""
    seen: set[str] = set()
    result: list[str] = []
    # Match bare paths: word chars / dots / hyphens ending in a known extension
    for m in _CANDIDATE_FILE_RE.finditer(recall_output):
        p = m.group(0).lstrip("./")
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


def _read_candidate_files(paths: list[str]) -> str:
    """Read file contents for each candidate path; return formatted block."""
    from pathlib import Path as _Path

    parts: list[str] = []
    total = 0
    for path_str in paths:
        if total >= _FILTER_MAX_TOTAL_CHARS:
            remaining = len(paths) - len(parts)
            if remaining:
                parts.append(f"*({remaining} more file(s) omitted — context cap)*")
            break
        try:
            raw = _Path(path_str).read_text()
        except OSError:
            continue
        snippet = raw[:_FILTER_MAX_FILE_CHARS]
        note = " ← truncated" if len(raw) > _FILTER_MAX_FILE_CHARS else ""
        parts.append(f"### {path_str}{note}\n```\n{snippet}\n```")
        total += len(snippet)
    return "\n\n".join(parts)


def _anchors_from_recall(recall_output: str) -> list[CodeAnchor]:
    """Parse file paths from recall plain-text output into minimal CodeAnchors."""
    files = _extract_candidate_files(recall_output)
    return [CodeAnchor(file=f, symbol="", reason="", confidence=1.0) for f in files]


_META_ANCHOR_REASONS: dict[str, str] = {
    "AGENTS.md": "project agent conventions and runner instructions",
    "CLAUDE.md": "project-level Claude Code instructions and conventions",
}


def _meta_anchors(repo_path: str, existing_files: set[str]) -> list[CodeAnchor]:
    """Build hot CodeAnchors for meta files not already captured by recall."""
    from pathlib import Path as _Path

    result: list[CodeAnchor] = []
    for mf in _find_meta_files(repo_path):
        if mf in existing_files:
            continue
        p = (_Path(repo_path) / mf) if not _Path(mf).is_absolute() else _Path(mf)
        if not p.exists():
            continue
        name = _Path(mf).name
        reason = _META_ANCHOR_REASONS.get(name, f"skill definition: {_Path(mf).parent.name}")
        result.append(
            CodeAnchor(file=mf, symbol="", reason=reason, confidence=0.95, relevance="hot")
        )
    return result


def filter_task_context(
    task: object,
    ladder: Ladder,
    session: object = None,
) -> str:
    """Per-task filter: cheap LLM reads the located files and summarizes what's useful.

    Receives:
      - task.target_files  — paths the locator found for this task
      - task.description   — this task only (not the full PRD)

    Reads those files and asks the cheap recall model to describe what's
    interesting in each file for THIS specific task.  The summary is stored on
    task.filtered_context and passed directly to the planner.
    """
    target_files = getattr(task, "target_files", None)
    description = getattr(task, "description", "")

    if not target_files:
        return ""

    file_contents = _read_candidate_files(target_files)
    if not file_contents:
        return ""

    prompt = (
        "You are a code context assistant. Below is a specific task and the source files "
        "identified as relevant to it. Read the files and describe — in plain language — "
        "what is interesting in each file for understanding and implementing this task.\n\n"
        f"## Task\n{description}\n\n"
        f"## Source Files\n{file_contents}\n\n"
        "For each file that has relevant content, write a short paragraph explaining:\n"
        "- What this file does\n"
        "- Which parts (functions, types, routes, configs) matter for this task and why\n"
        "- Any gotchas, patterns, or constraints the implementor should know\n\n"
        "Be concise but complete. Skip files with no relevant content. "
        "The planner will use this summary to understand the codebase before writing code."
    )
    try:
        return run_text(
            prompt,
            ladder.localizer_recall_model,
            variant=ladder.localizer_recall_variant,
            timeout=ladder.localizer_recall_timeout,
            agent=ladder.localizer_agent,
            session=session,
        )
    except (ProviderGapError, RuntimeError) as e:
        log.warning("filter model failed (%s), retrying with fallback", type(e).__name__)
        return run_text(
            prompt,
            ladder.localizer_recall_fallback_model,
            variant=ladder.localizer_recall_variant,
            timeout=ladder.localizer_recall_timeout,
            agent=ladder.localizer_agent,
            session=session,
        )


def localize(
    prd_text: str,
    session: Session,
    ladder: Ladder,
    *,
    repo_path: str = ".",
    force: bool = False,
) -> list[CodeAnchor]:
    """Recall-only localization: grep → cheap LLM → candidate file list.

    Returns CodeAnchors (file paths). The per-task filter (``filter_task_context``)
    runs in the pipeline immediately after this, before planning.
    """
    if not force and session.has("knowledge/localization.md"):
        existing = session.read("knowledge/localization.md")
        anchors = _parse_anchors(
            existing,
            hot=ladder.localizer_relevance_hot,
            medium=ladder.localizer_relevance_medium,
        )
        if anchors:
            return anchors

    # Run deterministic search tools FIRST, then feed results to LLM
    log.info("localize: running search tools")
    search_results = _run_search_tools(prd_text, repo_path)

    # Big repos blow past a small context — switch to the large-context model.
    recall_model = ladder.localizer_recall_model
    recall_variant = ladder.localizer_recall_variant
    recall_timeout = ladder.localizer_recall_timeout
    if len(search_results) > LARGE_CONTEXT_CHARS and ladder.localizer_recall_large_model:
        recall_model = ladder.localizer_recall_large_model
        recall_variant = ladder.localizer_recall_large_variant
        recall_timeout = ladder.localizer_recall_large_timeout
        log.info("localize: large search context → %s", recall_model)

    log.info("localize: recall via %s", recall_model)
    try:
        recall_output = _recall_phase(
            prd_text,
            search_results,
            recall_model,
            recall_variant,
            recall_timeout,
            agent=ladder.localizer_agent,
            session=session,
        )
    except (ProviderGapError, RuntimeError) as e:
        log.warning("recall model failed (%s), retrying with fallback model", type(e).__name__)
        recall_output = _recall_phase(
            prd_text,
            search_results,
            ladder.localizer_recall_fallback_model,
            recall_variant,
            recall_timeout,
            agent=ladder.localizer_agent,
            session=session,
        )

    # Prefer structured anchors (file + symbol + reason) so per-task targeting in
    # assign_target_files can actually match; fall back to bare file paths if the
    # cheap model didn't emit parseable JSON.
    anchors = _parse_anchors(
        recall_output,
        hot=ladder.localizer_relevance_hot,
        medium=ladder.localizer_relevance_medium,
    ) or _anchors_from_recall(recall_output)

    # Deterministically inject AGENTS.md / CLAUDE.md / skill files the LLM missed.
    existing_files = {a.file for a in anchors}
    injected = _meta_anchors(repo_path, existing_files)
    if injected:
        anchors = anchors + injected
        log.info("localize: injected %d meta anchor(s) (AGENTS.md/CLAUDE.md/skills)", len(injected))

    log.info("localize: %d candidate anchor(s) total", len(anchors))

    if not anchors:
        # Recall returned nothing — leave any existing localization.md intact rather
        # than overwriting it with an empty header.
        log.warning("localize: no anchors found — skipping write to preserve existing")
        return anchors

    # Write in the key:value block format _parse_anchors reads,
    # including line_start/line_end when the LLM provided them.
    lines: list[str] = ["# Localization\n"]
    for a in anchors:
        block = f"file: {a.file}\nsymbol: {a.symbol}\n"
        if a.line_start is not None:
            block += f"line_start: {a.line_start}\n"
        if a.line_end is not None:
            block += f"line_end: {a.line_end}\n"
        block += f"reason: {a.reason}\nconfidence: {a.confidence}\n"
        if a.relevance:
            block += f"relevance: {a.relevance}\n"
        lines.append(block)

    # Localization lives in the run's knowledge store (knowledge/localization.md) —
    # the single home for run-valuable memory — not loose at the session root.
    ks = KnowledgeStore(session)
    ks.write_note("localization", "\n".join(lines) + "\n")

    return anchors


def grounding_block(anchors: list[CodeAnchor]) -> str:
    """Build compact grounding string from hot/medium/low anchors.

    Hot anchors render inline as <file>:L<a>-L<b> — <symbol> with one-line insight.
    Medium/low anchors collapse under a single pointer line.
    Empty list yields "" (no header noise).
    """
    if not anchors:
        return ""

    hot_lines: list[str] = []
    has_medium_low = False

    for a in anchors:
        if a.relevance.lower() == "hot":
            line_range = ""
            if a.line_start is not None and a.line_end is not None:
                line_range = f":L{a.line_start}-{a.line_end}"
            elif a.line_start is not None:
                line_range = f":L{a.line_start}"

            location = f"{a.file}{line_range}"

            insight = ""
            if a.reason:
                insight = a.reason.split("\n")[0].strip()

            if insight:
                hot_lines.append(f"{location} — {a.symbol}\n  {insight}")
            else:
                hot_lines.append(f"{location} — {a.symbol}")
        else:
            has_medium_low = True

    result_parts: list[str] = []
    if hot_lines:
        result_parts.extend(hot_lines)

    if has_medium_low:
        result_parts.append("deeper context lives in knowledge/localization.md")

    return "\n".join(result_parts)
