"""Two-phase LLM-driven localizer: deterministic search first, then LLM filter."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from splinter.memory.knowledge import KnowledgeStore
from splinter.memory.session import Session
from splinter.models.roster import Ladder
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


def _extract_search_terms(prd_text: str) -> list[str]:
    """Extract likely search terms from the PRD."""
    text = _strip_frontmatter(prd_text)
    # Extract keywords from the description and acceptance
    words = re.findall(r"[A-Za-z][A-Za-z0-9_]+(?:\.[A-Za-z][A-Za-z0-9_]*)*", text)
    # Filter out common stop words and short words
    stop_words = (
        "the and for are but not you all can had her was one our out day get "
        "has him his how its may new now old see two way who boy did she use "
        "than them well were with have from they know want been good much "
        "some time very when come here just like long make many over such "
        "take will"
    )
    stop = set(stop_words.split())
    terms = [w for w in words if w.lower() not in stop and len(w) > 2]
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for w in terms:
        key = w.lower()
        if key not in seen:
            seen.add(key)
            unique.append(w)
    return unique[:10]


def _parse_anchors(text: str) -> list[CodeAnchor]:
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
            result.append(
                CodeAnchor(
                    file=item.get("file", ""),
                    symbol=item.get("symbol", ""),
                    reason=item.get("reason", ""),
                    confidence=float(item.get("confidence", 0.5)),
                    line_start=int(ls) if ls is not None else None,
                    line_end=int(le) if le is not None else None,
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
        anchors.append(
            CodeAnchor(
                file=file_m.group(1).strip(),
                symbol=symbol_m.group(1).strip() if symbol_m else "",
                reason=reason_m.group(1).strip() if reason_m else "",
                confidence=float(conf_m.group(1)) if conf_m else 0.5,
                line_start=int(ls_m.group(1)) if ls_m else None,
                line_end=int(le_m.group(1)) if le_m else None,
            )
        )
    return anchors


def _run_search_tools(prd_text: str, repo_path: str = ".") -> str:
    """Run deterministic search tools and return a structured report."""
    text = _strip_frontmatter(prd_text)
    terms = _extract_search_terms(text)

    lines: list[str] = []
    lines.append("# Search Tool Results\n")

    # List all Python files
    fl_result = search_tools.file_list(repo_path, "*.py")
    if fl_result.output:
        lines.append("## Python Files")
        lines.append(fl_result.output[:2000])
        lines.append("")

    # Grep for each key term
    for term in terms:
        g_result = search_tools.grep(term, repo_path)
        if g_result.output and g_result.exit_code == 0:
            lines.append(f"## grep for '{term}'")
            lines.append(g_result.output[:1500])
            lines.append("")

    return "\n".join(lines)


def _recall_phase(
    prd_text: str,
    search_results: str,
    model: str,
    variant: str = "minimal",
    timeout: int | None = None,
    agent: str = "build",
) -> str:
    """Cheap model filters the raw search results to a candidate list."""
    text = _strip_frontmatter(prd_text)
    prompt = (
        "You are a code search assistant. Given a feature description and raw search "
        "tool results, identify all relevant files, functions, classes, and symbols.\n\n"
        f"## Feature Description\n{text}\n\n"
        f"## Raw Search Results\n{search_results}\n\n"
        "Return a JSON array of candidate locations. Each item MUST have keys: "
        '"file" (path), "symbol" (function/class/symbol name, or "" if file-level), '
        '"reason" (why it is relevant — include the feature keywords it relates to), '
        '"confidence" (0.0–1.0), '
        '"line_start" (integer line number where the symbol starts, or null if unknown), '
        '"line_end" (integer line number where the symbol ends, or null if unknown). '
        "Use grep output lines (format: file:line:content) to populate line_start/line_end. "
        "Be thorough — coverage over precision. Output ONLY the JSON array, no prose."
    )
    return run_text(
        prompt, model, variant=variant, output_format="text", timeout=timeout, agent=agent
    )


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


def filter_task_context(
    task: object,
    ladder: Ladder,
) -> str:
    """Per-task filter run by the harness after localize, before planning.

    Receives:
      - task.target_files  — paths the locator found for this task
      - task.description   — this task only (not the full PRD)

    Reads those files, passes actual source to the precision model, and returns
    a focused context string stored on task.filtered_context.
    """
    target_files = getattr(task, "target_files", None)
    description = getattr(task, "description", "")

    if not target_files:
        return ""

    file_contents = _read_candidate_files(target_files)
    if not file_contents:
        return ""

    prompt = (
        "You are a code context agent. Below is a task description and the source files "
        "the locator identified as relevant. Read the code and extract the sections that "
        "matter for implementing this task.\n\n"
        f"## Task\n{description}\n\n"
        f"## Source Files\n{file_contents}\n\n"
        "For each relevant section, output a block in this EXACT format:\n\n"
        "### <file_path>:L<start>-L<end> — <symbol_or_description>\n"
        "rtk: rtk read <file_path>\n"
        "<one-line note on why this section matters for the task>\n"
        "```\n"
        "<the relevant code snippet>\n"
        "```\n\n"
        "Include exact line numbers. If uncertain about line numbers, estimate from the "
        "file content shown. Be specific — the planner will navigate directly to these locations."
    )
    try:
        return run_text(
            prompt,
            ladder.localizer_precision_model,
            variant=ladder.localizer_precision_variant,
            timeout=ladder.localizer_precision_timeout,
            agent=ladder.localizer_agent,
        )
    except (ProviderGapError, RuntimeError) as e:
        log.warning("precision model failed (%s), retrying with fallback model", type(e).__name__)
        return run_text(
            prompt,
            ladder.localizer_recall_fallback_model,
            variant=ladder.localizer_precision_variant,
            timeout=ladder.localizer_precision_timeout,
            agent=ladder.localizer_agent,
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
        anchors = _parse_anchors(existing)
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
        )

    # Prefer structured anchors (file + symbol + reason) so per-task targeting in
    # assign_target_files can actually match; fall back to bare file paths if the
    # cheap model didn't emit parseable JSON.
    anchors = _parse_anchors(recall_output) or _anchors_from_recall(recall_output)
    log.info("localize: %d candidate anchor(s) from recall", len(anchors))

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
        lines.append(block)

    # Localization lives in the run's knowledge store (knowledge/localization.md) —
    # the single home for run-valuable memory — not loose at the session root.
    ks = KnowledgeStore(session)
    ks.write_note("localization", "\n".join(lines) + "\n")

    return anchors
