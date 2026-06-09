"""Two-phase LLM-driven localizer: deterministic search first, then LLM filter."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from splinter.memory.knowledge import KnowledgeStore
from splinter.memory.session import Session
from splinter.models.roster import Ladder
from splinter.providers import claude_cli
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
    try:
        data = json.loads(text)
        if isinstance(data, list):
            for item in data:
                anchors.append(
                    CodeAnchor(
                        file=item.get("file", ""),
                        symbol=item.get("symbol", ""),
                        reason=item.get("reason", ""),
                        confidence=float(item.get("confidence", 0.5)),
                    )
                )
            return anchors
    except (json.JSONDecodeError, TypeError):
        pass

    pattern = re.compile(
        r"file:\s*(.+?)\s*\n\s*symbol:\s*(.+?)\s*\n\s*reason:\s*(.+?)\s*\n\s*confidence:\s*([\d.]+)",
        re.MULTILINE,
    )
    for m in pattern.finditer(text):
        anchors.append(
            CodeAnchor(
                file=m.group(1).strip(),
                symbol=m.group(2).strip(),
                reason=m.group(3).strip(),
                confidence=float(m.group(4)),
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


def _recall_phase(prd_text: str, search_results: str, model: str) -> str:
    """Cheap model filters the raw search results to a candidate list."""
    text = _strip_frontmatter(prd_text)
    prompt = (
        "You are a code search assistant. Given a feature description and raw search "
        "tool results, identify all relevant files, functions, classes, and symbols.\n\n"
        f"## Feature Description\n{text}\n\n"
        f"## Raw Search Results\n{search_results}\n\n"
        "Return a concise list of candidate locations. For each, note: file, symbol, "
        "and why it is relevant. Be thorough — coverage over precision."
    )
    result = claude_cli.run(prompt, model, effort="minimal", output_format="text")
    return result.text


def _precision_phase(recall_output: str, prd_text: str, model: str) -> list[CodeAnchor]:
    """Mid-tier model filters recall results to structured CodeAnchors."""
    text = _strip_frontmatter(prd_text)
    prompt = (
        "You are a code analysis agent. Given a feature description and a list of "
        "candidate code locations, filter and rank the results.\n\n"
        f"## Feature Description\n{text}\n\n"
        f"## Candidates\n{recall_output}\n\n"
        "Return ONLY a JSON array of objects with keys: file, symbol, reason, confidence "
        "(0.0-1.0). Include only truly relevant results. Example:\n"
        '[{"file": "src/foo.py", "symbol": "Foo.bar", "reason": "handles X", "confidence": 0.9}]'
    )
    result = claude_cli.run(prompt, model, effort="low", output_format="json")
    return _parse_anchors(result.text)


def localize(
    prd_text: str,
    session: Session,
    ladder: Ladder,
    *,
    repo_path: str = ".",
    force: bool = False,
) -> list[CodeAnchor]:
    if not force and session.has("localization.md"):
        existing = session.read("localization.md")
        anchors = _parse_anchors(existing)
        if anchors:
            return anchors

    precision_model = ladder.localizer_precision_model

    # Run deterministic search tools FIRST, then feed results to LLM
    log.info("localize: running search tools")
    search_results = _run_search_tools(prd_text, repo_path)

    # Big repos blow past a small context — switch to the large-context model.
    recall_model = ladder.localizer_recall_model
    if len(search_results) > LARGE_CONTEXT_CHARS and ladder.localizer_recall_large_model:
        recall_model = ladder.localizer_recall_large_model
        log.info("localize: large search context → %s", recall_model)

    log.info("localize: recall via %s", recall_model)
    recall_output = _recall_phase(prd_text, search_results, recall_model)
    log.info("localize: precision via %s", precision_model)
    anchors = _precision_phase(recall_output, prd_text, precision_model)
    log.info("localize: %d anchor(s)", len(anchors))

    lines: list[str] = ["# Localization\n"]
    for a in anchors:
        lines.append(f"- **{a.file}** :: `{a.symbol}` — {a.reason} (conf: {a.confidence})")
    session.write("localization.md", "\n".join(lines) + "\n")

    ks = KnowledgeStore(session)
    ks.write_note("localization", "\n".join(lines) + "\n")

    return anchors
