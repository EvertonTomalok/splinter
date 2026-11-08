"""Context for one open question: where the PRD talks about it, and a plain explanation.

A question from the PRD writer can read vague out of context ("Gate G0: which
approval starts US-002?"). Two helps sit behind ``?`` on the questions screen:

* :func:`prd_excerpt` — free and instant: the PRD sections that name the
  question's ids (``US-002``, ``G0``, ``B4``) and words, with their best lines.
* :func:`explain` — on demand: the cheapest model of the ladder (tier 0)
  explains the question in the language it is written in, in plain spoken
  words, from the question, its options and that excerpt only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from splinter.providers.dispatch import run_provider_session
from splinter.tui_questions import Question

#: ``US-002``, ``G0``, ``B4``, ``C7``: ids the writer quotes from the PRD.
_ID_RE = re.compile(r"\b(?:US-\d{1,4}|[A-Z]{1,2}\d{1,2})\b")
_WORD_RE = re.compile(r"[^\W\d_]{5,}", re.UNICODE)
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,4})\s+(.+?)\s*#*\s*$")
_STOPWORDS = frozenset(
    """
    about after again being before between could every first other should their there
    these those under where which while would whose what when with without into from
    have this that than then them they your yours
    quais qual quando quanto como onde porque sobre entre antes depois ainda todos todas
    cada mesmo mesma deve devem pode podem sendo seria seriam essa esse esta este isso
    aquele aquela outro outra outros outras nosso nossa nenhum nenhuma apenas somente
    usuario usuário sistema dados campo campos valor valores
    """.split()
)
_LINE_MAX = 160
_EXPLAIN_TIMEOUT = 120


@dataclass(frozen=True)
class Section:
    """A PRD section that talks about the question, with its best lines."""

    heading: str
    lines: tuple[str, ...]


@dataclass(frozen=True)
class Explanation:
    """What the cheap model said, and what it cost."""

    text: str
    model: str
    cost: float = 0.0
    tokens: dict[str, int] = field(default_factory=dict)


def question_terms(question: Question) -> tuple[set[str], set[str]]:
    """The ids and the meaningful words of a question (text, label, why, options)."""
    parts = [question.text, question.label, question.context]
    parts.extend(option.text for option in question.options)
    blob = " ".join(part for part in parts if part)
    ids = set(_ID_RE.findall(blob))
    head = " ".join(part for part in (question.text, question.label, question.context) if part)
    words = {w.lower() for w in _WORD_RE.findall(head)} - _STOPWORDS
    return ids, words


def _sections(prd: str) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = [("", [])]
    in_front = prd.lstrip().startswith("---")
    for index, raw in enumerate(prd.splitlines()):
        if in_front:
            if index and raw.strip() == "---":
                in_front = False
            continue
        heading = _HEADING_RE.match(raw)
        if heading:
            sections.append((heading.group(2).strip(), []))
        elif raw.strip():
            sections[-1][1].append(raw.strip())
    return [(head, lines) for head, lines in sections if head or lines]


def _score(text: str, ids: set[str], words: set[str]) -> int:
    found_ids = set(_ID_RE.findall(text)) & ids
    lowered = text.lower()
    return 3 * len(found_ids) + sum(1 for word in words if word in lowered)


def _clip(text: str) -> str:
    plain = " ".join(text.split())
    return plain if len(plain) <= _LINE_MAX else plain[: _LINE_MAX - 1] + "…"


def prd_excerpt(
    prd: str, question: Question, *, max_sections: int = 2, max_lines: int = 4
) -> list[Section]:
    """The PRD sections that name the question's ids and words, best first.

    Each section keeps its best ``max_lines`` lines in PRD order. A section
    whose only hit is one common word is dropped: it is noise, not context.
    """
    ids, words = question_terms(question)
    if not ids and not words:
        return []
    scored: list[tuple[int, int, Section]] = []
    for order, (heading, lines) in enumerate(_sections(prd)):
        line_scores = [(_score(line, ids, words), index) for index, line in enumerate(lines)]
        hits = sorted((s for s in line_scores if s[0] > 0), reverse=True)[:max_lines]
        total = 2 * _score(heading, ids, words) + sum(score for score, _ in hits)
        if total < 2:
            continue
        kept = tuple(_clip(lines[index]) for _, index in sorted(hits, key=lambda hit: hit[1]))
        scored.append((total, -order, Section(heading=heading or "(intro)", lines=kept)))
    scored.sort(reverse=True)
    return [section for _, _, section in scored[:max_sections]]


def excerpt_text(sections: list[Section]) -> str:
    """The excerpt as plain text for the prompt."""
    if not sections:
        return "(nothing in the PRD names this question directly)"
    blocks = []
    for section in sections:
        body = "\n".join(f"- {line}" for line in section.lines)
        blocks.append(f"[{section.heading}]\n{body}" if body else f"[{section.heading}]")
    return "\n\n".join(blocks)


def explain_prompt(question: Question, sections: list[Section], prd_title: str = "") -> str:
    """The prompt the cheap model gets: plain spoken words, no filler, no invention."""
    options = "\n".join(f"{option.letter}) {option.text}" for option in question.options)
    why = f"Why the writer asked: {question.context}\n" if question.context else ""
    title = f"PRD: {prd_title}\n" if prd_title else ""
    return (
        "Someone reviewing a PRD must answer the question below and finds it vague. "
        "Explain it the way a senior engineer would, sitting next to them: plain spoken "
        "words, concrete, about this project only.\n\n"
        "Rules:\n"
        "- Write in the same language the question is written in.\n"
        "- At most 110 words. Plain text: no markdown headings, no bold, no tables.\n"
        "- No greeting, no preamble, no praise of the question, no closing summary, "
        "no hedging.\n"
        "- No generic advice that would fit any project.\n"
        "- Use only the question, its options and the PRD excerpt. If something is not "
        "there, say it is not in the PRD; never invent it.\n\n"
        "Write, in this order:\n"
        "1. One or two sentences: what is actually being decided, and what it changes "
        "here.\n"
        "2. One line per option, starting with its letter: what happens if it is picked.\n"
        "3. Only if the PRD already leans to one option: one line saying which, and "
        "where the PRD says so.\n\n"
        f"Question: {question.text}\n"
        f"{why}"
        f"Options:\n{options}\n\n"
        f"{title}"
        f"PRD excerpt:\n{excerpt_text(sections)}\n"
    )


def explain_model() -> tuple[str, list[str]]:
    """The cheapest model of the ladder (tier 0) and its pool."""
    from splinter.models.roster import load_ladder

    tiers = load_ladder().tiers
    if not tiers or not tiers[0].models:
        raise RuntimeError("no tier-0 model configured to explain questions")
    models = list(tiers[0].models)
    return models[0], models


def clean_explanation(text: str) -> str:
    """Drop code fences, markdown emphasis and runs of blank lines."""
    body = text.strip()
    body = re.sub(r"^```[a-z]*\s*|\s*```$", "", body)
    body = re.sub(r"\*\*(.+?)\*\*", r"\1", body)
    body = re.sub(r"^#+\s*", "", body, flags=re.MULTILINE)
    return re.sub(r"\n{3,}", "\n\n", body).strip()


def explain(question: Question, sections: list[Section], prd_title: str = "") -> Explanation:
    """Ask the tier-0 model to explain ``question`` in plain words."""
    model, pool = explain_model()
    result, _sid = run_provider_session(
        explain_prompt(question, sections, prd_title),
        model,
        output_format="json",
        timeout=_EXPLAIN_TIMEOUT,
        role="explain",
        pool=pool,
    )
    tokens = {
        "input": result.tokens.get("input", 0) or 0,
        "output": result.tokens.get("output", 0) or 0,
    }
    text = clean_explanation(result.text)
    if not text:
        raise RuntimeError(f"{model} returned an empty answer")
    return Explanation(text=text, model=model, cost=result.cost, tokens=tokens)
