"""Context for one open question: where the PRD talks about it, and a plain explanation.

A question from the PRD writer can read vague out of context ("Gate G0: which
approval starts US-002?"). Two helps sit behind ``?`` on the questions screen:

* :func:`prd_excerpt` — free and instant: the PRD sections that name the
  question's ids (``US-002``, ``G0``, ``B4``) and words, with their best lines.
* :func:`explain` — on demand: a quick model (``models.explain``, else Haiku,
  then the ladder's tier 0, all at low effort) explains the question in the
  language it is written in, as a colleague would, from the question, its
  options and that excerpt only.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass, field

from splinter.models.roster import parse_pool_entry
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
_EXPLAIN_TIMEOUT = 90
#: The explanation is a quick read, not a reasoning task: the lowest effort every
#: provider accepts (``none``/``minimal`` are not offered by every opencode model).
_EXPLAIN_EFFORT = "low"


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
    """The excerpt as plain text for the prompt: section title, then its lines."""
    if not sections:
        return "(the PRD says nothing about this question directly)"
    blocks = []
    for section in sections:
        body = "\n".join(f"  {line}" for line in section.lines)
        head = f"Section {section.heading}"
        blocks.append(f"{head}:\n{body}" if body else head)
    return "\n\n".join(blocks)


#: Tone sample for the explainer: same facts, written for a machine and for a person.
_TONE_SAMPLE = (
    "Not like this: \"Decide-se o provider primário; fallback condicionado ao 3DS.\"\n"
    "Like this: \"A pergunta é por onde o pagamento passa primeiro. Hoje tudo vai pelo "
    "Stripe; se ele cair, a compra falha e o cliente vai embora.\""
)


def explain_prompt(question: Question, sections: list[Section], prd_title: str = "") -> str:
    """The prompt the cheap model gets: a person explaining to a person, no jargon."""
    options = "\n".join(f"{option.letter}) {option.text}" for option in question.options)
    why = f"Why it was asked: {question.context}\n" if question.context else ""
    title = f"Product spec: {prd_title}\n" if prd_title else ""
    return (
        "Answer straight from the text below. Do not run tools, do not open or search "
        "files, do not plan: write the answer and stop.\n\n"
        "A person must answer one question about their product spec and finds it "
        "confusing. Explain it to them the way a colleague would at their desk.\n\n"
        "How to write:\n"
        "- Use the language the question is written in, the way people really talk in "
        "it, and speak to the reader directly (in Portuguese: \"você\").\n"
        "- Short everyday sentences, active voice: say who does what and what happens.\n"
        "- The first time a code appears (US-002, G0, A1-A4, B1), say in words what it is, "
        "e.g. \"US-002 (o relatório de agenda)\". If the spec does not say what a code "
        "means, say that plainly.\n"
        "- Never write: excerpt, trecho, prompt, baseline, congelar, escopo, alinhar, "
        "decide-se, \"em suma\", \"vale ressaltar\". Never mention the material you were "
        "given; talk about the spec itself.\n"
        "- Plain text. No headings, no bold, no bullet symbols, no brackets around "
        "section names.\n"
        "- No greeting, no praise, no closing summary. Never say \"depende\" without "
        "saying on what.\n"
        "- Use only what is below. If the spec does not say something, say it does not; "
        "never invent it.\n"
        "- At most 140 words.\n\n"
        f"Tone:\n{_TONE_SAMPLE}\n\n"
        "What to write, in this order:\n"
        "1. What the question is really asking, in one or two sentences anyone on the "
        "team would get, with a concrete example from this spec.\n"
        "2. One line per option, starting with its letter and \")\": what happens in "
        "practice if they pick it (what gets built now, what waits, what risk they take).\n"
        "3. One last line: if the spec already points to one option, which one and why, in "
        "plain words; if not, what they need to know to choose.\n\n"
        f"Question: {question.text}\n"
        f"{why}"
        f"Options:\n{options}\n\n"
        f"{title}"
        f"What the spec says about it:\n{excerpt_text(sections)}\n"
    )


#: Default explainer: Claude Haiku first (most setups have the claude CLI), then
#: the ladder's cheapest tier. ``models.explain`` in the project config wins.
_DEFAULT_EXPLAINER = "haiku"


def explain_model() -> tuple[str, list[str]]:
    """The explainer model and its fallback pool, every entry at low effort.

    ``models.explain`` (a model or a list) in ``.splinter/config.yaml`` wins;
    otherwise Haiku, then the tier-0 models of the ladder.
    """
    from splinter.models.roster import _project_config, load_ladder

    models = _project_config().get("models")
    raw = models.get("explain") if isinstance(models, dict) else None
    entries = (
        [str(x).strip() for x in raw if str(x).strip()]
        if isinstance(raw, (list, tuple))
        else [str(raw).strip()] if raw else []
    )
    if not entries:
        tiers = load_ladder().tiers
        tier_zero = list(tiers[0].models) if tiers and tiers[0].models else []
        entries = [_DEFAULT_EXPLAINER, *[m for m in tier_zero if m != _DEFAULT_EXPLAINER]]
    pool = [entry if "@" in entry else f"{entry}@{_EXPLAIN_EFFORT}" for entry in entries]
    return parse_pool_entry(pool[0])[0], pool


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
        variant=parse_pool_entry(pool[0])[1] or _EXPLAIN_EFFORT,
        output_format="json",
        timeout=_EXPLAIN_TIMEOUT,
        role="explain",
        pool=pool,
        cwd=tempfile.gettempdir(),
    )
    tokens = {
        "input": result.tokens.get("input", 0) or 0,
        "output": result.tokens.get("output", 0) or 0,
    }
    text = clean_explanation(result.text)
    if not text:
        raise RuntimeError(f"{model} returned an empty answer")
    return Explanation(text=text, model=model, cost=result.cost, tokens=tokens)
