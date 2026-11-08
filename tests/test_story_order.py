"""Story order for the agents strategy — the planner call is a plain fake."""

from __future__ import annotations

import json

from splinter.agents.brief_io import saved_briefs_usable as _saved_briefs_usable
from splinter.agents.runner import Task
from splinter.agents.story_order import delivery_section, parse_deps, with_story_deps

PRD = """# PRD

## 3. Stories

### US-001: one

## 7. Estratégia de entrega

### Onda 1
- contrato

### Onda 2
- telas

## 8. Testes

- nada
"""


def _tasks(n: int = 3) -> list[Task]:
    return [
        Task(id=f"US-00{i}", description=f"story {i}", acceptance="ok") for i in range(1, n + 1)
    ]


def test_delivery_section_takes_the_rollout_heading_and_its_subsections() -> None:
    text = delivery_section(PRD)
    assert text.startswith("## 7. Estratégia de entrega")
    assert "Onda 2" in text and "telas" in text
    assert "Testes" not in text


def test_parse_keeps_only_known_earlier_stories() -> None:
    raw = json.dumps(
        {"deps": {"US-002": ["US-001", "US-003", "US-404"], "us-003": ["us-001", "US-001"]}}
    )
    assert parse_deps(raw, _tasks()) == {"US-002": ["US-001"], "US-003": ["US-001"]}


def test_story_deps_come_from_the_planner_with_titles_and_delivery_only() -> None:
    prompts: list[str] = []

    def ask(prompt: str) -> str:
        prompts.append(prompt)
        return '```json\n{"deps": {"US-002": ["US-001"], "US-003": ["US-001"]}}\n```'

    tasks = with_story_deps(_tasks(), PRD, ask)
    assert [t.deps for t in tasks] == [[], ["US-001"], ["US-001"]]
    assert "- US-002: story 2" in prompts[0]
    assert "Onda 1" in prompts[0] and "## 3. Stories" not in prompts[0]


def test_unusable_answer_runs_the_stories_in_prd_order() -> None:
    tasks = with_story_deps(_tasks(), PRD, lambda _prompt: "sure, here you go")
    assert [t.deps for t in tasks] == [[], ["US-001"], ["US-002"]]


def test_explicit_deps_or_a_single_story_skip_the_call() -> None:
    def ask(_prompt: str) -> str:
        raise AssertionError("no call expected")

    given = _tasks()
    given[2] = Task(id="US-003", description="s", acceptance="ok", deps=["US-001"])
    assert with_story_deps(given, PRD, ask) is given
    single = _tasks(1)
    assert with_story_deps(single, PRD, ask) is single


def test_resume_refines_again_over_a_copied_fallback_table() -> None:
    tasks = _tasks(2)
    copied = json.dumps([{"id": "us-001", "goal": "story 1"}, {"id": "us-002", "goal": "story 2"}])
    refined = json.dumps([{"id": "us-001-a", "goal": "write the proto"}])
    assert not _saved_briefs_usable(copied, tasks, set())
    assert _saved_briefs_usable(copied, tasks, {"us-001"})
    assert _saved_briefs_usable(refined, tasks, set())
    assert not _saved_briefs_usable("not json", tasks, set())
