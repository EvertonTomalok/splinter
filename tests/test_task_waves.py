from __future__ import annotations

from splinter.agents.refiner import Brief, brief_waves
from splinter.task_waves import (
    dependency_waves,
    facts_line,
    format_plan_steps,
    wave_facts,
)


def test_58_tasks_in_waves() -> None:
    wave_sizes = [8, 14, 10, 9, 7, 6, 4]
    deps: dict[str, list[str]] = {}
    prev_first: str | None = None
    for wave_idx, size in enumerate(wave_sizes):
        first_id = f"t{wave_idx}_0"
        for item_idx in range(size):
            task_id = f"t{wave_idx}_{item_idx}"
            task_deps = [prev_first] if prev_first is not None else []
            deps[task_id] = task_deps
        prev_first = first_id

    assert len(deps) == 58
    waves = dependency_waves(deps)
    assert [len(w) for w in waves] == [8, 14, 10, 9, 7, 6, 4]

    facts = wave_facts(deps)
    assert facts.stories == 58
    assert facts.waves == 7
    assert facts.widest == 14

    line_off = facts_line(facts, parallel=False)
    assert (
        line_off
        == "58 stories in 7 dependency waves · widest wave has 14 stories · parallel off"
    )

    line_on = facts_line(facts, parallel=True)
    assert (
        line_on
        == "58 stories in 7 dependency waves · widest wave has 14 stories · parallel on"
    )


def test_no_deps_gives_one_wave() -> None:
    deps = {f"t{i}": [] for i in range(10)}
    waves = dependency_waves(deps)
    assert len(waves) == 1
    assert len(waves[0]) == 10
    facts = wave_facts(deps)
    assert facts.stories == 10
    assert facts.waves == 1
    assert facts.widest == 10
    assert (
        facts_line(facts, parallel=False)
        == "10 stories in 1 dependency wave · widest wave has 10 stories · parallel off"
    )


def test_single_story_singular_grammar() -> None:
    deps = {"US-001": []}
    facts = wave_facts(deps)
    assert facts.stories == 1
    assert facts.waves == 1
    assert facts.widest == 1
    assert (
        facts_line(facts, parallel=False)
        == "1 story in 1 dependency wave · widest wave has 1 story · parallel off"
    )


def test_empty_deps_facts() -> None:
    facts = wave_facts({})
    assert facts.stories == 0
    assert facts.waves == 0
    assert facts.widest == 0
    assert facts_line(facts, parallel=False) == ""


def test_matches_brief_waves_fixture() -> None:
    briefs = [
        Brief(id="a"),
        Brief(id="b", deps=["a"]),
        Brief(id="c", deps=["a"]),
        Brief(id="d", deps=["b", "c"]),
    ]
    deps = {b.id: b.deps for b in briefs}
    waves = dependency_waves(deps)
    expected = [[b.id for b in w] for w in brief_waves(briefs)]
    assert waves == expected
    assert waves == [["a"], ["b", "c"], ["d"]]


def test_format_plan_steps() -> None:
    assert format_plan_steps("1. Step one\n2. Step two\n") == "2 steps"
    assert format_plan_steps("1. Only step\n") == "1 step"
    assert format_plan_steps("## T01 First\n## T02 Second\n") == "2 task plans"
    assert format_plan_steps("## T01 Only\n") == "1 task plan"
