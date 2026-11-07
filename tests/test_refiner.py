"""Tests for the agents-strategy brief refiner — the LLM seam is always mocked."""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from splinter.agents import refiner
from splinter.agents.refiner import (
    Brief,
    _sanitize_brief_id,
    brief_to_task,
    brief_waves,
    refine_briefs,
    validate_briefs,
)
from splinter.agents.runner import Task
from splinter.models.roster import Ladder


def test_fallback_preserves_sanitized_dependencies() -> None:
    briefs = refiner._fallback_briefs([
        Task(id="A/one", description="first", acceptance="ok"),
        Task(id="B", description="second", acceptance="ok", deps=["A/one"]),
    ])
    assert briefs[1].deps == [briefs[0].id]


def _ladder() -> Ladder:
    return Ladder(
        tiers=[],
        effort_map={},
        eval_model="sonnet",
        eval_effort="high",
        planner_model="sonnet",
        planner_effort="high",
        localizer_recall_model="recall-model",
        localizer_recall_large_model="recall-large-model",
        localizer_precision_model="precision-model",
    )


def _task(
    tid: str = "US-001",
    description: str = "build the thing",
    acceptance: str = "it works",
    target_files: list[str] | None = None,
) -> Task:
    return Task(
        description=description,
        acceptance=acceptance,
        id=tid,
        target_files=target_files,
    )


def _fake_run_text(
    result: str | BaseException,
) -> tuple[Callable[..., str], list[dict[str, object]]]:
    calls: list[dict[str, object]] = []

    def fake(prompt: str, model: str, **kwargs: object) -> str:
        calls.append({"prompt": prompt, "model": model, **kwargs})
        if isinstance(result, BaseException):
            raise result
        return result

    return fake, calls


class TestSanitizeBriefId:
    def test_keeps_slug_chars_and_lowercases(self) -> None:
        assert _sanitize_brief_id("q1") == "q1"
        assert _sanitize_brief_id("PB3a") == "pb3a"

    def test_sanitizes_task_style_ids(self) -> None:
        assert _sanitize_brief_id("US-001") == "us-001"

    def test_collapses_runs_and_strips_dashes(self) -> None:
        assert _sanitize_brief_id("--Q  1!!") == "q-1"
        assert _sanitize_brief_id("a/b") == "a-b"

    def test_unusable_input_falls_back_to_b(self) -> None:
        assert _sanitize_brief_id("") == "b"
        assert _sanitize_brief_id("---") == "b"
        assert _sanitize_brief_id("###") == "b"


class TestValidateBriefs:
    def test_valid_set_passes(self) -> None:
        validate_briefs(
            [Brief(id="a"), Brief(id="b", deps=["a"]), Brief(id="c", deps=["a", "b"])]
        )

    def test_empty_id_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            validate_briefs([Brief(id="")])

    def test_duplicate_ids_raise(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            validate_briefs([Brief(id="a"), Brief(id="a")])

    def test_unknown_dep_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown dep"):
            validate_briefs([Brief(id="a", deps=["missing"])])

    def test_two_brief_cycle_raises(self) -> None:
        with pytest.raises(ValueError, match="cycle"):
            validate_briefs([Brief(id="a", deps=["b"]), Brief(id="b", deps=["a"])])

    def test_self_dependency_raises(self) -> None:
        with pytest.raises(ValueError, match="cycle"):
            validate_briefs([Brief(id="a", deps=["a"])])


class TestRefineBriefs:
    def test_briefs_object_shape_parsed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = {
            "briefs": [
                {
                    "id": "q1",
                    "task": "US-001",
                    "goal": "add the endpoint",
                    "files": ["api/user.py", "tests/test_user.py"],
                    "sketch": "copy api/pet.py",
                    "acceptance": ["route exists", "test passes"],
                    "deps": [],
                    "effort": "hard",
                },
                {
                    "id": "q2",
                    "task": "US-001",
                    "goal": "wire the screen",
                    "files": ["web/user.tsx"],
                    "sketch": "copy web/pet.tsx",
                    "acceptance": ["screen renders"],
                    "deps": ["q1"],
                    "effort": "turbo",
                },
            ]
        }
        fake, calls = _fake_run_text(json.dumps(payload))
        monkeypatch.setattr(refiner, "run_text", fake)

        briefs = refine_briefs([_task()], _ladder(), prd_text="prd body", task_index=3)

        assert [b.id for b in briefs] == ["q1", "q2"]
        assert briefs[0].goal == "add the endpoint"
        assert briefs[0].files == ["api/user.py", "tests/test_user.py"]
        assert briefs[0].sketch == "copy api/pet.py"
        assert briefs[0].acceptance == ["route exists", "test passes"]
        assert briefs[0].effort == "hard"
        assert briefs[1].effort == "normal"
        assert briefs[0].task_id == "US-001"
        assert briefs[1].deps == ["q1"]
        assert len(calls) == 1
        call = calls[0]
        assert call["model"] == "opus"
        assert call["variant"] == "high"
        assert call["output_format"] == "json"
        assert call["timeout"] == 3600
        assert call["role"] == "prd"
        assert call["task_index"] == 3
        assert "US-001" in str(call["prompt"])
        assert "prd body" in str(call["prompt"])

    def test_bare_list_shape_parsed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = [
            {
                "id": "a1",
                "task": "US-404",
                "goal": "unrelated",
                "files": [],
                "sketch": "",
                "acceptance": [],
                "deps": [],
                "effort": "",
            }
        ]
        fake, _ = _fake_run_text(json.dumps(payload))
        monkeypatch.setattr(refiner, "run_text", fake)

        briefs = refine_briefs([_task()], _ladder())

        assert len(briefs) == 1
        assert briefs[0].id == "a1"
        assert briefs[0].task_id == "US-404"

    def test_acceptance_and_files_as_newline_strings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = {
            "briefs": [
                {
                    "id": "s1",
                    "task": "US-001",
                    "goal": "g",
                    "files": "a.py\nb.py",
                    "sketch": "k",
                    "acceptance": "line one\nline two",
                    "deps": "",
                    "effort": "",
                }
            ]
        }
        fake, _ = _fake_run_text(json.dumps(payload))
        monkeypatch.setattr(refiner, "run_text", fake)

        briefs = refine_briefs([_task()], _ladder())

        assert briefs[0].files == ["a.py", "b.py"]
        assert briefs[0].acceptance == ["line one", "line two"]
        assert briefs[0].deps == []

    def test_llm_exception_falls_back_one_to_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake, _ = _fake_run_text(RuntimeError("provider down"))
        monkeypatch.setattr(refiner, "run_text", fake)
        tasks = [
            _task("US-001", "first task", "acc one", ["x.py"]),
            _task("US-002", "second task", "acc two\nacc three", ["y.py", "z.py"]),
        ]

        briefs = refine_briefs(tasks, _ladder())

        assert [b.id for b in briefs] == ["us-001", "us-002"]
        assert [b.goal for b in briefs] == ["first task", "second task"]
        assert briefs[0].files == ["x.py"]
        assert briefs[1].files == ["y.py", "z.py"]
        assert briefs[0].acceptance == ["acc one"]
        assert briefs[1].acceptance == ["acc two", "acc three"]
        assert [b.task_id for b in briefs] == ["US-001", "US-002"]
        assert all(b.deps == [] for b in briefs)
        assert all(b.effort == "normal" for b in briefs)

    def test_malformed_json_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake, _ = _fake_run_text("sure! here are some briefs...")
        monkeypatch.setattr(refiner, "run_text", fake)

        briefs = refine_briefs([_task()], _ladder())

        assert [b.id for b in briefs] == ["us-001"]
        assert briefs[0].goal == "build the thing"

    def test_empty_output_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake, _ = _fake_run_text("   ")
        monkeypatch.setattr(refiner, "run_text", fake)

        briefs = refine_briefs([_task()], _ladder())

        assert [b.id for b in briefs] == ["us-001"]

    def test_validation_failing_output_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = {
            "briefs": [
                {"id": "q1", "task": "US-001", "goal": "g", "deps": ["ghost"]},
            ]
        }
        fake, _ = _fake_run_text(json.dumps(payload))
        monkeypatch.setattr(refiner, "run_text", fake)

        briefs = refine_briefs([_task()], _ladder())

        assert [b.id for b in briefs] == ["us-001"]
        assert briefs[0].task_id == "US-001"

    def test_task_without_id_gets_index_slug(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake, _ = _fake_run_text(ValueError("boom"))
        monkeypatch.setattr(refiner, "run_text", fake)

        briefs = refine_briefs([Task(description="anon", acceptance="ok")], _ladder())

        assert [b.id for b in briefs] == ["t0"]
        assert briefs[0].task_id == ""

    def test_no_tasks_returns_empty_without_llm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake, calls = _fake_run_text("{}")
        monkeypatch.setattr(refiner, "run_text", fake)

        assert refine_briefs([], _ladder()) == []
        assert calls == []


class TestBriefWaves:
    def test_independent_briefs_single_wave(self) -> None:
        briefs = [Brief(id="a"), Brief(id="b"), Brief(id="c")]
        waves = brief_waves(briefs)
        assert [[b.id for b in w] for w in waves] == [["a", "b", "c"]]

    def test_chain_yields_three_waves_in_order(self) -> None:
        briefs = [
            Brief(id="a"),
            Brief(id="b", deps=["a"]),
            Brief(id="c", deps=["b"]),
        ]
        waves = brief_waves(briefs)
        assert [[b.id for b in w] for w in waves] == [["a"], ["b"], ["c"]]

    def test_mixed_deps(self) -> None:
        briefs = [
            Brief(id="a"),
            Brief(id="b", deps=["a"]),
            Brief(id="c", deps=["a"]),
            Brief(id="d", deps=["b", "c"]),
        ]
        waves = brief_waves(briefs)
        assert [[b.id for b in w] for w in waves] == [["a"], ["b", "c"], ["d"]]

    def test_stable_input_order_within_wave(self) -> None:
        briefs = [Brief(id="z"), Brief(id="a"), Brief(id="m")]
        waves = brief_waves(briefs)
        assert [b.id for b in waves[0]] == ["z", "a", "m"]

    def test_unknown_dep_deferred_to_last_wave(self) -> None:
        briefs = [
            Brief(id="a"),
            Brief(id="b", deps=["a"]),
            Brief(id="x", deps=["ghost"]),
        ]
        waves = brief_waves(briefs)
        assert [[b.id for b in w] for w in waves] == [["a"], ["b"], ["x"]]

    def test_cycle_collapses_to_single_wave(self) -> None:
        briefs = [Brief(id="a", deps=["b"]), Brief(id="b", deps=["a"]), Brief(id="c")]
        waves = brief_waves(briefs)
        assert [[b.id for b in w] for w in waves] == [["a", "b", "c"]]

    def test_empty_input(self) -> None:
        assert brief_waves([]) == []


class TestBriefToTask:
    def test_maps_fields(self) -> None:
        brief = Brief(
            id="q1",
            goal="do the thing",
            files=["a.py", "b.py"],
            sketch="s",
            acceptance=["first", "second"],
            deps=["q0"],
            effort="hard",
            task_id="US-001",
        )
        task = brief_to_task(brief)
        assert task.id == "q1"
        assert task.description == "do the thing"
        assert task.acceptance == "first\nsecond"
        assert task.target_files == ["a.py", "b.py"]
        assert task.deps == ["q0"]
        assert task.reasoning_effort == "auto"
        assert task.effort == "hard"

    def test_empty_deps_become_none(self) -> None:
        task = brief_to_task(Brief(id="q2", goal="g"))
        assert task.deps is None

    def test_empty_acceptance_joins_to_empty_string(self) -> None:
        task = brief_to_task(Brief(id="q3", goal="g"))
        assert task.acceptance == ""
