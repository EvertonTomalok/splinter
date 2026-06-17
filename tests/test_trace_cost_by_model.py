"""Tests for Trace.cost_by_model and per-model summary."""

import pytest

from splinter.obs.trace import RunEntry, Trace


def _entry(model: str, cost: float, inp: int = 10, out: int = 5, task: int = 0) -> RunEntry:
    return RunEntry(
        model=model,
        tier=1,
        iteration=0,
        tokens={"input": inp, "output": out},
        cost=cost,
        latency_s=0.1,
        task=task,
    )


def _trace(*entries: RunEntry) -> Trace:
    t = Trace()
    t.entries.extend(entries)
    return t


class TestCostByModel:
    def test_empty(self) -> None:
        t = _trace()
        assert t.cost_by_model == {}
        assert t.total_cost == 0.0

    def test_single_model(self) -> None:
        t = _trace(_entry("gpt-4o", 0.01), _entry("gpt-4o", 0.02))
        cbm = t.cost_by_model
        assert set(cbm.keys()) == {"gpt-4o"}
        assert cbm["gpt-4o"] == pytest.approx(0.03)
        assert sum(cbm.values()) == pytest.approx(t.total_cost)

    def test_multiple_models_grouping(self) -> None:
        t = _trace(
            _entry("modelA", 0.10),
            _entry("modelA", 0.05),
            _entry("modelB", 0.20),
        )
        cbm = t.cost_by_model
        assert set(cbm.keys()) == {"modelA", "modelB"}
        assert cbm["modelA"] == pytest.approx(0.15)
        assert cbm["modelB"] == pytest.approx(0.20)
        assert sum(cbm.values()) == pytest.approx(t.total_cost)

    def test_sum_equals_total_cost(self) -> None:
        t = _trace(
            _entry("a", 0.001),
            _entry("b", 0.002),
            _entry("a", 0.003),
            _entry("c", 0.0001),
        )
        assert sum(t.cost_by_model.values()) == pytest.approx(t.total_cost)

    def test_no_mutation_of_entries(self) -> None:
        t = _trace(_entry("modelA", 0.1), _entry("modelB", 0.2))
        before_len = len(t.entries)
        before_ids = [id(e) for e in t.entries]
        _ = t.cost_by_model
        assert len(t.entries) == before_len
        assert [id(e) for e in t.entries] == before_ids

    def test_returns_new_dict_each_call(self) -> None:
        t = _trace(_entry("modelA", 0.1))
        d1 = t.cost_by_model
        d2 = t.cost_by_model
        assert d1 == d2
        assert d1 is not d2


class TestModelEntries:
    def test_filters_by_model(self) -> None:
        ea = _entry("modelA", 0.1)
        eb = _entry("modelB", 0.2)
        ec = _entry("modelA", 0.3)
        t = _trace(ea, eb, ec)
        assert t.model_entries("modelA") == [ea, ec]
        assert t.model_entries("modelB") == [eb]
        assert t.model_entries("missing") == []

    def test_returns_new_list(self) -> None:
        t = _trace(_entry("modelA", 0.1))
        assert t.model_entries("modelA") is not t.model_entries("modelA")


class TestSummaryPerModel:
    def test_per_model_section_present(self) -> None:
        t = _trace(_entry("modelA", 0.1), _entry("modelB", 0.2))
        s = t.summary()
        assert "## Per-model" in s
        assert "modelA" in s
        assert "modelB" in s

    def test_per_model_absent_when_empty(self) -> None:
        t = _trace()
        assert "## Per-model" not in t.summary()

    def test_per_model_single_entry(self) -> None:
        t = _trace(_entry("solo-model", 0.05, inp=100, out=50))
        s = t.summary()
        assert "## Per-model" in s
        assert "solo-model" in s
        assert "1 runs" in s

    def test_per_model_token_aggregation(self) -> None:
        t = _trace(
            _entry("m", 0.01, inp=10, out=5),
            _entry("m", 0.02, inp=20, out=10),
        )
        s = t.summary()
        assert "'input': 30" in s
        assert "'output': 15" in s

    def test_from_markdown_roundtrip_unaffected(self) -> None:
        t = _trace(
            _entry("modelA", 0.10, inp=100, out=50, task=1),
            _entry("modelB", 0.20, inp=200, out=100, task=2),
        )
        md = t.summary()
        t2 = Trace.from_markdown(md)
        assert len(t2.entries) == 2
        assert sum(t2.cost_by_model.values()) == pytest.approx(t2.total_cost)


class TestTotalTokensAllTypes:
    def test_includes_cache_keys(self) -> None:
        t = Trace()
        t.entries = [
            RunEntry(
                model="m", tier=1, iteration=1,
                tokens={"input": 100, "output": 50, "cache_read": 25, "cache_write": 10},
                cost=0.12, latency_s=1.0,
            ),
            RunEntry(
                model="m", tier=1, iteration=2,
                tokens={"input": 80, "output": 40, "cached_input": 20},
                cost=0.10, latency_s=1.0,
            ),
        ]
        total = t.total_tokens
        assert total["input"] == 180
        assert total["output"] == 90
        assert total["cache_read"] == 25
        assert total["cache_write"] == 10
        assert total["cached_input"] == 20

    def test_empty_dict_when_no_entries(self) -> None:
        t = Trace()
        assert t.total_tokens == {}

    def test_various_cache_key_types(self) -> None:
        t = Trace()
        t.entries = [
            RunEntry(
                model="m", tier=1, iteration=1,
                tokens={
                    "input": 100,
                    "output": 50,
                    "cache_creation_input_tokens": 30,
                    "cache_read_input_tokens": 20,
                },
                cost=0.12, latency_s=1.0,
            ),
        ]
        total = t.total_tokens
        assert total["cache_creation_input_tokens"] == 30
        assert total["cache_read_input_tokens"] == 20


class TestSummaryPerModelBullets:
    def test_model_cost_bullets_in_header(self) -> None:
        t = _trace(_entry("modelA", 0.15), _entry("modelB", 0.05))
        s = t.summary()
        assert "- modelA: $0.1500" in s
        assert "- modelB: $0.0500" in s

    def test_bullets_before_per_task_section(self) -> None:
        t = _trace(_entry("m", 0.1, task=1), _entry("m", 0.2, task=2))
        s = t.summary()
        idx_bullet = s.index("- m: $")
        idx_per_task = s.index("## Per-task")
        assert idx_bullet < idx_per_task

    def test_bullets_after_header_lines(self) -> None:
        t = _trace(_entry("m", 0.12))
        s = t.summary()
        idx_elapsed = s.index("- elapsed:")
        idx_bullet = s.index("- m: $")
        assert idx_elapsed < idx_bullet

    def test_no_model_bullets_when_empty(self) -> None:
        t = Trace()
        s = t.summary()
        lines = s.split("\n")
        model_bullets = [
            line for line in lines
            if line.startswith("- ") and ": $" in line and "total" not in line
        ]
        assert len(model_bullets) == 0

    def test_bullets_sorted_by_model_name(self) -> None:
        t = _trace(_entry("zebra", 0.1), _entry("alpha", 0.2), _entry("beta", 0.15))
        s = t.summary()
        lines = s.split("\n")
        model_bullets = [
            line for line in lines
            if line.startswith("- ") and ": $" in line and "total" not in line
        ]
        assert len(model_bullets) == 3
        assert "alpha" in model_bullets[0]
        assert "beta" in model_bullets[1]
        assert "zebra" in model_bullets[2]

    def test_cost_line_value_matches_total_cost(self) -> None:
        t = _trace(_entry("m", 0.1234), _entry("m2", 0.5678))
        s = t.summary()
        expected = f"- total cost: ${t.total_cost:.4f}"
        assert expected in s
