from __future__ import annotations

from pathlib import Path

import pytest

from splinter.memory.session import Session


def test_single_model_total_equals_per_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_llm_single")

    session.log_llm_usage("claude-x", {"input": 100, "output": 50}, 0.12)
    data = session.read_pre_run_usage()

    assert data["cost"] == pytest.approx(data["models"]["claude-x"]["cost"])
    assert data["input"] == data["models"]["claude-x"]["input"] == 100
    assert data["output"] == data["models"]["claude-x"]["output"] == 50


def test_multi_model_sum_equals_total(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_llm_multi")

    session.log_llm_usage("model-a", {"input": 200, "output": 100}, 0.12)
    session.log_llm_usage("model-b", {"input": 50, "output": 25}, 0.07)
    session.log_llm_usage("model-a", {"input": 80, "output": 40}, 0.05)

    data = session.read_pre_run_usage()
    per_model_cost = sum(m["cost"] for m in data["models"].values())
    per_model_input = sum(m["input"] for m in data["models"].values())
    per_model_output = sum(m["output"] for m in data["models"].values())

    assert data["cost"] == pytest.approx(per_model_cost)
    assert data["input"] == per_model_input
    assert data["output"] == per_model_output


def test_accumulation_across_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_llm_accum")

    session.log_llm_usage("model-a", {"input": 100, "output": 50}, 0.10)
    session.log_llm_usage("model-a", {"input": 200, "output": 75}, 0.20)

    data = session.read_pre_run_usage()
    ma = data["models"]["model-a"]

    assert ma["cost"] == pytest.approx(0.30)
    assert data["cost"] == pytest.approx(0.30)
    assert ma["input"] == 300
    assert ma["output"] == 125


def test_cost_not_rederived(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_llm_no_rederive")

    session.log_llm_usage("model-x", {"input": 10, "output": 5}, 0.99)
    data = session.read_pre_run_usage()

    assert data["cost"] == pytest.approx(0.99)
    assert data["models"]["model-x"]["cost"] == pytest.approx(0.99)


def test_concurrent_log_llm_usage_no_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_llm_race")

    n_threads = 20
    barrier = threading.Barrier(n_threads)
    errors: list[Exception] = []

    def _work(tid: int) -> None:
        try:
            barrier.wait()
            for _ in range(50):
                session.log_llm_usage(
                    f"model-{tid % 4}",
                    {"input": 10, "output": 5},
                    0.01,
                )
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_work, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"unexpected errors: {errors}"

    data = session.read_pre_run_usage()
    expected_total_calls = n_threads * 50
    expected_total_input = expected_total_calls * 10
    expected_total_output = expected_total_calls * 5
    expected_total_cost = expected_total_calls * 0.01

    assert data["input"] == expected_total_input, (
        f"input: {data['input']} != {expected_total_input}"
    )
    assert data["output"] == expected_total_output, (
        f"output: {data['output']} != {expected_total_output}"
    )
    assert data["cost"] == pytest.approx(expected_total_cost), (
        f"cost: {data['cost']} != {expected_total_cost}"
    )

    per_model_input = sum(m["input"] for m in data["models"].values())
    per_model_output = sum(m["output"] for m in data["models"].values())
    per_model_cost = sum(m["cost"] for m in data["models"].values())

    assert per_model_input == expected_total_input
    assert per_model_output == expected_total_output
    assert per_model_cost == pytest.approx(expected_total_cost)
