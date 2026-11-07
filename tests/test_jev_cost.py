from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from splinter.agents import localizer
from splinter.agents.localizer import CodeAnchor, _jev_rank_anchors, _jev_rank_files
from splinter.memory.session import Session
from splinter.providers.decisions import DecisionAnswer, DecisionResult


def _result(probs: list[float]) -> DecisionResult:
    return DecisionResult(
        model="typesafe/jev-1.14",
        answers={f"q{i}": DecisionAnswer("noul", {"noul": p}) for i, p in enumerate(probs)},
        input_tokens=120,
        output_tokens=8,
        cost=0.0021,
    )


@pytest.fixture
def session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Session:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    return Session("ses_jev_cost")


@pytest.fixture
def decide(monkeypatch: pytest.MonkeyPatch) -> None:
    from splinter.providers import decisions

    class _Opts:
        providers: tuple[()] = ()
        threshold = 0.5

    monkeypatch.setattr(localizer, "_jev_decision_options", lambda: _Opts())
    monkeypatch.setattr(
        decisions, "decide_chain", lambda state, questions, slots, **kw: _result([0.9, 0.8])
    )


def _anchors() -> list[CodeAnchor]:
    return [
        CodeAnchor(file="a.py", symbol="", reason="ra", confidence=0.9),
        CodeAnchor(file="b.py", symbol="", reason="rb", confidence=0.9),
    ]


class TestRecordUsage:
    def test_two_calls_sum(self, session: Session, decide: None) -> None:
        _jev_rank_anchors("t", _anchors(), hot=0.8, medium=0.4, session=session)
        _jev_rank_files("t", ["a.py", "b.py"], session=session)
        usage = session.read_jev_usage()
        assert usage["calls"] == 2
        assert usage["input"] == 240
        assert usage["output"] == 16
        assert usage["cost"] == pytest.approx(0.0042)
        models = session.read_pre_run_usage()["models"]
        assert list(models) == ["jev:typesafe/jev-1.14"]

    def test_no_session_no_write(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, decide: None
    ) -> None:
        monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
        ranked = _jev_rank_anchors("t", _anchors(), hot=0.8, medium=0.4)
        assert [a.file for a in ranked] == ["a.py", "b.py"]
        assert _jev_rank_files("t", ["a.py", "b.py"]) == ["a.py", "b.py"]
        assert list(tmp_path.rglob("pre_run_usage.json")) == []

    def test_raising_logger_keeps_ranking(self, session: Session, decide: None) -> None:
        def boom(*args: Any, **kw: Any) -> None:
            raise OSError("disk full")

        session.log_llm_usage = boom  # type: ignore[method-assign]
        ranked = _jev_rank_anchors("t", _anchors(), hot=0.8, medium=0.4, session=session)
        assert [a.file for a in ranked] == ["a.py", "b.py"]
        assert _jev_rank_files("t", ["a.py", "b.py"], session=session) == ["a.py", "b.py"]


class TestSessionJevUsage:
    def test_empty(self, session: Session) -> None:
        assert session.read_jev_usage() == {"cost": 0.0, "input": 0, "output": 0, "calls": 0}

    def test_old_file_without_calls_key(self, session: Session) -> None:
        session._ensure_dir()
        old = {
            "input": 5,
            "output": 1,
            "cost": 0.5,
            "models": {
                "jev:x": {"input": 5, "output": 1, "cost": 0.5},
                "sonnet": {"input": 99, "output": 99, "cost": 9.0},
            },
        }
        (session.dir / "pre_run_usage.json").write_text(json.dumps(old))
        assert session.read_jev_usage() == {"cost": 0.5, "input": 5, "output": 1, "calls": 0}
        session.log_llm_usage("jev:x", {"input": 1, "output": 1}, 0.1)
        usage = session.read_jev_usage()
        assert usage["calls"] == 1
        assert usage["cost"] == pytest.approx(0.6)

    def test_ignores_non_jev_models(self, session: Session) -> None:
        session.log_llm_usage("sonnet", {"input": 10, "output": 10}, 1.0)
        assert session.read_jev_usage()["calls"] == 0
        assert session.read_pre_run_usage()["models"]["sonnet"]["calls"] == 1
