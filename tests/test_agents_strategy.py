"""Integration tests for the agents strategy (all I/O boundaries mocked)."""

from __future__ import annotations

import json
import threading
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import MagicMock, patch

import pytest

import splinter.agents.brief_io as brief_io
import splinter.strategies.agents as agents_mod
import splinter.strategies.agents_brief as brief_mod
from splinter.agents.acceptance import AcceptanceMode, AcceptanceResult
from splinter.agents.codegen import CodegenResult
from splinter.agents.gate import GateResult
from splinter.agents.progress import read_attempts, read_done
from splinter.agents.refiner import Brief
from splinter.agents.runner import RunResult, Task
from splinter.configure import AgentsOptions
from splinter.engines.catalog import Engine, engine_by_key
from splinter.engines.health import EngineProbe
from splinter.engines.launch import EngineLaunch
from splinter.engines.pool import EnginePool
from splinter.memory.session import Session
from splinter.models.roster import Ladder
from splinter.procreg import IdleTimeoutError
from splinter.strategies.agents import AgentsStrategy
from splinter.vcs.guard import GuardEnv
from splinter.vcs.worktree import WorktreeHandle, WorktreeMergeConflict

_LAUNCH = EngineLaunch(
    text="done\nAGENT_RESULT: DONE q1",
    model="minimax/MiniMax-M3",
    session_id="sess-1",
    tokens={"input": 10, "output": 5},
    cost=0.001,
    cost_indeterminate=False,
)
_ACCEPT = AcceptanceResult(
    passed=True, mode=AcceptanceMode.GATE_EVAL, gate_passed=True, detail="ok"
)
_FAIL = AcceptanceResult(
    passed=False, mode=AcceptanceMode.GATE_EVAL, gate_passed=False, detail="gate failed"
)


def _chain() -> tuple[Engine, ...]:
    engines = [engine_by_key(k) for k in ("mcode", "agy", "sonnet")]
    return tuple(e for e in engines if e is not None)


def _session(tmp_path: Path) -> Session:
    session = Session.__new__(Session)
    session.id = "test"
    session.dir = tmp_path
    (tmp_path / "knowledge").mkdir(parents=True, exist_ok=True)
    return session


def _ladder() -> Ladder:
    return Ladder.__new__(Ladder)


def _briefs() -> list[Brief]:
    return [
        Brief(
            id="q1",
            goal="add greeting module",
            files=["splinter_greeting.py"],
            sketch="def greet(name): return f'hello, {name}'",
            acceptance=["greet('x') returns hello, x"],
            task_id="US-101",
        )
    ]


def _tasks() -> list[Task]:
    return [Task(description="add greeting module", acceptance="- [ ] greet works", id="US-101")]


def _opts(**overrides: Any) -> AgentsOptions:
    values: dict[str, Any] = {
        "acceptance_check": "gate+eval",
        "stall_min": 0.1,
        "max_attempts": 3,
        "probe_on_start": False,
    }
    values.update(overrides)
    return AgentsOptions(**values)


@contextmanager
def _mocked(
    tmp_path: Path,
    *,
    opts: AgentsOptions | None = None,
    run_engine: MagicMock | None = None,
    check_acceptance: MagicMock | None = None,
    task_loop: MagicMock | None = None,
) -> Iterator[dict[str, MagicMock]]:
    guard_env = GuardEnv(
        run_dir=tmp_path / "run",
        guard_script=tmp_path / "run" / "guard.sh",
        shims_dir=tmp_path / "run" / "shims",
    )
    patches: dict[str, MagicMock] = {
        "worktree_supported": MagicMock(return_value=True),
        "create_integration_worktree": MagicMock(
            return_value=WorktreeHandle(
                path=tmp_path / "int", branch="splinter/us-101-int", task_id="us-101-int"
            )
        ),
        "create_worktree": MagicMock(
            return_value=WorktreeHandle(
                path=tmp_path / "wt-q1", branch="splinter/us-101-q1", task_id="us-101-q1"
            )
        ),
        "commit_worktree": MagicMock(return_value=True),
        "merge_no_ff": MagicMock(return_value=True),
        "teardown_worktree": MagicMock(return_value=None),
        "setup_guard": MagicMock(return_value=guard_env),
        "configured_agents_options": MagicMock(return_value=opts or _opts()),
        "configured_timeout": MagicMock(return_value=600),
        "load_standards": MagicMock(return_value=""),
        "refine_briefs": MagicMock(return_value=_briefs()),
        "chain_for_effort": MagicMock(return_value=_chain()),
        "run_engine": run_engine or MagicMock(return_value=_LAUNCH),
        "check_acceptance": check_acceptance or MagicMock(return_value=_ACCEPT),
        "worktree_diff": MagicMock(return_value="diff --git a/x b/x"),
        "reattach_worktree": MagicMock(return_value=None),
        "run_codegen": MagicMock(return_value=CodegenResult()),
        "read_report": MagicMock(return_value=""),
        "_sleep": MagicMock(return_value=None),
    }
    in_brief = {
        "load_standards", "chain_for_effort", "run_engine", "check_acceptance",
        "worktree_diff", "run_codegen", "read_report", "_sleep",
    }
    with ExitStack() as stack:
        for name, mock in patches.items():
            module = brief_mod if name in in_brief else agents_mod
            stack.enter_context(patch.object(module, name, mock))
        stack.enter_context(
            patch("splinter.agents.gate.run_gate", return_value=GateResult(True, []))
        )
        if task_loop is not None:
            stack.enter_context(patch.object(AgentsStrategy, "_run_task_loop", task_loop))
        yield patches


class TestHappyPath:
    def test_brief_effort_picks_the_chain(self, tmp_path: Path) -> None:
        session = _session(tmp_path)
        hard = [Brief(id="q1", goal="g", effort="hard", task_id="US-101")]
        with _mocked(tmp_path) as patches:
            patches["refine_briefs"].return_value = hard
            AgentsStrategy().execute(_tasks(), session, _ladder(), max_iterations=2)

        assert patches["chain_for_effort"].call_args.args[1] == "hard"
        launched = patches["run_engine"].call_args
        assert launched.args[0].key == "mcode"
        assert launched.kwargs["effort"] == launched.args[0].default_effort

    def test_pass_merges_and_checkpoints(self, tmp_path: Path) -> None:
        session = _session(tmp_path)
        with _mocked(tmp_path) as patches:
            results = AgentsStrategy().execute(_tasks(), session, _ladder(), max_iterations=2)

        assert len(results) == 1
        patches["commit_worktree"].assert_called_once()
        merge_calls = patches["merge_no_ff"].call_args_list
        assert merge_calls[0].args[0].task_id == "us-101-q1"
        assert str(merge_calls[0].args[1]).endswith("int")
        assert merge_calls[1].args[0].task_id == "us-101-int"
        assert len(merge_calls) == 2
        patches["create_worktree"].assert_called_once_with(
            "us-101-q1", base_ref="splinter/us-101-int"
        )
        checkpoint = (tmp_path / "agents-run" / "checkpoint.json").read_text()
        assert "q1" in checkpoint
        state = (tmp_path / "agents-run" / "STATE.md").read_text()
        assert "merged q1" in state
        brief_doc = (tmp_path / "agents-run" / "briefs" / "q1.md").read_text()
        assert "q1" in brief_doc
        assert (tmp_path / "agents-run" / "README.md").exists()

    def test_composer_directive_reaches_the_brief(self, tmp_path: Path) -> None:
        session = _session(tmp_path)
        directive_id = session.queue_live_command("keep the public API", task_no=1)
        with _mocked(tmp_path) as patches:
            AgentsStrategy().execute(_tasks(), session, _ladder(), max_iterations=2)

        prompt = patches["run_engine"].call_args.args[1]
        assert "## User guidance" in prompt and "keep the public API" in prompt
        assert session.receipt_exists(directive_id)

    def test_engine_prompt_includes_brief_and_result_line(self, tmp_path: Path) -> None:
        session = _session(tmp_path)
        with _mocked(tmp_path) as patches:
            AgentsStrategy().execute(_tasks(), session, _ladder(), max_iterations=2)
        prompt = patches["run_engine"].call_args[0][1]
        assert "AGENT_RESULT: DONE q1" in prompt
        assert "workhorse" in prompt.lower()


class TestReroute:
    def test_stall_reroutes_to_next_engine_without_burning_attempts(
        self, tmp_path: Path
    ) -> None:
        engine_calls: list[str] = []

        def fake_run(engine: Any, prompt: str, **kw: Any) -> EngineLaunch:
            engine_calls.append(engine.key)
            if len(engine_calls) == 1:
                raise IdleTimeoutError(6.0, ["mcode", "exec"])
            return _LAUNCH

        session = _session(tmp_path)
        with _mocked(tmp_path, run_engine=MagicMock(side_effect=fake_run)) as patches:
            results = AgentsStrategy().execute(_tasks(), session, _ladder(), max_iterations=2)

        assert engine_calls == ["mcode", "agy"]
        assert len(results) == 1
        merge_ids = [c.args[0].task_id for c in patches["merge_no_ff"].call_args_list]
        assert "us-101-q1" in merge_ids

    def test_crash_reroutes_and_pool_exhaustion_burns_attempt(self, tmp_path: Path) -> None:
        engine_calls: list[str] = []

        def fake_run(engine: Any, prompt: str, **kw: Any) -> EngineLaunch:
            engine_calls.append(engine.key)
            if len(engine_calls) == 1:
                raise RuntimeError("agy exited 1: boom")
            return _LAUNCH

        session = _session(tmp_path)
        with _mocked(tmp_path, run_engine=MagicMock(side_effect=fake_run)):
            results = AgentsStrategy().execute(_tasks(), session, _ladder(), max_iterations=2)
        assert engine_calls == ["mcode", "agy"]
        assert len(results) == 1


class TestAssume:
    def test_acceptance_exhaustion_falls_back_to_direct_loop(self, tmp_path: Path) -> None:
        captured: dict[str, Any] = {}

        def fake_loop(
            task: Task,
            session: Session,
            ladder: Ladder,
            trace: Any,
            knowledge: Any,
            **kw: Any,
        ) -> RunResult:
            captured["cwd"] = kw.get("cwd")
            outcome: Any = kw.get("outcome")
            if outcome is not None:
                outcome.passed = True
            return RunResult(
                text="assumed", model="opus", tier=0, tokens={}, cost=0.0, raw={}
            )

        session = _session(tmp_path)
        with _mocked(
            tmp_path,
            check_acceptance=MagicMock(return_value=_FAIL),
            task_loop=MagicMock(side_effect=fake_loop),
        ) as patches:
            results = AgentsStrategy().execute(_tasks(), session, _ladder(), max_iterations=2)

        assert patches["run_engine"].call_count == 3
        assert captured["cwd"] == str(tmp_path / "wt-q1")
        merge_ids = [c.args[0].task_id for c in patches["merge_no_ff"].call_args_list]
        assert "us-101-q1" in merge_ids
        assert len(results) == 1
        fix_doc = (tmp_path / "agents-run" / "briefs" / "q1.md").read_text()
        assert "Fix round" in fix_doc


class TestModes:
    def test_skip_eval_forces_gate_only_mode(self, tmp_path: Path) -> None:
        session = _session(tmp_path)
        with _mocked(tmp_path) as patches:
            AgentsStrategy().execute(
                _tasks(), session, _ladder(), max_iterations=2, skip_eval=True
            )
        assert patches["check_acceptance"].call_args.kwargs["mode"] == AcceptanceMode.GATE

    def test_configured_mode_used_by_default(self, tmp_path: Path) -> None:
        session = _session(tmp_path)
        with _mocked(tmp_path) as patches:
            AgentsStrategy().execute(_tasks(), session, _ladder(), max_iterations=2)
        assert (
            patches["check_acceptance"].call_args.kwargs["mode"] == AcceptanceMode.GATE_EVAL
        )


class TestFallbackWithoutWorktrees:
    def test_runs_direct_loop_when_unsupported(self, tmp_path: Path) -> None:
        def fake_loop(
            task: Task,
            session: Session,
            ladder: Ladder,
            trace: Any,
            knowledge: Any,
            **kw: Any,
        ) -> RunResult:
            return RunResult(text="direct", model="sonnet", tier=0, tokens={}, cost=0.0, raw={})

        session = _session(tmp_path)
        with _mocked(
            tmp_path,
            task_loop=MagicMock(side_effect=fake_loop),
        ) as patches:
            patches["worktree_supported"] = MagicMock(return_value=False)
            with patch.object(agents_mod, "worktree_supported", patches["worktree_supported"]):
                results = AgentsStrategy().execute(
                    _tasks(), session, _ladder(), max_iterations=2
                )
        assert len(results) == 1
        assert results[0].text == "direct"
        patches["create_integration_worktree"].assert_not_called()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("US-101", "us-101"), ("", "run"), ("Fix/UP: stray", "fix-up-stray")],
)
class TestSlug:
    def test_slug_from_task_id(self, raw: str, expected: str) -> None:
        tasks = [Task(description="x", acceptance="- [ ] y", id=raw)]
        assert brief_io.slug_from(tasks) == expected


def test_merge_failure_blocks_checkpoint_and_dependents(tmp_path: Path) -> None:
    session = _session(tmp_path)
    with _mocked(tmp_path) as mocks:
        mocks["refine_briefs"].return_value = [
            Brief(id="q1", goal="first"), Brief(id="q2", goal="second", deps=["q1"]),
        ]
        mocks["merge_no_ff"].side_effect = WorktreeMergeConflict("dirty")
        with pytest.raises(RuntimeError, match="incomplete briefs"):
            AgentsStrategy().execute(_tasks(), session, _ladder())
        assert mocks["run_engine"].call_count == 1
        assert not (tmp_path / "agents-run" / "checkpoint.json").exists()


def test_assumed_brief_still_requires_merge(tmp_path: Path) -> None:
    def assume(*args: Any, **kwargs: Any) -> None:
        kwargs["outcome"].passed = True

    session = _session(tmp_path)
    with _mocked(
        tmp_path, check_acceptance=MagicMock(return_value=_FAIL),
        task_loop=MagicMock(side_effect=assume),
    ) as mocks:
        mocks["merge_no_ff"].side_effect = WorktreeMergeConflict("dirty")
        with pytest.raises(RuntimeError, match="incomplete briefs"):
            AgentsStrategy().execute(_tasks(), session, _ladder())
        assert not (tmp_path / "agents-run" / "checkpoint.json").exists()


def test_resume_reuses_integration_and_pending_worktree(tmp_path: Path) -> None:
    session = _session(tmp_path)
    for ident, directory in [("us-101-int", "int"), ("q1", "wt-q1")]:
        path = tmp_path / directory
        path.mkdir()
        session.set_worktree(ident, str(path), f"splinter/{ident}")
    with _mocked(tmp_path) as mocks:
        mocks["reattach_worktree"].side_effect = lambda ident: WorktreeHandle(
            tmp_path / ident, f"splinter/{ident}", ident
        )
        AgentsStrategy().execute(_tasks(), session, _ladder(), resume=True)
        mocks["create_integration_worktree"].assert_not_called()
        mocks["create_worktree"].assert_not_called()


@pytest.mark.parametrize("failure", ["gate", "merge"])
def test_finalization_failure_is_propagated(tmp_path: Path, failure: str) -> None:
    with _mocked(tmp_path) as mocks:
        with patch("splinter.agents.gate.run_gate", return_value=GateResult(failure != "gate", [])):
            if failure == "merge":
                mocks["merge_no_ff"].side_effect = [True, WorktreeMergeConflict("dirty")]
            with pytest.raises((RuntimeError, WorktreeMergeConflict)):
                AgentsStrategy().execute(_tasks(), _session(tmp_path), _ladder())


def test_budget_does_not_queue_all_ready_briefs(tmp_path: Path) -> None:
    with _mocked(tmp_path) as mocks:
        mocks["refine_briefs"].return_value = [Brief(id=f"q{i}", goal="g") for i in range(5)]
        with pytest.raises(RuntimeError, match="incomplete briefs"):
            AgentsStrategy().execute(
                _tasks(), _session(tmp_path), _ladder(), budget=0.0005, max_concurrency=1,
            )
        assert mocks["run_engine"].call_count == 1


def test_parallel_is_opt_in(tmp_path: Path) -> None:
    with _mocked(tmp_path), patch.object(AgentsStrategy, "_dispatch") as dispatch:
        def finish(*args: Any) -> list[RunResult]:
            assert args[-2].max_concurrency() == 1
            args[-1].add("q1")
            return []

        dispatch.side_effect = finish
        AgentsStrategy().execute(
            _tasks(), _session(tmp_path), _ladder(), parallel=False, max_concurrency=2,
        )


def test_diff_includes_untracked_content(tmp_path: Path) -> None:
    (tmp_path / "new.py").write_text("def greet(): return 'hello'\n")
    with patch.object(brief_io.subprocess, "run") as run:
        run.side_effect = [
            MagicMock(stdout="?? new.py"), MagicMock(stdout=""),
            MagicMock(stdout="new.py\0"),
        ]
        diff = brief_io.worktree_diff(tmp_path)
    assert "def greet(): return 'hello'" in diff
    assert run.call_args_list[0].args[0][-1] == "--untracked-files=all"


def _attempts(tmp_path: Path) -> list[Any]:
    return read_attempts((tmp_path / "agents-run" / "attempts.jsonl").read_text())


def test_blocked_workhorse_goes_to_the_orchestrator_without_acceptance(tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def assume(*args: Any, **kwargs: Any) -> None:
        captured.update(kwargs)
        kwargs["outcome"].passed = True

    blocked = EngineLaunch(text="AGENT_RESULT: BLOCKED q1", model="kimi")
    with _mocked(
        tmp_path, run_engine=MagicMock(return_value=blocked),
        task_loop=MagicMock(side_effect=assume),
    ) as mocks:
        mocks["read_report"].return_value = "protoc is blocked by the guard"
        AgentsStrategy().execute(_tasks(), _session(tmp_path), _ladder())
    assert mocks["run_engine"].call_count == 1
    mocks["check_acceptance"].assert_not_called()
    assert "protoc is blocked by the guard" in captured["plan_text"]
    assert "greet('x') returns hello, x" in captured["plan_text"]
    verdicts = [(a.engine, a.verdict) for a in _attempts(tmp_path)]
    assert verdicts == [("mcode", "RETRY"), ("orchestrator", "PASS")]


def test_codegen_failure_is_a_fix_round_and_success_reaches_acceptance(tmp_path: Path) -> None:
    with _mocked(tmp_path) as mocks:
        mocks["run_codegen"].side_effect = [
            CodegenResult(("make gen",), "make gen exited 2: report.proto:3: syntax error"),
            CodegenResult(("make gen",)),
        ]
        AgentsStrategy().execute(_tasks(), _session(tmp_path), _ladder())
    assert mocks["run_engine"].call_count == 2
    assert mocks["check_acceptance"].call_count == 1
    doc = (tmp_path / "agents-run" / "briefs" / "q1.md").read_text()
    assert "syntax error" in doc
    assert [a.verdict for a in _attempts(tmp_path)] == ["RETRY", "PASS"]


def test_cooling_chain_waits_instead_of_burning_attempts(tmp_path: Path) -> None:
    calls: list[str] = []

    def flaky(engine: Any, prompt: str, **kw: Any) -> EngineLaunch:
        calls.append(engine.key)
        if len(calls) <= 6:
            raise RuntimeError(f"{engine.key} exited 1: boom")
        return _LAUNCH

    clock = [1000.0]

    def sleep(seconds: float) -> None:
        clock[0] += seconds

    with (
        _mocked(tmp_path, run_engine=MagicMock(side_effect=flaky)) as mocks,
        patch("splinter.engines.health.time.monotonic", side_effect=lambda: clock[0]),
    ):
        mocks["_sleep"].side_effect = sleep
        AgentsStrategy().execute(_tasks(), _session(tmp_path), _ladder())
    assert mocks["_sleep"].call_count == 2
    assert len(calls) == 7
    assert read_done((tmp_path / "agents-run" / "checkpoint.json").read_text()) == {"q1"}


def test_attempts_are_logged_per_story_with_their_tier(tmp_path: Path) -> None:
    with _mocked(tmp_path, check_acceptance=MagicMock(side_effect=[_FAIL, _ACCEPT])):
        AgentsStrategy().execute(_tasks(), _session(tmp_path), _ladder())
    attempts = _attempts(tmp_path)
    assert [(a.story, a.n, a.brief, a.verdict) for a in attempts] == [
        (1, 1, "q1", "RETRY"), (1, 2, "q1", "PASS"),
    ]
    assert (tmp_path / "agents-run" / "logs" / "q1-1.log").exists()
    stories = (tmp_path / "agents-run" / "stories.json").read_text()
    assert '"q1": 1' in stories and "add greeting module" in stories


class TestStopRequested:
    def test_stop_before_dispatch_submits_nothing(self, tmp_path: Path) -> None:
        from splinter import procreg

        procreg.request_stop()
        try:
            with _mocked(tmp_path) as mocks:
                with pytest.raises(RuntimeError, match="incomplete briefs"):
                    AgentsStrategy().execute(_tasks(), _session(tmp_path), _ladder())
        finally:
            procreg.clear_stop()
        assert mocks["run_engine"].call_count == 0

    def test_stop_after_launch_skips_the_orchestrator(self, tmp_path: Path) -> None:
        from splinter import procreg

        def fake_run(engine: Any, prompt: str, **kw: Any) -> EngineLaunch:
            procreg.request_stop()
            return _LAUNCH

        loop = MagicMock(return_value=None)
        try:
            with _mocked(
                tmp_path,
                run_engine=MagicMock(side_effect=fake_run),
                check_acceptance=MagicMock(return_value=_FAIL),
                task_loop=loop,
            ):
                with pytest.raises(RuntimeError, match="incomplete briefs"):
                    AgentsStrategy().execute(_tasks(), _session(tmp_path), _ladder())
        finally:
            procreg.clear_stop()
        assert loop.call_count == 0

    def test_shutting_down_propagates_instead_of_rerouting(self, tmp_path: Path) -> None:
        from splinter.procreg import ShuttingDownError

        def fake_run(engine: Any, prompt: str, **kw: Any) -> EngineLaunch:
            raise ShuttingDownError("splinter is exiting")

        with _mocked(tmp_path, run_engine=MagicMock(side_effect=fake_run)) as mocks:
            with pytest.raises(RuntimeError, match="incomplete briefs"):
                AgentsStrategy().execute(_tasks(), _session(tmp_path), _ladder())
        assert mocks["run_engine"].call_count == 1


def test_infra_failure_writes_an_engine_log(tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_run(engine: Any, prompt: str, **kw: Any) -> EngineLaunch:
        calls.append(engine.key)
        if len(calls) == 1:
            raise IdleTimeoutError(480.0, ["mcode"])
        return _LAUNCH

    with _mocked(tmp_path, run_engine=MagicMock(side_effect=fake_run)):
        AgentsStrategy().execute(_tasks(), _session(tmp_path), _ladder())
    first = (tmp_path / "agents-run" / "logs" / "q1-1.log").read_text()
    assert "STALL" in first


def test_failed_gate_builds_no_diff(tmp_path: Path) -> None:
    with _mocked(tmp_path, check_acceptance=MagicMock(side_effect=[_FAIL, _ACCEPT])) as mocks:
        AgentsStrategy().execute(_tasks(), _session(tmp_path), _ladder())
    assert mocks["worktree_diff"].call_count == 0
    assert "diff" not in mocks["check_acceptance"].call_args.kwargs


class TestFixRoundDelta:
    def _prompts(self, mocks: dict[str, MagicMock]) -> list[str]:
        return [call.args[1] for call in mocks["run_engine"].call_args_list]

    def test_resumed_session_gets_only_the_fix_round(self, tmp_path: Path) -> None:
        with _mocked(tmp_path, check_acceptance=MagicMock(side_effect=[_FAIL, _ACCEPT])) as mocks:
            AgentsStrategy().execute(_tasks(), _session(tmp_path), _ladder())
        first, second = self._prompts(mocks)
        assert "--- WORKHORSE CONTRACT ---" in first
        assert "--- WORKHORSE CONTRACT ---" not in second
        assert "add greeting module" not in second
        assert "Fix round 1" in second
        assert "AGENT_RESULT: DONE q1" in second
        assert mocks["run_engine"].call_args.kwargs["session"] == "sess-1"

    def test_engine_without_continue_gets_the_whole_document(self, tmp_path: Path) -> None:
        agy = engine_by_key("agy")
        assert agy is not None and not agy.supports_continue
        with _mocked(tmp_path, check_acceptance=MagicMock(side_effect=[_FAIL, _ACCEPT])) as mocks:
            mocks["chain_for_effort"].return_value = (agy,)
            AgentsStrategy().execute(_tasks(), _session(tmp_path), _ladder())
        second = self._prompts(mocks)[1]
        assert "--- WORKHORSE CONTRACT ---" in second
        assert "Fix round 1" in second

    def test_guidance_sent_between_rounds_reaches_the_resumed_session(
        self, tmp_path: Path
    ) -> None:
        session = _session(tmp_path)

        def fake_run(engine: Any, prompt: str, **kw: Any) -> EngineLaunch:
            session.queue_live_command("keep the public API", task_no=1)
            return _LAUNCH

        with _mocked(
            tmp_path,
            run_engine=MagicMock(side_effect=fake_run),
            check_acceptance=MagicMock(side_effect=[_FAIL, _ACCEPT]),
        ) as mocks:
            AgentsStrategy().execute(_tasks(), session, _ladder())
        second = self._prompts(mocks)[1]
        assert "keep the public API" in second
        assert "--- WORKHORSE CONTRACT ---" not in second

    def test_fix_not_delivered_by_a_failed_launch_is_sent_again(self, tmp_path: Path) -> None:
        calls: list[str] = []

        def fake_run(engine: Any, prompt: str, **kw: Any) -> EngineLaunch:
            calls.append(prompt)
            if len(calls) == 2:
                raise RuntimeError("mcode exited 1: boom")
            return _LAUNCH

        only = engine_by_key("mcode")
        with _mocked(
            tmp_path,
            run_engine=MagicMock(side_effect=fake_run),
            check_acceptance=MagicMock(side_effect=[_FAIL, _ACCEPT]),
        ) as mocks:
            mocks["chain_for_effort"].return_value = (only,)
            AgentsStrategy().execute(_tasks(), _session(tmp_path), _ladder())
        assert len(calls) == 3
        assert "Fix round 1" in calls[2]
        assert "--- WORKHORSE CONTRACT ---" not in calls[2]

    def test_reroute_never_resumes_the_previous_engine_session(self, tmp_path: Path) -> None:
        sessions: list[str | None] = []

        def fake_run(engine: Any, prompt: str, **kw: Any) -> EngineLaunch:
            sessions.append(kw["session"])
            if len(sessions) == 2:
                raise RuntimeError("mcode exited 1: boom")
            if engine.key == "kimi":
                return EngineLaunch(text="AGENT_RESULT: DONE q1", model="kimi")
            return _LAUNCH

        chain = tuple(e for e in (engine_by_key("mcode"), engine_by_key("kimi")) if e)
        with _mocked(
            tmp_path,
            run_engine=MagicMock(side_effect=fake_run),
            check_acceptance=MagicMock(side_effect=[_FAIL, _FAIL, _ACCEPT]),
        ) as mocks:
            mocks["chain_for_effort"].return_value = chain
            AgentsStrategy().execute(_tasks(), _session(tmp_path), _ladder())
        assert sessions[-1] is None


def _run_entries(tmp_path: Path) -> list[dict[str, Any]]:
    lines = (tmp_path / "events.jsonl").read_text().splitlines()
    return [e for e in map(json.loads, lines) if e.get("type") == "run"]


def test_failed_launch_that_spent_tokens_is_traced(tmp_path: Path) -> None:
    silent = EngineLaunch(
        text="no result line", model="m", tokens={"input": 7, "output": 3}, cost=0.002
    )
    with _mocked(tmp_path, run_engine=MagicMock(side_effect=[silent, _LAUNCH])):
        AgentsStrategy().execute(_tasks(), _session(tmp_path), _ladder())
    assert len(_run_entries(tmp_path)) == 2


def test_chain_with_no_open_breaker_backs_off(tmp_path: Path) -> None:
    calls: list[str] = []

    def flaky(engine: Any, prompt: str, **kw: Any) -> EngineLaunch:
        calls.append(engine.key)
        if len(calls) <= 6:
            raise RuntimeError(f"{engine.key} exited 1: boom")
        return _LAUNCH

    with (
        _mocked(tmp_path, run_engine=MagicMock(side_effect=flaky)) as mocks,
        patch.object(EnginePool, "report_infra_failure"),
    ):
        AgentsStrategy().execute(_tasks(), _session(tmp_path), _ladder())
    assert [call.args[0] for call in mocks["_sleep"].call_args_list] == [5.0, 10.0]


def test_open_breaker_wait_is_not_capped_by_the_floor(tmp_path: Path) -> None:
    only = engine_by_key("mcode")
    assert only is not None
    pool = MagicMock()
    pool.wait_s.return_value = 300.0
    ctx = MagicMock(pool=pool)
    with patch.object(brief_mod, "_sleep") as sleep:
        brief_mod._wait_for_chain(ctx, (only,), "q1", 1)
        pool.wait_s.return_value = 0.0
        brief_mod._wait_for_chain(ctx, (only,), "q1", 12)
    assert [call.args[0] for call in sleep.call_args_list] == [300.0, 120.0]


def test_probes_run_at_once_and_trip_the_down_engines() -> None:
    keys = ("mcode", "agy", "sonnet", "codex")
    engines = tuple(e for e in map(engine_by_key, keys) if e is not None)
    assert len(engines) == 4
    barrier = threading.Barrier(4, timeout=5)
    threads: set[str] = set()

    def fake_probe(engine: Engine) -> EngineProbe:
        threads.add(threading.current_thread().name)
        barrier.wait()
        return EngineProbe(key=engine.key, up=engine.key != "agy", binary="x", detail="d")

    health = MagicMock()
    with (
        patch.object(agents_mod, "chain_for_effort", return_value=engines),
        patch.object(agents_mod, "probe", side_effect=fake_probe) as probe,
    ):
        AgentsStrategy()._probe_into(health, _ladder())
    assert len(threads) == 4
    assert probe.call_count == 4
    health.trip.assert_called_once_with("agy")


def test_crashing_probe_does_not_stop_the_others() -> None:
    keys = ("mcode", "agy")
    engines = tuple(e for e in map(engine_by_key, keys) if e is not None)

    def fake_probe(engine: Engine) -> EngineProbe:
        if engine.key == "mcode":
            raise OSError("no pty")
        return EngineProbe(key=engine.key, up=False, binary="x", detail="d")

    health = MagicMock()
    with (
        patch.object(agents_mod, "chain_for_effort", return_value=engines),
        patch.object(agents_mod, "probe", side_effect=fake_probe),
    ):
        AgentsStrategy()._probe_into(health, _ladder())
    health.trip.assert_called_once_with("agy")


class TestResumeByBranch:
    def _save(self, session: Session, tmp_path: Path) -> None:
        session.set_worktree("us-101-int", str(tmp_path / "gone-int"), "splinter/us-101-int")
        session.set_worktree("q1", str(tmp_path / "gone-q1"), "splinter/us-101-q1")

    def test_record_with_a_lost_path_recreates_on_the_same_task_id(self, tmp_path: Path) -> None:
        session = _session(tmp_path)
        self._save(session, tmp_path)
        with _mocked(tmp_path) as mocks:
            AgentsStrategy().execute(_tasks(), session, _ladder(), resume=True)
        mocks["create_worktree"].assert_called_once_with(
            "us-101-q1", base_ref="splinter/us-101-int"
        )
        mocks["create_integration_worktree"].assert_called_once_with("us-101")
        reattached = [call.args[0] for call in mocks["reattach_worktree"].call_args_list]
        assert reattached == ["us-101-int", "us-101-q1"]

    def test_reattached_worktree_is_not_created_again(self, tmp_path: Path) -> None:
        session = _session(tmp_path)
        self._save(session, tmp_path)
        with _mocked(tmp_path) as mocks:
            mocks["reattach_worktree"].side_effect = lambda ident: WorktreeHandle(
                tmp_path / ident, f"splinter/{ident}", ident
            )
            AgentsStrategy().execute(_tasks(), session, _ladder(), resume=True)
        mocks["create_worktree"].assert_not_called()
        mocks["create_integration_worktree"].assert_not_called()

    def test_fresh_run_never_adopts_a_stale_worktree(self, tmp_path: Path) -> None:
        with _mocked(tmp_path) as mocks:
            AgentsStrategy().execute(_tasks(), _session(tmp_path), _ladder())
        mocks["reattach_worktree"].assert_not_called()


def test_refine_workers_reach_the_refiner(tmp_path: Path) -> None:
    with _mocked(tmp_path, opts=_opts(refine_workers=7)) as mocks:
        AgentsStrategy().execute(_tasks(), _session(tmp_path), _ladder())
    assert mocks["refine_briefs"].call_args.kwargs["max_workers"] == 7
