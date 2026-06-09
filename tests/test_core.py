from __future__ import annotations

import random
from pathlib import Path

import pytest

from splinter.agents.gate import GateResult
from splinter.agents.runner import Task, resolve_model, resolve_variant
from splinter.analyze import (
    _iterations,
    _prd_phases,
    _run_state,
    _trace_metrics,
    render_iteration,
    render_overview,
    render_trajectory,
)
from splinter.configure import DEFAULT_CONFIG, init_prompt_templates
from splinter.enums import Decision
from splinter.memory.knowledge import KnowledgeStore
from splinter.memory.session import Session, list_sessions, new_session_id
from splinter.models.roster import load_ladder
from splinter.obs.trace import Trace, log_run
from splinter.providers.registry import available_providers, get_provider
from splinter.strategies.base import EvalVerdict
from splinter.strategies.registry import available_strategies, get_strategy
from splinter.strategies.stages import _parse_verdict
from splinter.templating import TEMPLATE_NAMES, packaged_template, render, section


def test_new_session_id_format() -> None:
    sid = new_session_id()
    assert sid.startswith("ses_")
    assert len(sid) > 10


def test_ladder_loads() -> None:
    ladder = load_ladder()
    assert len(ladder.tiers) == 5
    assert ladder.tiers[0].name == "easy"
    assert ladder.tiers[4].name == "top"
    assert len(ladder.all_model_ids()) > 0
    # floor is glm-5.1 (no weak models)
    assert ladder.tiers[0].models[0] == "opencode-go/glm-5.1"


def test_ladder_tier_by_level() -> None:
    ladder = load_ladder()
    t0 = ladder.tier_by_level(0)
    assert t0.name == "easy"
    t4 = ladder.tier_by_level(4)
    assert t4.name == "top"


def test_normal_effort_defaults_to_deepseek_v4_pro() -> None:
    ladder = load_ladder()
    em = ladder.effort_mapping("normal")
    assert em is not None and em.start_tier == 1
    model_id, provider = resolve_model(em.start_tier, ladder)
    assert provider == "opencode"
    assert model_id == "opencode-go/deepseek-v4-pro"


def test_localizer_roster() -> None:
    ladder = load_ladder()
    assert ladder.localizer_recall_model == "opencode-go/deepseek-v4-flash"
    assert ladder.localizer_recall_large_model == "opencode-go/minimax-m3"
    assert ladder.localizer_precision_model == "opencode-go/kimi-k2.6"


def test_config_model_overrides_apply_to_ladder(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from splinter.configure import write_model_config

    monkeypatch.chdir(tmp_path)
    write_model_config(
        {
            "planner": "opus-4.8",
            "localizer_precision": "opencode-go/minimax-m3",
            "tiers": ["opencode-go/kimi-k2.6"],
        }
    )
    ladder = load_ladder()
    assert ladder.planner_model == "opus-4.8"
    assert ladder.localizer_precision_model == "opencode-go/minimax-m3"
    t0 = ladder.tier_by_level(0)
    assert t0.models[0] == "opencode-go/kimi-k2.6"
    assert t0.provider == "opencode"
    assert ladder.eval_model == "sonnet"  # untouched step keeps its default


def test_config_effort_overrides_apply_to_ladder(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from splinter.configure import write_model_config

    monkeypatch.chdir(tmp_path)
    write_model_config(
        {},
        {
            "planner": "max",
            "eval": "low",
            "localizer_recall": "high",
            "tiers": ["high", "", "", "", ""],  # blank tiers fall back to effort_map
        },
    )
    ladder = load_ladder()
    assert ladder.planner_effort == "max"
    assert ladder.eval_effort == "low"
    assert ladder.localizer_recall_variant == "high"
    assert ladder.tier_variant(0) == "high"
    assert ladder.tier_variant(1) is None  # blank → no override


def test_configure_tui_saves_models_and_efforts(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    import asyncio

    from textual.widgets import Select

    from splinter.tui import ConfigureApp

    monkeypatch.chdir(tmp_path)

    async def drive() -> None:
        app = ConfigureApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("#planner__eff", Select).value = "max"
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()
        assert app.saved

    asyncio.run(drive())
    text = (tmp_path / ".splinter" / "config.yaml").read_text()
    assert "Select.NULL" not in text and "NoSelection" not in text
    ladder = load_ladder()
    assert ladder.planner_effort == "max"


def test_ladder_effort_mapping() -> None:
    ladder = load_ladder()
    em = ladder.effort_mapping("trivial")
    assert em is not None
    assert em.start_tier == 0
    assert em.variant == "minimal"


def test_resolve_variant_auto() -> None:
    ladder = load_ladder()
    task = Task(description="test", acceptance="test", effort="hard", reasoning_effort="auto")
    v = resolve_variant(task, None, ladder)
    assert v == "high"


def test_resolve_variant_override() -> None:
    ladder = load_ladder()
    task = Task(description="test", acceptance="test", effort="normal", reasoning_effort="auto")
    v = resolve_variant(task, "max", ladder)
    assert v == "max"


def test_resolve_model() -> None:
    ladder = load_ladder()
    model_id, provider = resolve_model(0, ladder)
    assert provider == "opencode"
    assert model_id.startswith("opencode-go/")


def test_resolve_model_claude() -> None:
    ladder = load_ladder()
    model_id, provider = resolve_model(3, ladder)
    assert provider == "claude"
    assert model_id == "sonnet"


def test_session_write_read(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_test")
    session.write("test.md", "hello world")
    assert session.read("test.md") == "hello world"


def test_session_append(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_test")
    session.append("log.md", "line 1")
    session.append("log.md", "line 2")
    content = session.read("log.md")
    assert "line 1" in content
    assert "line 2" in content


def test_list_sessions_newest_first(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    import os
    import time

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    assert list_sessions() == []
    for sid in ("ses_a", "ses_b", "ses_c"):
        Session(sid).update_index(f"# {sid}\n")
        # bump mtime so ordering is deterministic
        os.utime((tmp_path / "sessions" / sid))
        time.sleep(0.01)
    listed = list_sessions()
    assert set(listed) == {"ses_a", "ses_b", "ses_c"}
    assert listed[0] == "ses_c"  # newest


def test_session_has(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_test")
    assert not session.has("plan.md")
    session.write("plan.md", "# Plan")
    assert session.has("plan.md")


def test_knowledge_store(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_test")
    ks = KnowledgeStore(session)
    ks.write_note("test-note", "# Test\nSome content")
    assert "test-note" in ks.list_notes()
    assert "Some content" in ks.read_note("test-note")


def test_knowledge_query(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_test")
    ks = KnowledgeStore(session)
    ks.write_note("auth-module", "# Authentication\nLogin flow")
    ks.write_note("db-schema", "# Database\nTables and columns")
    results = ks.query("auth")
    assert "auth-module" in results
    assert "db-schema" not in results


def test_trace_summary() -> None:
    trace = Trace()
    from splinter.agents.runner import RunResult
    result = RunResult(
        text="output", model="test-model", tier=0,
        tokens={"input": 100, "output": 50}, cost=0.01, raw={},
    )
    log_run(trace, result, 1)
    summary = trace.summary()
    assert "test-model" in summary
    assert "total runs: 1" in summary


def test_gate_result_dataclass() -> None:
    gr = GateResult(passed=True, checks=[("ruff", True, "ok")])
    assert gr.passed
    assert len(gr.checks) == 1


def test_eval_verdict_dataclass() -> None:
    v = EvalVerdict(decision="PASS", reason="all good")
    assert v.decision == "PASS"
    assert v.passed


def test_default_config() -> None:
    assert "gate_checks" in DEFAULT_CONFIG
    assert "defaults" in DEFAULT_CONFIG


# --- design-pattern wiring -------------------------------------------------


def test_strategy_registry_resolves_name_and_alias() -> None:
    assert type(get_strategy("direct")) is type(get_strategy("raphael"))
    assert "direct" in available_strategies()
    assert "raphael" in available_strategies()


def test_strategy_registry_unknown() -> None:
    with pytest.raises(ValueError):
        get_strategy("nonesuch")


def test_opencode_extract_tokens_handles_nested() -> None:
    from splinter.providers.opencode import _extract_tokens

    raw = {
        "tokens": {
            "input": 120,
            "output": "45",          # string count
            "reasoning": 3.0,         # float
            "cache": {"read": 0, "write": 0},  # nested dict — must be skipped
            "bogus": None,            # must be skipped
        }
    }
    tokens = _extract_tokens(raw)
    assert tokens["input"] == 120
    assert tokens["output"] == 45
    assert tokens["reasoning"] == 3
    assert "cache" not in tokens
    assert "bogus" not in tokens


def test_provider_registry() -> None:
    assert set(available_providers()) == {"claude", "opencode"}
    assert get_provider("claude").name == "claude"
    with pytest.raises(ValueError):
        get_provider("bogus")


def test_decision_strenum_compares_to_str() -> None:
    assert Decision.PASS == "PASS"
    assert Decision("ESCALATE") is Decision.ESCALATE


def test_parse_verdict_escalate() -> None:
    v = _parse_verdict("VERDICT: ESCALATE\nREASON: too hard\nCORRECTIONS: rewrite parser")
    assert v.decision == Decision.ESCALATE
    assert not v.passed
    assert v.corrections == "rewrite parser"


def test_parse_verdict_pass() -> None:
    v = _parse_verdict("VERDICT: PASS\nREASON: meets criteria\nCORRECTIONS: none")
    assert v.passed


# --- prompt templates ------------------------------------------------------


def test_section_omits_blank_body() -> None:
    assert section("Code Context", "") == ""
    assert section("Plan", "do x") == "## Plan\ndo x"


def test_render_collapses_empty_sections() -> None:
    prompt = render(
        "run",
        plan_section=section("Plan", "p"),
        task_section=section("Task", "t"),
        acceptance_section=section("Acceptance Criteria", "a"),
        code_context_section=section("Code Context", ""),
    )
    assert "Code Context" not in prompt
    assert "\n\n\n" not in prompt


def test_all_templates_packaged() -> None:
    for name in TEMPLATE_NAMES:
        assert packaged_template(name).strip()


def test_render_prefers_override(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    monkeypatch.chdir(tmp_path)
    init_prompt_templates()
    (tmp_path / ".splinter" / "prompts" / "plan.md").write_text("OVERRIDE {task_section}")
    out = render("plan", task_section=section("Task", "t"))
    assert out.startswith("OVERRIDE")


def test_init_prompt_templates_no_overwrite(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    monkeypatch.chdir(tmp_path)
    first = init_prompt_templates()
    assert len(first) == len(TEMPLATE_NAMES)
    second = init_prompt_templates()
    assert second == []  # already present, not overwritten


# --- analyze: status + trajectory -----------------------------------------


def test_session_status_roundtrip(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_test")
    assert session.read_status() == {}
    session.set_status("running", pid=4321, strategy="raphael")
    status = session.read_status()
    assert status["state"] == "running"
    assert status["pid"] == 4321
    session.set_status("completed")
    assert session.read_status()["state"] == "completed"


def _free_pid() -> int:
    """Return a pid that is not currently in use."""
    import os

    while True:
        pid = random.randint(100_000, 2_000_000_000)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return pid  # no such process — free
        except PermissionError:
            continue  # exists, owned by another user


def test_run_state_alive_vs_dead(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    import os

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_test")
    session.set_status("running", pid=os.getpid())
    assert _run_state(session) == "RUNNING"
    session.set_status("running", pid=_free_pid())  # not a live pid
    assert _run_state(session) == "INTERRUPTED"
    session.set_status("completed")
    assert _run_state(session) == "COMPLETED"


def test_iterations_parse_trajectory() -> None:
    loop = (
        "## Iteration 1\n- model: flash (tier 0)\n- verdict: RETRY — x\n\n"
        "## Iteration 2\n- model: qwen (tier 1)\n- verdict: PASS — ok\n\n"
    )
    iters = _iterations(loop)
    assert iters == [(1, "T0", "RETRY"), (2, "T1", "PASS")]


def test_trace_metrics_parse() -> None:
    trace = (
        "# Trace\n- total runs: 3\n- total cost: $0.0123\n"
        "- total tokens: {'input': 900, 'output': 400}\n- elapsed: 4.2s\n"
    )
    m = _trace_metrics(trace)
    assert m["cost"] == "0.0123"
    assert m["runs"] == "3"
    assert m["elapsed"] == "4.2s"


def test_render_trajectory_lists_iterations(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_test")
    session.append("loop.md", "## Iteration 1\n- model: flash (tier 0)\n- verdict: RETRY — x\n\n")
    session.append("loop.md", "## Iteration 2\n- model: qwen (tier 1)\n- verdict: PASS — ok\n\n")
    out = render_trajectory(session)
    assert "1. T0 · RETRY" in out
    assert "2. T1 · PASS" in out


def test_prd_phases_parse() -> None:
    md = "- clarify\n- finalize · 3 stories\n- run · direct\n"
    assert _prd_phases(md) == [
        ("clarify", ""),
        ("finalize", "3 stories"),
        ("run", "direct"),
    ]


def test_render_trajectory_includes_prd_phases(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_test")
    session.append("prd_phases.md", "- clarify")
    session.append("prd_phases.md", "- finalize · 2 stories")
    out = render_trajectory(session)
    assert "P1. clarify" in out
    assert "P2. finalize · 2 stories" in out


def test_render_trajectory_prd_then_iterations(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_test")
    session.append("prd_phases.md", "- run · direct")
    session.append("loop.md", "## Iteration 1\n- model: flash (tier 0)\n- verdict: PASS — ok\n\n")
    out = render_trajectory(session)
    assert "P1. run · direct" in out
    assert "1. T0 · PASS" in out


def test_render_overview_trajectory_shows_prd_phases_without_iterations(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_test")
    session.append("prd_phases.md", "- clarify")
    session.append("prd_phases.md", "- finalize · 1 stories")
    out = render_overview(session, "REFINING")
    assert "Trajectory: clarify → finalize" in out


def test_render_iteration_includes_runner_and_eval(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_test")
    session.append("loop.md", "## Iteration 2\n- model: qwen (tier 1)\n- verdict: PASS — ok\n\n")
    session.write("runs/iter-2.md", "# Run output\nprint('hello world')\n")
    session.append("eval.md", "### Iter 2: PASS\n**Reason:** prints hello\nRAW2\n\n")
    out = render_iteration(session, 2)
    assert "runner output" in out
    assert "print('hello world')" in out
    assert "prints hello" in out
    assert render_iteration(session, 99) == "no iteration 99."


# --- progress + TUI --------------------------------------------------------


def _seed_session(session: Session) -> None:
    session.update_index("# Session\n")
    session.set_status("completed", strategy="raphael", tasks=1, max_iterations=5, stage="done")
    session.write("plan.md", "# Plan\n\n1. write hello.py\n")
    session.append("loop.md", "## Iteration 1\n- model: flash (tier 0)\n- verdict: RETRY — x\n\n")
    session.append("loop.md", "## Iteration 2\n- model: qwen (tier 1)\n- verdict: PASS — ok\n\n")
    session.write("runs/iter-2.md", "# Run output\nprint('hello world')\n")
    session.append("eval.md", "### Iter 2: PASS\n**Reason:** prints hello\nRAW\n\n")
    session.write("trace.md", "# Trace\n- total runs: 2\n- total cost: $0.0020\n")


def test_analyze_tui_headless(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
    import asyncio

    from splinter.tui import AnalyzeApp

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_test")
    _seed_session(session)

    async def drive() -> None:
        app = AnalyzeApp(session)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("down", "down", "down", "down")
            await pilot.pause()
            await pilot.press("r")
            await pilot.pause()
            await pilot.press("q")

    asyncio.run(drive())


def test_run_tui_streams_log_and_overview(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    import asyncio
    import logging

    from splinter.tui import RunApp

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_run")
    session.update_index("# run\n")
    session.set_status("running", strategy="raphael", tasks=1, stage="run")

    # Stub pipeline work: emit a couple of log lines, then complete.
    def fake_pipeline(**kwargs: object) -> int:
        log = logging.getLogger("splinter.test")
        log.info("doing step one")
        log.info("doing step two")
        return 0

    monkeypatch.setattr("splinter.pipeline.run_pipeline", fake_pipeline)

    async def drive() -> None:
        app = RunApp(session, {})
        async with app.run_test() as pilot:
            await pilot.pause()
            for _ in range(10):
                await pilot.pause(0.05)
                if app.workers and all(
                    w.state.name in ("SUCCESS", "ERROR") for w in app.workers
                ):
                    break
            await pilot.pause()
            assert app.rc == 0
            await pilot.press("q")

    asyncio.run(drive())


def test_run_tui_captures_failure(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    import asyncio

    from splinter.tui import RunApp

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_fail")
    session.update_index("# f\n")
    session.set_status("running", strategy="raphael", tasks=1, stage="run")

    def boom(**kwargs: object) -> int:
        raise RuntimeError("provider exploded")

    monkeypatch.setattr("splinter.pipeline.run_pipeline", boom)

    async def drive() -> None:
        app = RunApp(session, {})
        async with app.run_test() as pilot:
            await pilot.pause()
            for _ in range(10):
                await pilot.pause(0.05)
                if app.workers and all(
                    w.state.name in ("SUCCESS", "ERROR") for w in app.workers
                ):
                    break
            await pilot.pause()
            assert app.rc == 1
            assert "provider exploded" in app.error
            await pilot.press("q")

    asyncio.run(drive())


def test_session_picker_navigate_delete_open(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    import asyncio

    from textual.widgets import DataTable

    from splinter.tui import SessionPicker

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    for sid in ("ses_a", "ses_b", "ses_c"):
        s = Session(sid)
        s.update_index(f"# {sid}\n")
        s.set_status("completed")

    async def drive() -> None:
        app = SessionPicker()
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.query_one(DataTable)
            assert table.row_count == 3
            await pilot.press("down")  # select the 2nd row
            await pilot.press("d")  # delete it
            await pilot.pause()
            assert table.row_count == 2
            assert set(list_sessions()) == {"ses_a", "ses_c"}
            await pilot.press("enter")  # open highlighted session
            await pilot.pause()
        assert app.return_value in {"ses_a", "ses_c"}

    asyncio.run(drive())
