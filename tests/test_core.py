from __future__ import annotations

import random
from pathlib import Path

import pytest

from splinter.agents.gate import GateResult
from splinter.agents.runner import Task, resolve_model, resolve_variant
from splinter.analyze import (
    _collapse_phases,
    _escalations,
    _iterations,
    _prd_phases,
    _run_state,
    _task_iters,
    _trace_metrics,
    _trajectory_lines,
    render_iteration,
    render_overview,
    render_trajectory,
)
from splinter.configure import (
    DEFAULT_CONFIG,
    gate_default_for,
    gate_default_languages,
    init_prompt_templates,
)
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


@pytest.fixture
def isolated_ladder(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> "object":
    """load_ladder() with NO project config override — pins ladder.yaml defaults.

    Runs from an empty cwd so the developer's untracked ./.splinter/config.yaml
    can't bleed its personal picks into these default-asserting tests.
    """
    monkeypatch.chdir(tmp_path)
    return load_ladder()


def test_ladder_loads(isolated_ladder: "object") -> None:
    ladder = isolated_ladder
    assert len(ladder.tiers) == 6
    assert ladder.tiers[0].name == "easy"
    assert ladder.tiers[4].name == "critical"
    assert ladder.tiers[5].name == "last-resort"
    assert len(ladder.all_model_ids()) > 0
    # floor is deepseek-v4-pro; workhorse rung (T1) switches to minimax-m3
    assert ladder.tiers[0].models[0] == "opencode-go/deepseek-v4-pro"
    assert ladder.tiers[1].models[0] == "opencode-go/minimax-m3"
    # per-tier reasoning variants: cheap floor, then climb to max on the open rungs
    assert ladder.tier_variant(0) == "low"
    assert ladder.tier_variant(1) == "high"
    assert ladder.tier_variant(2) == "high"
    assert ladder.tier_variant(3) == "max"
    assert ladder.tier_variant(4) == "high"
    assert ladder.tier_variant(5) == "max"
    # T4 critical is the strong open runner (qwen); T5 last-resort is Claude
    assert ladder.tier_by_level(4).models[0] == "opencode-go/qwen3.7-plus"
    assert ladder.tier_by_level(4).provider == "opencode"
    assert ladder.tier_by_level(5).models[0] == "sonnet"
    assert ladder.tier_by_level(5).provider == "claude"


def test_ladder_tier_by_level() -> None:
    ladder = load_ladder()
    t0 = ladder.tier_by_level(0)
    assert t0.name == "easy"
    t4 = ladder.tier_by_level(4)
    assert t4.name == "critical"


def test_normal_effort_starts_at_minimax_m3(isolated_ladder: "object") -> None:
    ladder = isolated_ladder
    em = ladder.effort_mapping("normal")
    assert em is not None and em.start_tier == 1
    model_id, provider = resolve_model(em.start_tier, ladder)
    assert provider == "opencode"
    assert model_id == "opencode-go/minimax-m3"


def test_localizer_roster(isolated_ladder: "object") -> None:
    ladder = isolated_ladder
    assert ladder.localizer_recall_model == "opencode/deepseek-v4-flash-free"
    assert ladder.localizer_recall_large_model == "opencode-go/minimax-m3"
    assert ladder.localizer_precision_model == "opencode/deepseek-v4-flash-free"
    assert ladder.localizer_recall_fallback_model == "haiku"
    assert ladder.localizer_agent == "explore"


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
    assert ladder.eval_model == "opus"  # untouched step keeps its ladder.yaml default


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
            # tier 0 overridden to high; blank tiers keep the ladder.yaml default.
            "tiers": ["high", "", "", "", ""],
        },
    )
    ladder = load_ladder()
    assert ladder.planner_effort == "max"
    assert ladder.eval_effort == "low"
    assert ladder.localizer_recall_variant == "high"
    assert ladder.tier_variant(0) == "high"  # config override wins
    assert ladder.tier_variant(1) == "high"  # blank → ladder.yaml default (high)


def test_write_gate_checks_roundtrip(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
    from splinter.agents.gate import _config_gate_checks
    from splinter.configure import write_gate_checks

    monkeypatch.chdir(tmp_path)
    checks = [
        {"name": "ruff", "cmd": "ruff check", "when": "always"},
        {"name": "mypy", "cmd": "mypy", "when": "always"},
    ]
    write_gate_checks(checks)
    result = _config_gate_checks(str(tmp_path))
    assert result == checks


def test_write_gate_checks_empty_list(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
    from splinter.configure import load_config, write_gate_checks

    monkeypatch.chdir(tmp_path)
    write_gate_checks([])
    config = load_config()
    assert config["gate_checks"] == []


def test_write_gate_checks_preserves_model_blocks(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from splinter.configure import load_config, write_gate_checks, write_model_config

    monkeypatch.chdir(tmp_path)
    models = {"planner": "opus", "tiers": ["haiku", "sonnet"]}
    efforts = {"planner": "high", "tiers": ["low", "high"]}
    timeouts = {"planner": 3600, "tiers": [1800, 3600]}
    write_model_config(models, efforts, timeouts=timeouts)

    checks = [{"name": "test", "cmd": "pytest", "when": "always"}]
    write_gate_checks(checks)

    config = load_config()
    assert config["models"] == models
    assert config["efforts"] == efforts
    assert config["timeouts"] == timeouts
    assert config["gate_checks"] == checks


def test_write_model_then_gate_save_order(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from splinter.configure import load_config, write_gate_checks, write_model_config

    monkeypatch.chdir(tmp_path)
    models = {"planner": "sonnet", "tiers": ["haiku"]}
    write_model_config(models)

    checks = [{"name": "lint", "cmd": "ruff", "when": "always"}]
    write_gate_checks(checks)

    config = load_config()
    assert "models" in config
    assert "gate_checks" in config
    assert config["models"]["planner"] == "sonnet"
    assert config["gate_checks"] == checks


def test_write_gate_checks_go_preset(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
    from splinter.configure import load_config, write_gate_checks

    monkeypatch.chdir(tmp_path)
    go_checks = gate_default_for("go")
    write_gate_checks(go_checks)
    config = load_config()
    assert config["gate_checks"] == go_checks


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


def test_ladder_effort_mapping(isolated_ladder: "object") -> None:
    ladder = isolated_ladder
    em = ladder.effort_mapping("trivial")
    assert em is not None
    assert em.start_tier == 0
    # trivial tasks start on the cheapest rung at low reasoning.
    assert em.variant == "low"


def test_resolve_variant_auto(isolated_ladder: "object") -> None:
    ladder = isolated_ladder
    task = Task(description="test", acceptance="test", effort="hard", reasoning_effort="auto")
    v = resolve_variant(task, None, ladder)
    # hard → starts at T3, max reasoning.
    assert v == "max"


def test_resolve_variant_override() -> None:
    ladder = load_ladder()
    task = Task(description="test", acceptance="test", effort="normal", reasoning_effort="auto")
    v = resolve_variant(task, "max", ladder)
    assert v == "max"


def test_resolve_model(isolated_ladder: "object") -> None:
    ladder = isolated_ladder
    model_id, provider = resolve_model(0, ladder)
    assert provider == "opencode"
    assert model_id.startswith("opencode-go/")


def test_resolve_model_claude(isolated_ladder: "object") -> None:
    ladder = isolated_ladder
    # sonnet (claude) is the last-resort top rung (level 5); T4 is qwen (opencode).
    model_id, provider = resolve_model(5, ladder)
    assert provider == "claude"
    assert model_id == "sonnet"
    assert resolve_model(4, ladder)[1] == "opencode"


def test_rewrite_runners_claude(isolated_ladder: "object") -> None:
    from splinter.enums import Variant
    from splinter.models.roster import rewrite_runners_claude

    ladder = isolated_ladder
    rewrite_runners_claude(ladder)
    for level in range(6):
        model_id, provider = resolve_model(level, ladder)
        assert provider == "claude"
        assert model_id == "sonnet"
        assert ladder.tier_variant(level) == Variant.HIGH


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


def test_list_sessions_newest_first(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
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
        text="output",
        model="test-model",
        tier=0,
        tokens={"input": 100, "output": 50},
        cost=0.01,
        raw={},
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


def test_gate_default_for_python() -> None:
    checks = gate_default_for("python")
    assert len(checks) == 3
    names = [c["name"] for c in checks]
    assert "ruff" in names
    assert "mypy" in names
    assert "pytest" in names
    # Verify commands and when conditions
    ruff_check = next(c for c in checks if c["name"] == "ruff")
    assert ruff_check["cmd"] == "ruff check ."
    assert ruff_check["when"] == "always"
    mypy_check = next(c for c in checks if c["name"] == "mypy")
    assert mypy_check["cmd"] == "mypy ."
    assert mypy_check["when"] == "always"
    pytest_check = next(c for c in checks if c["name"] == "pytest")
    assert pytest_check["cmd"] == "pytest"
    assert pytest_check["when"] == "tests_exist"


def test_gate_default_for_go() -> None:
    checks = gate_default_for("go")
    assert len(checks) == 3
    names = [c["name"] for c in checks]
    assert "gofmt" in names
    assert "go-vet" in names
    assert "go-test" in names
    # All should be "always"
    for check in checks:
        assert check["when"] == "always"


def test_gate_default_for_rust() -> None:
    checks = gate_default_for("rust")
    assert len(checks) == 3
    names = [c["name"] for c in checks]
    assert "fmt" in names
    assert "clippy" in names
    assert "test" in names
    # All should be "always"
    for check in checks:
        assert check["when"] == "always"


def test_gate_default_for_typescript() -> None:
    checks = gate_default_for("typescript")
    assert len(checks) == 2
    names = [c["name"] for c in checks]
    assert "tsc" in names
    assert "eslint" in names
    # All should be "always"
    for check in checks:
        assert check["when"] == "always"


def test_gate_default_for_javascript_npm() -> None:
    checks = gate_default_for("javascript-npm")
    assert len(checks) == 2
    names = [c["name"] for c in checks]
    assert "lint" in names
    assert "test" in names


def test_gate_default_for_javascript_pnpm() -> None:
    checks = gate_default_for("javascript-pnpm")
    assert len(checks) == 2
    names = [c["name"] for c in checks]
    assert "biome" in names
    assert "test" in names
    biome_check = next(c for c in checks if c["name"] == "biome")
    assert biome_check["cmd"] == "biome check ."


def test_gate_default_for_javascript_yarn() -> None:
    checks = gate_default_for("javascript-yarn")
    assert len(checks) == 2
    names = [c["name"] for c in checks]
    assert "lint" in names
    assert "test" in names


def test_gate_default_for_node() -> None:
    checks = gate_default_for("node")
    assert len(checks) == 1
    assert checks[0]["name"] == "test"
    assert checks[0]["cmd"] == "npm test"


def test_gate_default_for_unknown_language() -> None:
    checks = gate_default_for("unknown")
    assert checks == []
    checks = gate_default_for("cobol")
    assert checks == []


def test_gate_default_for_copy_semantics() -> None:
    """Mutating the returned list/dict does not affect subsequent calls."""
    first = gate_default_for("python")
    first.append({"name": "extra", "cmd": "extra cmd", "when": "always"})
    first[0]["cmd"] = "modified"
    second = gate_default_for("python")
    assert len(second) == 3
    assert second[0]["cmd"] == "ruff check ."
    assert all(c.get("name") != "extra" for c in second)


def test_gate_default_languages() -> None:
    langs = gate_default_languages()
    assert isinstance(langs, list)
    assert len(langs) == 16
    expected = [
        "cpp",
        "csharp",
        "go",
        "java",
        "javascript-npm",
        "javascript-pnpm",
        "javascript-yarn",
        "kotlin",
        "node",
        "php",
        "python",
        "ruby",
        "rust",
        "rust-proto",
        "swift",
        "typescript",
    ]
    assert langs == expected


def test_gate_defaults_no_multicommand_cmds() -> None:
    """Every cmd must be a single bare command (no && ; |)."""
    from splinter.configure import LANGUAGE_GATE_DEFAULTS

    for lang, checks in LANGUAGE_GATE_DEFAULTS.items():
        for check in checks:
            cmd = check.get("cmd", "")
            assert "&&" not in cmd, f"{lang} check '{check.get('name')}' contains &&"
            assert ";" not in cmd, f"{lang} check '{check.get('name')}' contains ;"
            assert "|" not in cmd, f"{lang} check '{check.get('name')}' contains |"


def test_gate_defaults_required_fields() -> None:
    """Every entry has name, cmd, when fields with valid when values."""
    from splinter.configure import LANGUAGE_GATE_DEFAULTS

    for lang, checks in LANGUAGE_GATE_DEFAULTS.items():
        for check in checks:
            assert "name" in check, f"{lang} check missing 'name'"
            assert "cmd" in check, f"{lang} check missing 'cmd'"
            assert "when" in check, f"{lang} check missing 'when'"
            assert check["when"] in (
                "always",
                "tests_exist",
                "proto_changed",
            ), f"{lang} check has invalid 'when': {check['when']}"


def test_gate_defaults_tests_exist_only_in_python() -> None:
    """Only python language may use when: tests_exist."""
    from splinter.configure import LANGUAGE_GATE_DEFAULTS

    _MULTI_LANG_TESTS_EXIST = {
        "python",
        "ruby",
        "cpp",
        "swift",
        "csharp",
        "php",
        "java",
        "kotlin",
    }
    for lang, checks in LANGUAGE_GATE_DEFAULTS.items():
        for check in checks:
            if check.get("when") == "tests_exist":
                assert lang in _MULTI_LANG_TESTS_EXIST, (
                    f"Language '{lang}' uses tests_exist but is not in the allowed set"
                )


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
            "output": "45",  # string count
            "reasoning": 3.0,  # float
            "cache": {"read": 0, "write": 0},  # nested dict — must be skipped
            "bogus": None,  # must be skipped
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


def test_render_prefers_override(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
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


def test_session_status_roundtrip(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
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


def test_run_state_alive_vs_dead(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
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
    assert "TRAJECTORY" in out
    assert "clarify" in out and "finalize" in out
    assert "→" in out


# --- _collapse_phases, _task_iters, _escalations, new trajectory layout ----


def test_collapse_phases_consecutive() -> None:
    phases = [("refine", ""), ("refine", ""), ("refine", "")]
    assert _collapse_phases(phases) == [("refine", 3)]


def test_collapse_phases_non_consecutive() -> None:
    phases = [("a", ""), ("b", ""), ("a", "")]
    assert _collapse_phases(phases) == [("a", 1), ("b", 1), ("a", 1)]


def test_collapse_phases_single() -> None:
    assert _collapse_phases([("clarify", "detail")]) == [("clarify", 1)]


def test_collapse_phases_empty() -> None:
    assert _collapse_phases([]) == []


def test_task_iters_single_task() -> None:
    loop = (
        "## Iteration 1\n- model: flash (tier 0)\n- verdict: RETRY\n\n"
        "## Iteration 2\n- model: qwen (tier 1)\n- verdict: PASS\n\n"
    )
    result = _task_iters(loop)
    assert len(result) == 1
    task_no, _title, iters = result[0]
    assert task_no == 1
    assert iters == [(1, "T0", "RETRY"), (2, "T1", "PASS")]


def test_task_iters_multi_task() -> None:
    loop = (
        "# Task 1/2: first\n"
        "## Iteration 1\n- model: flash (tier 0)\n- verdict: PASS\n\n"
        "# Task 2/2: second\n"
        "## Iteration 1\n- model: qwen (tier 1)\n- verdict: RETRY\n\n"
        "## Iteration 2\n- model: qwen (tier 1)\n- verdict: PASS\n\n"
    )
    result = _task_iters(loop)
    assert len(result) == 2
    assert result[0][0] == 1
    assert result[0][2] == [(1, "T0", "PASS")]
    assert result[1][0] == 2
    assert result[1][2] == [(1, "T1", "RETRY"), (2, "T1", "PASS")]


def test_task_iters_reindexes_from_one() -> None:
    loop = (
        "# Task 1/2: a\n"
        "## Iteration 5\n- model: flash (tier 0)\n- verdict: PASS\n\n"
        "# Task 2/2: b\n"
        "## Iteration 6\n- model: qwen (tier 1)\n- verdict: PASS\n\n"
    )
    result = _task_iters(loop)
    assert result[0][2][0][0] == 1
    assert result[1][2][0][0] == 1


def test_escalations_detects_tier_change() -> None:
    iters = [(1, "T1", "PASS"), (2, "T1", "PASS"), (3, "T3", "PASS")]
    assert _escalations(iters) == {2}


def test_escalations_no_false_positive() -> None:
    iters = [(1, "T1", "PASS"), (2, "T1", "RETRY"), (3, "T1", "PASS")]
    assert _escalations(iters) == set()


def test_escalations_empty() -> None:
    assert _escalations([]) == set()
    assert _escalations([(1, "T1", "PASS")]) == set()


def test_trajectory_lines_multi_task_headers(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_test")
    loop = (
        "# Task 1/2: first\n"
        "## Iteration 1\n- model: flash (tier 0)\n- verdict: PASS\n\n"
        "# Task 2/2: second\n"
        "## Iteration 1\n- model: qwen (tier 1)\n- verdict: PASS\n\n"
    )
    session.write("loop.md", loop)
    iters = _iterations(loop)
    out = "\n".join(_trajectory_lines(session, iters))
    assert "task 1" in out
    assert "task 2" in out


def test_trajectory_lines_single_task_no_header(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_test")
    loop = "## Iteration 1\n- model: flash (tier 0)\n- verdict: PASS\n\n"
    session.write("loop.md", loop)
    iters = _iterations(loop)
    out = "\n".join(_trajectory_lines(session, iters))
    assert "task 1" not in out


def test_trajectory_lines_escalation_marker(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_test")
    loop = (
        "## Iteration 1\n- model: flash (tier 0)\n- verdict: RETRY\n\n"
        "## Iteration 2\n- model: qwen (tier 2)\n- verdict: PASS\n\n"
    )
    session.write("loop.md", loop)
    iters = _iterations(loop)
    out = "\n".join(_trajectory_lines(session, iters))
    assert "⤴" in out


def test_trajectory_lines_collapse_phases(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_test")
    session.write("prd_phases.md", "- refine\n- refine\n- refine\n")
    out = "\n".join(_trajectory_lines(session, []))
    assert "x3" in out


def test_overview_md_trajectory_task_bullets(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from splinter.tui import _overview_md

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_test")
    loop = (
        "# Task 1/2: first\n"
        "## Iteration 1\n- model: flash (tier 0)\n- verdict: PASS\n\n"
        "# Task 2/2: second\n"
        "## Iteration 1\n- model: qwen (tier 1)\n- verdict: RETRY\n\n"
    )
    session.write("loop.md", loop)
    out = _overview_md(session, "RUNNING")
    assert "**Task 1**" in out
    assert "**Task 2**" in out
    assert "**Run**" in out
    assert "T0·PASS" not in out


def test_overview_md_trajectory_prd_line(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from splinter.tui import _overview_md

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_test")
    session.write("prd_phases.md", "- clarify\n- refine\n- refine\n- finalize\n")
    out = _overview_md(session, "RUNNING")
    assert "**PRD**" in out
    assert "refine x2" in out
    assert "clarify" in out
    assert "finalize" in out


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
                if app.workers and all(w.state.name in ("SUCCESS", "ERROR") for w in app.workers):
                    break
            await pilot.pause()
            assert app.rc == 0
            await pilot.press("q")

    asyncio.run(drive())


def test_run_tui_captures_failure(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
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
                if app.workers and all(w.state.name in ("SUCCESS", "ERROR") for w in app.workers):
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


# --- planner: parse_stories + assign_target_files ----------------------------


_MULTI_STORY_PRD = """\
---
feature: multi
strategy: direct
---

# Multi-Story Feature

### US-001: Create data model
**Description:** Define the core data model with id and name fields.

**Splinter hints:**
- effort: trivial
- eval_skill: run_python

**Acceptance Criteria:**
- [ ] Model class exists
- [ ] Has id and name fields

### US-002: Add validation
**Description:** Add input validation for the data model.

Depends on US-001

**Splinter hints:**
- effort: normal
- eval_skill: run_python

**Acceptance Criteria:**
- [ ] Validation rejects empty names
- [ ] Validation rejects duplicate ids

### US-003: Build API endpoint
**Description:** Expose the data model via an API endpoint.

Depends on US-001
Blocked until US-002

**Splinter hints:**
- effort: hard
- eval_skill: run_python

**Acceptance Criteria:**
- [ ] GET endpoint returns model
- [ ] POST endpoint creates model
"""


def test_planner_parse_stories_sets_id() -> None:
    from splinter.agents.planner import parse_stories

    tasks = parse_stories(_MULTI_STORY_PRD)
    assert len(tasks) == 3
    assert tasks[0].id == "US-001"
    assert tasks[1].id == "US-002"
    assert tasks[2].id == "US-003"


def test_planner_parse_stories_deps() -> None:
    from splinter.agents.planner import parse_stories

    tasks = parse_stories(_MULTI_STORY_PRD)
    assert tasks[0].deps is None
    assert tasks[1].deps == ["US-001"]
    assert tasks[2].deps == ["US-001", "US-002"]


def test_planner_parse_stories_back_compat() -> None:
    from splinter.agents.planner import parse_stories

    tasks = parse_stories(_MULTI_STORY_PRD)
    assert "US-001" in tasks[0].description
    assert "core data model" in tasks[0].description
    assert tasks[0].effort == "trivial"
    assert tasks[0].eval_skill == "run_python"
    assert "Model class exists" in tasks[0].acceptance
    assert tasks[2].effort == "hard"


def test_planner_parse_stories_hello_world_prd() -> None:
    from splinter.agents.planner import parse_stories

    prd_text = Path("samples/hello-world-prd.md").read_text()
    tasks = parse_stories(prd_text)
    assert len(tasks) == 1
    assert tasks[0].id == "US-001"
    assert tasks[0].effort == "trivial"
    assert tasks[0].eval_skill == "run_python"
    assert "hello" in tasks[0].acceptance.lower()


def test_planner_assign_target_files_keyword_match() -> None:
    from splinter.agents.localizer import CodeAnchor
    from splinter.agents.planner import assign_target_files, parse_stories

    tasks = parse_stories(_MULTI_STORY_PRD)
    anchors = [
        CodeAnchor(
            file="models/user.py",
            symbol="UserModel",
            reason="core data model definition",
            confidence=0.9,
        ),
        CodeAnchor(
            file="validators/user.py",
            symbol="validate_user",
            reason="input validation for model",
            confidence=0.8,
        ),
        CodeAnchor(
            file="api/routes.py",
            symbol="api_endpoint",
            reason="API endpoint for data model",
            confidence=0.7,
        ),
    ]
    assign_target_files(tasks, anchors)
    assert tasks[0].target_files is not None
    assert "models/user.py" in tasks[0].target_files
    assert tasks[2].target_files is not None
    assert "api/routes.py" in tasks[2].target_files


def test_planner_assign_target_files_fallback() -> None:
    from splinter.agents.localizer import CodeAnchor
    from splinter.agents.planner import assign_target_files

    tasks = [Task(description="unrelated task", acceptance="works")]
    anchors = [
        CodeAnchor(file="foo.py", symbol="Foo", reason="something else", confidence=0.5),
        CodeAnchor(file="bar.py", symbol="Bar", reason="another thing", confidence=0.5),
    ]
    assign_target_files(tasks, anchors)
    assert tasks[0].target_files == ["foo.py", "bar.py"]


def test_planner_assign_target_files_empty_anchors() -> None:
    from splinter.agents.planner import assign_target_files

    tasks = [Task(description="test", acceptance="test")]
    assign_target_files(tasks, [])
    assert tasks[0].target_files is None


def test_planner_plan_function(tmp_path: Path) -> None:
    from splinter.agents.localizer import CodeAnchor
    from splinter.agents.planner import plan

    prd_file = tmp_path / "test-prd.md"
    prd_file.write_text(_MULTI_STORY_PRD)
    anchors = [
        CodeAnchor(
            file="models/user.py",
            symbol="UserModel",
            reason="core data model",
            confidence=0.9,
        ),
    ]
    tasks, strategy = plan(str(prd_file), anchors)
    assert strategy == "direct"
    assert len(tasks) == 3
    assert tasks[0].id == "US-001"
    assert tasks[0].target_files is not None


# ── US-003: PRD grounding before Q&A ─────────────────────────────────────────


def test_ground_localization_returns_grounding_string(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """ground_localization returns non-empty string when localize returns hot anchors."""
    from splinter import prd_session
    from splinter.agents.localizer import CodeAnchor
    from splinter.memory.session import Session

    monkeypatch.chdir(tmp_path)
    session = Session()
    ladder = load_ladder()
    anchor = CodeAnchor(
        file="app.py", symbol="main", reason="entry point", confidence=0.9, relevance="hot"
    )
    monkeypatch.setattr("splinter.agents.localizer.localize", lambda *a, **kw: [anchor])
    result = prd_session.ground_localization(session, ladder, "build something")
    assert "app.py" in result
    assert "main" in result


def test_ground_localization_returns_empty_on_failure(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """ground_localization returns '' when localize raises — never blocks PRD flow."""
    from splinter import prd_session
    from splinter.memory.session import Session

    monkeypatch.chdir(tmp_path)
    session = Session()
    ladder = load_ladder()

    monkeypatch.setattr(
        "splinter.agents.localizer.localize",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("model down")),
    )
    result = prd_session.ground_localization(session, ladder, "build something")
    assert result == ""


def test_open_questions_embeds_grounding(monkeypatch: "pytest.MonkeyPatch") -> None:
    """open_questions prompt contains grounding section when localization= non-empty."""
    from splinter import prd_session

    captured: list[str] = []

    def _fake_ask(prompt: str, *, resume: str | None, session: object = None) -> prd_session.Turn:
        captured.append(prompt)
        return prd_session.Turn(text="Q1?", session_id="sid")

    monkeypatch.setattr(prd_session, "_ask", _fake_ask)
    prd_session.open_questions("## Draft\ndo stuff", localization="app.py — main\n  entry")
    assert "## Codebase Localization (grounding)" in captured[0]
    assert "app.py — main" in captured[0]


def test_open_questions_omits_grounding_when_empty(monkeypatch: "pytest.MonkeyPatch") -> None:
    """open_questions prompt omits grounding section when localization=''."""
    from splinter import prd_session

    captured: list[str] = []

    def _fake_ask(prompt: str, *, resume: str | None, session: object = None) -> prd_session.Turn:
        captured.append(prompt)
        return prd_session.Turn(text="Q1?", session_id="sid")

    monkeypatch.setattr(prd_session, "_ask", _fake_ask)
    prd_session.open_questions("## Draft\ndo stuff", localization="")
    assert "## Codebase Localization" not in captured[0]


def test_generate_prd_embeds_grounding(monkeypatch: "pytest.MonkeyPatch") -> None:
    """generate_prd prompt contains grounding section when localization= non-empty."""
    from splinter import prd_session

    captured: list[str] = []

    def _fake_ask(prompt: str, *, resume: str | None, session: object = None) -> prd_session.Turn:
        captured.append(prompt)
        return prd_session.Turn(text="---\nfeature: x\n---\n", session_id="sid")

    monkeypatch.setattr(prd_session, "_ask", _fake_ask)
    prd_session.generate_prd("build an API", localization="api.py — router")
    assert "## Codebase Localization (grounding)" in captured[0]
    assert "api.py — router" in captured[0]


def test_generate_prd_omits_grounding_when_empty(monkeypatch: "pytest.MonkeyPatch") -> None:
    """generate_prd prompt omits grounding section when localization=''."""
    from splinter import prd_session

    captured: list[str] = []

    def _fake_ask(prompt: str, *, resume: str | None, session: object = None) -> prd_session.Turn:
        captured.append(prompt)
        return prd_session.Turn(text="---\nfeature: x\n---\n", session_id="sid")

    monkeypatch.setattr(prd_session, "_ask", _fake_ask)
    prd_session.generate_prd("build an API", localization="")
    assert "## Codebase Localization" not in captured[0]


def test_finalize_embeds_grounding(monkeypatch: "pytest.MonkeyPatch") -> None:
    """finalize prompt contains grounding section when localization= non-empty."""
    from splinter import prd_session

    captured: list[str] = []

    def _fake_ask(prompt: str, *, resume: str | None, session: object = None) -> prd_session.Turn:
        captured.append(prompt)
        return prd_session.Turn(text="---\nfeature: x\n---\n", session_id="sid")

    monkeypatch.setattr(prd_session, "_ask", _fake_ask)
    prd_session.finalize(
        resume="sid", strategy="direct", autodecide=False, localization="config.py — Config"
    )
    assert "## Codebase Localization (grounding)" in captured[0]
    assert "config.py — Config" in captured[0]


def test_finalize_omits_grounding_when_empty(monkeypatch: "pytest.MonkeyPatch") -> None:
    """finalize prompt omits grounding section when localization=''."""
    from splinter import prd_session

    captured: list[str] = []

    def _fake_ask(prompt: str, *, resume: str | None, session: object = None) -> prd_session.Turn:
        captured.append(prompt)
        return prd_session.Turn(text="---\nfeature: x\n---\n", session_id="sid")

    monkeypatch.setattr(prd_session, "_ask", _fake_ask)
    prd_session.finalize(resume="sid", strategy="direct", autodecide=False, localization="")
    assert "## Codebase Localization" not in captured[0]


def test_refine_embeds_grounding(monkeypatch: "pytest.MonkeyPatch") -> None:
    """refine prompt contains grounding section when localization= non-empty."""
    from splinter import prd_session

    captured: list[str] = []

    def _fake_ask(prompt: str, *, resume: str | None, session: object = None) -> prd_session.Turn:
        captured.append(prompt)
        return prd_session.Turn(
            text="## Working Draft\n```\n---\nfeature: x\n---\n```\n## Open Questions\nNone",
            session_id="sid",
        )

    monkeypatch.setattr(prd_session, "_ask", _fake_ask)
    prd_session.refine("A, B, C", resume="sid", localization="router.py — handle")
    assert "## Codebase Localization (grounding)" in captured[0]
    assert "router.py — handle" in captured[0]


def test_refine_omits_grounding_when_empty(monkeypatch: "pytest.MonkeyPatch") -> None:
    """refine prompt omits grounding section when localization=''."""
    from splinter import prd_session

    captured: list[str] = []

    def _fake_ask(prompt: str, *, resume: str | None, session: object = None) -> prd_session.Turn:
        captured.append(prompt)
        return prd_session.Turn(
            text="## Working Draft\n```\n---\nfeature: x\n---\n```\n## Open Questions\nNone",
            session_id="sid",
        )

    monkeypatch.setattr(prd_session, "_ask", _fake_ask)
    prd_session.refine("A, B, C", resume="sid", localization="")
    assert "## Codebase Localization" not in captured[0]


def test_localize_cache_skips_search(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
    """localize short-circuits without calling _run_search_tools when cache exists."""
    from splinter.agents import localizer
    from splinter.memory.knowledge import KnowledgeStore
    from splinter.memory.session import Session

    monkeypatch.chdir(tmp_path)
    session = Session()
    ladder = load_ladder()

    # Pre-populate session cache
    cache_md = (
        "# Localization\n\n"
        "file: app.py\n"
        "symbol: main\n"
        "reason: entry\n"
        "confidence: 0.9\n"
        "relevance: hot\n"
    )
    KnowledgeStore(session).write_note("localization", cache_md)

    search_called = False

    def _spy_search(*a: object, **kw: object) -> str:
        nonlocal search_called
        search_called = True
        return ""

    monkeypatch.setattr(localizer, "_run_search_tools", _spy_search)
    result = localizer.localize("rebuild everything", session, ladder)
    assert not search_called, "_run_search_tools must not fire when cache exists"
    assert len(result) > 0
    assert result[0].file == "app.py"


def test_no_ground_flag_skips_ground_localization(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """run_prd with no_ground=True never calls ground_localization."""
    from splinter import prd_session
    from splinter.prd import run_prd

    monkeypatch.chdir(tmp_path)

    ground_called = False

    def _spy_ground(*a: object, **kw: object) -> str:
        nonlocal ground_called
        ground_called = True
        return "app.py — main"

    monkeypatch.setattr(prd_session, "ground_localization", _spy_ground)

    fake_result = type("R", (), {"text": "Q1?", "raw": {"_session_id": "sid"}, "usage": {"input_tokens": 0, "output_tokens": 0}})()
    _prd_body = (
        "---\nfeature: x\nstrategy: direct\nkind: feature\ncreated: 2026-06-10\n---\n"
        "### US-001: X\n**Description:** d\n**Acceptance Criteria:**\n- [ ] c\n"
    )
    fake_result2 = type("R", (), {"text": _prd_body, "raw": {"_session_id": "sid"}, "usage": {"input_tokens": 0, "output_tokens": 0}})()

    import splinter.providers.claude_cli as cc

    def _fake_cc_run(*a: object, **kw: object) -> object:
        return fake_result if not kw.get("resume") else fake_result2

    monkeypatch.setattr(cc, "run", _fake_cc_run)
    monkeypatch.setattr("builtins.input", lambda *a: "1A")

    run_prd(description="add auth", no_ground=True)
    assert not ground_called, "ground_localization must not be called with --no-ground"


def _fake_cc_result(text: str) -> object:
    return type(
        "R",
        (),
        {
            "text": text,
            "raw": {"_session_id": "sid"},
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )()


def test_run_prd_abort_no_answers_leaves_no_empty_session(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Aborting at the answers prompt must garbage-collect the half-built session."""
    from splinter.memory.session import list_sessions
    from splinter.prd import run_prd

    monkeypatch.chdir(tmp_path)

    import splinter.providers.claude_cli as cc

    monkeypatch.setattr(cc, "run", lambda *a, **kw: _fake_cc_result("Q1?"))
    monkeypatch.setattr(cc, "_calc_cost", lambda *a, **kw: 0.0)
    monkeypatch.setattr("builtins.input", lambda *a: "")  # user gives no answers

    rc = run_prd(description="add auth", no_ground=True)
    assert rc == 1
    assert list_sessions() == [], "aborted PRD run must not litter an empty session"


def test_run_prd_keyboardinterrupt_cleans_up_empty_session(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Ctrl+C mid-run must clean up the empty session and re-raise."""
    import pytest

    from splinter.memory.session import list_sessions
    from splinter.prd import run_prd

    monkeypatch.chdir(tmp_path)

    import splinter.providers.claude_cli as cc

    monkeypatch.setattr(cc, "run", lambda *a, **kw: _fake_cc_result("Q1?"))
    monkeypatch.setattr(cc, "_calc_cost", lambda *a, **kw: 0.0)

    def _interrupt(*a: object) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", _interrupt)

    with pytest.raises(KeyboardInterrupt):
        run_prd(description="add auth", no_ground=True)
    assert list_sessions() == [], "interrupted PRD run must not litter an empty session"


_STUB_PRD = "---\nstrategy: cascade\n---\n# Test"
_REAL_PRD = "---\nstrategy: cascade\n---\n### US-001: Login\n**Description:** d\n"


def _seed_refining(session_dir_factory: "object", sid: str, prd: str, *, age_s: float) -> "object":
    """Build a refining session on disk with a backdated mtime."""
    import os
    from datetime import datetime, timezone

    from splinter.memory.session import Session

    s = Session(sid)
    s.write("prd.md", prd)
    s.set_status("refining", source="prd", phase="run")
    old = datetime.now(timezone.utc).timestamp() - age_s
    os.utime(s.dir, (old, old))
    return s


def test_prd_session_is_resumable_distinguishes_stub_from_real(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """A stub PRD with no user stories is not resumable; one with US-NNN is."""
    monkeypatch.chdir(tmp_path)
    from splinter.prd_session import prd_session_is_resumable

    stub = _seed_refining(None, "ses_stub", _STUB_PRD, age_s=0)
    real = _seed_refining(None, "ses_real", _REAL_PRD, age_s=0)
    assert prd_session_is_resumable(stub) is False
    assert prd_session_is_resumable(real) is True


def test_prune_dead_prd_sessions_removes_only_old_stub_refinements(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """GC drops aged stub refinements; spares real PRDs and freshly-touched ones."""
    monkeypatch.chdir(tmp_path)
    from splinter.memory.session import Session, list_sessions
    from splinter.prd_session import prune_dead_prd_sessions

    _seed_refining(None, "ses_old_stub", _STUB_PRD, age_s=300)  # junk → prune
    _seed_refining(None, "ses_real", _REAL_PRD, age_s=300)  # has stories → keep
    _seed_refining(None, "ses_fresh_stub", _STUB_PRD, age_s=5)  # too new → keep
    # A completed run with a stub PRD must never be touched.
    done = Session("ses_done")
    done.write("trace.md", "some run output")
    done.set_status("completed")

    pruned = prune_dead_prd_sessions(min_age_seconds=60.0)

    assert pruned == ["ses_old_stub"]
    remaining = set(list_sessions())
    assert remaining == {"ses_real", "ses_fresh_stub", "ses_done"}


# --- US-003: ConfigureApp gates section -------------------------------------


def test_configure_tui_lists_gate_checks(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
    import asyncio

    from splinter.tui import ConfigureApp

    monkeypatch.chdir(tmp_path)

    async def drive() -> None:
        app = ConfigureApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app._gate_checks == DEFAULT_CONFIG["gate_checks"]
            gate_rows = app.query("#gate-rows .gate-row")
            assert len(gate_rows) == len(DEFAULT_CONFIG["gate_checks"])

    asyncio.run(drive())


def test_configure_save_preserves_gate_checks(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    import asyncio

    from splinter.configure import load_config
    from splinter.tui import ConfigureApp

    monkeypatch.chdir(tmp_path)

    async def drive() -> None:
        app = ConfigureApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()
        assert app.saved

    asyncio.run(drive())
    config = load_config()
    assert config["gate_checks"] == DEFAULT_CONFIG["gate_checks"]


def test_configure_save_with_seeded_config(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    import asyncio

    import yaml

    from splinter.configure import load_config
    from splinter.tui import ConfigureApp

    monkeypatch.chdir(tmp_path)
    custom_checks = [
        {"name": "lint", "cmd": "ruff check .", "when": "always"},
        {"name": "types", "cmd": "mypy src", "when": "always"},
    ]
    splinter_dir = tmp_path / ".splinter"
    splinter_dir.mkdir()
    (splinter_dir / "config.yaml").write_text(
        yaml.dump({"gate_checks": custom_checks}, default_flow_style=False)
    )

    async def drive() -> None:
        app = ConfigureApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app._gate_checks == custom_checks
            gate_rows = app.query("#gate-rows .gate-row")
            assert len(gate_rows) == len(custom_checks)
            await pilot.press("s")
            await pilot.pause()
        assert app.saved

    asyncio.run(drive())
    config = load_config()
    assert config["gate_checks"] == custom_checks


def test_configure_tui_gate_delete(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
    import asyncio

    from textual.widgets import Button

    from splinter.configure import load_config
    from splinter.tui import ConfigureApp

    monkeypatch.chdir(tmp_path)

    async def drive() -> None:
        app = ConfigureApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            initial_count = len(app._gate_checks)
            btn = app.query_one("#gate_del_0", Button)
            btn.press()
            await pilot.pause()
            assert len(app._gate_checks) == initial_count - 1
            gate_rows = app.query("#gate-rows .gate-row")
            assert len(gate_rows) == initial_count - 1
            await pilot.press("s")
            await pilot.pause()
        assert app.saved

    asyncio.run(drive())
    config = load_config()
    assert len(config["gate_checks"]) == len(DEFAULT_CONFIG["gate_checks"]) - 1


def test_write_gate_checks_preserves_models(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from splinter.configure import load_config, write_gate_checks, write_model_config

    monkeypatch.chdir(tmp_path)
    write_model_config({"planner": "opus", "tiers": ["haiku"]})
    write_gate_checks([{"name": "ruff", "cmd": "ruff check", "when": "always"}])
    config = load_config()
    assert "models" in config
    assert "gate_checks" in config
    assert config["models"]["planner"] == "opus"
    assert config["gate_checks"][0]["name"] == "ruff"


def test_append_language_preset_rust(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
    import asyncio

    from textual.widgets import Button

    from splinter.configure import load_config
    from splinter.tui import ConfigureApp

    monkeypatch.chdir(tmp_path)

    async def drive() -> None:
        app = ConfigureApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            original_count = len(app._gate_checks)
            rust_preset = gate_default_for("rust")
            btn = app.query_one("#gate_preset", Button)
            btn.press()
            await pilot.pause()
            languages = gate_default_languages()
            rust_idx = languages.index("rust")
            for _ in range(rust_idx):
                await pilot.press("down")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert len(app._gate_checks) == original_count + len(rust_preset)
            rust_in_checks = any(
                c["name"] == "clippy" and "cargo clippy" in c["cmd"]
                for c in app._gate_checks
            )
            assert rust_in_checks
            await pilot.press("s")
            await pilot.pause()
        assert app.saved

    asyncio.run(drive())
    config = load_config()
    rust_checks = gate_default_for("rust")
    assert len(config["gate_checks"]) == len(DEFAULT_CONFIG["gate_checks"]) + len(
        rust_checks
    )


def test_append_two_languages_rust_then_go(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    import asyncio

    from textual.widgets import Button

    from splinter.tui import ConfigureApp

    monkeypatch.chdir(tmp_path)

    async def drive() -> None:
        app = ConfigureApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            base_count = len(app._gate_checks)
            rust_preset = gate_default_for("rust")
            go_preset = gate_default_for("go")
            btn = app.query_one("#gate_preset", Button)
            languages = gate_default_languages()
            rust_idx = languages.index("rust")
            go_idx = languages.index("go")
            btn.press()
            await pilot.pause()
            for _ in range(rust_idx):
                await pilot.press("down")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            count_after_rust = len(app._gate_checks)
            assert count_after_rust == base_count + len(rust_preset)
            btn.press()
            await pilot.pause()
            for _ in range(go_idx):
                await pilot.press("down")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            final_count = len(app._gate_checks)
            expected = base_count + len(rust_preset) + len(go_preset)
            assert final_count == expected
            go_in_checks = any(
                c["name"] == "go-test" and "go test" in c["cmd"] for c in app._gate_checks
            )
            assert go_in_checks

    asyncio.run(drive())


def test_append_duplicate_languages(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
    import asyncio

    from textual.widgets import Button

    from splinter.tui import ConfigureApp

    monkeypatch.chdir(tmp_path)

    async def drive() -> None:
        app = ConfigureApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            base_count = len(app._gate_checks)
            python_preset = gate_default_for("python")
            btn = app.query_one("#gate_preset", Button)
            languages = gate_default_languages()
            python_idx = languages.index("python")
            for append_num in range(2):
                btn.press()
                await pilot.pause()
                for _ in range(python_idx):
                    await pilot.press("down")
                await pilot.pause()
                await pilot.press("enter")
                await pilot.pause()
                expected_count = base_count + (append_num + 1) * len(python_preset)
                assert len(app._gate_checks) == expected_count

    asyncio.run(drive())


def test_append_cancel_leaves_unchanged(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
    import asyncio

    from textual.widgets import Button

    from splinter.tui import ConfigureApp

    monkeypatch.chdir(tmp_path)

    async def drive() -> None:
        app = ConfigureApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            original_count = len(app._gate_checks)
            original_checks = [dict(c) for c in app._gate_checks]
            btn = app.query_one("#gate_preset", Button)
            btn.press()
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert len(app._gate_checks) == original_count
            assert app._gate_checks == original_checks

    asyncio.run(drive())


def test_append_preset_persists_on_save(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
    import asyncio

    from textual.widgets import Button

    from splinter.configure import load_config
    from splinter.tui import ConfigureApp

    monkeypatch.chdir(tmp_path)

    async def drive() -> None:
        app = ConfigureApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            btn = app.query_one("#gate_preset", Button)
            btn.press()
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            saved_count = len(app._gate_checks)
            await pilot.press("s")
            await pilot.pause()
        assert app.saved
        assert saved_count > len(DEFAULT_CONFIG["gate_checks"])

    asyncio.run(drive())
    config = load_config()
    assert len(config["gate_checks"]) == len(DEFAULT_CONFIG["gate_checks"]) + len(
        gate_default_for("python")
    )


# --- US-005: inline gate edit / add / delete ----------------------------------


def test_parse_gate_spec_single_cmd() -> None:
    from splinter.agents.gate import parse_gate_spec

    checks = parse_gate_spec("npm run build")
    assert checks == [{"name": "npm", "cmd": "npm run build", "when": "always"}]


def test_parse_gate_spec_multi_cmd() -> None:
    from splinter.agents.gate import parse_gate_spec

    checks = parse_gate_spec("npm run lint; npm test")
    assert len(checks) == 2
    assert checks[0] == {"name": "npm", "cmd": "npm run lint", "when": "always"}
    assert checks[1] == {"name": "npm", "cmd": "npm test", "when": "always"}


def test_write_model_config_gate_checks_empty(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from splinter.configure import load_config, write_model_config

    monkeypatch.chdir(tmp_path)
    write_model_config({"planner": "opus", "tiers": ["haiku"]}, gate_checks=[])
    config = load_config()
    assert config["gate_checks"] == []


def test_configure_tui_add_custom_gate(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    import asyncio

    from textual.widgets import Button, Input

    from splinter.configure import load_config
    from splinter.tui import ConfigureApp

    monkeypatch.chdir(tmp_path)

    async def drive() -> None:
        app = ConfigureApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("#gate_add_input", Input).value = "npm run build"
            await pilot.pause()
            app.query_one("#gate_add", Button).press()
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()
        assert app.saved

    asyncio.run(drive())
    config = load_config()
    assert any(
        c["name"] == "npm" and c["cmd"] == "npm run build" and c["when"] == "always"
        for c in config["gate_checks"]
    )


def test_configure_tui_edit_gate_cmd(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    import asyncio

    from textual.widgets import Input

    from splinter.configure import load_config
    from splinter.tui import ConfigureApp

    monkeypatch.chdir(tmp_path)

    async def drive() -> None:
        app = ConfigureApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("#gate_cmd_0", Input).value = "uv run ruff check --strict"
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()
        assert app.saved

    asyncio.run(drive())
    config = load_config()
    assert config["gate_checks"][0]["cmd"] == "uv run ruff check --strict"
    assert config["gate_checks"][1] == DEFAULT_CONFIG["gate_checks"][1]
    assert config["gate_checks"][2] == DEFAULT_CONFIG["gate_checks"][2]


def test_configure_tui_delete_all_gates(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    import asyncio

    from textual.widgets import Button

    from splinter.configure import load_config
    from splinter.tui import ConfigureApp

    monkeypatch.chdir(tmp_path)

    async def drive() -> None:
        app = ConfigureApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            total = len(app._gate_checks)
            for _ in range(total):
                app.query_one("#gate_del_0", Button).press()
                await pilot.pause()
            assert app._gate_checks == []
            await pilot.press("s")
            await pilot.pause()
        assert app.saved

    asyncio.run(drive())
    config = load_config()
    assert config["gate_checks"] == []
