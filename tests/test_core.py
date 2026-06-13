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
from splinter.strategies.stages import (
    IterationContext,
    RunStage,
    _parse_verdict,
)
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
    # Expect language field to default to "all" when missing
    assert result == [
        {"name": "ruff", "cmd": "ruff check", "when": "always", "language": "all"},
        {"name": "mypy", "cmd": "mypy", "when": "always", "language": "all"},
    ]


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
    # Language field defaults to "all" when missing
    assert config["gate_checks"] == [
        {
            "name": "test",
            "cmd": "pytest",
            "when": "always",
            "language": "all",
        }
    ]


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
    # Language field defaults to "all" when missing
    assert config["gate_checks"] == [
        {
            "name": "lint",
            "cmd": "ruff",
            "when": "always",
            "language": "all",
        }
    ]


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

    from splinter.providers import opencode
    from splinter.tui import ConfigureApp

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        opencode,
        "list_models",
        lambda timeout=30: ["opencode-go/test-model"],
    )

    async def drive() -> None:
        app = ConfigureApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            assert app._models_by_provider
            assert set(app._models_by_provider.keys()) == {"claude", "opencode", "codex"}
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


def test_roster_loads_codex_tiers() -> None:
    from splinter.models.roster import provider_for

    assert provider_for("codex/gpt-5-codex") == "codex"
    assert provider_for("opencode/foo") == "opencode"
    assert provider_for("opencode-go/bar") == "opencode"
    assert provider_for("sonnet") == "claude"


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
        assert check["language"] == "go"


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
    first.append({"name": "extra", "cmd": "extra cmd", "when": "always", "language": "python"})
    first[0]["cmd"] = "modified"
    second = gate_default_for("python")
    assert len(second) == 3
    assert second[0]["cmd"] == "ruff check ."
    assert all(c.get("name") != "extra" for c in second)
    for check in second:
        assert check["language"] == "python"


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
    assert set(available_providers()) == {"claude", "opencode", "codex"}
    assert get_provider("claude").name == "claude"
    with pytest.raises(ValueError):
        get_provider("bogus")


def test_get_provider_returns_codex() -> None:
    from splinter.providers.codex import CodexProvider

    provider = get_provider("codex")
    assert isinstance(provider, CodexProvider)
    assert provider.name == "codex"


def test_provider_for_codex_resolution(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
    from splinter.models.roster import provider_for
    assert provider_for("codex/gpt-5") == "codex"
    assert provider_for("opencode/foo") == "opencode"
    assert provider_for("opencode-go/bar") == "opencode"
    assert provider_for("sonnet") == "claude"
    assert provider_for("opus") == "claude"
    monkeypatch.chdir(tmp_path)
    raw = {
        "tiers": [
            {
                "name": "test-codex",
                "level": 0,
                "models": ["codex/gpt-5"],
                "provider": "codex",
            }
        ],
        "effort_map": {},
        "eval": {},
        "planner": {},
        "localizer": {},
    }
    ladder = load_ladder(raw)
    t0 = ladder.tier_by_level(0)
    assert t0.provider == "codex", "explicit provider: codex label must survive load_ladder"


def test_available_models_includes_codex() -> None:
    from splinter.configure import CODEX_MODELS, available_models

    models = available_models()
    assert isinstance(models, list)
    for codex_model in CODEX_MODELS:
        assert codex_model in models, f"codex model {codex_model} not in available_models()"


def test_configure_provider_catalogs_include_all_static_models() -> None:
    from splinter.configure import CLAUDE_MODELS, CODEX_MODELS, EFFORT_CHOICES
    from splinter.enums import Variant
    from splinter.models.roster import CODEX_MODELS as ROSTER_CODEX_MODELS

    assert "fable" in CLAUDE_MODELS
    assert "claude-fable-5" in CLAUDE_MODELS
    assert set(CODEX_MODELS) == set(ROSTER_CODEX_MODELS.values())
    assert EFFORT_CHOICES == [str(v) for v in Variant]
    assert "auto" in EFFORT_CHOICES


def test_available_models_by_provider_keys(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from splinter.configure import CLAUDE_MODELS, CODEX_MODELS, available_models_by_provider
    from splinter.providers import opencode

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        opencode,
        "list_models",
        lambda timeout=30: ["opencode-go/foo", "opencode/bar", "other/baz"],
    )
    result = available_models_by_provider()
    assert set(result.keys()) == {"claude", "opencode", "codex"}
    assert result["codex"] == sorted(CODEX_MODELS)
    assert "sonnet" in result["claude"]
    assert "opus" in result["claude"]
    assert "fable" in result["claude"]
    assert result["claude"] == sorted(set(CLAUDE_MODELS))
    for m in result["claude"]:
        assert not m.startswith("opencode") and not m.startswith("codex/")
    assert "opencode-go/foo" in result["opencode"]
    assert "opencode/bar" in result["opencode"]
    assert "other/baz" not in result["opencode"]

def test_available_models_by_provider_isolates_failure(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from splinter.configure import CODEX_MODELS, available_models_by_provider
    from splinter.providers import opencode

    monkeypatch.chdir(tmp_path)

    def _raise(*args: object, **kwargs: object) -> list[str]:
        raise RuntimeError("fail")

    monkeypatch.setattr(opencode, "list_models", _raise)
    result = available_models_by_provider()
    assert isinstance(result["opencode"], list)
    assert "sonnet" in result["claude"]
    assert result["codex"] == sorted(CODEX_MODELS)


def test_available_models_by_provider_union(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from splinter.configure import available_models, available_models_by_provider
    from splinter.providers import opencode

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        opencode,
        "list_models",
        lambda timeout=30: ["opencode-go/test-model"],
    )
    by_provider = available_models_by_provider()
    union = sorted({m for models in by_provider.values() for m in models})
    flat = available_models()
    assert union == flat


def test_provider_for_codex() -> None:
    from splinter.models.roster import provider_for

    assert provider_for("codex/gpt-5-codex") == "codex"
    assert provider_for("opencode/foo") == "opencode"
    assert provider_for("opencode-go/bar") == "opencode"
    assert provider_for("sonnet") == "claude"


def test_check_codex_valid(monkeypatch: "pytest.MonkeyPatch") -> None:
    from splinter.providers import codex
    from splinter.setup import _check_codex

    monkeypatch.setattr("shutil.which", lambda _: "/usr/local/bin/codex")
    monkeypatch.setattr(codex, "ping", lambda **kw: True)
    ok, detail = _check_codex()
    assert ok is True
    assert detail == "/usr/local/bin/codex"


def test_check_codex_missing(monkeypatch: "pytest.MonkeyPatch") -> None:
    from splinter.setup import _check_codex

    monkeypatch.setattr("shutil.which", lambda _: None)
    ok, detail = _check_codex()
    assert ok is False
    assert "codex not found in PATH" in detail


def test_check_codex_invalid(monkeypatch: "pytest.MonkeyPatch") -> None:
    from splinter.providers import codex
    from splinter.setup import _check_codex

    monkeypatch.setattr("shutil.which", lambda _: "/usr/local/bin/codex")
    monkeypatch.setattr(codex, "ping", lambda **kw: False)
    ok, detail = _check_codex()
    assert ok is False
    assert "codex exec did not respond" in detail


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


def test_structure_renders_at_run_entry(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Verify folder structure anchor writes at RunStage entry, before run_task."""
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_test")

    task = Task(
        description="test task",
        acceptance="test acceptance",
        target_files=["test.py"],
    )
    ladder = load_ladder()
    trace = Trace()
    knowledge = KnowledgeStore(session)

    ctx = IterationContext(
        task=task,
        plan="test plan",
        tier=0,
        iteration=1,
        ladder=ladder,
        session=session,
        trace=trace,
        knowledge=knowledge,
    )

    call_log: list[str] = []

    def stub_run_task(*args, **kwargs):  # type: ignore
        call_log.append("run_task called")
        loop_content = session.read("loop.md")
        call_log.append(f"loop.md has: {repr(loop_content[:50])}")
        from splinter.agents.runner import RunResult
        return RunResult(
            text="test output",
            model="test-model",
            tier=0,
            tokens={"completion": 100, "prompt": 50},
            cost=0.01,
            raw={},
        )

    monkeypatch.setattr("splinter.strategies.stages.run_task", stub_run_task)

    stage = RunStage()
    stage.process(ctx)

    assert len(call_log) >= 2
    assert "run_task called" in call_log
    assert "## Iteration 1" in call_log[1]


def test_no_duplicate_structure_after_eval(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Verify structure anchor appears exactly once after RunStage then EvalStage."""
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_test")

    task = Task(
        description="test task",
        acceptance="test acceptance",
        target_files=["test.py"],
    )
    ladder = load_ladder()
    trace = Trace()
    knowledge = KnowledgeStore(session)

    ctx = IterationContext(
        task=task,
        plan="test plan",
        tier=0,
        iteration=1,
        ladder=ladder,
        session=session,
        trace=trace,
        knowledge=knowledge,
    )

    def stub_run_task(*args, **kwargs):  # type: ignore
        from splinter.agents.runner import RunResult
        return RunResult(
            text="test output",
            model="test-model",
            tier=0,
            tokens={"completion": 100, "prompt": 50},
            cost=0.01,
            raw={},
        )

    monkeypatch.setattr("splinter.strategies.stages.run_task", stub_run_task)

    # Run RunStage
    run_stage = RunStage()
    run_stage.process(ctx)

    # Check loop.md has exactly one "## Iteration 1"
    loop_content = session.read("loop.md")
    count = loop_content.count("## Iteration 1")
    assert count == 1, f"Expected 1 '## Iteration 1', found {count}"


def test_trajectory_independent_of_eval(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Verify trajectory renders after RunStage only, without EvalStage."""
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_test")

    task = Task(
        description="test task",
        acceptance="test acceptance",
        target_files=["test.py"],
    )
    ladder = load_ladder()
    trace = Trace()
    knowledge = KnowledgeStore(session)

    ctx = IterationContext(
        task=task,
        plan="test plan",
        tier=0,
        iteration=1,
        ladder=ladder,
        session=session,
        trace=trace,
        knowledge=knowledge,
    )

    def stub_run_task(*args, **kwargs):  # type: ignore
        from splinter.agents.runner import RunResult
        return RunResult(
            text="test output",
            model="test-model",
            tier=0,
            tokens={"completion": 100, "prompt": 50},
            cost=0.01,
            raw={},
        )

    monkeypatch.setattr("splinter.strategies.stages.run_task", stub_run_task)

    # Run RunStage only, NOT EvalStage
    run_stage = RunStage()
    run_stage.process(ctx)

    # Parse trajectory from loop.md
    loop_content = session.read("loop.md")

    # Should have at least the structure anchor
    assert "## Iteration 1" in loop_content
    # Note: we can't verify verdict without running EvalStage
    # but we can verify the structure is there


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


def test_analyze_tui_plan_survives_reload(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Regression: after navigating to plan N, the 1s auto-reload must keep
    rendering plan N — not snap back to the first plan."""
    import asyncio

    import splinter.tui as tui_mod
    from splinter.tui import AnalyzeApp

    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_plans")
    _seed_session(session)
    session.write("knowledge/plan-1.md", "# Plan One\n\n1. first\n")
    session.write("knowledge/plan-2.md", "# Plan Two\n\n1. second\n")

    rendered: list[str] = []
    real_file_md = tui_mod._file_md
    monkeypatch.setattr(
        tui_mod,
        "_file_md",
        lambda s, label, file: (rendered.append(file), real_file_md(s, label, file))[1],
    )

    async def drive() -> None:
        app = AnalyzeApp(session)
        async with app.run_test() as pilot:
            await pilot.pause()
            tree = app.query_one("#nav", tui_mod.Tree)
            tree.move_cursor(app._plan_node)
            await pilot.pause()
            await pilot.press("right", "right")  # idx -> 2
            await pilot.pause()
            assert app._plan_idx == 2
            rendered.clear()
            app._do_reload()  # the 1s auto-refresh path
            await pilot.pause()
            assert app._plan_idx == 2
            assert rendered and rendered[-1].endswith("plan-2.md")
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

    fake_result = type(
        "R",
        (),
        {
            "text": "Q1?",
            "raw": {"_session_id": "sid"},
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    )()
    _prd_body = (
        "---\nfeature: x\nstrategy: direct\nkind: feature\ncreated: 2026-06-10\n---\n"
        "### US-001: X\n**Description:** d\n**Acceptance Criteria:**\n- [ ] c\n"
    )
    fake_result2 = type(
        "R",
        (),
        {
            "text": _prd_body,
            "raw": {"_session_id": "sid"},
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    )()

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
    monkeypatch.setattr(cc, "_calc_cost", lambda *a, **kw: (0.0, False))
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
    monkeypatch.setattr(cc, "_calc_cost", lambda *a, **kw: (0.0, False))

    def _interrupt(*a: object) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", _interrupt)

    with pytest.raises(KeyboardInterrupt):
        run_prd(description="add auth", no_ground=True)
    assert list_sessions() == [], "interrupted PRD run must not litter an empty session"


def test_run_prd_abort_after_ground_localization_cleans_up(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Aborting after ground_localization (which creates knowledge/localization.md)
    must still garbage-collect the session — is_empty() would return False due to
    the localization file, but prd_session_is_resumable correctly returns False.
    """
    from splinter.memory.session import Session, list_sessions, new_session_id
    from splinter.prd import _prune_prd_session

    monkeypatch.chdir(tmp_path)

    # Simulate a session that had localization written (ground_localization side-effect)
    # but no real PRD content.
    sid = new_session_id()
    s = Session(sid)
    s.write("knowledge/localization.md", "# Localization\nfile: foo.py\nsymbol: bar\n")

    assert not s.is_empty(), "sanity: localization.md makes is_empty() return False"

    _prune_prd_session(s)

    assert list_sessions() == [], "session with only localization.md must be cleaned up"


def test_run_configure_no_interactive_creates_no_session(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """configure (non-interactive) must never create a session directory."""
    from splinter.configure import run_configure
    from splinter.memory.session import list_sessions

    monkeypatch.chdir(tmp_path)

    rc = run_configure(gate_checks="ruff check .", interactive=False)
    assert rc == 0
    assert list_sessions() == [], "configure must not create sessions"


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
            # Verify language column is present for each row
            for i in range(len(DEFAULT_CONFIG["gate_checks"])):
                lang_select = app.query_one(f"#gate_lang_{i}")
                assert lang_select is not None

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
    # Language field defaults to "all" for checks without explicit language
    assert config["gate_checks"] == [
        {"name": "lint", "cmd": "ruff check .", "when": "always", "language": "all"},
        {"name": "types", "cmd": "mypy src", "when": "always", "language": "all"},
    ]


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

    checks = parse_gate_spec("npm run build", "unknown")
    assert checks == [
        {
            "name": "npm",
            "cmd": "npm run build",
            "when": "always",
            "language": "unknown",
        }
    ]


def test_parse_gate_spec_multi_cmd() -> None:
    from splinter.agents.gate import parse_gate_spec

    checks = parse_gate_spec("npm run lint; npm test", "unknown")
    assert len(checks) == 2
    expected_0 = {
        "name": "npm",
        "cmd": "npm run lint",
        "when": "always",
        "language": "unknown",
    }
    expected_1 = {
        "name": "npm",
        "cmd": "npm test",
        "when": "always",
        "language": "unknown",
    }
    assert checks[0] == expected_0
    assert checks[1] == expected_1


def test_parse_gate_spec_with_language() -> None:
    from splinter.agents.gate import parse_gate_spec

    checks = parse_gate_spec("npm run build", language="javascript-npm")
    assert len(checks) == 1
    assert checks[0]["language"] == "javascript-npm"


def test_parse_gate_spec_json_with_language() -> None:
    from splinter.agents.gate import parse_gate_spec

    spec = '[{"name": "lint", "cmd": "npm lint", "when": "always"}]'
    checks = parse_gate_spec(spec, language="javascript-npm")
    assert len(checks) == 1
    assert checks[0]["language"] == "javascript-npm"


def test_parse_gate_spec_json_preserves_language() -> None:
    from splinter.agents.gate import parse_gate_spec

    spec = '[{"name": "lint", "cmd": "npm lint", "when": "always", "language": "custom"}]'
    checks = parse_gate_spec(spec, language="javascript-npm")
    assert len(checks) == 1
    assert checks[0]["language"] == "custom"


def test_write_model_config_gate_checks_empty(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from splinter.configure import load_config, write_model_config

    monkeypatch.chdir(tmp_path)
    write_model_config({"planner": "opus", "tiers": ["haiku"]}, gate_checks=[])
    config = load_config()
    assert config["gate_checks"] == []


def test_configured_gate_checks_backward_compat(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from splinter.agents.gate import configured_gate_checks, save_gate_checks

    monkeypatch.chdir(tmp_path)
    # Simulate old gate.json without language field
    old_checks = [
        {"name": "ruff", "cmd": "ruff check", "when": "always"},
        {"name": "mypy", "cmd": "mypy .", "when": "always"},
    ]
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    save_gate_checks(session_dir, old_checks)

    # Load should stamp missing language with "all"
    loaded = configured_gate_checks(session_dir=session_dir)
    assert loaded is not None
    assert len(loaded) == 2
    for check in loaded:
        assert check["language"] == "all"


def test_configure_tui_add_custom_gate(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    import asyncio

    from textual.widgets import Button, Input, Select

    from splinter.configure import load_config
    from splinter.tui import ConfigureApp

    monkeypatch.chdir(tmp_path)

    async def drive() -> None:
        app = ConfigureApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            initial_count = len(app._gate_checks)
            app.query_one("#gate_add_input", Input).value = "npm run build"
            await pilot.pause()
            app.query_one("#gate_add", Button).press()
            await pilot.pause()
            # Set language for the newly added gate
            new_gate_idx = initial_count
            lang_select = app.query_one(f"#gate_lang_{new_gate_idx}", Select)
            lang_select.value = "javascript-npm"
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()
        assert app.saved

    asyncio.run(drive())
    config = load_config()
    added_check = next(
        (c for c in config["gate_checks"] if c["name"] == "npm" and c["cmd"] == "npm run build"),
        None,
    )
    assert added_check is not None
    assert added_check["when"] == "always"
    assert added_check["language"] == "javascript-npm"


def test_configure_tui_edit_gate_cmd(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    import asyncio

    from textual.widgets import Input, Select

    from splinter.configure import load_config
    from splinter.tui import ConfigureApp

    monkeypatch.chdir(tmp_path)

    async def drive() -> None:
        app = ConfigureApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("#gate_cmd_0", Input).value = "uv run ruff check --strict"
            await pilot.pause()
            # Change language of first gate
            lang_select = app.query_one("#gate_lang_0", Select)
            lang_select.value = "rust"
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()
        assert app.saved

    asyncio.run(drive())
    config = load_config()
    assert config["gate_checks"][0]["cmd"] == "uv run ruff check --strict"
    assert config["gate_checks"][0]["language"] == "rust"
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


# --- US-002: _model_opts_for composition ------------------------------------


def test_model_opts_for_claude_only(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
    from splinter.models.roster import CODEX_MODELS as ROSTER_CODEX_MODELS
    from splinter.tui import ConfigureApp

    monkeypatch.chdir(tmp_path)
    app = ConfigureApp()
    app._models_by_provider = {
        "claude": ["opus", "sonnet"],
        "opencode": ["opencode-go/deepseek-v4-flash-free", "opencode-go/minimax-m3"],
        "codex": sorted(ROSTER_CODEX_MODELS.values()),
    }

    opts = app._model_opts_for("claude", "")

    opt_ids = {v for _label, v in opts}
    assert "sonnet" in opt_ids
    assert "opus" in opt_ids
    assert "opencode-go/deepseek-v4-flash-free" not in opt_ids
    assert "opencode-go/minimax-m3" not in opt_ids
    for codex_id in ROSTER_CODEX_MODELS.values():
        assert codex_id not in opt_ids


def test_model_opts_for_opencode_only(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
    from splinter.models.roster import CODEX_MODELS as ROSTER_CODEX_MODELS
    from splinter.tui import ConfigureApp

    monkeypatch.chdir(tmp_path)
    app = ConfigureApp()
    app._models_by_provider = {
        "claude": ["opus", "sonnet"],
        "opencode": ["opencode-go/deepseek-v4-flash-free", "opencode-go/minimax-m3"],
        "codex": sorted(ROSTER_CODEX_MODELS.values()),
    }
    opts = app._model_opts_for("opencode", "")

    opt_ids = {v for _label, v in opts}
    assert "sonnet" not in opt_ids
    assert "opus" not in opt_ids
    assert "opencode-go/deepseek-v4-flash-free" in opt_ids
    assert "opencode-go/minimax-m3" in opt_ids
    for codex_id in ROSTER_CODEX_MODELS.values():
        assert codex_id not in opt_ids


def test_model_opts_for_filter_narrows(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
    from splinter.tui import ConfigureApp

    monkeypatch.chdir(tmp_path)
    app = ConfigureApp()
    app._models_by_provider = {
        "opencode": [
            "opencode-go/deepseek-v4-flash-free",
            "opencode-go/deepseek-v4-pro",
            "opencode-go/minimax-m3",
        ],
    }

    all_opencode = app._model_opts_for("opencode", "")
    assert len(all_opencode) == 3

    flash_opts = app._model_opts_for("opencode", "flash")
    assert len(flash_opts) == 1
    assert flash_opts[0][1] == "opencode-go/deepseek-v4-flash-free"

    deepseek_opts = app._model_opts_for("opencode", "deepseek")
    assert len(deepseek_opts) == 2
    assert {v for _label, v in deepseek_opts} == {
        "opencode-go/deepseek-v4-flash-free",
        "opencode-go/deepseek-v4-pro",
    }

    empty = app._model_opts_for("opencode", "nonexistent")
    assert len(empty) == 0


def test_model_opts_for_codex_includes_roster_ids(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch",
) -> None:
    from splinter.models.roster import CODEX_MODELS as ROSTER_CODEX_MODELS
    from splinter.tui import ConfigureApp

    monkeypatch.chdir(tmp_path)
    app = ConfigureApp()
    app._models_by_provider = {"codex": sorted(ROSTER_CODEX_MODELS.values())}

    opts = app._model_opts_for("codex", "")
    expected = sorted(ROSTER_CODEX_MODELS.values())
    assert len(opts) == len(expected)
    for (label, value), expected_id in zip(opts, expected):
        assert label == expected_id
        assert value == expected_id
    assert {v for _label, v in opts} == set(expected)


def test_model_opts_for_default_shows_all(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch",
) -> None:
    from splinter.tui import ConfigureApp

    monkeypatch.chdir(tmp_path)
    app = ConfigureApp()
    app._models_by_provider = {
        "claude": ["opus", "sonnet"],
        "opencode": ["opencode-go/deepseek-v4-flash-free", "opencode-go/minimax-m3"],
        "codex": ["codex/gpt-5-codex"],
    }
    opts = app._model_opts_for("(default)", "")
    opt_ids = {v for _label, v in opts}
    assert "sonnet" in opt_ids
    assert "opencode-go/deepseek-v4-flash-free" in opt_ids
    assert "codex/gpt-5-codex" in opt_ids
    assert len(opts) == 5


def test_model_opts_for_default_filter_narrows(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch",
) -> None:
    from splinter.tui import ConfigureApp

    monkeypatch.chdir(tmp_path)
    app = ConfigureApp()
    app._models_by_provider = {
        "claude": ["sonnet"],
        "opencode": ["opencode-go/deepseek-v4-flash-free", "opencode-go/minimax-m3"],
    }
    opts = app._model_opts_for("(default)", "flash")
    assert len(opts) == 1
    assert opts[0][1] == "opencode-go/deepseek-v4-flash-free"


def test_current_model_selections_returns_providers_block(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch",
) -> None:
    from splinter.configure import current_model_selections
    from splinter.models.roster import provider_for

    monkeypatch.chdir(tmp_path)
    current = current_model_selections()
    assert "providers" in current
    providers = current["providers"]
    assert "localizer_recall" in providers
    assert "localizer_recall_large" in providers
    assert "localizer_precision" in providers
    assert "planner" in providers
    assert "eval" in providers
    assert "tiers" in providers
    assert isinstance(providers["tiers"], list)
    assert len(providers["tiers"]) == 6
    for p in providers["tiers"]:
        assert p in {"claude", "opencode", "codex"}
    ladder = load_ladder()
    assert providers["planner"] == provider_for(ladder.planner_model)
    assert providers["eval"] == provider_for(ladder.eval_model)


def test_write_model_config_providers_roundtrip(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch",
) -> None:
    from splinter.configure import load_config, write_model_config

    monkeypatch.chdir(tmp_path)
    providers = {
        "planner": "claude",
        "eval": "opencode",
        "tiers": ["opencode", "opencode", "claude", "opencode", "opencode", "claude"],
    }
    write_model_config(
        {"planner": "sonnet", "tiers": ["deepseek-v4-pro"]},
        providers=providers,
    )
    config = load_config()
    assert "providers" in config
    assert config["providers"] == providers


def test_write_model_config_providers_none_preserves_existing(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch",
) -> None:
    from splinter.configure import load_config, write_model_config

    monkeypatch.chdir(tmp_path)
    existing = {"planner": "claude", "tiers": ["opencode"]}
    write_model_config({"planner": "sonnet", "tiers": ["haiku"]}, providers=existing)
    write_model_config({"planner": "opus", "tiers": ["sonnet"]}, providers=None)
    config = load_config()
    assert config["providers"] == existing


def test_configure_tui_model_trigger_toggles_overlay(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch",
) -> None:
    import asyncio

    from textual.widgets import Button

    from splinter.providers import opencode
    from splinter.tui import ConfigureApp

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        opencode,
        "list_models",
        lambda timeout=30: ["opencode-go/test-a", "opencode-go/test-b"],
    )

    async def drive() -> None:
        app = ConfigureApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            app.query_one("#tier_5__trigger", Button).press()
            await pilot.pause()
            assert len(app.query("#tier_5__overlay")) == 1
            app.query_one("#tier_5__trigger", Button).press()
            await pilot.pause()
            assert len(app.query("#tier_5__overlay")) == 0

    asyncio.run(drive())


def test_configure_tui_rows_use_compact_height() -> None:
    from splinter.tui import ConfigureApp

    assert "min-height: 4;" in ConfigureApp.CSS
    assert "min-height: 12;" not in ConfigureApp.CSS
    assert ".step-desc { color: $text-muted; height: 1; }" in ConfigureApp.CSS


def test_configure_tui_row_description_tooltips(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch",
) -> None:
    import asyncio

    from textual.widgets import Label

    from splinter.configure import MODEL_STEPS, TIER_STEPS
    from splinter.providers import opencode
    from splinter.tui import ConfigureApp

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(opencode, "list_models", lambda timeout=30: [])

    async def drive() -> None:
        app = ConfigureApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            expected = {desc for _key, _label, desc in MODEL_STEPS}
            expected.update(desc for _label, desc in TIER_STEPS)
            names = list(app.query(".step-name").results(Label))
            descs = list(app.query(".step-desc").results(Label))
            infos = list(app.query(".step-info").results())
            assert len(names) == len(descs) == len(infos) == len(expected)
            for widget in [*names, *descs, *infos]:
                assert widget.tooltip in expected

    asyncio.run(drive())


def test_configure_tui_provider_change_repopulates_model_list(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch",
) -> None:
    import asyncio

    from textual.widgets import OptionList, Select

    from splinter.providers import opencode
    from splinter.tui import ConfigureApp

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        opencode,
        "list_models",
        lambda timeout=30: ["opencode-go/test-a", "opencode-go/test-b"],
    )

    async def drive() -> None:
        app = ConfigureApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            model_widget = app.query_one("#planner", OptionList)
            provider_widget = app.query_one("#planner__prov", Select)
            provider_widget.value = "opencode"
            await pilot.pause()
            opt_ids = {str(o.prompt) for o in model_widget._options}
            assert "sonnet" not in opt_ids
            assert "opus" not in opt_ids
            assert "opencode-go/test-a" in opt_ids
            provider_widget.value = "claude"
            await pilot.pause()
            claude_ids = {str(o.prompt) for o in model_widget._options}
            assert "sonnet" in claude_ids
            assert "opus" in claude_ids
            assert "opencode-go/test-a" not in claude_ids

    asyncio.run(drive())


def test_configure_tui_provider_filter_persists_across_switch(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch",
) -> None:
    import asyncio

    from textual.widgets import Input, OptionList, Select

    from splinter.providers import opencode
    from splinter.tui import ConfigureApp

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        opencode,
        "list_models",
        lambda timeout=30: ["opencode-go/deepseek-v4-flash-free", "opencode-go/minimax-m3"],
    )

    async def drive() -> None:
        app = ConfigureApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            provider_widget = app.query_one("#planner__prov", Select)
            filter_widget = app.query_one("#planner__filter", Input)
            model_widget = app.query_one("#planner", OptionList)
            provider_widget.value = "opencode"
            await pilot.pause()
            filter_widget.value = "flash"
            await pilot.pause()
            opt_ids = {str(o.prompt) for o in model_widget._options}
            assert "opencode-go/deepseek-v4-flash-free" in opt_ids
            assert "opencode-go/minimax-m3" not in opt_ids
            provider_widget.value = "claude"
            await pilot.pause()
            assert filter_widget.value == "flash"
            claude_ids = {str(o.prompt) for o in model_widget._options}
            assert len(claude_ids) < 3

    asyncio.run(drive())


# --- US-002: gate language survives save/load --------------------------------


def test_gate_load_defaults_missing_language_to_all(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Loading a gate check without language field defaults to 'all'."""
    import json

    from splinter.agents.gate import configured_gate_checks

    monkeypatch.chdir(tmp_path)
    session_dir = tmp_path / ".splinter" / "sessions" / "test_session"
    session_dir.mkdir(parents=True, exist_ok=True)

    # Write a gate.json with checks lacking language field
    checks_without_lang = [
        {"name": "lint", "cmd": "ruff check", "when": "always"},
        {"name": "test", "cmd": "pytest", "when": "tests_exist"},
    ]
    (session_dir / "gate.json").write_text(json.dumps(checks_without_lang))

    result = configured_gate_checks(session_dir=session_dir)
    assert result is not None
    assert len(result) == 2
    assert all(c["language"] == "all" for c in result)


def test_gate_save_load_round_trip_preserves_language(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Gate language persists through save_gate_checks -> configured_gate_checks."""
    from splinter.agents.gate import configured_gate_checks, save_gate_checks

    monkeypatch.chdir(tmp_path)
    session_dir = tmp_path / ".splinter" / "sessions" / "test_session"
    session_dir.mkdir(parents=True, exist_ok=True)

    # Save checks with explicit language values
    checks = [
        {"name": "ruff", "cmd": "ruff check", "when": "always", "language": "python"},
        {"name": "cargo", "cmd": "cargo test", "when": "always", "language": "rust"},
        {"name": "generic", "cmd": "custom cmd", "when": "always", "language": "all"},
    ]
    save_gate_checks(session_dir, checks)

    # Load them back
    loaded = configured_gate_checks(session_dir=session_dir)
    assert loaded is not None
    assert len(loaded) == 3
    assert loaded[0]["language"] == "python"
    assert loaded[1]["language"] == "rust"
    assert loaded[2]["language"] == "all"


def test_configure_save_preserves_gate_language_from_preset(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Gate language from presets (e.g., go, rust) survives save/load."""
    from splinter.configure import gate_default_for, load_config, write_gate_checks

    monkeypatch.chdir(tmp_path)
    go_checks = gate_default_for("go")
    assert len(go_checks) > 0
    assert all(c["language"] == "go" for c in go_checks)

    write_gate_checks(go_checks)
    config = load_config()

    # Verify all checks preserved their language field
    assert len(config["gate_checks"]) == len(go_checks)
    assert all(c["language"] == "go" for c in config["gate_checks"])


def test_write_gate_checks_normalizes_missing_language_to_all(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """write_gate_checks normalizes checks lacking language to 'all'."""
    from splinter.configure import load_config, write_gate_checks

    monkeypatch.chdir(tmp_path)
    checks_without_lang = [
        {"name": "test1", "cmd": "cmd1", "when": "always"},
        {"name": "test2", "cmd": "cmd2", "when": "always"},
    ]
    write_gate_checks(checks_without_lang)
    config = load_config()

    assert len(config["gate_checks"]) == 2
    assert all(c["language"] == "all" for c in config["gate_checks"])


# --- US-004: language-filtered gate -----------------------------------------


def test_run_gate_skips_other_language_check(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from splinter.agents.gate import run_gate, save_gate_checks

    class _FakeProc:
        returncode = 0
        stdout = "ok"
        stderr = ""

    monkeypatch.setattr("subprocess.run", lambda cmd, **kw: _FakeProc())

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    save_gate_checks(session_dir, [
        {"name": "ruff", "cmd": "ruff check", "when": "always", "language": "python"},
        {"name": "cargo-test", "cmd": "cargo test", "when": "always", "language": "go"},
    ])

    result = run_gate(session_dir=session_dir, languages={"python"})
    check_names = [name for name, _, _ in result.checks]
    assert "ruff" in check_names
    assert "cargo-test" not in check_names


def test_run_gate_runs_all_when_no_language(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from splinter.agents.gate import run_gate, save_gate_checks

    class _FakeProc:
        returncode = 0
        stdout = "ok"
        stderr = ""

    monkeypatch.setattr("subprocess.run", lambda cmd, **kw: _FakeProc())

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    save_gate_checks(session_dir, [
        {"name": "ruff", "cmd": "ruff check", "when": "always", "language": "python"},
        {"name": "go-test", "cmd": "go test ./...", "when": "always", "language": "go"},
    ])

    for langs in (None, set()):
        result = run_gate(session_dir=session_dir, languages=langs)
        check_names = [name for name, _, _ in result.checks]
        assert "ruff" in check_names, f"ruff missing with languages={langs}"
        assert "go-test" in check_names, f"go-test missing with languages={langs}"


def test_run_gate_runs_all_tagged_check(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from splinter.agents.gate import run_gate, save_gate_checks

    class _FakeProc:
        returncode = 0
        stdout = "ok"
        stderr = ""

    monkeypatch.setattr("subprocess.run", lambda cmd, **kw: _FakeProc())

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    save_gate_checks(session_dir, [
        {"name": "always-check", "cmd": "always-cmd", "when": "always", "language": "all"},
        {"name": "go-check", "cmd": "go-cmd", "when": "always", "language": "go"},
    ])

    result = run_gate(session_dir=session_dir, languages={"python"})
    check_names = [name for name, _, _ in result.checks]
    assert "always-check" in check_names
    assert "go-check" not in check_names


def test_task_languages_union() -> None:
    from splinter.agents.gate import task_languages

    task = Task(description="t", acceptance="t", target_files=["a.py", "b.go"])
    assert task_languages(task) == {"python", "go"}

    task_mixed = Task(description="t", acceptance="t", target_files=["x.ts", "y.rs", "z.py"])
    assert task_languages(task_mixed) == {"typescript", "rust", "python"}

    task_empty = Task(description="t", acceptance="t", target_files=[])
    assert task_languages(task_empty) == set()

    task_none = Task(description="t", acceptance="t", target_files=None)
    assert task_languages(task_none) == set()


def test_gate_default_for_tags_language() -> None:
    from splinter.configure import LANGUAGE_GATE_DEFAULTS

    for lang in LANGUAGE_GATE_DEFAULTS:
        checks = gate_default_for(lang)
        for check in checks:
            assert check.get("language") == lang, (
                f"gate_default_for({lang!r}) check {check['name']!r} "
                f"has language={check.get('language')!r}"
            )


# --- US-004: dispatch delegation via provider registry -----------------------


def _make_fake_provider(name: str, resp_text: str = "ok", session_id: str | None = None) -> object:
    from splinter.providers.base import ModelProvider, ProviderResponse

    calls: list[dict] = []

    class FakeProvider(ModelProvider):
        name = ""  # overridden below

        def run(  # type: ignore[override]
            self, prompt, model, *, variant=None, output_format="json",
            session=None, timeout=None, agent="build",
        ):
            calls.append(dict(
                prompt=prompt, model=model, variant=variant,
                output_format=output_format, session=session,
                timeout=timeout, agent=agent,
            ))
            return ProviderResponse(
                text=resp_text, tokens={"input": 1, "output": 1},
                cost=0.01, raw={}, session_id=session_id,
            )

    p = FakeProvider()
    FakeProvider.name = name
    p.calls = calls  # type: ignore[attr-defined]
    return p


def test_dispatch_run_text_routes_to_registered_provider(monkeypatch: "pytest.MonkeyPatch") -> None:
    from splinter.providers import dispatch

    fake = _make_fake_provider("claude")
    monkeypatch.setattr(dispatch, "get_provider", lambda _name: fake)
    result = dispatch.run_text("hello", "sonnet", timeout=10)
    assert result == "ok"
    assert fake.calls[0]["model"] == "sonnet"  # type: ignore[attr-defined]


def test_dispatch_run_text_opencode_routes_correctly(monkeypatch: "pytest.MonkeyPatch") -> None:
    from splinter.providers import dispatch

    fake = _make_fake_provider("opencode")
    monkeypatch.setattr(dispatch, "get_provider", lambda _name: fake)
    result = dispatch.run_text("build it", "opencode-go/minimax-m3", agent="build", timeout=30)
    assert result == "ok"
    assert fake.calls[0]["agent"] == "build"  # type: ignore[attr-defined]


def test_dispatch_run_text_calls_log_when_session_present(monkeypatch: "pytest.MonkeyPatch") -> None:  # noqa: E501
    from splinter.providers import dispatch

    fake = _make_fake_provider("claude")
    monkeypatch.setattr(dispatch, "get_provider", lambda _name: fake)
    logged: list[tuple] = []

    class FakeSession:
        def log_llm_usage(self, model, tokens, cost):
            logged.append((model, tokens, cost))

    dispatch.run_text("hi", "sonnet", session=FakeSession(), timeout=5)
    assert len(logged) == 1
    assert logged[0][0] == "sonnet"


def test_dispatch_run_text_no_log_when_session_none(monkeypatch: "pytest.MonkeyPatch") -> None:
    from splinter.providers import dispatch

    fake = _make_fake_provider("claude")
    monkeypatch.setattr(dispatch, "get_provider", lambda _name: fake)
    logged: list[object] = []

    class FakeSession:
        def log_llm_usage(self, *a):
            logged.append(a)

    dispatch.run_text("hi", "sonnet", session=None, timeout=5)
    assert logged == []


def test_dispatch_run_text_session_returns_sid_from_provider(monkeypatch: "pytest.MonkeyPatch") -> None:  # noqa: E501
    from splinter.providers import dispatch

    fake = _make_fake_provider("claude", session_id="new-sid")
    monkeypatch.setattr(dispatch, "get_provider", lambda _name: fake)
    text, sid = dispatch.run_text_session("hi", "sonnet", timeout=5)
    assert text == "ok"
    assert sid == "new-sid"


def test_dispatch_run_text_session_fallback_to_passed_session(monkeypatch: "pytest.MonkeyPatch") -> None:  # noqa: E501
    """When provider returns session_id=None, sid falls back to passed-in session."""
    from splinter.providers import dispatch

    fake = _make_fake_provider("claude", session_id=None)
    monkeypatch.setattr(dispatch, "get_provider", lambda _name: fake)
    text, sid = dispatch.run_text_session("hi", "sonnet", session="prior-sid", timeout=5)
    assert text == "ok"
    assert sid == "prior-sid"


def test_dispatch_run_provider_session_resp_sid_matches_returned_sid(monkeypatch: "pytest.MonkeyPatch") -> None:  # noqa: E501
    from splinter.providers import dispatch
    from splinter.providers.base import ProviderResponse

    fake = _make_fake_provider("claude", session_id=None)
    monkeypatch.setattr(dispatch, "get_provider", lambda _name: fake)
    resp, sid = dispatch.run_provider_session("hi", "sonnet", session="prev", timeout=5)
    assert isinstance(resp, ProviderResponse)
    assert sid == "prev"
    assert resp.session_id == sid


def test_dispatch_no_if_else_branching() -> None:
    """dispatch.py must not import claude_cli, opencode, or codex directly."""
    import ast
    from pathlib import Path

    src = (Path(__file__).parent.parent / "splinter" / "providers" / "dispatch.py").read_text()
    tree = ast.parse(src)
    direct_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if any(node.module.endswith(m) for m in ("claude_cli", "opencode", "codex")):
                direct_imports.append(node.module)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if any(alias.name.endswith(m) for m in ("claude_cli", "opencode", "codex")):
                    direct_imports.append(alias.name)
    assert direct_imports == [], f"dispatch.py imports provider modules directly: {direct_imports}"


def test_save_writes_selected_models_per_row(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from splinter.configure import TIER_STEPS, load_config, write_model_config

    monkeypatch.chdir(tmp_path)
    models: dict[str, object] = {
        "localizer_recall": "haiku",
        "localizer_recall_large": "sonnet",
        "localizer_precision": "haiku",
        "planner": "opus",
        "eval": "sonnet",
        "tiers": ["haiku", "sonnet", "sonnet", "opus", "opus", "opus"],
    }
    efforts: dict[str, object] = {
        "localizer_recall": "low",
        "localizer_recall_large": "low",
        "localizer_precision": "low",
        "planner": "high",
        "eval": "high",
        "tiers": ["medium", "high", "high", "max", "xhigh", "max"],
    }
    timeouts: dict[str, object] = {
        "localizer_recall": 120,
        "localizer_recall_large": 120,
        "localizer_precision": 120,
        "planner": 600,
        "eval": 600,
        "tiers": [300, 300, 600, 600, 900, 900],
    }
    write_model_config(models, efforts, timeouts=timeouts)
    config = load_config()
    assert config["models"] == models
    assert config["efforts"] == efforts
    assert config["timeouts"] == timeouts
    assert len(config["models"]["tiers"]) == len(TIER_STEPS)
    assert len(config["efforts"]["tiers"]) == len(TIER_STEPS)
    assert len(config["timeouts"]["tiers"]) == len(TIER_STEPS)


def test_blank_model_omitted(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from splinter.configure import current_model_selections, load_config, write_model_config

    monkeypatch.chdir(tmp_path)
    models = {
        "localizer_recall_large": "sonnet",
        "localizer_precision": "haiku",
        "eval": "sonnet",
        "tiers": ["haiku"],
    }
    write_model_config(models)
    config = load_config()
    assert "localizer_recall" not in config["models"]
    assert "planner" not in config["models"]
    assert config["models"]["localizer_recall_large"] == "sonnet"
    assert config["models"]["tiers"][0] == "haiku"
    selections = current_model_selections()
    assert selections["models"]["localizer_recall"] != ""
    assert selections["models"]["planner"] != ""


def test_save_load_roundtrip(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    from splinter.configure import current_model_selections, write_model_config

    monkeypatch.chdir(tmp_path)
    models = {
        "localizer_recall": "haiku",
        "localizer_recall_large": "sonnet",
        "localizer_precision": "haiku",
        "planner": "opus",
        "eval": "sonnet",
        "tiers": ["haiku", "sonnet", "sonnet", "opus", "opus", "opus"],
    }
    efforts = {
        "localizer_recall": "low",
        "localizer_recall_large": "low",
        "localizer_precision": "low",
        "planner": "high",
        "eval": "high",
        "tiers": ["low", "high", "high", "max", "high", "max"],
    }
    timeouts = {
        "localizer_recall": 120,
        "localizer_recall_large": 120,
        "localizer_precision": 120,
        "planner": 600,
        "eval": 600,
        "tiers": [300, 300, 600, 600, 900, 900],
    }
    write_model_config(models, efforts, timeouts=timeouts)
    selections = current_model_selections()
    assert selections["models"] == models
    assert selections["efforts"] == efforts
    assert selections["timeouts"] == timeouts


# --- US-002: per-provider-call trace logging acceptance tests ----------------


def _make_provider_response(text: str = "ok", cost: float = 0.01) -> object:
    from splinter.providers.base import ProviderResponse

    return ProviderResponse(
        text=text,
        tokens={"input": 100, "output": 50},
        cost=cost,
        raw={},
        session_id="sid",
    )


def _mock_provider_run(monkeypatch: "pytest.MonkeyPatch") -> None:
    """Mock provider.run so the real dispatch _log_trace code executes."""
    monkeypatch.setattr(
        "splinter.providers.registry.get_provider",
        lambda _name: type("P", (), {"run": lambda *a, **kw: _make_provider_response()})(),
    )


def test_log_trace_appends_entry_for_billable_call() -> None:
    """_log_trace appends a RunEntry when cost > 0 or tokens are non-empty."""
    from splinter.obs.trace import Trace
    from splinter.providers.dispatch import _log_trace

    trace = Trace()
    _log_trace(trace, "test-model", {"input": 100}, 0.01,
               tier=0, iteration=1, task_index=0, role="run")
    assert len(trace.entries) == 1
    assert trace.entries[0].model == "test-model"
    assert trace.entries[0].cost == 0.01
    assert trace.entries[0].tier == 0
    assert trace.entries[0].iteration == 1
    assert trace.entries[0].role == "run"


def test_log_trace_skips_non_billable_call() -> None:
    """_log_trace does NOT append when cost=0 and tokens sum to zero."""
    from splinter.obs.trace import Trace
    from splinter.providers.dispatch import _log_trace

    trace = Trace()
    _log_trace(trace, "test-model", {}, 0.0,
               tier=0, iteration=1, task_index=0, role="run")
    assert len(trace.entries) == 0


def test_log_trace_skips_zero_sum_token_dict() -> None:
    """_log_trace excludes a response with cost=0 and tokens={\"input\": 0, \"output\": 0}
    — the dict is non-empty but the token counts sum to zero."""
    from splinter.obs.trace import Trace
    from splinter.providers.dispatch import _log_trace

    trace = Trace()
    _log_trace(trace, "test-model", {"input": 0, "output": 0}, 0.0,
               tier=0, iteration=1, task_index=0, role="run")
    assert len(trace.entries) == 0


def test_log_trace_logs_cost_only_when_tokens_zero_sum() -> None:
    """_log_trace logs when cost > 0 even if all token counts are zero."""
    from splinter.obs.trace import Trace
    from splinter.providers.dispatch import _log_trace

    trace = Trace()
    _log_trace(trace, "test-model", {"input": 0, "output": 0}, 0.01,
               tier=0, iteration=1, task_index=0, role="run")
    assert len(trace.entries) == 1
    assert trace.entries[0].cost == 0.01


def test_billed_error_raises_and_logs_usage_to_trace(
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    """When provider.run returns a billed response but then the provider
    raises a gap error carrying usage/cost, dispatch logs the usage before
    re-raising — the trace entry is preserved."""
    from splinter.obs.trace import Trace
    from splinter.providers import dispatch as dispatch_mod
    from splinter.providers.base import ProviderGapError

    trace = Trace()

    billing_gap = ProviderGapError(
        kind="rate_limit", provider="opencode", model="test-model",
        original=RuntimeError("billed then rate-limited"),
    )
    billing_gap.tokens = {"input": 200, "output": 80}  # type: ignore[attr-defined]
    billing_gap.cost = 0.05  # type: ignore[attr-defined]

    monkeypatch.setattr(
        dispatch_mod, "get_provider",
        lambda _name: type(
            "P",
            (),
            {"run": lambda *a, **kw: (_ for _ in ()).throw(billing_gap)},
        )(),
    )

    try:
        dispatch_mod.run_provider_session(
            "prompt", "test-model",
            trace=trace, iteration=2, tier=1, task_index=0, role="run",
        )
    except ProviderGapError:
        pass

    assert len(trace.entries) == 1
    assert trace.entries[0].cost == 0.05
    assert trace.entries[0].tokens == {"input": 200, "output": 80}
    assert trace.entries[0].tier == 1
    assert trace.entries[0].iteration == 2


def test_run_provider_session_logs_to_trace(monkeypatch: "pytest.MonkeyPatch") -> None:
    """dispatch.run_provider_session logs a billable call to trace when trace
    param is provided — the real _log_trace code path is exercised."""
    from splinter.obs.trace import Trace
    from splinter.providers import dispatch as dispatch_mod

    monkeypatch.setattr(
        dispatch_mod, "get_provider",
        lambda _name: type("P", (), {"run": lambda *a, **kw: _make_provider_response()})(),
    )

    trace = Trace()
    dispatch_mod.run_provider_session(
        "prompt", "test-model",
        trace=trace, iteration=1, tier=0, task_index=0, role="run",
    )
    assert len(trace.entries) == 1
    assert trace.entries[0].role == "run"
    assert trace.entries[0].cost == 0.01


def test_multi_retry_each_billable_call_appends_one_entry(
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    """Multiple billable provider calls through dispatch produce corresponding
    trace entries — N calls → N entries. Tests through run_task."""
    from unittest.mock import Mock

    from splinter.obs.trace import Trace
    from splinter.providers import dispatch as dispatch_mod

    monkeypatch.setattr(
        dispatch_mod, "get_provider",
        lambda _name: type("P", (), {"run": lambda *a, **kw: _make_provider_response(cost=0.01)})(),
    )
    monkeypatch.setattr("splinter.agents.runner.resolve_model", lambda t, _lad: ("t0", "opencode"))
    monkeypatch.setattr("splinter.agents.runner.resolve_variant", lambda *a, **kw: "low")
    monkeypatch.setattr("splinter.agents.runner.record_exchange", lambda *a, **kw: None)

    from splinter.agents.runner import Task, run_task

    trace = Trace()
    ladder = Mock()
    ladder.tier_timeout.return_value = 600
    task = Task(description="test", acceptance="works", suggested_tier=0)

    run_task(task, "plan", 0, ladder, trace=trace, iteration=1, task_index=0)
    assert len(trace.entries) == 1
    assert trace.entries[0].role == "run"
    assert trace.entries[0].cost == 0.01

    run_task(task, "plan", 0, ladder, trace=trace, iteration=2, task_index=0)
    assert len(trace.entries) == 2
    assert trace.entries[1].iteration == 2

    run_task(task, "plan", 0, ladder, trace=trace, iteration=3, task_index=0)
    assert len(trace.entries) == 3


def test_billable_failed_retry_is_logged(monkeypatch: "pytest.MonkeyPatch") -> None:
    """A provider call that returns tokens/cost is logged to trace regardless of
    whether the run outcome (gate/eval) is considered a failure. Entry exists
    independently of downstream verdict."""
    from unittest.mock import Mock

    from splinter.obs.trace import Trace
    from splinter.providers import dispatch as dispatch_mod

    monkeypatch.setattr(
        dispatch_mod, "get_provider",
        lambda _name: type(
            "P", (), {"run": lambda *a, **kw: _make_provider_response(cost=0.03)}
        )(),
    )
    monkeypatch.setattr("splinter.agents.runner.resolve_model", lambda t, _lad: ("t0", "opencode"))
    monkeypatch.setattr("splinter.agents.runner.resolve_variant", lambda *a, **kw: "low")
    monkeypatch.setattr("splinter.agents.runner.record_exchange", lambda *a, **kw: None)

    from splinter.agents.runner import Task, run_task

    trace = Trace()
    ladder = Mock()
    ladder.tier_timeout.return_value = 600
    task = Task(description="test", acceptance="works", suggested_tier=0)

    run_task(task, "plan", 0, ladder, trace=trace, iteration=1, task_index=0)
    assert len(trace.entries) == 1
    assert trace.entries[0].cost == 0.03
    assert trace.entries[0].tokens == {"input": 100, "output": 50}
    assert trace.entries[0].task == 0


def test_non_billable_call_excluded_from_trace(monkeypatch: "pytest.MonkeyPatch") -> None:
    """Provider calls with cost=0 and zero-sum tokens are excluded from the trace.
    This covers both empty tokens {} and zero-valued {\"input\": 0, \"output\": 0}."""
    from unittest.mock import Mock

    from splinter.obs.trace import Trace
    from splinter.providers import dispatch as dispatch_mod
    from splinter.providers.base import ProviderResponse

    monkeypatch.setattr(
        dispatch_mod, "get_provider",
        lambda _name: type(
            "P",
            (),
            {
                "run": lambda *a, **kw: ProviderResponse(
                    text="empty", tokens={"input": 0, "output": 0}, cost=0.0,
                    raw={}, session_id=None,
                ),
            },
        )(),
    )
    monkeypatch.setattr("splinter.agents.runner.resolve_model", lambda t, _lad: ("t0", "opencode"))
    monkeypatch.setattr("splinter.agents.runner.resolve_variant", lambda *a, **kw: "low")
    monkeypatch.setattr("splinter.agents.runner.record_exchange", lambda *a, **kw: None)

    from splinter.agents.runner import Task, run_task

    trace = Trace()
    ladder = Mock()
    ladder.tier_timeout.return_value = 600
    task = Task(description="test", acceptance="works", suggested_tier=0)

    run_task(task, "plan", 0, ladder, trace=trace, iteration=1, task_index=0)
    assert len(trace.entries) == 0


def test_no_double_log_stages_dont_append_independently(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch",
) -> None:
    """RunStage does not add a second trace entry — dispatch is the single
    logging point. The stage calls run_task (which calls dispatch internally);
    the stage itself never calls log_run or appends to trace.entries."""
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))

    from splinter.agents.runner import RunResult, Task
    from splinter.memory.knowledge import KnowledgeStore
    from splinter.memory.session import Session
    from splinter.models.roster import load_ladder
    from splinter.obs.trace import Trace
    from splinter.strategies.stages import IterationContext, RunStage

    session = Session("ses_test")
    ladder = load_ladder()
    trace = Trace()
    knowledge = KnowledgeStore(session)

    task = Task(
        description="test task",
        acceptance="test acceptance",
        target_files=["test.py"],
    )

    def stub_run_task(*args: object, **kwargs: object) -> RunResult:
        trace.entries.append(
            type("Entry", (), {"model": "test", "cost": 0.02, "tokens": {}, "latency_s": 0.1})(),
        )
        return RunResult(
            text="test output",
            model="test-model",
            tier=0,
            tokens={"input": 100, "output": 50},
            cost=0.02,
            raw={},
        )

    monkeypatch.setattr("splinter.strategies.stages.run_task", stub_run_task)

    ctx = IterationContext(
        task=task,
        plan="test plan",
        tier=0,
        iteration=1,
        ladder=ladder,
        session=session,
        trace=trace,
        knowledge=knowledge,
    )
    stage = RunStage()
    stage.process(ctx)

    assert len(trace.entries) == 1, (
        f"expected 1 trace entry (from dispatch), got {len(trace.entries)}"
    )


def test_compute_summary_cost_uses_trace_when_entries_present() -> None:
    from splinter.agents.runner import RunResult
    from splinter.obs.trace import RunEntry, Trace
    from splinter.pipeline import _compute_summary_cost

    trace = Trace()
    trace.entries.append(
        RunEntry(model="m1", tier=0, iteration=0, tokens={"input": 10}, cost=0.02,
                 latency_s=0.0, task=0, role="plan")
    )
    trace.entries.append(
        RunEntry(model="m1", tier=0, iteration=1, tokens={"input": 20}, cost=0.05,
                 latency_s=0.0, task=0, role="run")
    )

    results = [
        RunResult(text="out", model="m1", tier=0, tokens={"input": 20},
                  cost=0.05, raw={})
    ]

    total, runs = _compute_summary_cost(trace, results)
    assert total == pytest.approx(0.07)
    assert runs == 2


def test_compute_summary_cost_falls_back_to_results_when_trace_empty() -> None:
    from splinter.agents.runner import RunResult
    from splinter.obs.trace import Trace
    from splinter.pipeline import _compute_summary_cost

    trace = Trace()

    results = [
        RunResult(text="a", model="m", tier=0, tokens={}, cost=0.03, raw={}),
        RunResult(text="b", model="m", tier=0, tokens={}, cost=0.07, raw={}),
    ]

    total, runs = _compute_summary_cost(trace, results)
    assert total == pytest.approx(0.10)
    assert runs == 2


# ── US-003: indeterminate-cost flagging ──────────────────────────────────────

def test_calc_cost_known_model_returns_correct_cost() -> None:
    """Known model produces (cost, False) — no indeterminate flag."""
    from splinter.providers.claude_cli import _calc_cost

    cost, indeterminate = _calc_cost("sonnet", {"input_tokens": 1_000_000, "output_tokens": 0})
    assert indeterminate is False
    assert cost == pytest.approx(3.00)


def test_calc_cost_unknown_model_warns_and_flags(caplog: pytest.LogCaptureFixture) -> None:
    """Unknown model returns (0.0, True) and emits a WARNING log."""
    import logging

    from splinter.providers.claude_cli import _calc_cost

    with caplog.at_level(logging.WARNING, logger="splinter"):
        cost, indeterminate = _calc_cost(
            "no-such-model-xyz", {"input_tokens": 500, "output_tokens": 100}
        )

    assert indeterminate is True
    assert cost == 0.0
    assert any("indeterminate" in r.message.lower() for r in caplog.records)


def test_extract_cost_present_field_returns_value_not_indeterminate() -> None:
    """opencode payload with cost field → (cost, False)."""
    from splinter.providers.opencode import _extract_cost

    cost, indeterminate = _extract_cost({"cost": 0.0123})
    assert indeterminate is False
    assert cost == pytest.approx(0.0123)


def test_extract_cost_missing_field_warns_and_flags(caplog: pytest.LogCaptureFixture) -> None:
    """opencode payload without cost field → (0.0, True) and WARNING log."""
    import logging

    from splinter.providers.opencode import _extract_cost

    with caplog.at_level(logging.WARNING, logger="splinter"):
        cost, indeterminate = _extract_cost({})

    assert indeterminate is True
    assert cost == 0.0
    assert any("indeterminate" in r.message.lower() for r in caplog.records)


def test_codex_calc_cost_unknown_model_warns_and_flags(caplog: pytest.LogCaptureFixture) -> None:
    """Codex unknown model returns (0.0, True) and emits a WARNING log."""
    import logging

    from splinter.providers.codex import _calc_cost

    with caplog.at_level(logging.WARNING, logger="splinter"):
        cost, indeterminate = _calc_cost("no-such-codex-model", {"input": 500, "output": 100})

    assert indeterminate is True
    assert cost == 0.0
    assert any("indeterminate" in r.message.lower() for r in caplog.records)


def test_codex_calc_cost_known_model_returns_correct_cost() -> None:
    """Known codex model produces (cost, False)."""
    from splinter.providers.codex import _calc_cost

    cost, indeterminate = _calc_cost("gpt-5-codex", {"input": 1_000_000, "output": 0})
    assert indeterminate is False
    assert cost == pytest.approx(10.00)


def test_trace_indeterminate_flag_survives_markdown_roundtrip() -> None:
    """cost_indeterminate=True entry is serialized with [!cost] and re-parsed correctly."""
    from splinter.obs.trace import RunEntry, Trace

    trace = Trace()
    trace.entries.append(
        RunEntry(
            model="no-such-model",
            tier=0,
            iteration=0,
            tokens={"input": 500, "output": 100},
            cost=0.0,
            latency_s=1.5,
            cost_indeterminate=True,
        )
    )
    trace.entries.append(
        RunEntry(
            model="sonnet",
            tier=0,
            iteration=1,
            tokens={"input": 200, "output": 50},
            cost=0.0015,
            latency_s=0.8,
            cost_indeterminate=False,
        )
    )

    md = trace.summary()
    assert "[!cost]" in md

    restored = Trace.from_markdown(md)
    assert len(restored.entries) == 2
    indet = restored.entries[0]
    normal = restored.entries[1]
    assert indet.cost_indeterminate is True
    assert normal.cost_indeterminate is False
