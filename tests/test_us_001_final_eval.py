"""US-001: Final eval gate config schema and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from splinter.configure import (
    FinalEvalEntry,
    dump_final_eval,
    load_final_eval,
    write_gate_checks,
    write_model_config,
)
from splinter.enums import FinalEvalKind, Variant


class TestLoadFinalEval:
    """Test loading final eval list from config."""

    def test_load_final_eval_empty_when_absent(self) -> None:
        config = {}
        result = load_final_eval(config)
        assert result == []

    def test_load_final_eval_single_entry(self) -> None:
        config = {"final_eval": [{"name": "gate1", "kind": "ask_user"}]}
        result = load_final_eval(config)
        assert len(result) == 1
        assert result[0].name == "gate1"
        assert result[0].kind == FinalEvalKind.ASK_USER
        assert result[0].skill is None
        assert result[0].cmd is None
        assert result[0].variant is None

    def test_load_final_eval_skill_kind(self) -> None:
        config = {
            "final_eval": [
                {
                    "name": "review",
                    "kind": "skill",
                    "skill": "cursor_review",
                    "model": "opus",
                    "variant": "high",
                }
            ]
        }
        result = load_final_eval(config)
        assert len(result) == 1
        assert result[0].kind == FinalEvalKind.SKILL
        assert result[0].skill == "cursor_review"
        assert result[0].model == "opus"
        assert result[0].variant == Variant.HIGH

    def test_load_final_eval_command_kind(self) -> None:
        config = {
            "final_eval": [
                {
                    "name": "lint",
                    "kind": "command",
                    "cmd": "ruff check",
                }
            ]
        }
        result = load_final_eval(config)
        assert len(result) == 1
        assert result[0].kind == FinalEvalKind.COMMAND
        assert result[0].cmd == "ruff check"

    def test_load_final_eval_preserves_order(self) -> None:
        config = {
            "final_eval": [
                {"name": "gate1", "kind": "ask_user"},
                {"name": "gate2", "kind": "skill", "skill": "review"},
                {"name": "gate3", "kind": "command", "cmd": "check"},
            ]
        }
        result = load_final_eval(config)
        assert len(result) == 3
        assert result[0].name == "gate1"
        assert result[1].name == "gate2"
        assert result[2].name == "gate3"

    def test_load_final_eval_invalid_kind_raises(self) -> None:
        config = {"final_eval": [{"name": "bad", "kind": "invalid_kind"}]}
        with pytest.raises(ValueError):
            load_final_eval(config)

    def test_load_final_eval_invalid_variant_raises(self) -> None:
        config = {
            "final_eval": [
                {
                    "name": "bad_variant",
                    "kind": "skill",
                    "skill": "review",
                    "variant": "invalid_variant",
                }
            ]
        }
        with pytest.raises(ValueError):
            load_final_eval(config)

    def test_load_final_eval_optional_variant(self) -> None:
        config = {
            "final_eval": [
                {
                    "name": "review",
                    "kind": "skill",
                    "skill": "cursor_review",
                }
            ]
        }
        result = load_final_eval(config)
        assert result[0].variant is None


class TestDumpFinalEval:
    """Test dumping final eval list to config dict."""

    def test_dump_final_eval_empty_list(self) -> None:
        entries: list[FinalEvalEntry] = []
        result = dump_final_eval(entries)
        assert result == []

    def test_dump_final_eval_single_entry(self) -> None:
        entries = [
            FinalEvalEntry(
                name="gate1",
                kind=FinalEvalKind.ASK_USER,
            )
        ]
        result = dump_final_eval(entries)
        assert len(result) == 1
        assert result[0]["name"] == "gate1"
        assert result[0]["kind"] == "ask_user"
        assert result[0]["skill"] is None
        assert result[0]["cmd"] is None
        assert result[0]["variant"] is None

    def test_dump_final_eval_with_variant(self) -> None:
        entries = [
            FinalEvalEntry(
                name="review",
                kind=FinalEvalKind.SKILL,
                skill="cursor_review",
                model="opus",
                variant=Variant.HIGH,
            )
        ]
        result = dump_final_eval(entries)
        assert result[0]["kind"] == "skill"
        assert result[0]["skill"] == "cursor_review"
        assert result[0]["model"] == "opus"
        assert result[0]["variant"] == "high"

    def test_dump_final_eval_preserves_order(self) -> None:
        entries = [
            FinalEvalEntry(name="gate1", kind=FinalEvalKind.ASK_USER),
            FinalEvalEntry(name="gate2", kind=FinalEvalKind.SKILL, skill="review"),
            FinalEvalEntry(name="gate3", kind=FinalEvalKind.COMMAND, cmd="check"),
        ]
        result = dump_final_eval(entries)
        assert len(result) == 3
        assert result[0]["name"] == "gate1"
        assert result[1]["name"] == "gate2"
        assert result[2]["name"] == "gate3"


class TestFinalEvalRoundTrip:
    """Test round-trip preservation of final eval config."""

    def test_roundtrip_preserves_order(self) -> None:
        original_config = {
            "final_eval": [
                {"name": "gate1", "kind": "ask_user"},
                {"name": "gate2", "kind": "skill", "skill": "review", "variant": "high"},
                {"name": "gate3", "kind": "command", "cmd": "pytest"},
            ]
        }
        loaded = load_final_eval(original_config)
        dumped = dump_final_eval(loaded)
        reloaded = load_final_eval({"final_eval": dumped})

        assert len(reloaded) == len(loaded)
        for i, entry in enumerate(reloaded):
            assert entry.name == loaded[i].name
            assert entry.kind == loaded[i].kind
            assert entry.skill == loaded[i].skill
            assert entry.cmd == loaded[i].cmd
            assert entry.variant == loaded[i].variant


class TestFinalEvalGatePersistence:
    """Test that final_eval persists through config save/load."""

    def test_write_gate_checks_preserves_final_eval(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        original_final_eval = [{"name": "review", "kind": "skill", "skill": "cursor_review"}]
        original_config = {"final_eval": original_final_eval, "gate_checks": []}
        config_path = tmp_path / ".splinter" / "config.yaml"
        config_path.parent.mkdir(parents=True)
        import yaml

        with open(config_path, "w") as f:
            yaml.dump(original_config, f)

        checks = [{"name": "test", "cmd": "pytest", "when": "always"}]
        write_gate_checks(checks)

        from splinter.configure import load_config

        reloaded = load_config()
        assert "final_eval" in reloaded
        assert reloaded["final_eval"] == original_final_eval

    def test_write_model_config_preserves_final_eval(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        original_final_eval = [{"name": "review", "kind": "skill", "skill": "cursor_review"}]
        original_config = {"final_eval": original_final_eval}
        config_path = tmp_path / ".splinter" / "config.yaml"
        config_path.parent.mkdir(parents=True)
        import yaml

        with open(config_path, "w") as f:
            yaml.dump(original_config, f)

        models = {"planner": "opus", "tiers": ["haiku"]}
        write_model_config(models)

        from splinter.configure import load_config

        reloaded = load_config()
        assert "final_eval" in reloaded
        assert reloaded["final_eval"] == original_final_eval


class TestDisabledWhenAbsent:
    """Test that final_eval is disabled when config key is absent."""

    def test_config_without_final_eval_roundtrips_unchanged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        from splinter.configure import load_config

        original_config = {
            "defaults": {"timeout": 3600},
            "gate_checks": [{"name": "test", "cmd": "pytest", "when": "always"}],
        }
        config_path = tmp_path / ".splinter" / "config.yaml"
        config_path.parent.mkdir(parents=True)
        import yaml

        with open(config_path, "w") as f:
            yaml.dump(original_config, f)

        checks = [{"name": "test", "cmd": "pytest", "when": "always"}]
        write_gate_checks(checks)

        reloaded = load_config()
        assert "final_eval" not in reloaded
