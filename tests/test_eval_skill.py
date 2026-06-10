from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from splinter.agents.evaluator import Evaluator
from splinter.agents.runner import Task
from splinter.models.roster import Ladder, load_ladder
from splinter.providers.base import ProviderResponse
from splinter.skills import ResolvedSkill, resolve_eval_skill


def _ladder() -> Ladder:
    return load_ladder()


# --- resolve_eval_skill ------------------------------------------------------


def test_resolve_eval_skill_none_returns_none() -> None:
    assert resolve_eval_skill(None) is None


def test_resolve_eval_skill_empty_string_returns_none() -> None:
    assert resolve_eval_skill("") is None


def test_resolve_eval_skill_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown eval skill"):
        resolve_eval_skill("nonexistent_skill_name_xyz")


def test_resolve_eval_skill_from_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    skill_dir = tmp_path / "skills" / "run_python"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("""---
name: run_python
description: Run Python and check exit code
---
Execute the impl as a Python script and gate on exit code 0.
""")
    resolved = resolve_eval_skill("run_python")
    assert resolved is not None
    assert resolved.name == "run_python"
    assert resolved.description == "Run Python and check exit code"
    assert "exit code 0" in resolved.body


def test_resolve_eval_skill_from_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    pkg_dir = tmp_path / "splinter" / "skills" / "run_python"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "SKILL.md").write_text("""---
name: run_python
description: Run Python and check exit code
---
Execute the impl.
""")
    resolved = resolve_eval_skill("run_python")
    assert resolved is not None
    assert resolved.name == "run_python"


def test_resolve_eval_cwd_wins_over_pkg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "skills" / "run_python").mkdir(parents=True)
    (tmp_path / "skills" / "run_python" / "SKILL.md").write_text("cwd skill")
    (tmp_path / "splinter" / "skills" / "run_python").mkdir(parents=True)
    (tmp_path / "splinter" / "skills" / "run_python" / "SKILL.md").write_text("pkg skill")
    resolved = resolve_eval_skill("run_python")
    assert resolved is not None
    assert resolved.body == "cwd skill"


def test_resolve_eval_skill_no_frontmatter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "skills" / "plain").mkdir(parents=True)
    (tmp_path / "skills" / "plain" / "SKILL.md").write_text("just body, no frontmatter")
    resolved = resolve_eval_skill("plain")
    assert resolved is not None
    assert resolved.name == "plain"
    assert resolved.description == ""
    assert resolved.body == "just body, no frontmatter"


# --- CLI flags are parsed into run_kwargs ------------------------------------


def test_cli_run_kwargs_includes_eval_flags() -> None:
    from splinter.cli import app

    with patch("splinter.pipeline.run_pipeline", return_value=0) as mock_run:
        app(
            args=[
                "run",
                "--task",
                "samples/hello-world-task.yaml",
                "--eval",
                "run_python",
                "--eval-model",
                "sonnet",
                "--eval-effort",
                "max",
                "--quiet",
            ],
            standalone_mode=False,
        )

    mock_run.assert_called_once()
    kwargs = mock_run.call_args.kwargs
    assert kwargs["eval_skill"] == "run_python"
    assert kwargs["eval_model"] == "sonnet"
    assert kwargs["eval_effort"] == "max"


# --- eval_model / eval_effort override ladder in pipeline --------------------


def test_pipeline_override_eval_on_ladder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    from splinter.pipeline import run_pipeline

    task_file = tmp_path / "task.yaml"
    task_file.write_text("description: test\nacceptance: test\n")

    ladder = _ladder()
    with (
        patch("splinter.pipeline.load_ladder", return_value=ladder),
        patch("splinter.pipeline.get_strategy", side_effect=ValueError("stop early")),
    ):
        try:
            run_pipeline(task_path=str(task_file), eval_model="custom-eval", eval_effort="low")
        except ValueError:
            pass

    assert ladder.eval_model == "custom-eval"
    assert ladder.eval_effort == "low"


def test_pipeline_no_override_keeps_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    from splinter.pipeline import run_pipeline

    task_file = tmp_path / "task.yaml"
    task_file.write_text("description: test\nacceptance: test\n")

    ladder = _ladder()
    original_model = ladder.eval_model
    original_effort = ladder.eval_effort

    with (
        patch("splinter.pipeline.load_ladder", return_value=ladder),
        patch("splinter.pipeline.get_strategy", side_effect=ValueError("stop early")),
    ):
        try:
            run_pipeline(task_path=str(task_file))
        except ValueError:
            pass

    assert ladder.eval_model == original_model
    assert ladder.eval_effort == original_effort


# --- eval_effort CLI override wins on premium tiers --------------------------


def test_eval_effort_for_respects_ladder_effort() -> None:
    ladder = _ladder()
    ladder.eval_effort = "low"
    ev = Evaluator(ladder)
    assert ev.eval_effort_for(2) == "low"
    assert ev.eval_effort_for(3) == "low"
    assert ev.eval_effort_for(4) == "low"


# --- judge injects skill body into prompt ------------------------------------


def test_judge_injects_skill_into_prompt() -> None:
    ladder = _ladder()
    ev = Evaluator(ladder)
    task = Task(description="test task", acceptance="must work")
    skill = ResolvedSkill(name="run_python", description="desc", body="Check exit code 0.")

    with patch(
        "splinter.agents.evaluator.run_provider_session",
        return_value=(ProviderResponse(text="VERDICT: PASS\nREASON: ok\nCORRECTIONS: none"), "sid"),
    ) as mock_run:
        ev.judge(task, "some output", eval_skill=skill)

    mock_run.assert_called_once()
    prompt_arg = mock_run.call_args.args[0]
    assert "Eval Skill" in prompt_arg
    assert "Check exit code 0" in prompt_arg


def test_judge_no_skill_no_section_in_prompt() -> None:
    ladder = _ladder()
    ev = Evaluator(ladder)
    task = Task(description="test task", acceptance="must work")

    with patch(
        "splinter.agents.evaluator.run_provider_session",
        return_value=(ProviderResponse(text="VERDICT: PASS\nREASON: ok\nCORRECTIONS: none"), "sid"),
    ) as mock_run:
        ev.judge(task, "some output")

    mock_run.assert_called_once()
    prompt_arg = mock_run.call_args.args[0]
    assert "Eval Skill" not in prompt_arg


# --- cowabunga regression ----------------------------------------------------


def test_cowabunga_disables_ask_user() -> None:
    ev = Evaluator(_ladder())
    v = type(
        "V",
        (),
        {
            "decision": "ASK_USER",
            "passed": False,
            "reason": "ambiguous",
            "corrections": "",
            "raw": "",
        },
    )()
    action = ev.next_action(v, tier=1, max_tier=4, cowabunga=True)
    assert action.stop
    assert not action.ask_user


def test_cowabunga_off_surfaces_ask_user() -> None:
    ev = Evaluator(_ladder())
    v = type(
        "V",
        (),
        {
            "decision": "ASK_USER",
            "passed": False,
            "reason": "ambiguous",
            "corrections": "",
            "raw": "",
        },
    )()
    action = ev.next_action(v, tier=1, max_tier=4, cowabunga=False)
    assert action.ask_user
    assert action.stop


# --- eval_skill resolution precedence: CLI > task.eval_skill -----------------


def test_direct_resolves_cli_skill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    (tmp_path / "skills" / "cli_skill").mkdir(parents=True)
    (tmp_path / "skills" / "cli_skill" / "SKILL.md").write_text("CLI skill body")

    from splinter.memory.session import Session
    from splinter.strategies.direct import DirectStrategy

    strat = DirectStrategy()
    task = Task(description="test", acceptance="test", eval_skill="story_skill")
    session = Session("ses_test")
    ladder = _ladder()

    with patch.object(strat, "_run_task_loop", return_value=None) as mock_loop:
        strat.execute([task], session, ladder, eval_skill="cli_skill")
        call_kwargs = mock_loop.call_args.kwargs
        assert call_kwargs["eval_skill"] == "cli_skill"


# --- eval template has skill_section placeholder -----------------------------


def test_eval_template_has_skill_section_placeholder() -> None:
    from splinter.templating import packaged_template

    tmpl = packaged_template("eval")
    assert "{skill_section}" in tmpl


# --- stages: _evaluate shim still works -------------------------------------


def test_stages_evaluate_shim_still_works() -> None:
    from splinter.strategies.stages import _evaluate

    task = Task(description="test", acceptance="test")
    with patch(
        "splinter.agents.evaluator.run_provider_session",
        return_value=(ProviderResponse(text="VERDICT: PASS\nREASON: ok\nCORRECTIONS: none"), "sid"),
    ):
        verdict = _evaluate(task, "output", "sonnet", "high")
    assert verdict.decision == "PASS"
