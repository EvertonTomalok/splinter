"""opencode ≥2 takes the effort as ``-m provider/model#variant``, not ``--variant``."""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from splinter.providers import opencode
from splinter.providers.opencode import model_with_variant


@pytest.mark.parametrize(
    ("model", "variant", "expected"),
    [
        ("zai-coding-plan/glm-5.3-flash", "low", "zai-coding-plan/glm-5.3-flash#low"),
        ("opencode/deepseek-v4-flash-free", None, "opencode/deepseek-v4-flash-free"),
        ("opencode/deepseek-v4-flash-free", "auto", "opencode/deepseek-v4-flash-free"),
        ("zai-coding-plan/glm-5.3#high", "low", "zai-coding-plan/glm-5.3#high"),
    ],
)
def test_model_with_variant(model: str, variant: str | None, expected: str) -> None:
    assert model_with_variant(model, variant) == expected


def test_run_puts_variant_in_model_and_never_passes_variant_flag() -> None:
    done = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    with (
        patch.object(opencode, "run_subprocess", return_value=done) as run_sub,
        patch.object(opencode, "_parse_output", return_value={}),
        patch.object(opencode, "_extract_text", return_value="ok"),
    ):
        opencode.run("hi", "zai-coding-plan/glm-5.3-flash", variant="low", timeout=5)
    cmd = run_sub.call_args.args[0]
    assert "--variant" not in cmd
    assert cmd[cmd.index("-m") + 1] == "zai-coding-plan/glm-5.3-flash#low"


@pytest.mark.parametrize(
    ("role", "expected"),
    [
        ("prd", "build"),
        ("explain", "build"),
        ("eval", "build"),
        ("plan", "plan"),
        ("build", "build"),
    ],
)
def test_pipeline_roles_map_to_agents_opencode_knows(role: str, expected: str) -> None:
    done = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    with (
        patch.object(opencode, "run_subprocess", return_value=done) as run_sub,
        patch.object(opencode, "_parse_output", return_value={}),
        patch.object(opencode, "_extract_text", return_value="ok"),
    ):
        opencode.run("hi", "zai-coding-plan/glm-5.3-flash", agent=role, timeout=5)
    cmd = run_sub.call_args.args[0]
    assert cmd[cmd.index("--agent") + 1] == expected
