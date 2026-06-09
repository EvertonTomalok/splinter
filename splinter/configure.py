"""Reads and writes the project/user ``.splinter/config.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from splinter.templating import TEMPLATE_NAMES, packaged_template

DEFAULT_CONFIG: dict[str, Any] = {
    "gate_checks": [
        {"name": "ruff", "cmd": "uv run ruff check", "when": "always"},
        {"name": "mypy", "cmd": "uv run mypy splinter", "when": "always"},
        {"name": "pytest", "cmd": "uv run pytest", "when": "tests_exist"},
    ],
    "defaults": {
        "strategy": "direct",
        "effort": "auto",
        "max_iterations": 5,
    },
}


def _config_path(scope: str = "project") -> Path:
    if scope == "user":
        return Path.home() / ".splinter" / "config.yaml"
    return Path(".splinter") / "config.yaml"


def load_config(scope: str = "project") -> dict[str, Any]:
    for s in ("project", "user"):
        p = _config_path(s)
        if p.exists():
            with open(p) as f:
                loaded: dict[str, Any] = yaml.safe_load(f) or {}
                return loaded
    return DEFAULT_CONFIG.copy()


def _prompts_dir() -> Path:
    return Path(".splinter") / "prompts"


def init_prompt_templates(*, overwrite: bool = False) -> list[Path]:
    """Write the packaged prompt templates into ``./.splinter/prompts/`` for editing.

    Existing files are kept unless ``overwrite`` is set. Returns the paths written.
    """
    out_dir = _prompts_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for name in TEMPLATE_NAMES:
        dest = out_dir / f"{name}.md"
        if dest.exists() and not overwrite:
            continue
        dest.write_text(packaged_template(name))
        written.append(dest)
    return written


def run_configure(
    *, gate_checks: str | None = None, init_prompts: bool = False, force: bool = False
) -> int:
    config = load_config()

    if gate_checks:
        checks = []
        for cmd in gate_checks.split(","):
            cmd = cmd.strip()
            if cmd:
                name = cmd.split()[0] if cmd else "check"
                checks.append({"name": name, "cmd": cmd, "when": "always"})
        config["gate_checks"] = checks

    p = _config_path("project")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"config written to {p}")

    if init_prompts:
        written = init_prompt_templates(overwrite=force)
        if written:
            print(f"prompt templates written to {_prompts_dir()}/ (edit to customize):")
            for path in written:
                print(f"  {path.name}")
        else:
            print(
                f"prompt templates already present in {_prompts_dir()}/ "
                "(use --force to overwrite)"
            )

    return 0
