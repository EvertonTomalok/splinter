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
        # Per-model-call subprocess timeout in seconds. Reasoning models on hard
        # tasks routinely run many minutes; default to an hour so a slow call is
        # never killed mid-thought. Override with `splinter configure --timeout`.
        "timeout": 3600,
    },
}


def configured_timeout() -> int:
    """The per-call model timeout (seconds) from config; defaults to 1 hour."""
    try:
        value = load_config().get("defaults", {}).get("timeout", 3600)
        return int(value)
    except (ValueError, TypeError):
        return 3600


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


# Per-step model knobs the configure TUI exposes: (role key, label, description).
MODEL_STEPS: list[tuple[str, str, str]] = [
    (
        "localizer_recall",
        "Locate · recall",
        "Broad search: greps/reads the repo to list every candidate file & symbol "
        "(coverage over precision). Cheap & fast.",
    ),
    (
        "localizer_recall_large",
        "Locate · recall (large context)",
        "Same recall, used on big repos when the search output is huge — wants a "
        "large-context model.",
    ),
    (
        "localizer_precision",
        "Locate · filter",
        "Filters the recall candidates down to the relevant file/symbol anchors the "
        "planner will use.",
    ),
    (
        "planner",
        "Plan",
        "The sensei. Reads the localization map and writes the implementation plan — "
        "once per session.",
    ),
    (
        "eval",
        "Eval",
        "The judge. Checks the output against acceptance criteria and returns "
        "PASS / RETRY / ESCALATE plus concrete fixes. Runs on a different family than "
        "the coder.",
    ),
]

# Runner ladder tiers: (label, description). Index = tier level.
TIER_STEPS: list[tuple[str, str]] = [
    ("Run · T0 easy", "Floor runner — easy → moderate-easy tasks. Where most work starts."),
    ("Run · T1 moderate", "Default runner for normal work."),
    ("Run · T2 hard", "Runner for moderate+ work."),
    ("Run · T3 premium", "Premium runner reached by escalation."),
    ("Run · T4 top", "Ceiling runner — the last rung before giving up."),
]


def available_models() -> list[str]:
    """All selectable model ids: opencode-go/* (live) + claude + ladder defaults."""
    from splinter.models.roster import load_ladder

    models: set[str] = {"sonnet", "opus"}
    try:
        from splinter.providers import opencode

        models.update(m for m in opencode.list_models() if m.startswith("opencode-go/"))
    except Exception:
        pass
    models.update(load_ladder().all_model_ids())
    return sorted(models)


# Reasoning-effort levels accepted by both claude (--effort) and opencode (--variant).
EFFORT_CHOICES = ["minimal", "low", "high", "max"]


def current_model_selections() -> dict[str, Any]:
    """Current per-step model + effort picks (ladder + any existing config override)."""
    from splinter.models.roster import load_ladder

    ladder = load_ladder()
    tiers = sorted(ladder.tiers, key=lambda t: t.level)
    return {
        "models": {
            "localizer_recall": ladder.localizer_recall_model,
            "localizer_recall_large": ladder.localizer_recall_large_model,
            "localizer_precision": ladder.localizer_precision_model,
            "planner": ladder.planner_model,
            "eval": ladder.eval_model,
            "tiers": [t.models[0] for t in tiers],
        },
        "efforts": {
            "localizer_recall": ladder.localizer_recall_variant,
            "localizer_recall_large": ladder.localizer_recall_large_variant,
            "localizer_precision": ladder.localizer_precision_variant,
            "planner": ladder.planner_effort,
            "eval": ladder.eval_effort,
            "tiers": [ladder.tier_variant(t.level) or "" for t in tiers],
        },
        "timeouts": {
            "localizer_recall": ladder.localizer_recall_timeout,
            "localizer_recall_large": ladder.localizer_recall_large_timeout,
            "localizer_precision": ladder.localizer_precision_timeout,
            "planner": ladder.planner_timeout,
            "eval": ladder.eval_timeout,
            "tiers": [ladder.tier_timeout(t.level) for t in tiers],
        },
    }


def write_model_config(
    models: dict[str, Any],
    efforts: dict[str, Any] | None = None,
    timeout: int | None = None,
    timeouts: dict[str, Any] | None = None,
) -> Path:
    """Merge ``models`` (and optional ``efforts``/``timeouts``) into config.yaml.

    ``timeout`` sets the global ``defaults.timeout`` fallback; ``timeouts`` holds
    the per-step overrides. Gate checks and the rest of ``defaults`` are preserved.
    """
    config = load_config()
    config.setdefault("gate_checks", DEFAULT_CONFIG["gate_checks"])
    config.setdefault("defaults", DEFAULT_CONFIG["defaults"].copy())
    config["models"] = models
    if efforts is not None:
        config["efforts"] = efforts
    if timeouts is not None:
        config["timeouts"] = timeouts
    if timeout is not None:
        config["defaults"]["timeout"] = timeout

    p = _config_path("project")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return p


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
    *,
    gate_checks: str | None = None,
    timeout: int | None = None,
    init_prompts: bool = False,
    force: bool = False,
    interactive: bool | None = None,
) -> int:
    import sys

    if interactive is None:
        interactive = sys.stdin.isatty() and sys.stdout.isatty()

    # Interactive with no direct flags → open the model-selection TUI, which
    # writes config.yaml itself on save.
    if interactive and not gate_checks and not init_prompts and timeout is None:
        from splinter.tui import run_configure_tui

        return run_configure_tui()

    config = load_config()

    if timeout is not None:
        config.setdefault("defaults", DEFAULT_CONFIG["defaults"].copy())
        config["defaults"]["timeout"] = timeout

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
