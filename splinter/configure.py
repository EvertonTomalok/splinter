"""Reads and writes the project/user ``.splinter/config.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from splinter.templating import TEMPLATE_NAMES, packaged_template

LANGUAGE_GATE_DEFAULTS: dict[str, list[dict[str, str]]] = {
    "python": [
        {"name": "ruff", "cmd": "ruff check .", "when": "always"},
        {"name": "mypy", "cmd": "mypy .", "when": "always"},
        {"name": "pytest", "cmd": "pytest", "when": "tests_exist"},
    ],
    "go": [
        {"name": "gofmt", "cmd": "gofmt -l .", "when": "always"},
        {"name": "go-vet", "cmd": "go vet ./...", "when": "always"},
        {"name": "go-test", "cmd": "go test ./...", "when": "always"},
    ],
    "rust": [
        {"name": "fmt", "cmd": "cargo fmt -- --check", "when": "always"},
        {"name": "clippy", "cmd": "cargo clippy", "when": "always"},
        {"name": "test", "cmd": "cargo test", "when": "always"},
    ],
    "typescript": [
        {"name": "tsc", "cmd": "tsc --noEmit", "when": "always"},
        {"name": "eslint", "cmd": "eslint .", "when": "always"},
    ],
    "javascript-npm": [
        {"name": "lint", "cmd": "npm run lint", "when": "always"},
        {"name": "test", "cmd": "npm test", "when": "always"},
    ],
    "javascript-pnpm": [
        {"name": "biome", "cmd": "biome check .", "when": "always"},
        {"name": "test", "cmd": "pnpm test", "when": "always"},
    ],
    "javascript-yarn": [
        {"name": "lint", "cmd": "yarn lint", "when": "always"},
        {"name": "test", "cmd": "yarn test", "when": "always"},
    ],
    "node": [
        {"name": "test", "cmd": "npm test", "when": "always"},
    ],
    "ruby": [
        {"name": "rubocop", "cmd": "bundle exec rubocop", "when": "always"},
        {"name": "rspec", "cmd": "bundle exec rspec", "when": "tests_exist"},
        {"name": "proto-gen", "cmd": "grpc_tools_ruby_protoc --ruby_out=. --grpc_out=. proto/*.proto", "when": "proto_changed"},
    ],
    "cpp": [
        {"name": "cmake-build", "cmd": "cmake --build build", "when": "always"},
        {"name": "ctest", "cmd": "ctest --test-dir build", "when": "tests_exist"},
        {"name": "proto-gen", "cmd": "protoc --cpp_out=. --grpc_out=. --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` proto/*.proto", "when": "proto_changed"},
    ],
    "swift": [
        {"name": "build", "cmd": "swift build", "when": "always"},
        {"name": "test", "cmd": "swift test", "when": "tests_exist"},
        {"name": "proto-gen", "cmd": "protoc --swift_out=. --grpc-swift_out=. proto/*.proto", "when": "proto_changed"},
    ],
    "csharp": [
        {"name": "build", "cmd": "dotnet build", "when": "always"},
        {"name": "test", "cmd": "dotnet test", "when": "tests_exist"},
        {"name": "proto-gen", "cmd": "dotnet build", "when": "proto_changed"},
    ],
    "php": [
        {"name": "phpstan", "cmd": "vendor/bin/phpstan analyse", "when": "always"},
        {"name": "phpunit", "cmd": "vendor/bin/phpunit", "when": "tests_exist"},
        {"name": "proto-gen", "cmd": "protoc --php_out=. --grpc_out=. --plugin=protoc-gen-grpc=`which grpc_php_plugin` proto/*.proto", "when": "proto_changed"},
    ],
    "java": [
        {"name": "build", "cmd": "./gradlew build -x test", "when": "always"},
        {"name": "test", "cmd": "./gradlew test", "when": "tests_exist"},
        {"name": "proto-gen", "cmd": "./gradlew generateProto", "when": "proto_changed"},
    ],
    "kotlin": [
        {"name": "build", "cmd": "./gradlew build -x test", "when": "always"},
        {"name": "test", "cmd": "./gradlew test", "when": "tests_exist"},
        {"name": "proto-gen", "cmd": "./gradlew generateProto", "when": "proto_changed"},
    ],
    "rust-proto": [
        {"name": "fmt", "cmd": "cargo fmt -- --check", "when": "always"},
        {"name": "clippy", "cmd": "cargo clippy", "when": "always"},
        {"name": "test", "cmd": "cargo test", "when": "always"},
        {"name": "proto-gen", "cmd": "cargo build", "when": "proto_changed"},
    ],
}

DEFAULT_CONFIG: dict[str, Any] = {
    "gate_checks": [
        {"name": "ruff", "cmd": "uv run ruff check", "when": "always"},
        {"name": "mypy", "cmd": "uv run mypy splinter", "when": "always"},
        {"name": "pytest", "cmd": "uv run pytest", "when": "tests_exist"},
    ],
    "defaults": {
        "strategy": "cascade",
        "effort": "auto",
        "max_iterations": 5,
        # Per-model-call subprocess timeout in seconds. Reasoning models on hard
        # tasks routinely run many minutes; default to an hour so a slow call is
        # never killed mid-thought. Override with `splinter configure --timeout`.
        "timeout": 3600,
        "budget": None,
    },
}


def configured_timeout() -> int:
    """The per-call model timeout (seconds) from config; defaults to 1 hour."""
    try:
        value = load_config().get("defaults", {}).get("timeout", 3600)
        return int(value)
    except (ValueError, TypeError):
        return 3600


def configured_budget() -> float | None:
    """The session budget target (USD) from config; None means no limit."""
    try:
        value = load_config().get("defaults", {}).get("budget")
        if value is None:
            return None
        return float(value)
    except (ValueError, TypeError):
        return None


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

# Runner ladder tiers: (label, description). Index = tier level. Must stay in sync
# with the `tiers` list in models/ladder.yaml (one entry per tier, ordered by level).
TIER_STEPS: list[tuple[str, str]] = [
    ("Run · T0 easy", "Floor runner — real-easy tasks. Cheapest rung."),
    ("Run · T1 moderate", "The workhorse — default runner where most runs live."),
    ("Run · T2 moderate-hard", "Same workhorse model, maxed reasoning, before switching models."),
    ("Run · T3 hard", "Switches model: the stronger open runner at high reasoning."),
    (
        "Run · T4 critical",
        "Frontier Claude. Reached by escalation — avoid unless cheaper rungs failed.",
    ),
    (
        "Run · T5 last-resort",
        "The very last rung: Claude maxed out. Only if `critical` still failed.",
    ),
]


def gate_default_for(language: str) -> list[dict[str, str]]:
    """Return a copy of the gate-check preset for a language.

    Returns an empty list for unknown languages. The returned list and each dict
    are independent copies, so mutations don't affect the shared preset.
    """
    if language not in LANGUAGE_GATE_DEFAULTS:
        return []
    return [dict(entry) for entry in LANGUAGE_GATE_DEFAULTS[language]]


def gate_default_languages() -> list[str]:
    """Sorted list of language keys available for gate-check presets."""
    return sorted(LANGUAGE_GATE_DEFAULTS)


def available_models() -> list[str]:
    """All selectable model ids: opencode-go/* & opencode/* (live) + claude + ladder defaults."""
    from splinter.models.roster import load_ladder

    models: set[str] = {"sonnet", "opus"}
    try:
        from splinter.providers import opencode

        models.update(
            m
            for m in opencode.list_models()
            if m.startswith("opencode-go/") or m.startswith("opencode/")
        )
    except Exception:
        pass
    models.update(load_ladder().all_model_ids())
    return sorted(models)


# Reasoning-effort levels accepted by both claude (--effort) and opencode (--variant).
EFFORT_CHOICES = ["minimal", "low", "medium", "high", "xhigh", "max"]


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
    gate_checks: list[dict[str, str]] | None = None,
) -> Path:
    """Merge ``models`` (and optional ``efforts``/``timeouts``/``gate_checks``) into config.yaml.

    ``timeout`` sets the global ``defaults.timeout`` fallback; ``timeouts`` holds
    the per-step overrides. Pass ``gate_checks`` to overwrite gate checks atomically
    in the same write; ``None`` (default) preserves existing or sets the default.
    """
    config = load_config()
    config.setdefault("gate_checks", DEFAULT_CONFIG["gate_checks"])
    if gate_checks is not None:
        config["gate_checks"] = gate_checks
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


def write_gate_checks(checks: list[dict[str, str]]) -> Path:
    """Persist gate checks into config.yaml. Preserves models, efforts, timeouts, defaults."""
    config = load_config()
    config.setdefault("defaults", DEFAULT_CONFIG["defaults"].copy())
    config["gate_checks"] = checks

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


DEFAULT_CC_CONFIG: dict[str, Any] = {
    "defaults": {
        "strategy": "cascade",
        "effort": "auto",
        "max_iterations": 5,
        "timeout": 3600,
        "budget": None,
    },
    "gate_checks": DEFAULT_CONFIG["gate_checks"],
    "models": {
        "localizer_recall": "haiku",
        "localizer_recall_large": "haiku",
        "localizer_precision": "haiku",
        "planner": "opus",
        "eval": "opus",
        "tiers": [
            "haiku",  # T0 easy
            "haiku",  # T1 moderate
            "sonnet",  # T2 moderate-hard
            "sonnet",  # T3 hard
            "opus",  # T4 critical
            "opus",  # T5 last-resort
        ],
    },
    "efforts": {
        "localizer_recall": "low",
        "localizer_recall_large": "low",
        "localizer_precision": "low",
        "planner": "high",
        "eval": "high",
        "tiers": ["high", "max", "high", "max", "high", "max"],
    },
    "timeouts": {
        "localizer_recall": 3600,
        "localizer_recall_large": 3600,
        "localizer_precision": 3600,
        "planner": 3600,
        "eval": 3600,
        "tiers": [3600, 3600, 3600, 3600, 3600, 3600],
    },
}


def _swap_config(source: str) -> int:
    """Copy ``.splinter/<source>`` over ``.splinter/config.yaml``.

    For ``config.claude.yaml``: auto-generates the CC-only profile on first use
    so the flag works on a fresh clone (where ``.splinter/`` is gitignored).
    """
    src = _config_path("project").parent / source
    dst = _config_path("project")
    if not src.exists():
        if source == "config.claude.yaml":
            src.parent.mkdir(parents=True, exist_ok=True)
            with open(src, "w") as f:
                yaml.dump(DEFAULT_CC_CONFIG, f, default_flow_style=False, sort_keys=False)
            print(f"created {src} (default CC-only profile — haiku/sonnet/opus runners)")
        else:
            print(f"error: {src} not found — run `splinter configure` to generate it first")
            return 1
    import shutil

    shutil.copy2(src, dst)
    print(f"config.yaml updated from {source}")
    return 0


def run_configure(
    *,
    gate_checks: str | None = None,
    timeout: int | None = None,
    init_prompts: bool = False,
    force: bool = False,
    interactive: bool | None = None,
    use_default: bool = False,
    use_cc_only: bool = False,
) -> int:
    if use_cc_only:
        return _swap_config("config.claude.yaml")
    if use_default:
        return _swap_config("config.opencode.yaml")

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
                f"prompt templates already present in {_prompts_dir()}/ (use --force to overwrite)"
            )

    return 0
