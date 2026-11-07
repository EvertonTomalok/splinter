"""Reads and writes the project/user ``.splinter/config.yaml``."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from splinter.enums import FinalEvalKind, Variant
from splinter.templating import TEMPLATE_NAMES, packaged_template

_config_cache: dict[str, Any] | None = None


def invalidate_config_cache() -> None:
    global _config_cache
    _config_cache = None


@dataclass(frozen=True)
class FinalEvalEntry:
    """A single final eval gate in the ordered list."""

    name: str
    kind: FinalEvalKind
    skill: str | None = None
    cmd: str | None = None
    model: str | None = None
    variant: Variant | None = None
    # Explicit provider override. When None, derived from model via provider_for().
    # Supported values: "claude", "opencode" (covers codex via opencode/codex-* model ids).
    provider: str | None = None


LANGUAGE_GATE_DEFAULTS: dict[str, list[dict[str, str]]] = {
    "python": [
        {"name": "ruff", "cmd": "ruff check .", "when": "always", "language": "python"},
        {"name": "mypy", "cmd": "mypy .", "when": "always", "language": "python"},
        {"name": "pytest", "cmd": "pytest", "when": "tests_exist", "language": "python"},
    ],
    "go": [
        {"name": "gofmt", "cmd": "gofmt -l .", "when": "always", "language": "go"},
        {"name": "go-vet", "cmd": "go vet ./...", "when": "always", "language": "go"},
        {"name": "go-test", "cmd": "go test ./...", "when": "always", "language": "go"},
    ],
    "rust": [
        {"name": "fmt", "cmd": "cargo fmt -- --check", "when": "always", "language": "rust"},
        {"name": "clippy", "cmd": "cargo clippy", "when": "always", "language": "rust"},
        {"name": "test", "cmd": "cargo test", "when": "always", "language": "rust"},
    ],
    "typescript": [
        {"name": "tsc", "cmd": "tsc --noEmit", "when": "always", "language": "typescript"},
        {"name": "eslint", "cmd": "eslint .", "when": "always", "language": "typescript"},
    ],
    "javascript-npm": [
        {"name": "lint", "cmd": "npm run lint", "when": "always", "language": "javascript-npm"},
        {"name": "test", "cmd": "npm test", "when": "always", "language": "javascript-npm"},
    ],
    "javascript-pnpm": [
        {"name": "biome", "cmd": "biome check .", "when": "always", "language": "javascript-pnpm"},
        {"name": "test", "cmd": "pnpm test", "when": "always", "language": "javascript-pnpm"},
    ],
    "javascript-yarn": [
        {"name": "lint", "cmd": "yarn lint", "when": "always", "language": "javascript-yarn"},
        {"name": "test", "cmd": "yarn test", "when": "always", "language": "javascript-yarn"},
    ],
    "node": [
        {"name": "test", "cmd": "npm test", "when": "always", "language": "node"},
    ],
    "ruby": [
        {
            "name": "rubocop",
            "cmd": "bundle exec rubocop",
            "when": "always",
            "language": "ruby",
        },
        {
            "name": "rspec",
            "cmd": "bundle exec rspec",
            "when": "tests_exist",
            "language": "ruby",
        },
        {
            "name": "proto-gen",
            "cmd": "grpc_tools_ruby_protoc --ruby_out=. --grpc_out=. proto/*.proto",
            "when": "proto_changed",
            "language": "ruby",
        },
    ],
    "cpp": [
        {
            "name": "cmake-build",
            "cmd": "cmake --build build",
            "when": "always",
            "language": "cpp",
        },
        {
            "name": "ctest",
            "cmd": "ctest --test-dir build",
            "when": "tests_exist",
            "language": "cpp",
        },
        {
            "name": "proto-gen",
            "cmd": "protoc --cpp_out=. --grpc_out=. --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` proto/*.proto",  # noqa: E501
            "when": "proto_changed",
            "language": "cpp",
        },
    ],
    "swift": [
        {
            "name": "build",
            "cmd": "swift build",
            "when": "always",
            "language": "swift",
        },
        {
            "name": "test",
            "cmd": "swift test",
            "when": "tests_exist",
            "language": "swift",
        },
        {
            "name": "proto-gen",
            "cmd": "protoc --swift_out=. --grpc-swift_out=. proto/*.proto",
            "when": "proto_changed",
            "language": "swift",
        },
    ],
    "csharp": [
        {"name": "build", "cmd": "dotnet build", "when": "always", "language": "csharp"},
        {"name": "test", "cmd": "dotnet test", "when": "tests_exist", "language": "csharp"},
        {"name": "proto-gen", "cmd": "dotnet build", "when": "proto_changed", "language": "csharp"},
    ],
    "php": [
        {
            "name": "phpstan",
            "cmd": "vendor/bin/phpstan analyse",
            "when": "always",
            "language": "php",
        },
        {
            "name": "phpunit",
            "cmd": "vendor/bin/phpunit",
            "when": "tests_exist",
            "language": "php",
        },
        {
            "name": "proto-gen",
            "cmd": "protoc --php_out=. --grpc_out=. --plugin=protoc-gen-grpc=`which grpc_php_plugin` proto/*.proto",  # noqa: E501
            "when": "proto_changed",
            "language": "php",
        },
    ],
    "java": [
        {
            "name": "build",
            "cmd": "./gradlew build -x test",
            "when": "always",
            "language": "java",
        },
        {
            "name": "test",
            "cmd": "./gradlew test",
            "when": "tests_exist",
            "language": "java",
        },
        {
            "name": "proto-gen",
            "cmd": "./gradlew generateProto",
            "when": "proto_changed",
            "language": "java",
        },
    ],
    "kotlin": [
        {
            "name": "build",
            "cmd": "./gradlew build -x test",
            "when": "always",
            "language": "kotlin",
        },
        {
            "name": "test",
            "cmd": "./gradlew test",
            "when": "tests_exist",
            "language": "kotlin",
        },
        {
            "name": "proto-gen",
            "cmd": "./gradlew generateProto",
            "when": "proto_changed",
            "language": "kotlin",
        },
    ],
    "rust-proto": [
        {
            "name": "fmt",
            "cmd": "cargo fmt -- --check",
            "when": "always",
            "language": "rust-proto",
        },
        {
            "name": "clippy",
            "cmd": "cargo clippy",
            "when": "always",
            "language": "rust-proto",
        },
        {
            "name": "test",
            "cmd": "cargo test",
            "when": "always",
            "language": "rust-proto",
        },
        {
            "name": "proto-gen",
            "cmd": "cargo build",
            "when": "proto_changed",
            "language": "rust-proto",
        },
    ],
}

DEFAULT_CONFIG: dict[str, Any] = {
    "gate_checks": [
        {"name": "ruff", "cmd": "uv run ruff check", "when": "always", "language": "python"},
        {"name": "mypy", "cmd": "uv run mypy splinter", "when": "always", "language": "python"},
        {"name": "pytest", "cmd": "uv run pytest", "when": "tests_exist", "language": "python"},
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
        "soft_budget": True,
    },
    "agents": {
        "acceptance_check": "gate+eval",
        "stall_min": 8,
        "max_attempts": 3,
        "probe_on_start": False,
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


def configured_soft_budget() -> bool:
    """Whether budget caps escalation softly (True) or hard-stops (False); defaults True."""
    try:
        value = load_config().get("defaults", {}).get("soft_budget", True)
        return bool(value)
    except (ValueError, TypeError):
        return True


DECISION_USE_TARGETS = ("recall", "recall_large", "filter")


@dataclass(frozen=True)
class DecisionsOptions:
    """Decision model (jev) under ``decisions:``.

    ``providers`` is the ordered chain (primary, then fallback), each slot with
    its own model and token; ``use_for`` holds the per-step toggles (busca /
    filtro). The flat ``provider``/``model``/``url``/``api_key_env`` fields
    mirror the primary slot for older callers.
    """

    provider: str = "openrouter"
    model: str = "typesafe/jev-1.14"
    url: str = ""
    api_key_env: str = ""
    use_for: tuple[str, ...] = ()
    threshold: float = 0.5
    providers: tuple[Any, ...] = ()


def _decision_slots(raw: dict[str, Any]) -> tuple[Any, ...]:
    """Parse ``decisions.providers``; a legacy flat config becomes one slot."""
    from splinter.providers.decisions import DECISION_PROVIDERS, DecisionSlot

    entries = raw.get("providers")
    if not isinstance(entries, list) or not entries:
        entries = [
            {
                "provider": raw.get("provider", "openrouter"),
                "model": raw.get("model", ""),
                "url": raw.get("url", ""),
                "api_key_env": raw.get("api_key_env", ""),
            }
        ]
    slots: list[DecisionSlot] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        provider = str(entry.get("provider", "")).strip()
        if provider not in DECISION_PROVIDERS:
            continue
        slots.append(
            DecisionSlot(
                provider=provider,
                model=str(entry.get("model") or "").strip(),
                api_key=str(entry.get("api_key") or "").strip(),
                api_key_env=str(entry.get("api_key_env") or "").strip(),
                url=str(entry.get("url") or "").strip(),
            )
        )
    return tuple(slots) or (DecisionSlot(),)


def configured_decisions() -> DecisionsOptions:
    """Read ``decisions:`` from config; every field falls back on invalid values."""
    from splinter.providers.decisions import DEFAULT_DECISION_MODEL

    raw = load_config().get("decisions", {})
    if not isinstance(raw, dict):
        return DecisionsOptions()
    use_raw = raw.get("use_for", [])
    use_for = (
        tuple(
            str(u).strip().lower()
            for u in use_raw
            if str(u).strip().lower() in DECISION_USE_TARGETS
        )
        if isinstance(use_raw, (list, tuple))
        else ()
    )
    try:
        threshold = min(max(float(raw.get("threshold", 0.5)), 0.0), 1.0)
    except (ValueError, TypeError):
        threshold = 0.5
    slots = _decision_slots(raw)
    primary = slots[0]
    return DecisionsOptions(
        provider=primary.provider,
        model=primary.resolved_model() or DEFAULT_DECISION_MODEL,
        url=primary.url,
        api_key_env=primary.api_key_env,
        use_for=use_for,
        threshold=threshold,
        providers=slots,
    )


def write_decisions_config(
    *,
    provider: str | None = None,
    model: str | None = None,
    url: str | None = None,
    api_key_env: str | None = None,
    use_for: list[str] | tuple[str, ...] | None = None,
    threshold: float | None = None,
    providers: list[dict[str, str]] | None = None,
) -> Path:
    """Merge the ``decisions:`` section into config.yaml; ``None`` fields are kept.

    ``providers`` replaces the whole chain (empty fields are omitted) and drops
    the legacy flat ``provider``/``model``/``url``/``api_key_env`` keys.
    """
    config = load_config()
    section = dict(config.get("decisions") or {})
    if providers is not None:
        for legacy in ("provider", "model", "url", "api_key_env"):
            section.pop(legacy, None)
        section["providers"] = [
            {k: v for k, v in entry.items() if v} for entry in providers if entry.get("provider")
        ]
    elif isinstance(section.get("providers"), list) and section["providers"]:
        first = dict(section["providers"][0])
        flat = {"provider": provider, "model": model, "url": url, "api_key_env": api_key_env}
        first.update({k: v for k, v in flat.items() if v is not None})
        section["providers"] = [first, *section["providers"][1:]]
        provider = model = url = api_key_env = None
    if provider is not None:
        section["provider"] = provider
    if model is not None:
        section["model"] = model
    if url is not None:
        section["url"] = url
    if api_key_env is not None:
        section["api_key_env"] = api_key_env
    if use_for is not None:
        section["use_for"] = [u for u in use_for if u in DECISION_USE_TARGETS]
    if threshold is not None:
        section["threshold"] = min(max(float(threshold), 0.0), 1.0)
    config["decisions"] = section
    p = _config_path("project")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    invalidate_config_cache()
    return p


ACCEPTANCE_CHECK_VALUES = ("gate", "gate+eval", "gate+orchestrator")
_LEGACY_AGENTS_KEYS = frozenset({"engines", "engine_models", "engine_efforts"})


@dataclass(frozen=True)
class AgentsOptions:
    """Run knobs under ``agents:``; the model per brief comes from its effort chain."""

    acceptance_check: str = "gate+eval"
    stall_min: float = 8.0
    max_attempts: int = 3
    probe_on_start: bool = False


def configured_agents_options() -> AgentsOptions:
    """Read ``agents:`` from config, falling back per-field on invalid values."""
    raw = load_config().get("agents", {})
    if not isinstance(raw, dict):
        return AgentsOptions()
    acceptance = str(raw.get("acceptance_check", "gate+eval")).strip()
    if acceptance not in ACCEPTANCE_CHECK_VALUES:
        acceptance = "gate+eval"
    try:
        stall_min = max(float(raw.get("stall_min", 8.0)), 0.0)
    except (ValueError, TypeError):
        stall_min = 8.0
    try:
        max_attempts = max(int(raw.get("max_attempts", 3)), 1)
    except (ValueError, TypeError):
        max_attempts = 3
    return AgentsOptions(
        acceptance_check=acceptance,
        stall_min=stall_min,
        max_attempts=max_attempts,
        probe_on_start=bool(raw.get("probe_on_start", False)),
    )


def write_agents_config(
    *,
    acceptance_check: str | None = None,
    stall_min: float | None = None,
    max_attempts: int | None = None,
    probe_on_start: bool | None = None,
) -> Path:
    """Merge the ``agents:`` run knobs into config.yaml; ``None`` fields are kept.

    The model a brief runs on is not configured here: its effort picks the
    ladder tier and that tier's model chain. Legacy per-engine keys
    (``engines``/``engine_models``/``engine_efforts``) are dropped on write.
    """
    config = load_config()
    section = {
        k: v
        for k, v in dict(config.get("agents") or DEFAULT_CONFIG["agents"]).items()
        if k not in _LEGACY_AGENTS_KEYS
    }

    if acceptance_check is not None:
        if acceptance_check not in ACCEPTANCE_CHECK_VALUES:
            raise ValueError(
                f"invalid acceptance_check {acceptance_check!r}; "
                f"use one of: {', '.join(ACCEPTANCE_CHECK_VALUES)}"
            )
        section["acceptance_check"] = acceptance_check
    if stall_min is not None:
        section["stall_min"] = max(float(stall_min), 0.0)
    if max_attempts is not None:
        section["max_attempts"] = max(int(max_attempts), 1)
    if probe_on_start is not None:
        section["probe_on_start"] = probe_on_start

    config["agents"] = section
    p = _config_path("project")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    invalidate_config_cache()
    return p


def _config_path(scope: str = "project") -> Path:
    if scope == "user":
        return Path.home() / ".splinter" / "config.yaml"
    return Path(".splinter") / "config.yaml"


def load_config(scope: str = "project") -> dict[str, Any]:
    # Deep copies throughout: a shallow copy shares nested dicts, so a caller
    # mutating e.g. config["defaults"]["timeout"] would poison the cache (and,
    # on the fallback path, the module-level DEFAULT_CONFIG) for the whole process.
    import copy as _copy

    global _config_cache
    if _config_cache is not None:
        return _copy.deepcopy(_config_cache)
    for s in ("project", "user"):
        p = _config_path(s)
        if p.exists():
            with open(p) as f:
                loaded: dict[str, Any] = yaml.safe_load(f) or {}
                _config_cache = loaded
                return _copy.deepcopy(loaded)
    _config_cache = _copy.deepcopy(DEFAULT_CONFIG)
    return _copy.deepcopy(_config_cache)


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
        "prd",
        "PRD",
        "Interactive PRD generation/refinement model used in the question/answer "
        "phase before execution starts.",
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


CLAUDE_MODELS: list[str] = [
    "fable",
    "opus",
    "sonnet",
    "haiku",
    "claude-fable-5",
    "claude-fable-5-1",
    "claude-opus-5-5",
    "claude-sonnet-5-5",
    "claude-haiku-5-5",
]

_SNAPSHOT_SUFFIX = re.compile(r"-\d{8}$")
_RETIRED_PREFIXES = ("claude-2", "claude-3", "claude-instant")


def synced_models_by_provider() -> dict[str, list[str]]:
    """Model ids from the synced catalogue (``.splinter/pricing.json``), by provider.

    "Sync prices" refreshes this from the live public catalogue, so new releases
    reach the pickers without a code change. Dated snapshots and retired
    families are left out to keep the lists short.
    """
    from splinter.models.pricing_store import load_store
    from splinter.models.roster import provider_for

    groups: dict[str, set[str]] = {}
    for model_id in load_store():
        if _SNAPSHOT_SUFFIX.search(model_id) or model_id.startswith(_RETIRED_PREFIXES):
            continue
        groups.setdefault(provider_for(model_id), set()).add(model_id)
    return {k: sorted(v) for k, v in groups.items()}

CODEX_MODELS: list[str] = [
    "codex/gpt-5.5",
    "codex/gpt-5.4",
    "codex/gpt-5.4-mini",
    "codex/gpt-5.3-codex",
    "codex/gpt-5.2",
    "codex/gpt-5-codex",
    "codex/codex-auto-review",
]


def _cursor_models() -> list[str]:
    try:
        from splinter.providers.cursor import list_models as _cursor_list

        return _cursor_list()
    except Exception:
        return ["cursor/auto"]


def sync_prices() -> tuple[int, dict[str, str]]:
    """Fetch live provider pricing and persist to ``.splinter/pricing.json``.

    Returns ``(synced_model_count, per_provider_errors)``. Writes are immutable.
    """
    from splinter.models.pricing_store import load_store, merge, save_store
    from splinter.providers.registry import fetch_all_pricing

    fetched, failures = fetch_all_pricing()
    existing = load_store()
    merged = merge(existing, fetched)
    save_store(merged)
    return len(fetched), failures


def unpriced_models(model_ids: list[str]) -> list[str]:
    """Return models with available but not-yet-synced pricing.

    Only models that have a known seed price (from the provider's bundled
    pricing table) or that belong to a provider whose ``fetch_pricing`` can
    return live rates are considered.  Models that genuinely have no known
    pricing source are silently excluded.
    """
    from splinter.models.pricing_store import load_store
    from splinter.models.roster import provider_for
    from splinter.providers.claude_cli import _seed_price as claude_seed
    from splinter.providers.codex import _seed_price as codex_seed

    store = load_store()
    missing: list[str] = []
    for model_id in model_ids:
        provider = provider_for(model_id)
        if provider == "opencode":
            continue
        # Check store first (direct + prefix match)
        price = store.get(model_id)
        if price is None:
            for prefix, candidate in store.items():
                if model_id.startswith(prefix):
                    price = candidate
                    break
        if price is not None and price.input > 0 and price.output > 0:
            continue
        # Model is unpriced in store — but only flag it if there IS a seed
        # price available.  No seed = genuinely unpriceable = nothing to sync.
        if provider == "claude":
            if claude_seed(model_id) is None:
                continue
        elif provider == "codex":
            if codex_seed(model_id) is None:
                continue
        # cursor has no seed pricing; every cursor model is priced live via
        # ``fetch_pricing``, so if it's not in the store after sync it simply
        # has no known price — skip.
        elif provider == "cursor":
            continue
        missing.append(model_id)
    return sorted(missing)


def _engine_models_by_provider() -> dict[str, list[str]]:
    """Engine-CLI models as selectable ids, grouped by engine provider.

    Only the CLIs that are not otherwise listed (mcode, kimi, agy) are exposed —
    the claude/codex/opencode engines are already covered by their provider
    model lists. Ids carry the routing prefix so provider_for resolves them.
    """
    from splinter.engines.catalog import CATALOG

    groups: dict[str, set[str]] = {}
    for engine in CATALOG:
        if engine.provider in ("mcode", "kimi", "agy") and engine.model:
            groups.setdefault(engine.provider, set()).add(f"{engine.provider}/{engine.model}")
    return {k: sorted(v) for k, v in groups.items()}


def available_models() -> list[str]:
    """All selectable model ids: claude + opencode + ladder + codex + cursor + engines."""
    from splinter.models.roster import load_ladder

    models: set[str] = set(CLAUDE_MODELS)
    try:
        from splinter.providers import opencode

        models.update(opencode.list_models())
    except Exception:
        pass
    models.update(load_ladder().all_model_ids())
    models.update(CODEX_MODELS)
    try:
        models.update(_cursor_models())
    except Exception:
        pass
    for group in _engine_models_by_provider().values():
        models.update(group)
    return sorted(models)


def available_models_by_provider() -> dict[str, list[str]]:
    """Model ids grouped by provider. Remote fetches (opencode, cursor) run in parallel."""
    import concurrent.futures

    from splinter.models.roster import load_ladder, provider_for

    claude: set[str] = set()
    opencode_set: set[str] = set()
    try:
        ladder = load_ladder()
        claude = set(CLAUDE_MODELS)
        for m in ladder.all_model_ids():
            p = provider_for(m)
            if p == "claude":
                claude.add(m)
            elif p == "opencode":
                opencode_set.add(m)
    except Exception:
        claude = set()

    def _fetch_opencode() -> list[str]:
        try:
            from splinter.providers import opencode

            return list(opencode.list_models())
        except Exception:
            return []

    def _fetch_cursor() -> list[str]:
        try:
            return _cursor_models()
        except Exception:
            return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        oc_future = pool.submit(_fetch_opencode)
        cursor_future = pool.submit(_fetch_cursor)
        opencode_set.update(oc_future.result())
        cursor_set = cursor_future.result()

    synced = synced_models_by_provider()
    groups: dict[str, list[str]] = {
        "claude": sorted(claude | set(synced.get("claude", []))),
        "codex": sorted(set(CODEX_MODELS) | set(synced.get("codex", []))),
        "cursor": sorted(set(cursor_set) | set(synced.get("cursor", []))),
        "opencode": sorted(opencode_set),
    }
    from splinter.models.live_models import cached_provider_models

    for provider, engine_models in _engine_models_by_provider().items():
        live = cached_provider_models(provider)
        groups[provider] = sorted(set(engine_models) | set(synced.get(provider, [])) | set(live))
    return groups


# Reasoning-effort variants Splinter accepts and normalizes per provider.
EFFORT_CHOICES = [str(v) for v in Variant]


def current_model_selections() -> dict[str, Any]:
    """Current per-step model + effort picks (ladder + any existing config override)."""
    from splinter.models.roster import load_ladder, provider_for

    ladder = load_ladder()
    tiers = sorted(ladder.tiers, key=lambda t: t.level)
    return {
        "models": {
            "localizer_recall": ladder.localizer_recall_model,
            "localizer_recall_large": ladder.localizer_recall_large_model,
            "localizer_precision": ladder.localizer_precision_model,
            "planner": ladder.planner_model,
            "prd": ladder.prd_model,
            "eval": ladder.eval_model,
            "tiers": [t.models[0] for t in tiers],
        },
        "tier_pools": [
            ", ".join(
                f"{m}@{t.model_variants[m]}" if t.model_variants.get(m) else m
                for m in t.models[1:]
            )
            for t in tiers
        ],
        "step_pools": {
            step: ", ".join(ladder.step_pools.get(step, [])[1:])
            for step in (
                "localizer_recall",
                "localizer_recall_large",
                "localizer_precision",
                "planner",
                "prd",
                "eval",
            )
        },
        "efforts": {
            "localizer_recall": ladder.localizer_recall_variant,
            "localizer_recall_large": ladder.localizer_recall_large_variant,
            "localizer_precision": ladder.localizer_precision_variant,
            "planner": ladder.planner_effort,
            "prd": ladder.prd_effort,
            "eval": ladder.eval_effort,
            "tiers": [ladder.tier_variant(t.level) or "" for t in tiers],
        },
        "timeouts": {
            "localizer_recall": ladder.localizer_recall_timeout,
            "localizer_recall_large": ladder.localizer_recall_large_timeout,
            "localizer_precision": ladder.localizer_precision_timeout,
            "planner": ladder.planner_timeout,
            "prd": ladder.prd_timeout,
            "eval": ladder.eval_timeout,
            "tiers": [ladder.tier_timeout(t.level) for t in tiers],
        },
        "providers": {
            "localizer_recall": provider_for(ladder.localizer_recall_model),
            "localizer_recall_large": provider_for(ladder.localizer_recall_large_model),
            "localizer_precision": provider_for(ladder.localizer_precision_model),
            "planner": provider_for(ladder.planner_model),
            "prd": provider_for(ladder.prd_model),
            "eval": provider_for(ladder.eval_model),
            "tiers": [t.provider for t in tiers],
        },
    }


def _normalize_gate_checks(checks: list[dict[str, str]]) -> list[dict[str, str]]:
    """Ensure gate checks have language field."""
    return [{**c, "language": c.get("language", "all")} for c in checks]


def load_final_eval(config: dict[str, Any]) -> list[FinalEvalEntry]:
    """Load final eval list from config dict. Returns empty list if absent.

    Validates `kind` and `variant` at boundary; raises ValueError on invalid values.
    Preserves order. When `final_eval` key is absent, returns empty list (disabled).
    """
    raw = config.get("final_eval")
    if raw is None:
        return []

    entries: list[FinalEvalEntry] = []
    for item in raw:
        kind_str = item.get("kind")
        variant_str = item.get("variant") or item.get("effort")
        kind = FinalEvalKind(kind_str)
        variant = Variant(variant_str) if variant_str else None

        entries.append(
            FinalEvalEntry(
                name=item.get("name", ""),
                kind=kind,
                skill=item.get("skill"),
                cmd=item.get("cmd"),
                model=item.get("model"),
                variant=variant,
                provider=item.get("provider"),
            )
        )

    return entries


def dump_final_eval(entries: list[FinalEvalEntry]) -> list[dict[str, str | None]]:
    """Convert final eval list to config dict format for serialization.

    Preserves order. Converts enums to plain strings for YAML output.
    """
    return [
        {
            "name": e.name,
            "kind": str(e.kind),
            "skill": e.skill,
            "cmd": e.cmd,
            "model": e.model,
            "variant": str(e.variant) if e.variant else None,
            "provider": e.provider,
        }
        for e in entries
    ]


def write_model_config(
    models: dict[str, Any],
    efforts: dict[str, Any] | None = None,
    timeout: int | None = None,
    timeouts: dict[str, Any] | None = None,
    gate_checks: list[dict[str, str]] | None = None,
    providers: dict[str, Any] | None = None,
) -> Path:
    """Merge ``models`` (and optional ``efforts``/``timeouts``/``gate_checks``) into config.yaml.

    ``timeout`` sets the global ``defaults.timeout`` fallback; ``timeouts`` holds
    the per-step overrides. Pass ``gate_checks`` to overwrite gate checks atomically
    in the same write; ``None`` (default) preserves existing or sets the default.
    """
    config = load_config()
    config.setdefault("gate_checks", DEFAULT_CONFIG["gate_checks"])
    if gate_checks is not None:
        config["gate_checks"] = _normalize_gate_checks(gate_checks)
    else:
        config["gate_checks"] = _normalize_gate_checks(config["gate_checks"])
    config.setdefault("defaults", DEFAULT_CONFIG["defaults"].copy())
    config["models"] = models
    if efforts is not None:
        config["efforts"] = efforts
    if timeouts is not None:
        config["timeouts"] = timeouts
    if timeout is not None:
        config["defaults"]["timeout"] = timeout
    if providers is not None:
        config["providers"] = providers

    p = _config_path("project")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    invalidate_config_cache()
    return p


def write_gate_checks(checks: list[dict[str, str]]) -> Path:
    """Persist gate checks into config.yaml. Preserves models, efforts, timeouts, defaults."""
    config = load_config()
    config.setdefault("defaults", DEFAULT_CONFIG["defaults"].copy())
    config["gate_checks"] = _normalize_gate_checks(checks)

    p = _config_path("project")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    invalidate_config_cache()
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


# Templates editable via `splinter configure` → Templates screen: (key, label).
EDITABLE_TEMPLATES: list[tuple[str, str]] = [
    ("agents", "AGENTS.md"),
    ("prd", "PRD"),
    ("plan", "Planner"),
    ("eval", "Eval"),
]


def template_override_path(key: str) -> Path:
    """Path where the project override for ``key`` lives. May not exist."""
    if key == "agents":
        return Path(".splinter") / "AGENTS.md"
    return _prompts_dir() / f"{key}.md"


def template_default_text(key: str) -> str:
    """Packaged/fallback text for ``key``."""
    if key == "agents":
        # No packaged default and the root AGENTS.md is NOT read here — the user
        # writes their own, saved as the .splinter/AGENTS.md override.
        return ""
    # prd/plan/eval all ship as packaged templates under splinter/prompts/.
    return packaged_template(key)


def template_current_text(key: str) -> str:
    """Current text for ``key``: project override if present, else packaged default.

    A blank override (e.g. an accidental empty save) falls back to the packaged
    default — except for ``agents``, which is legitimately empty by default.
    """
    override = template_override_path(key)
    if override.exists():
        text = override.read_text()
        if text.strip() or key == "agents":
            return text
    return template_default_text(key)


def template_is_overridden(key: str) -> bool:
    """True when a project override file exists for ``key``."""
    return template_override_path(key).exists()


def write_template_override(key: str, text: str) -> Path:
    """Write ``text`` as the project override for ``key``. Returns path written."""
    path = template_override_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def reset_template_override(key: str) -> bool:
    """Delete the project override for ``key``. Returns True if it existed."""
    path = template_override_path(key)
    if path.exists():
        path.unlink()
        return True
    return False


DEFAULT_CC_CONFIG: dict[str, Any] = {
    "defaults": {
        "strategy": "cascade",
        "effort": "auto",
        "max_iterations": 5,
        "timeout": 3600,
        "budget": None,
        "soft_budget": True,
    },
    "gate_checks": DEFAULT_CONFIG["gate_checks"],
    "models": {
        "localizer_recall": "haiku",
        "localizer_recall_large": "haiku",
        "localizer_precision": "haiku",
        "planner": "opus",
        "prd": "opus",
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
        "prd": "high",
        "eval": "high",
        "tiers": ["high", "max", "high", "max", "high", "max"],
    },
    "timeouts": {
        "localizer_recall": 3600,
        "localizer_recall_large": 3600,
        "localizer_precision": 3600,
        "planner": 3600,
        "prd": 3600,
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
    invalidate_config_cache()
    print(f"config.yaml updated from {source}")
    return 0


def _prune_accidental_sessions(before: set[str]) -> None:
    """Remove any empty sessions created as a side effect of a non-run/prd command.

    ``before`` is the set of session ids that existed before the command ran.
    Any new id that is still empty is deleted.
    """
    from splinter.memory.session import Session, delete_session, list_sessions

    after = set(list_sessions())
    for sid in after - before:
        s = Session(sid)
        if s.is_empty():
            delete_session(sid)


def _agents_flags_present(**kwargs: object) -> bool:
    return any(v is not None for v in kwargs.values())


def run_configure(
    *,
    gate_checks: str | None = None,
    timeout: int | None = None,
    init_prompts: bool = False,
    force: bool = False,
    interactive: bool | None = None,
    use_default: bool = False,
    use_cc_only: bool = False,
    sync_prices_flag: bool = False,
    agents_acceptance: str | None = None,
    agents_stall_min: float | None = None,
    agents_max_attempts: int | None = None,
    agents_probe_on_start: bool | None = None,
    decisions_provider: str | None = None,
    decisions_model: str | None = None,
    jev_recall: bool | None = None,
    jev_filter: bool | None = None,
    jev_recall_large: bool | None = None,
    decisions_threshold: float | None = None,
) -> int:
    if use_cc_only:
        return _swap_config("config.claude.yaml")
    if use_default:
        return _swap_config("config.opencode.yaml")

    import sys

    from splinter.memory.session import list_sessions

    _before = set(list_sessions())

    agents_requested = _agents_flags_present(
        acceptance=agents_acceptance,
        stall_min=agents_stall_min,
        max_attempts=agents_max_attempts,
        probe_on_start=agents_probe_on_start,
    )
    decisions_requested = _agents_flags_present(
        provider=decisions_provider,
        model=decisions_model,
        recall=jev_recall,
        recall_large=jev_recall_large,
        filter=jev_filter,
        threshold=decisions_threshold,
    )

    try:
        if sync_prices_flag:
            from splinter.models.live_models import sync_model_lists

            count, failures = sync_prices()
            print(f"synced {count} model price(s) to .splinter/pricing.json")
            for provider, err in sorted(failures.items()):
                print(f"  {provider}: {err}")
            lists, list_failures = sync_model_lists()
            for group in ("providers", "decisions"):
                for name, models in sorted(lists.get(group, {}).items()):
                    print(f"  {name}: {len(models)} model(s)")
            for name, err in sorted(list_failures.items()):
                print(f"  {name}: {err}")
            return 0

        if interactive is None:
            interactive = sys.stdin.isatty() and sys.stdout.isatty()

        # Interactive with no direct flags → open the model-selection TUI, which
        # writes config.yaml itself on save.
        if (
            interactive
            and not gate_checks
            and not init_prompts
            and timeout is None
            and not agents_requested
            and not decisions_requested
        ):
            from splinter.tui_configure import run_configure_tui

            return run_configure_tui()

        if decisions_requested:
            use_for: list[str] | None = None
            flags = {"recall": jev_recall, "recall_large": jev_recall_large, "filter": jev_filter}
            if any(v is not None for v in flags.values()):
                current = configured_decisions().use_for
                use_for = [
                    t
                    for t in DECISION_USE_TARGETS
                    if (flags[t] if flags[t] is not None else t in current)
                ]
            write_decisions_config(
                provider=decisions_provider,
                model=decisions_model,
                use_for=use_for,
                threshold=decisions_threshold,
            )

        if agents_requested:
            write_agents_config(
                acceptance_check=agents_acceptance,
                stall_min=agents_stall_min,
                max_attempts=agents_max_attempts,
                probe_on_start=agents_probe_on_start,
            )

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
                    checks.append({"name": name, "cmd": cmd, "when": "always", "language": "all"})
            config["gate_checks"] = checks

        p = _config_path("project")
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        invalidate_config_cache()
        print(f"config written to {p}")

        if init_prompts:
            written = init_prompt_templates(overwrite=force)
            if written:
                print(f"prompt templates written to {_prompts_dir()}/ (edit to customize):")
                for path in written:
                    print(f"  {path.name}")
            else:
                print(
                    f"prompt templates already present in {_prompts_dir()}/"
                    " (use --force to overwrite)"
                )

        return 0
    finally:
        _prune_accidental_sessions(_before)
