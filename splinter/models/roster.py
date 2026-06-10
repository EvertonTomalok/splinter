from __future__ import annotations

import importlib.resources
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def provider_for(model_id: str) -> str:
    """Infer the provider from a model id."""
    return "opencode" if model_id.startswith("opencode-go/") else "claude"


@dataclass
class Tier:
    name: str
    level: int
    models: list[str]
    provider: str = "opencode"
    variants: dict[str, str | None] = field(default_factory=dict)


@dataclass
class EffortMapping:
    start_tier: int
    variant: str


@dataclass
class Ladder:
    tiers: list[Tier]
    effort_map: dict[str, EffortMapping]
    eval_model: str
    eval_effort: str
    planner_model: str
    planner_effort: str
    localizer_recall_model: str
    localizer_recall_large_model: str
    localizer_precision_model: str
    localizer_recall_variant: str = "minimal"
    localizer_recall_large_variant: str = "minimal"
    localizer_precision_variant: str = "low"
    # per-tier reasoning variant override, keyed by tier level (else effort_map)
    tier_variants: dict[int, str] = field(default_factory=dict)
    # per-step subprocess timeout (seconds); default filled from config in load_ladder
    default_timeout: int = 3600
    eval_timeout: int = 3600
    planner_timeout: int = 3600
    localizer_recall_timeout: int = 3600
    localizer_recall_large_timeout: int = 3600
    localizer_precision_timeout: int = 3600
    tier_timeouts: dict[int, int] = field(default_factory=dict)

    def tier_variant(self, level: int) -> str | None:
        return self.tier_variants.get(level)

    def tier_timeout(self, level: int) -> int:
        return self.tier_timeouts.get(level, self.default_timeout)

    def tier_by_level(self, level: int) -> Tier:
        for t in self.tiers:
            if t.level == level:
                return t
        raise ValueError(f"no tier at level {level}")

    def tier_by_name(self, name: str) -> Tier:
        for t in self.tiers:
            if t.name == name:
                return t
        raise ValueError(f"no tier named {name}")

    def all_model_ids(self) -> list[str]:
        ids: list[str] = []
        for t in self.tiers:
            ids.extend(t.models)
        return ids

    def opencode_model_ids(self) -> list[str]:
        return [m for t in self.tiers if t.provider == "opencode" for m in t.models]

    def effort_mapping(self, effort: str) -> EffortMapping | None:
        return self.effort_map.get(effort)


def _load_raw() -> dict[str, Any]:
    ref = importlib.resources.files("splinter.models") / "ladder.yaml"
    with importlib.resources.as_file(ref) as p:
        with open(p) as f:
            data: dict[str, Any] = yaml.safe_load(f)
            return data


def load_ladder(raw: dict[str, Any] | None = None) -> Ladder:
    if raw is None:
        raw = _load_raw()

    tiers: list[Tier] = []
    for td in raw["tiers"]:
        tiers.append(
            Tier(
                name=td["name"],
                level=td["level"],
                models=td["models"],
                provider=td.get("provider", "opencode"),
                variants=td.get("variants", {}),
            )
        )

    effort_map: dict[str, EffortMapping] = {}
    for name, em in raw.get("effort_map", {}).items():
        effort_map[name] = EffortMapping(start_tier=em["start_tier"], variant=em["variant"])

    eval_cfg = raw.get("eval", {})
    planner_cfg = raw.get("planner", {})
    loc_cfg = raw.get("localizer", {})

    ladder = Ladder(
        tiers=tiers,
        effort_map=effort_map,
        eval_model=eval_cfg.get("default_model", "sonnet"),
        eval_effort=eval_cfg.get("default_effort", "high"),
        planner_model=planner_cfg.get("model", "sonnet"),
        planner_effort=planner_cfg.get("effort", "high"),
        localizer_recall_model=loc_cfg.get("recall_model", "opencode-go/deepseek-v4-flash"),
        localizer_recall_large_model=loc_cfg.get(
            "recall_model_large", "opencode-go/minimax-m3"
        ),
        localizer_precision_model=loc_cfg.get("precision_model", "opencode-go/kimi-k2.6"),
    )
    # Seed every per-step timeout from the global default, then let per-step
    # config entries override individual steps in _apply_config_overrides.
    from splinter.configure import configured_timeout

    gdefault = configured_timeout()
    ladder.default_timeout = gdefault
    ladder.eval_timeout = gdefault
    ladder.planner_timeout = gdefault
    ladder.localizer_recall_timeout = gdefault
    ladder.localizer_recall_large_timeout = gdefault
    ladder.localizer_precision_timeout = gdefault

    # Per-tier reasoning variant declared in ladder.yaml (each tier is a fixed
    # model + reasoning level). Config `efforts.tiers` can still override below.
    for td in raw["tiers"]:
        if td.get("variant"):
            ladder.tier_variants[td["level"]] = td["variant"]

    _apply_config_overrides(ladder)
    return ladder


def _project_config() -> dict[str, Any]:
    p = Path(".splinter") / "config.yaml"
    if not p.exists():
        return {}
    cfg = yaml.safe_load(p.read_text()) or {}
    return cfg if isinstance(cfg, dict) else {}


def _apply_config_overrides(ladder: Ladder) -> None:
    """Apply per-step model + effort overrides from ./.splinter/config.yaml."""
    cfg = _project_config()
    m = cfg.get("models") if isinstance(cfg.get("models"), dict) else {}
    e = cfg.get("efforts") if isinstance(cfg.get("efforts"), dict) else {}
    t = cfg.get("timeouts") if isinstance(cfg.get("timeouts"), dict) else {}

    if m:
        if m.get("localizer_recall"):
            ladder.localizer_recall_model = m["localizer_recall"]
        if m.get("localizer_recall_large"):
            ladder.localizer_recall_large_model = m["localizer_recall_large"]
        if m.get("localizer_precision"):
            ladder.localizer_precision_model = m["localizer_precision"]
        if m.get("planner"):
            ladder.planner_model = m["planner"]
        if m.get("eval"):
            ladder.eval_model = m["eval"]
        for level, model_id in enumerate(m.get("tiers") or []):
            if not model_id:
                continue
            for tier in ladder.tiers:
                if tier.level == level:
                    tier.models = [model_id, *tier.models[1:]]
                    tier.provider = provider_for(model_id)
                    break

    if e:
        if e.get("localizer_recall"):
            ladder.localizer_recall_variant = e["localizer_recall"]
        if e.get("localizer_recall_large"):
            ladder.localizer_recall_large_variant = e["localizer_recall_large"]
        if e.get("localizer_precision"):
            ladder.localizer_precision_variant = e["localizer_precision"]
        if e.get("planner"):
            ladder.planner_effort = e["planner"]
        if e.get("eval"):
            ladder.eval_effort = e["eval"]
        for level, variant in enumerate(e.get("tiers") or []):
            if variant:
                ladder.tier_variants[level] = variant

    if t:
        def _to_int(value: Any) -> int | None:
            try:
                n = int(value)
                return n if n > 0 else None
            except (ValueError, TypeError):
                return None

        if _to_int(t.get("localizer_recall")):
            ladder.localizer_recall_timeout = _to_int(t["localizer_recall"])  # type: ignore[assignment]
        if _to_int(t.get("localizer_recall_large")):
            ladder.localizer_recall_large_timeout = _to_int(t["localizer_recall_large"])  # type: ignore[assignment]
        if _to_int(t.get("localizer_precision")):
            ladder.localizer_precision_timeout = _to_int(t["localizer_precision"])  # type: ignore[assignment]
        if _to_int(t.get("planner")):
            ladder.planner_timeout = _to_int(t["planner"])  # type: ignore[assignment]
        if _to_int(t.get("eval")):
            ladder.eval_timeout = _to_int(t["eval"])  # type: ignore[assignment]
        for level, raw_to in enumerate(t.get("tiers") or []):
            n = _to_int(raw_to)
            if n:
                ladder.tier_timeouts[level] = n
