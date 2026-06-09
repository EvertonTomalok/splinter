from __future__ import annotations

import importlib.resources
from dataclasses import dataclass, field
from typing import Any

import yaml


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

    return Ladder(
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
