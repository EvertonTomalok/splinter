"""jev on the configure screen: global provider chain + the per-step toggles.

The jev pane holds credentials only — primary and fallback provider, each with
its model and token. Whether jev helps a step is toggled on that step's own
pane (Busca, Filtro).
"""

from __future__ import annotations

from typing import Any

from textual.containers import Horizontal, Vertical
from textual.widget import Widget
from textual.widgets import Button, Checkbox, Input, Label, Select, Static

from splinter.providers.decisions import DECISION_PROVIDERS, DecisionSlot

SLOT_TITLES: tuple[str, ...] = ("Primary", "Fallback")
EYE_SHOW = "👁 show"
EYE_HIDE = "👁 hide"
NO_PROVIDER = "none"

#: Steps with a jev toggle: step key → (use_for target, label, note).
JEV_STEP_TOGGLES: dict[str, tuple[str, str, str]] = {
    "localizer_recall": (
        "recall",
        "Use jev to rank the candidates",
        "If jev fails, the LLM's list is kept.",
    ),
    "localizer_recall_large": (
        "recall_large",
        "Use jev to rank the candidates",
        "Runs when the search output is too big for Recall. If jev fails, the LLM's list "
        "is kept.",
    ),
    "localizer_precision": (
        "filter",
        "Use jev to pick the files before the summary",
        "Files below the cutoff are dropped before the LLM summarizes. If jev fails, none are.",
    ),
}


def _field(label: str, widget: Widget, unit: str = "") -> Horizontal:
    children: list[Widget] = [Label(label, classes="field-label"), widget]
    if unit:
        children.append(Label(unit, classes="field-unit"))
    return Horizontal(*children, classes="field")


def token_display(slot: DecisionSlot) -> str:
    if slot.api_key:
        return slot.api_key
    return f"${slot.api_key_env}" if slot.api_key_env else ""


def parse_token(text: str) -> dict[str, str]:
    """``$NAME`` → env var name; anything else → literal token; empty → provider default."""
    text = text.strip()
    if not text:
        return {}
    if text.startswith("$") and len(text) > 1:
        return {"api_key_env": text[1:]}
    return {"api_key": text}


def token_placeholder(provider: str) -> str:
    spec = DECISION_PROVIDERS.get(provider)
    if spec is None:
        return "pick a provider"
    return f"token or $ENV_VAR · empty uses ${spec.api_key_env}"


def model_options(provider: str, current: str = "") -> list[tuple[str, str]]:
    from splinter.models.live_models import cached_decision_models

    spec = DECISION_PROVIDERS.get(provider)
    curated = list(spec.models) if spec else []
    live = [m for m in cached_decision_models(provider) if m not in curated]
    models = curated + live
    if current and current not in models:
        models.insert(0, current)
    return [(m, m) for m in models]


def _slot_block(index: int, slot: DecisionSlot | None) -> Vertical:
    provider = slot.provider if slot else NO_PROVIDER
    choices = [(spec.label, key) for key, spec in DECISION_PROVIDERS.items()]
    if index > 0:
        choices.append(("none", NO_PROVIDER))
    models = model_options(provider, slot.model if slot else "")
    model_value = (slot.resolved_model() if slot else "") or (models[0][1] if models else "")
    model_select: Select[str] = Select(
        models or [("—", "")],
        value=model_value if models else "",
        allow_blank=False,
        id=f"jev-model-{index}",
        classes="wide-select",
    )
    model_select.disabled = provider == NO_PROVIDER
    token = Input(
        token_display(slot) if slot else "",
        password=True,
        placeholder=token_placeholder(provider),
        id=f"jev-token-{index}",
        classes="token",
    )
    token.disabled = provider == NO_PROVIDER
    eye = Button(EYE_SHOW, id=f"jev-eye-{index}", classes="eye")
    eye.disabled = token.disabled
    return Vertical(
        Static(SLOT_TITLES[index], classes="slot-title"),
        _field(
            "Provider",
            Select(choices, value=provider, allow_blank=False, id=f"jev-provider-{index}",
                   classes="wide-select"),
        ),
        _field("Model", model_select),
        _field("Token", Horizontal(token, eye, classes="token-row")),
        classes="slot",
    )


def jev_pane(providers: tuple[Any, ...], threshold: float) -> Vertical:
    slots: list[DecisionSlot | None] = [*providers[:2]]
    slots += [None] * (2 - len(slots))
    return Vertical(
        Static("jev · decision model", classes="pane-title"),
        Static(
            "jev credentials. The fallback runs when the primary fails (no token, rate "
            "limit, network). To use jev, turn it on in the Recall, Recall large and "
            "Filter steps.",
            classes="pane-desc",
        ),
        *[_slot_block(i, slot) for i, slot in enumerate(slots)],
        _field(
            "Probability cutoff",
            Input(f"{threshold:g}", type="number", id="jev-threshold", classes="num"),
            "0 to 1; candidates below it are dropped",
        ),
        Static(
            "Tokens are saved in .splinter/config.yaml (git-ignored). Prefer $ENV_VAR.",
            classes="hint",
        ),
        id="pane-jev",
        classes="pane",
    )


def step_toggle(key: str, use_for: tuple[str, ...]) -> Vertical | None:
    spec = JEV_STEP_TOGGLES.get(key)
    if spec is None:
        return None
    target, label, note = spec
    return Vertical(
        Checkbox(label, target in use_for, id=f"jev-{target}"),
        Static(note, classes="note"),
        classes="jev-toggle",
    )


def chain_summary_label(providers: tuple[Any, ...]) -> str:
    names = [s.provider for s in providers if isinstance(s, DecisionSlot)]
    return " → ".join(names) if names else "no provider"
