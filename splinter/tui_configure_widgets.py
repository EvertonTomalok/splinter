"""Widgets and copy for the configure screen: chain list, model picker, labels."""

from __future__ import annotations

import re

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Select, Static
from textual.widgets.option_list import Option

from splinter.configure_chain import (
    PROVIDERS,
    Chain,
    ChainEntry,
    display_model,
    efforts_for,
    with_provider,
)
from splinter.models.roster import provider_for

STEP_COPY: dict[str, tuple[str, str, str]] = {
    "localizer_recall": (
        "Recall",
        "Locate · recall",
        "Sweeps the repo and lists every candidate file and symbol. "
        "Coverage over precision: cheap and fast.",
    ),
    "localizer_recall_large": (
        "Recall large",
        "Locate · recall (large context)",
        "The same recall on big repos, when the search output does not fit. "
        "Wants a long-context model.",
    ),
    "localizer_precision": (
        "Filter",
        "Locate · filter",
        "Narrows the candidates down to the file and symbol anchors the plan will use.",
    ),
    "planner": (
        "Plan",
        "Plan",
        "Reads the localization map and writes the implementation plan, once per session.",
    ),
    "prd": (
        "PRD",
        "PRD",
        "Runs the PRD questions and splits tasks into briefs for the agents strategy.",
    ),
    "eval": (
        "Eval",
        "Eval",
        "The judge: checks the result against acceptance and returns PASS, RETRY or "
        "ESCALATE. Prefer a different model family than the runner.",
    ),
}

ACCEPTANCE_LABELS: dict[str, str] = {
    "gate": "gate only",
    "gate+eval": "gate + eval",
    "gate+orchestrator": "gate + orchestrator",
}
WHEN_LABELS: dict[str, str] = {
    "always": "always",
    "tests_exist": "if tests exist",
    "proto_changed": "if proto changed",
}
JEV_TARGET_LABELS: dict[str, str] = {
    "recall": "recall",
    "recall_large": "recall large",
    "filter": "filter",
}

_PROVIDER_COL = 10
_MODEL_COL = 34


def effort_label(entry: ChainEntry, index: int) -> str:
    if len(efforts_for(entry.model)) <= 1:
        return "—"
    if entry.effort:
        return entry.effort
    return "default" if index == 0 else "inherit"


def effort_choices(entry: ChainEntry, index: int) -> list[tuple[str, str]]:
    """Select options for one row: default plus the provider's native efforts."""
    native = list(efforts_for(entry.model)[1:])
    if entry.effort and entry.effort not in native:
        native.append(entry.effort)
    default = "default" if index == 0 else "inherit"
    return [(default, ""), *[(e, e) for e in native]]


def chain_option(entry: ChainEntry, index: int) -> Option:
    line = Text(no_wrap=True, overflow="ellipsis")
    line.append(f"{index + 1:>2}  ", style="dim")
    line.append(provider_for(entry.model).ljust(_PROVIDER_COL), style="dim")
    shown = display_model(entry.model)
    model = shown if len(shown) <= _MODEL_COL else shown[: _MODEL_COL - 1] + "…"
    line.append(model.ljust(_MODEL_COL), style="bold" if index == 0 else "")
    line.append("  ")
    line.append(effort_label(entry, index), style="" if entry.effort else "dim italic")
    return Option(line, id=f"entry-{index}")


def chain_head() -> Text:
    head = Text(no_wrap=True, style="dim")
    head.append("  #  ")
    head.append("provider".ljust(_PROVIDER_COL))
    head.append("model".ljust(_MODEL_COL))
    head.append("  effort")
    return head


def fallback_note(chain: Chain) -> str:
    if len(chain) <= 1:
        return "No fallback: a rate limit (429) has nowhere to reroute."
    order = ", then ".join(str(i) for i in range(2, len(chain) + 1))
    return f"If 1 is rate-limited (429), {order} runs next — each with its own effort."


def nav_option(option_id: str, label: str, summary: str) -> Option:
    line = Text(no_wrap=True, overflow="ellipsis")
    line.append(label.ljust(14))
    line.append(summary, style="dim")
    return Option(line, id=option_id)


def nav_header(option_id: str, title: str) -> Option:
    return Option(Text(title, style="bold"), id=option_id, disabled=True)


def newest_first(models: list[str]) -> list[str]:
    """Group by family, highest version first; pay-per-token gateways go last.

    opencode lists the active subscriptions (zai-coding-plan/…, minimax-coding-plan/…)
    next to hundreds of ``openrouter/…`` ids — the plans come first.
    """

    def key(model: str) -> tuple[bool, str, list[int]]:
        family = re.sub(r"[\d.]+", "", model).strip("-")
        return model.startswith("openrouter/"), family, [-int(n) for n in re.findall(r"\d+", model)]

    return sorted(models, key=key)


def match_models(models: list[str], query: str) -> list[str]:
    tokens = query.lower().split()
    return [m for m in models if all(t in m.lower() for t in tokens)]


class ChainList(OptionList):
    """The ordered models of one chain; posts edit requests to the app."""

    BINDINGS = [
        Binding("a", "request('add')", "fallback"),
        Binding("x,delete", "request('remove')", "remove"),
        Binding("right", "request('effort+')", "effort"),
        Binding("left", "request('effort-')", "effort", show=False),
        Binding("shift+up", "request('up')", "move"),
        Binding("shift+down", "request('down')", "move", show=False),
    ]

    class Request(Message):
        def __init__(self, chain_list: ChainList, action: str) -> None:
            super().__init__()
            self.chain_list = chain_list
            self.action = action

    def __init__(self, chain_key: str, *options: Option, id: str | None = None) -> None:
        super().__init__(*options, id=id)
        self.chain_key = chain_key

    def action_request(self, action: str) -> None:
        self.post_message(self.Request(self, action))


ALL_PROVIDERS = "*"


class ModelPicker(ModalScreen[str | None]):
    """Provider first, then a filterable model list; Enter on free text uses a custom id."""

    DEFAULT_CSS = """
    ModelPicker { align: center middle; background: $background 70%; }
    ModelPicker > Vertical {
        width: 76; max-width: 95%; height: auto; max-height: 85%;
        background: $surface; border: round $primary; padding: 1 2;
    }
    ModelPicker .picker-title { text-style: bold; margin-bottom: 1; }
    ModelPicker #picker-provider { width: 28; margin-bottom: 1; }
    ModelPicker Input { margin-bottom: 1; }
    ModelPicker OptionList { height: auto; max-height: 16; }
    ModelPicker OptionList, ModelPicker OptionList:focus { border: none; }
    ModelPicker .picker-hint { color: $text-muted; margin-top: 1; }
    """

    BINDINGS = [
        Binding("escape", "cancel", "cancel"),
        Binding("tab", "cycle_provider(1)", "next provider", priority=True),
        Binding("shift+tab", "cycle_provider(-1)", "prev provider", priority=True),
    ]

    def __init__(
        self,
        title: str,
        models_by_provider: dict[str, list[str]],
        provider: str = ALL_PROVIDERS,
        current: str = "",
    ) -> None:
        super().__init__()
        self._title = title
        self._groups = models_by_provider
        self._provider = provider if provider in models_by_provider else ALL_PROVIDERS
        self._current = current
        self._loading = not any(models_by_provider.values())

    def _models(self) -> list[str]:
        if self._provider == ALL_PROVIDERS:
            return newest_first(list({m for group in self._groups.values() for m in group}))
        return newest_first(list(self._groups.get(self._provider, [])))

    def compose(self) -> ComposeResult:
        providers = [p for p in PROVIDERS if p in self._groups] or list(PROVIDERS)
        self._choices = [ALL_PROVIDERS, *providers]
        with Vertical():
            yield Static(self._title, classes="picker-title")
            yield Select(
                [("all providers", ALL_PROVIDERS), *[(p, p) for p in providers]],
                value=self._provider,
                allow_blank=False,
                compact=True,
                id="picker-provider",
            )
            yield Input(placeholder="filter, or type a model id", id="picker-filter")
            yield OptionList(*self._models(), id="picker-list")
            hint = (
                "Loading models… type an id and press enter to use it anyway."
                if self._loading
                else "tab / shift+tab switch provider · ↑↓ move · enter pick · esc cancel"
            )
            yield Static(hint, classes="picker-hint")

    def on_mount(self) -> None:
        models = self._models()
        if self._current in models:
            self.query_one("#picker-list", OptionList).highlighted = models.index(self._current)
        self.query_one("#picker-filter", Input).focus()

    def _refill(self) -> None:
        """Rebuild the list for the provider/filter, keeping the current model highlighted."""
        query = self.query_one("#picker-filter", Input).value
        options = self.query_one("#picker-list", OptionList)
        keep = self._current
        if options.highlighted is not None and options.option_count:
            keep = str(options.get_option_at_index(options.highlighted).prompt)
        models = match_models(self._models(), query)
        options.clear_options()
        options.add_options(models)
        if models:
            options.highlighted = models.index(keep) if keep in models else 0

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "picker-provider" and isinstance(event.value, str):
            self._provider = event.value
            self._refill()

    def on_input_changed(self, event: Input.Changed) -> None:
        self._refill()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        options = self.query_one("#picker-list", OptionList)
        if options.highlighted is not None and options.option_count:
            option = options.get_option_at_index(options.highlighted)
            self.dismiss(str(option.prompt))
            return
        typed = with_provider(event.value, self._provider)
        self.dismiss(typed or None)

    def on_key(self, event: events.Key) -> None:
        if event.key not in ("down", "up"):
            return
        if self.focused is not self.query_one("#picker-filter", Input):
            return
        options = self.query_one("#picker-list", OptionList)
        if event.key == "down":
            options.action_cursor_down()
        else:
            options.action_cursor_up()
        event.stop()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(str(event.option.prompt))

    def action_cycle_provider(self, step: int) -> None:
        """Switch to the next/previous provider; focus stays in the filter."""
        current = self._provider if self._provider in self._choices else ALL_PROVIDERS
        nxt = self._choices[(self._choices.index(current) + step) % len(self._choices)]
        self.query_one("#picker-provider", Select).value = nxt
        self.query_one("#picker-filter", Input).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)
