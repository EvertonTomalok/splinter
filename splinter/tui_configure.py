"""``splinter configure`` — list + detail: pick a step, edit its fallback chain.

Left: steps, effort tiers (effort → model) and run settings, each with a
one-line summary. Right: the selected item. A chain is edited in place —
models listed in fallback order, each with its own effort.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widget import Widget
from textual.widgets import (
    Button,
    Checkbox,
    ContentSwitcher,
    Footer,
    Input,
    Label,
    OptionList,
    Select,
    Static,
)
from textual.widgets.option_list import Option

from splinter.configure_chain import (
    PROVIDERS,
    STEP_KEYS,
    Chain,
    ChainEntry,
    append_entry,
    chain_summary,
    chains_from_ladder,
    config_payload,
    cycle_effort,
    efforts_for,
    efforts_starting_at,
    move_entry,
    remove_entry,
    replace_model,
    set_effort,
    tier_key,
    timeouts_from_ladder,
)
from splinter.models.roster import provider_for
from splinter.tui_configure_jev import (
    EYE_HIDE,
    EYE_SHOW,
    JEV_STEP_TOGGLES,
    NO_PROVIDER,
    chain_summary_label,
    jev_pane,
    model_options,
    parse_token,
    step_toggle,
    token_placeholder,
)
from splinter.tui_configure_widgets import (
    ACCEPTANCE_LABELS,
    STEP_COPY,
    WHEN_LABELS,
    ChainList,
    ModelPicker,
    chain_head,
    chain_option,
    effort_choices,
    fallback_note,
    nav_header,
    nav_option,
)

_WHEN_CHOICES = list(WHEN_LABELS)
_RUN_ITEMS = ("gates", "agents", "jev")


class ConfigureApp(App[bool]):
    """Steps and effort tiers on the left, the selected chain on the right."""

    CSS = """
    Screen { background: $background; }
    #body { height: 1fr; }
    #side { width: 40; height: 1fr; border-right: solid $panel; }
    #nav { height: 1fr; border: none; background: $background; padding: 1 1 0 1; }
    #nav:focus { background-tint: $foreground 3%; }
    #tools { height: auto; padding: 0 1 1 2; }
    #tool-buttons { height: auto; margin-top: 1; }
    #tool-buttons Button { margin-right: 2; }
    #unpriced-banner { color: $warning; text-style: bold; height: auto; }
    #sync-status { color: $text-muted; height: auto; }
    #detail { width: 1fr; height: 1fr; padding: 1 4 1 3; }
    .pane { height: auto; }
    .pane-title { text-style: bold; }
    .pane-desc { color: $text-muted; margin-bottom: 1; max-width: 88; }
    .chain-head { margin-top: 1; }
    ChainList { height: auto; max-height: 14; border: none; background: $background; }
    ChainList:focus { background-tint: $foreground 3%; }
    .chain-actions { height: auto; margin-top: 1; }
    .chain-actions Button { margin-right: 2; }
    .effort-label { width: auto; margin-left: 2; margin-right: 1; color: $text-muted; }
    .effort-select { width: 18; }
    .note { color: $text-muted; margin-top: 1; }
    .hint { color: $text-muted; margin-top: 1; }
    .field { height: auto; margin-top: 1; }
    .field-label { width: 30; height: 1; }
    .field-unit { width: auto; height: 1; margin-left: 2; color: $text-muted; }
    .num { width: 12; }
    .wide-select { width: 32; }
    #gate-rows { height: auto; margin-top: 1; }
    .gate-row { height: auto; margin-bottom: 1; }
    .gate-name { width: 8; text-style: bold; }
    .gate-cmd { width: 1fr; }
    .gate-when { width: 19; margin-left: 2; }
    .gate-lang { width: 13; margin-left: 2; }
    .gate-del { width: auto; min-width: 9; margin-left: 2; }
    .gate-actions { height: auto; }
    .gate-add-input { width: 1fr; margin-right: 2; }
    .gate-actions Button { margin-left: 1; }
    .slot { height: auto; margin-top: 1; }
    .slot-title { text-style: bold; color: $text-muted; }
    .token-row { height: auto; width: 1fr; }
    .token { width: 1fr; max-width: 48; }
    .eye { width: auto; margin-left: 2; }
    .jev-toggle { height: auto; margin-top: 1; }
    """

    BINDINGS = [
        Binding("s", "save", "save"),
        Binding("q", "quit", "quit"),
        Binding("ctrl+q", "quit", "quit", show=False),
        Binding("escape", "focus_nav", "back", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        from splinter.configure import (
            DEFAULT_CONFIG,
            configured_agents_options,
            configured_decisions,
            load_config,
        )
        from splinter.models.roster import load_ladder

        self.saved = False
        self.saved_path = ""
        self._ladder = load_ladder()
        self._tier_levels = sorted(t.level for t in self._ladder.tiers)
        self._chains: dict[str, Chain] = chains_from_ladder(self._ladder)
        self._timeouts = timeouts_from_ladder(self._ladder)
        self._models_by_provider: dict[str, list[str]] = {}
        self._agents = configured_agents_options()
        self._decisions = configured_decisions()
        self._gate_checks: list[dict[str, str]] = copy.deepcopy(
            load_config().get("gate_checks") or DEFAULT_CONFIG["gate_checks"]
        )
        self._sync_message = ""

    # ── layout ────────────────────────────────────────────────────────────

    def _prev_level(self, level: int) -> int | None:
        idx = self._tier_levels.index(level)
        return self._tier_levels[idx - 1] if idx > 0 else None

    def _inherited_effort(self, level: int) -> str:
        """The effort whose failures land on ``level`` (nearest lower starting tier)."""
        prev = self._prev_level(level)
        while prev is not None:
            efforts = efforts_starting_at(self._ladder, prev)
            if efforts:
                return efforts[0]
            prev = self._prev_level(prev)
        return ""

    def _tier_label(self, level: int) -> str:
        efforts = efforts_starting_at(self._ladder, level)
        if efforts:
            return f"T{level} {efforts[0]}"
        origin = self._inherited_effort(level)
        return f"T{level} {origin} ↑" if origin else f"T{level}"

    def _nav_options(self) -> list[Option]:
        options: list[Option] = [nav_header("h-steps", "STEPS")]
        options += [
            nav_option(key, STEP_COPY[key][0], self._step_summary(key))
            for key in STEP_KEYS
        ]
        options += [Option("", id="s-1", disabled=True), nav_header("h-tiers", "EFFORT → MODEL")]
        options += [
            nav_option(key, self._tier_label(level), chain_summary(self._chains[key]))
            for level in self._tier_levels
            for key in (tier_key(level),)
        ]
        options += [Option("", id="s-2", disabled=True), nav_header("h-run", "RUN")]
        options += [nav_option(item, *self._run_summary(item)) for item in _RUN_ITEMS]
        return options

    def _run_summary(self, item: str) -> tuple[str, str]:
        if item == "gates":
            return "Gates", f"{len(self._gate_checks)} checks"
        if item == "agents":
            return "Agents", ACCEPTANCE_LABELS.get(self._agents.acceptance_check, "")
        return "jev", chain_summary_label(self._decisions.providers)

    def _step_summary(self, key: str) -> str:
        toggle = JEV_STEP_TOGGLES.get(key)
        if toggle and toggle[0] in self._decisions.use_for:
            return f"{chain_summary(self._chains[key], width=12)} · jev"
        return chain_summary(self._chains[key])

    def compose(self) -> ComposeResult:
        with Horizontal(id="body"):
            with Vertical(id="side"):
                yield OptionList(*self._nav_options(), id="nav")
                with Vertical(id="tools"):
                    yield Static("", id="unpriced-banner")
                    yield Static("", id="sync-status")
                    with Horizontal(id="tool-buttons"):
                        yield Button("Sync models", id="sync-prices")
                        yield Button("Templates ▸", id="goto-templates")
            with VerticalScroll(id="detail"):
                with ContentSwitcher(initial=f"pane-{STEP_KEYS[0]}", id="switcher"):
                    for key in STEP_KEYS:
                        _, title, desc = STEP_COPY[key]
                        yield self._chain_pane(key, title, desc)
                    for level in self._tier_levels:
                        yield self._chain_pane(
                            tier_key(level), *self._tier_copy(level)
                        )
                    yield self._gates_pane()
                    yield self._agents_pane()
                    yield self._jev_pane()
        yield Footer()

    def _tier_copy(self, level: int) -> tuple[str, str]:
        efforts = efforts_starting_at(self._ladder, level)
        prev = self._prev_level(level)
        if efforts:
            title = f"T{level} · {', '.join(efforts)}"
            desc = f"{' and '.join(efforts).capitalize()} tasks start here."
            if prev is not None:
                desc += f" Also receives whatever fails twice in a row on T{prev}."
            desc += " Agents-strategy briefs with this effort run on this chain."
        else:
            origin = self._inherited_effort(level)
            title = f"T{level} · {origin} after 2 failures"
            desc = (
                f"Second chance for {origin} tasks: receives what failed twice in a row "
                f"on T{prev}. No task starts here."
            )
        if level != self._tier_levels[-1]:
            nxt = self._tier_levels[self._tier_levels.index(level) + 1]
            desc += f" Two failures here escalate to T{nxt}."
        else:
            desc += " Last rung: if it fails here, the loop stops."
        return title, desc

    def _chain_pane(self, key: str, title: str, desc: str) -> Vertical:
        chain = self._chains[key]
        timeout = self._timeouts.get(key)
        return Vertical(
            Static(title, classes="pane-title"),
            Static(desc, classes="pane-desc"),
            Static(chain_head(), classes="chain-head"),
            ChainList(
                key,
                *[chain_option(e, i) for i, e in enumerate(chain)],
                id=f"chain-{key}",
            ),
            Horizontal(
                Button("+ fallback", id=f"add-{key}"),
                Button("Swap model", id=f"swap-{key}"),
                Label("Effort", classes="effort-label"),
                self._effort_select(key, chain[0], 0),
                classes="chain-actions",
            ),
            Static(fallback_note(chain), id=f"note-{key}", classes="note"),
            Static(
                "Keys on the list:  ↑↓ select row · ←→ change effort · shift+↑↓ reorder · "
                "a add fallback · x remove · enter swap model",
                classes="hint",
            ),
            *[w for w in (step_toggle(key, self._decisions.use_for),) if w is not None],
            Horizontal(
                Label("Timeout per call", classes="field-label"),
                Input(
                    value=str(timeout) if timeout else "",
                    placeholder="3600",
                    type="integer",
                    id=f"timeout-{key}",
                    classes="num",
                ),
                Label("seconds", classes="field-unit"),
                classes="field",
            ),
            id=f"pane-{key}",
            classes="pane",
        )

    def _gate_row(self, index: int, check: dict[str, str]) -> Horizontal:
        from splinter.configure import gate_default_languages

        lang_choices = gate_default_languages()
        lang = check.get("language", "")
        when = check.get("when", "always")
        return Horizontal(
            Label(check.get("name", ""), classes="gate-name"),
            Input(value=check.get("cmd", ""), id=f"gate_cmd_{index}", classes="gate-cmd"),
            Select(
                [(WHEN_LABELS[w], w) for w in _WHEN_CHOICES],
                value=when if when in _WHEN_CHOICES else "always",
                allow_blank=False,
                id=f"gate_when_{index}",
                classes="gate-when",
            ),
            Select(
                [("all", "all"), *[(lc, lc) for lc in lang_choices]],
                value=lang if lang in lang_choices else "all",
                allow_blank=False,
                id=f"gate_lang_{index}",
                classes="gate-lang",
            ),
            Button("remove", id=f"gate_del_{index}", classes="gate-del"),
            classes="gate-row",
        )

    def _gates_pane(self) -> Vertical:
        return Vertical(
            Static("Gates", classes="pane-title"),
            Static(
                "Deterministic checks after every run. If one fails, the result goes "
                "back to the runner before eval.",
                classes="pane-desc",
            ),
            Horizontal(
                Input(
                    id="gate_add_input",
                    placeholder="command; another command",
                    classes="gate-add-input",
                ),
                Button("Add", id="gate_add"),
                Button("Language preset…", id="gate_preset"),
                classes="gate-actions",
            ),
            Vertical(*[self._gate_row(i, c) for i, c in enumerate(self._gate_checks)],
                     id="gate-rows"),
            id="pane-gates",
            classes="pane",
        )

    def _field(self, label: str, widget: Widget, unit: str = "") -> Horizontal:
        children: list[Widget] = [Label(label, classes="field-label"), widget]
        if unit:
            children.append(Label(unit, classes="field-unit"))
        return Horizontal(*children, classes="field")

    def _agents_pane(self) -> Vertical:
        opts = self._agents
        return Vertical(
            Static("Agents", classes="pane-title"),
            Static(
                "Each brief's model comes from its effort, through the EFFORT → MODEL "
                "chains. Only the run rules live here.",
                classes="pane-desc",
            ),
            self._field(
                "Brief acceptance",
                Select(
                    [(label, value) for value, label in ACCEPTANCE_LABELS.items()],
                    value=opts.acceptance_check,
                    allow_blank=False,
                    id="agents-acceptance",
                    classes="wide-select",
                ),
            ),
            self._field(
                "No output for",
                Input(f"{opts.stall_min:g}", type="number", id="agents-stall-min",
                      classes="num"),
                "min, then STALL and switch model",
            ),
            self._field(
                "Attempts before the orchestrator",
                Input(str(opts.max_attempts), type="integer", id="agents-max-attempts",
                      classes="num"),
                "takes the brief over",
            ),
            self._field(
                "Probe models first",
                Checkbox("da 1ª onda (gasta cota)", opts.probe_on_start, id="agents-probe"),
            ),
            id="pane-agents",
            classes="pane",
        )

    def _jev_pane(self) -> Vertical:
        return jev_pane(self._decisions.providers, self._decisions.threshold)

    # ── lifecycle ─────────────────────────────────────────────────────────

    def _compact(self, root: Widget) -> None:
        for widget in root.query("Button, Input, Select, Checkbox"):
            widget.compact = True  # type: ignore[attr-defined]

    def on_mount(self) -> None:
        self.title = "splinter · configure"
        self._compact(self.screen)
        nav = self.query_one("#nav", OptionList)
        nav.highlighted = nav.get_option_index(STEP_KEYS[0])
        nav.focus()
        self.query_one("#unpriced-banner", Static).display = False
        self.run_worker(self._fetch_models, thread=True, name="fetch-models", exclusive=True)

    def _fetch_models(self) -> None:
        from splinter.configure import available_models_by_provider

        self._models_by_provider = available_models_by_provider()
        self.call_from_thread(self._update_unpriced_banner)

    def _all_models(self) -> list[str]:
        models = {m for group in self._models_by_provider.values() for m in group}
        models.update(e.model for chain in self._chains.values() for e in chain)
        return sorted(models) if self._models_by_provider else []

    def _models_grouped(self) -> dict[str, list[str]]:
        """Every known model under its routing provider (empty lists while loading)."""
        groups: dict[str, list[str]] = {p: [] for p in PROVIDERS}
        for model in self._all_models():
            groups.setdefault(provider_for(model), []).append(model)
        return groups

    # ── navigation ────────────────────────────────────────────────────────

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option_list.id == "nav" and event.option_id:
            self.query_one("#switcher", ContentSwitcher).current = f"pane-{event.option_id}"
            self.query_one("#detail", VerticalScroll).scroll_home(animate=False)
        elif isinstance(event.option_list, ChainList):
            self._sync_effort_select(event.option_list.chain_key)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "nav" and event.option_id:
            self._focus_pane(event.option_id)
        elif isinstance(event.option_list, ChainList):
            self._pick_model(event.option_list.chain_key, "swap")

    def _focus_pane(self, item: str) -> None:
        targets = {
            "gates": "#gate_add_input",
            "agents": "#agents-acceptance",
            "jev": "#jev-provider-0",
        }
        self.query_one(targets.get(item, f"#chain-{item}")).focus()

    def action_focus_nav(self) -> None:
        self.query_one("#nav", OptionList).focus()

    # ── chain editing ─────────────────────────────────────────────────────

    def _effort_select(self, key: str, entry: ChainEntry, index: int) -> Select[str]:
        select: Select[str] = Select(
            effort_choices(entry, index),
            value=entry.effort,
            allow_blank=False,
            id=f"effort-{key}",
            classes="effort-select",
        )
        select.disabled = len(efforts_for(entry.model)) <= 1
        return select

    def _sync_effort_select(self, key: str) -> None:
        """Point the effort selector at the highlighted row (no Changed echo)."""
        chain = self._chains[key]
        index = min(self._chain_list(key).highlighted or 0, len(chain) - 1)
        entry = chain[index]
        select = self.query_one(f"#effort-{key}", Select)
        with select.prevent(Select.Changed):
            select.set_options(effort_choices(entry, index))
            select.value = entry.effort
        select.disabled = len(efforts_for(entry.model)) <= 1

    def _chain_list(self, key: str) -> ChainList:
        return self.query_one(f"#chain-{key}", ChainList)

    def _set_chain(self, key: str, chain: Chain, highlight: int) -> None:
        self._chains = {**self._chains, key: chain}
        chain_list = self._chain_list(key)
        chain_list.clear_options()
        chain_list.add_options([chain_option(e, i) for i, e in enumerate(chain)])
        chain_list.highlighted = max(0, min(highlight, len(chain) - 1))
        self.query_one(f"#note-{key}", Static).update(fallback_note(chain))
        self._sync_effort_select(key)
        self._refresh_nav(key)

    def _refresh_nav(self, item: str) -> None:
        if item in _RUN_ITEMS:
            label, summary = self._run_summary(item)
        elif item in STEP_COPY:
            label, summary = STEP_COPY[item][0], self._step_summary(item)
        else:
            level = int(item.removeprefix("tier_"))
            label, summary = self._tier_label(level), chain_summary(self._chains[item])
        self.query_one("#nav", OptionList).replace_option_prompt(
            item, nav_option(item, label, summary).prompt
        )

    def on_chain_list_request(self, message: ChainList.Request) -> None:
        key = message.chain_list.chain_key
        index = message.chain_list.highlighted or 0
        chain = self._chains[key]
        if message.action == "add":
            self._pick_model(key, "add")
        elif message.action == "remove":
            self._set_chain(key, remove_entry(chain, index), index)
        elif message.action in ("effort+", "effort-"):
            step = 1 if message.action == "effort+" else -1
            self._set_chain(key, cycle_effort(chain, index, step), index)
        elif message.action in ("up", "down"):
            delta = -1 if message.action == "up" else 1
            moved = move_entry(chain, index, delta)
            self._set_chain(key, moved, index + delta if moved != chain else index)

    def _pick_model(self, key: str, mode: str) -> None:
        chain = self._chains[key]
        index = self._chain_list(key).highlighted or 0
        current = chain[index].model if mode == "swap" else ""
        title = "Add fallback" if mode == "add" else f"Swap model {index + 1}"

        def done(model: str | None) -> None:
            if not model:
                self._chain_list(key).focus()
                return
            if mode == "add":
                effort = chain[-1].effort
                self._set_chain(key, append_entry(chain, model, effort), len(chain))
            else:
                self._set_chain(key, replace_model(chain, index, model), index)
            self._chain_list(key).focus()

        anchor = chain[index] if mode == "swap" else chain[-1]
        self.push_screen(
            ModelPicker(title, self._models_grouped(), provider_for(anchor.model), current),
            done,
        )

    # ── buttons, gates, run settings ──────────────────────────────────────

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid.startswith("jev-eye-"):
            self._toggle_token(int(bid.removeprefix("jev-eye-")))
        elif bid == "sync-prices":
            self._do_sync_prices()
        elif bid == "goto-templates":
            from splinter.tui import TemplatesScreen

            self.push_screen(TemplatesScreen())
        elif bid.startswith("add-"):
            self._pick_model(bid.removeprefix("add-"), "add")
        elif bid.startswith("swap-"):
            self._pick_model(bid.removeprefix("swap-"), "swap")
        elif bid == "gate_preset":
            from splinter.tui import _GateLangModal

            self.push_screen(_GateLangModal(), self._on_lang_picked)
        elif bid == "gate_add":
            await self._add_gates()
        elif bid.startswith("gate_del_"):
            self._capture_gates()
            index = int(bid.removeprefix("gate_del_"))
            self._gate_checks = [c for j, c in enumerate(self._gate_checks) if j != index]
            await self._rebuild_gates()

    async def _add_gates(self) -> None:
        from splinter.agents.gate import parse_gate_spec

        self._capture_gates()
        field = self.query_one("#gate_add_input", Input)
        raw = field.value.strip()
        if raw:
            self._gate_checks = self._gate_checks + parse_gate_spec(raw, "all")
        field.value = ""
        await self._rebuild_gates()

    async def _on_lang_picked(self, language: str | None) -> None:
        if language is None:
            return
        from splinter.configure import gate_default_for

        self._capture_gates()
        self._gate_checks = self._gate_checks + [dict(c) for c in gate_default_for(language)]
        await self._rebuild_gates()

    async def _rebuild_gates(self) -> None:
        rows = self.query_one("#gate-rows", Vertical)
        await rows.remove_children()
        await rows.mount_all([self._gate_row(i, c) for i, c in enumerate(self._gate_checks)])
        self._compact(rows)
        self._refresh_nav("gates")

    def _capture_gates(self) -> None:
        checks: list[dict[str, str]] = []
        for i, original in enumerate(self._gate_checks):
            try:
                cmd = self.query_one(f"#gate_cmd_{i}", Input).value.strip()
                when = self._select_value(f"gate_when_{i}") or "always"
                language = self._select_value(f"gate_lang_{i}") or "all"
            except Exception:
                checks.append(dict(original))
                continue
            if cmd:
                name = original["name"] if cmd == original["cmd"] else cmd.split()[0]
                checks.append({"name": name, "cmd": cmd, "when": when, "language": language})
        self._gate_checks = checks

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        if event.checkbox.id in {f"jev-{t}" for t, *_ in JEV_STEP_TOGGLES.values()}:
            self._sync_jev_summary()

    def on_select_changed(self, event: Select.Changed) -> None:
        sid = event.select.id or ""
        if sid.startswith("effort-") and isinstance(event.value, str):
            key = sid.removeprefix("effort-")
            index = self._chain_list(key).highlighted or 0
            chain = self._chains[key]
            if index < len(chain) and chain[index].effort != event.value:
                self._set_chain(key, set_effort(chain, index, event.value), index)
        elif sid.startswith("jev-provider-") and isinstance(event.value, str):
            self._on_jev_provider(int(sid.removeprefix("jev-provider-")), event.value)
        elif sid == "agents-acceptance" and isinstance(event.value, str):
            self._agents = replace(self._agents, acceptance_check=event.value)
            self._refresh_nav("agents")

    def _toggle_token(self, index: int, reveal: bool | None = None) -> None:
        """Token stays masked; the eye button reveals it until pressed again."""
        token = self.query_one(f"#jev-token-{index}", Input)
        token.password = (not token.password) if reveal is None else not reveal
        eye = self.query_one(f"#jev-eye-{index}", Button)
        eye.label = EYE_SHOW if token.password else EYE_HIDE

    def _on_jev_provider(self, index: int, provider: str) -> None:
        model = self.query_one(f"#jev-model-{index}", Select)
        token = self.query_one(f"#jev-token-{index}", Input)
        off = provider == NO_PROVIDER
        model.disabled = off
        token.disabled = off
        self.query_one(f"#jev-eye-{index}", Button).disabled = off
        if off:
            self._toggle_token(index, reveal=False)
        token.placeholder = token_placeholder(provider)
        if not off:
            options = model_options(provider)
            model.set_options(options)
            if options:
                model.value = options[0][1]
        self._decisions = replace(self._decisions, providers=tuple(self._jev_slots()))
        self._refresh_nav("jev")

    def _jev_slots(self) -> list[Any]:
        from splinter.providers.decisions import DecisionSlot

        slots: list[DecisionSlot] = []
        for entry in self._jev_provider_entries():
            slots.append(DecisionSlot(**entry))
        return slots

    def _jev_provider_entries(self) -> list[dict[str, str]]:
        entries: list[dict[str, str]] = []
        for index in (0, 1):
            provider = self._select_value(f"jev-provider-{index}")
            if not provider or provider == NO_PROVIDER:
                continue
            entry = {"provider": provider, "model": self._select_value(f"jev-model-{index}")}
            entry.update(parse_token(self._input_value(f"jev-token-{index}")))
            entries.append(entry)
        return entries

    def _jev_targets(self) -> list[str]:
        return [
            target
            for target, *_ in JEV_STEP_TOGGLES.values()
            if self._checkbox_value(f"jev-{target}")
        ]

    def _sync_jev_summary(self) -> None:
        self._decisions = replace(self._decisions, use_for=tuple(self._jev_targets()))
        for key in JEV_STEP_TOGGLES:
            self._refresh_nav(key)

    # ── pricing ───────────────────────────────────────────────────────────

    def _do_sync_prices(self) -> None:
        self.query_one("#sync-status", Static).update("syncing models & prices…")
        self.run_worker(self._sync_worker, thread=True, name="sync-prices", exclusive=True)

    def _sync_worker(self) -> None:
        """Refresh the live catalogue, then reload the pickers' model lists from it."""
        from splinter.configure import available_models_by_provider, sync_prices
        from splinter.models.live_models import sync_model_lists

        count, failures = sync_prices()
        lists, list_failures = sync_model_lists()
        live = sum(len(v) for group in lists.values() for v in group.values())
        parts = [f"synced {count} prices", f"{live} live models"]
        parts += [f"{provider}: {err}" for provider, err in sorted(failures.items())]
        parts += [f"{name}: {err}" for name, err in sorted(list_failures.items())]
        self._sync_message = " · ".join(parts)
        self._models_by_provider = available_models_by_provider()
        self.call_from_thread(self._after_sync)

    def _after_sync(self) -> None:
        self.query_one("#sync-status", Static).update(self._sync_message)
        self._update_unpriced_banner()

    def _update_unpriced_banner(self) -> None:
        from splinter.configure import unpriced_models

        missing = unpriced_models(self._all_models())
        banner = self.query_one("#unpriced-banner", Static)
        banner.update("Some models have no price: sync." if missing else "")
        banner.display = bool(missing)

    # ── save ──────────────────────────────────────────────────────────────

    def _input_value(self, wid: str) -> str:
        try:
            return self.query_one(f"#{wid}", Input).value.strip()
        except Exception:
            return ""

    def _select_value(self, wid: str) -> str:
        raw: Any = self.query_one(f"#{wid}", Select).value
        return raw if isinstance(raw, str) else ""

    def _checkbox_value(self, wid: str) -> bool:
        return bool(self.query_one(f"#{wid}", Checkbox).value)

    def _number(self, wid: str, kind: type[float] | type[int]) -> float | int | None:
        raw = self._input_value(wid)
        try:
            return kind(raw) if raw else None
        except ValueError:
            return None

    def action_save(self) -> None:
        from splinter.configure import (
            write_agents_config,
            write_decisions_config,
            write_model_config,
        )

        self._capture_gates()
        timeouts: dict[str, int | None] = {}
        for key in self._chains:
            value = self._number(f"timeout-{key}", int)
            timeouts[key] = int(value) if value and value > 0 else None
        models, efforts, out_timeouts, providers = config_payload(
            self._chains, timeouts, len(self._tier_levels)
        )
        self.saved_path = str(
            write_model_config(
                models,
                efforts,
                timeouts=out_timeouts,
                gate_checks=self._gate_checks,
                providers=providers,
            )
        )
        stall = self._number("agents-stall-min", float)
        attempts = self._number("agents-max-attempts", int)
        write_agents_config(
            acceptance_check=self._select_value("agents-acceptance") or None,
            stall_min=float(stall) if stall is not None else None,
            max_attempts=int(attempts) if attempts is not None else None,
            probe_on_start=self._checkbox_value("agents-probe"),
        )
        threshold = self._number("jev-threshold", float)
        write_decisions_config(
            providers=self._jev_provider_entries(),
            use_for=self._jev_targets(),
            threshold=float(threshold) if threshold is not None else None,
        )
        self.saved = True
        self.exit(True)


def run_configure_tui() -> int:
    app = ConfigureApp()
    app.run()
    if app.saved:
        print(f"config written to {app.saved_path}")
    else:
        print("configure cancelled — nothing written.")
    return 0
