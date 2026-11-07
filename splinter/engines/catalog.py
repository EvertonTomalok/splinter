"""Static engine catalog, ported from the Go agent-engine routing table."""

from __future__ import annotations

import re
from dataclasses import dataclass

PROBE_PROMPT = "Reply with the word ok only."
PROBE_UP_PATTERN = re.compile(r"(?i)\bok\b")


@dataclass(frozen=True)
class Engine:
    """One row of the static routing table, available without a ping."""

    key: str
    label: str
    provider: str
    model: str
    role: str
    default_effort: str = "low"
    supports_continue: bool = False
    supports_model_override: bool = True
    supports_effort: bool = False
    unset_env: tuple[str, ...] = ()


CATALOG: tuple[Engine, ...] = (
    Engine(
        key="mcode",
        label="MiniMax M3 (mcode)",
        provider="mcode",
        model="minimax/MiniMax-M3",
        role="main workhorse: multi-file wiring and implementation",
        supports_continue=True,
    ),
    Engine(
        key="kimi",
        label="Kimi K3.1 Pro (kimi)",
        provider="kimi",
        model="kimi-code/k3.1-pro",
        role="workhorse: fast agentic implementation and reroutes",
        supports_continue=True,
    ),
    Engine(
        key="kimi-flash",
        label="Kimi highspeed (kimi)",
        provider="kimi",
        model="kimi-code/kimi-for-coding-highspeed",
        role="faster, shallower Kimi variant",
        supports_continue=True,
    ),
    Engine(
        key="mcode-flash",
        label="MiniMax M3.1 Flash (mcode)",
        provider="mcode",
        model="minimax/MiniMax-M3.1-Flash",
        role="faster, shallower MiniMax variant",
        supports_continue=True,
    ),
    Engine(
        key="agy",
        label="Gemini 3.8 Flash (agy, edit-only)",
        provider="agy",
        model="gemini-3.8-flash",
        role="edit-only briefs naming every file",
        supports_model_override=False,
        supports_effort=True,
    ),
    Engine(
        key="sonnet",
        label="Sonnet 5.5 (claude-credit)",
        provider="claude",
        model="claude-sonnet-5-5",
        role="hard briefs and reroutes; auto credit/plan routing",
        supports_effort=True,
    ),
    Engine(
        key="claude-plan",
        label="Sonnet 5.5 (plain claude, plan quota)",
        provider="claude",
        model="claude-sonnet-5-5",
        role="subscription quota with API credentials stripped",
        supports_effort=True,
        unset_env=("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
    ),
    Engine(
        key="codex",
        label="GPT (codex)",
        provider="codex",
        model="",
        role="mechanical briefs; workspace-write sandbox, network off",
        supports_continue=True,
    ),
    Engine(
        key="glm",
        label="GLM 5.3 Flash (opencode)",
        provider="opencode",
        model="zai-coding-plan/glm-5.3-flash",
        role="narrow mechanical briefs with an exact sketch",
        supports_model_override=False,
    ),
    Engine(
        key="opencode-minimax",
        label="MiniMax M3 (opencode)",
        provider="opencode",
        model="minimax-coding-plan/MiniMax-M3",
        role="fallback when mcode is down",
        supports_model_override=False,
    ),
    Engine(
        key="reviewer-opus",
        label="Final reviewer: Opus high (claude-credit)",
        provider="claude",
        model="claude-opus-5-5",
        role="final review with fixes",
        default_effort="high",
        supports_effort=True,
    ),
    Engine(
        key="reviewer-fable",
        label="Final reviewer: Fable high (claude-credit)",
        provider="claude",
        model="claude-fable-5",
        role="final review with fixes",
        default_effort="high",
        supports_effort=True,
    ),
    Engine(
        key="reviewer-sonnet",
        label="Final reviewer: Sonnet high (claude-credit)",
        provider="claude",
        model="claude-sonnet-5-5",
        role="final review with fixes",
        default_effort="high",
        supports_effort=True,
    ),
)


def engine_by_key(key: str) -> Engine | None:
    """Return the catalog engine with ``key``, or None when unknown."""
    for engine in CATALOG:
        if engine.key == key:
            return engine
    return None


def catalog_table() -> str:
    """Markdown listing of key, label and role for every catalog engine."""
    lines = ["| Engine | Launch | State |", "|---|---|---|"]
    for engine in CATALOG:
        name = f"{engine.label} — {engine.role}".replace("|", "\\|")
        launch = f"`agent-engine launch {engine.key} <wt> <brief> <log>`"
        lines.append(f"| {name} | {launch} | catalog |")
    return "\n".join(lines) + "\n"
