"""Engine health: per-engine circuit breakers and cheap provider probes."""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from enum import StrEnum

from splinter.engines.catalog import CATALOG, PROBE_PROMPT, PROBE_UP_PATTERN, Engine
from splinter.procreg import run_subprocess

_PROBE_EFFORT = "low"
_DETAIL_LIMIT = 120
_REVIEWER_KEYS: tuple[str, ...] = ("reviewer-opus", "reviewer-fable", "reviewer-sonnet")


def _now(now: float | None) -> float:
    return time.monotonic() if now is None else now


class BreakerState(StrEnum):
    """Lifecycle states of one engine's circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half-open"


@dataclass
class CircuitBreaker:
    """Open after ``failure_threshold`` failures; allow one trial after cooldown."""

    failure_threshold: int = 2
    cooldown_s: float = 300.0
    failures: int = 0
    opened_at: float | None = None

    def record_success(self) -> None:
        """Reset the breaker to CLOSED with no failures."""
        self.failures = 0
        self.opened_at = None

    def record_failure(self, now: float | None = None) -> None:
        """Count a failure; tripping the threshold starts (or restarts) the cooldown."""
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.opened_at = _now(now)

    def state(self, now: float | None = None) -> BreakerState:
        """Current state, derived from failure count and elapsed cooldown."""
        if self.failures < self.failure_threshold:
            return BreakerState.CLOSED
        if self.opened_at is None:
            return BreakerState.OPEN
        if _now(now) - self.opened_at >= self.cooldown_s:
            return BreakerState.HALF_OPEN
        return BreakerState.OPEN

    def allow(self, now: float | None = None) -> bool:
        """True unless the breaker is OPEN (HALF_OPEN grants a single trial)."""
        return self.state(now) is not BreakerState.OPEN


class EngineHealth:
    """Circuit-breaker registry, one breaker per engine key."""

    def __init__(self) -> None:
        self._breakers: dict[str, CircuitBreaker] = {}

    def _breaker(self, key: str) -> CircuitBreaker:
        breaker = self._breakers.get(key)
        if breaker is None:
            breaker = CircuitBreaker()
            self._breakers[key] = breaker
        return breaker

    def allow(self, key: str, now: float | None = None) -> bool:
        """Whether an engine may be launched right now."""
        return self._breaker(key).allow(now)

    def trip(self, key: str) -> None:
        """Record an infra failure against the engine's breaker."""
        self._breaker(key).record_failure()

    def reset(self, key: str) -> None:
        """Record a success, closing the engine's breaker."""
        self._breaker(key).record_success()

    def snapshot(self) -> dict[str, BreakerState]:
        """Current breaker state for every known engine key."""
        return {key: breaker.state() for key, breaker in self._breakers.items()}


@dataclass(frozen=True)
class EngineProbe:
    """Result of one cheap provider ping."""

    key: str
    up: bool
    binary: str
    detail: str = ""


def binary_for(engine: Engine) -> str:
    """CLI binary a probe of this engine invokes (sonnet pings claude-credit)."""
    if engine.provider == "mcode":
        return "mcode"
    if engine.provider == "kimi":
        return "kimi"
    if engine.provider == "agy":
        return "agy"
    if engine.key == "sonnet":
        return "claude-credit"
    if engine.provider == "claude":
        return "claude"
    if engine.provider == "codex":
        return "codex"
    if engine.provider == "opencode":
        return "opencode"
    return ""


def _probe_argv(engine: Engine, prompt: str, tmpdir: str) -> list[str] | None:
    if engine.provider == "mcode":
        args = [
            "exec",
            "--cwd",
            tmpdir,
            "--model",
            engine.model,
            "--permission",
            "smart",
            "--timeout",
            "90s",
            "--output-format",
            "text",
            prompt,
        ]
    elif engine.provider == "kimi":
        args = ["-p", prompt, "--model", engine.model, "--output-format", "text"]
    elif engine.provider == "agy":
        args = [
            "-p",
            prompt,
            "--model",
            f"{engine.model}-{_PROBE_EFFORT}",
            "--mode",
            "accept-edits",
            "--print-timeout",
            "90s",
        ]
    elif engine.key == "sonnet":
        args = ["--via", "auto", prompt, "--model", engine.model, "--effort", _PROBE_EFFORT]
    elif engine.provider == "claude":
        args = ["-p", prompt, "--model", engine.model, "--effort", _PROBE_EFFORT]
    elif engine.provider == "codex":
        args = [
            "exec",
            "-m",
            engine.model.removeprefix("codex/"),
            "--skip-git-repo-check",
            "-s",
            "read-only",
            "-c",
            f'model_reasoning_effort="{_PROBE_EFFORT}"',
            "-C",
            tmpdir,
            prompt,
        ]
    elif engine.provider == "opencode":
        args = ["run", "--standalone", "-m", engine.model, prompt]
    else:
        return None
    return [binary_for(engine), *args]


def _run_probe(engine: Engine, prompt: str, timeout: int) -> tuple[int, str]:
    """Invoke the engine's provider cheaply; returns (returncode, combined output)."""
    with tempfile.TemporaryDirectory(prefix="splinter-engine-probe-") as tmpdir:
        argv = _probe_argv(engine, prompt, tmpdir)
        if argv is None:
            return 1, "unsupported probe provider"
        env: dict[str, str] | None = None
        if engine.unset_env:
            blocked = set(engine.unset_env)
            env = {k: v for k, v in os.environ.items() if k not in blocked}
        completed = run_subprocess(argv, timeout=timeout, cwd=tmpdir, env=env)
        return completed.returncode, completed.stdout + completed.stderr


def _probe_detail(returncode: int, output: str) -> str:
    detail = output.strip().split("\n", 1)[0]
    if returncode != 0:
        detail = f"{detail} exit status {returncode}".strip()
    if not detail:
        detail = "engine returned no ok response"
    return detail[:_DETAIL_LIMIT]


def probe(engine: Engine, *, timeout: int = 120) -> EngineProbe:
    """Ping one engine's provider with a one-word prompt and classify the reply."""
    name = binary_for(engine)
    path = shutil.which(name) if name else None
    if path is None:
        return EngineProbe(key=engine.key, up=False, binary="", detail="absent")
    if is_reviewer(engine.key) or engine.provider not in {
        "mcode", "kimi", "agy", "claude", "codex", "opencode"
    }:
        return EngineProbe(
            key=engine.key, up=False, binary=path, detail="unsupported probe provider"
        )
    returncode, output = _run_probe(engine, PROBE_PROMPT, timeout)
    up = returncode == 0 and PROBE_UP_PATTERN.search(output) is not None
    detail = "" if up else _probe_detail(returncode, output)
    return EngineProbe(key=engine.key, up=up, binary=path, detail=detail)


def probe_report(probes: list[EngineProbe], *, codex_model: str = "") -> str:
    """Markdown report of probe results; reviewers inherit the sonnet state."""
    states = {item.key: ("up" if item.up else "down") for item in probes}
    lines = [
        "Reviewer availability is inferred from the Sonnet ping; "
        "reviewer models are not pinged.",
        "",
        "| Engine | Launch | State |",
        "|---|---|---|",
    ]
    for engine in CATALOG:
        state_key = "sonnet" if engine.default_effort == "high" else engine.key
        label = engine.label
        if engine.key == "codex":
            label = f"GPT {codex_model or 'default'} (codex)"
        escaped = label.replace("|", "\\|")
        launch = f"`agent-engine launch {engine.key} <wt> <brief> <log>`"
        state = states.get(state_key, "absent")
        lines.append(f"| {escaped} | {launch} | {state} |")
    return "\n".join(lines) + "\n"


def reviewer_keys() -> tuple[str, ...]:
    """Engine keys of the final reviewers (not probed directly)."""
    return _REVIEWER_KEYS


def is_reviewer(key: str) -> bool:
    """Whether ``key`` names a final reviewer engine."""
    return key in _REVIEWER_KEYS
