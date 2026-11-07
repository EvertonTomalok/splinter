"""Launch a catalog engine's provider CLI with engine-specific flags and guards.

The strategy layer speaks only in :class:`~splinter.engines.catalog.Engine` keys;
this module is the single seam that maps an engine onto its provider module
invocation — model id, effort/variant, sandbox and deny flags, resume sessions —
plus the guarded child environment and the STALL idle watchdog.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from splinter.engines.catalog import Engine
from splinter.engines.health import is_reviewer

_CODEx_DEFAULT_MODEL = "codex/gpt-5.5"


@dataclass(frozen=True)
class EngineLaunch:
    """Normalised result of one engine invocation, provider-agnostic."""

    text: str
    model: str
    session_id: str | None = None
    tokens: dict[str, int] | None = None
    cost: float = 0.0
    cost_indeterminate: bool = False


def _claude_launch(
    engine: Engine,
    prompt: str,
    *,
    cwd: str | None,
    env: dict[str, str] | None,
    session: str | None,
    timeout: int | None,
    idle_timeout: float | None,
    effort: str | None,
) -> EngineLaunch:
    from splinter.providers import claude_cli
    from splinter.vcs.guard import reviewer_deny, workhorse_deny

    deny = reviewer_deny() if is_reviewer(engine.key) else workhorse_deny()
    result = claude_cli.run(
        prompt,
        engine.model,
        effort=effort or engine.default_effort,
        resume=session,
        timeout=timeout,
        cwd=cwd,
        env=env,
        idle_timeout=idle_timeout,
        disallowed_tools=deny,
        permission_mode="bypassPermissions",
        unset_env=engine.unset_env,
    )
    return EngineLaunch(
        text=result.text,
        model=engine.model,
        tokens=dict(result.usage.get("usage") or {}) if result.usage else None,
        cost_indeterminate=True,
    )


def _mcode_launch(
    engine: Engine,
    prompt: str,
    *,
    cwd: str | None,
    env: dict[str, str] | None,
    session: str | None,
    timeout: int | None,
    idle_timeout: float | None,
    files: Sequence[str],
) -> EngineLaunch:
    from splinter.providers import mcode

    result = mcode.run(
        prompt,
        f"mcode/{engine.model}" if engine.model else engine.model,
        session=session,
        timeout=timeout,
        cwd=cwd,
        env=env,
        idle_timeout=idle_timeout,
        files=files,
    )
    return EngineLaunch(
        text=result.text,
        model=engine.model,
        session_id=result.session_id,
        tokens=result.tokens,
        cost=result.cost,
        cost_indeterminate=result.cost_indeterminate,
    )


def _kimi_launch(
    engine: Engine,
    prompt: str,
    *,
    cwd: str | None,
    env: dict[str, str] | None,
    session: str | None,
    timeout: int | None,
    idle_timeout: float | None,
    add_dirs: Sequence[str],
) -> EngineLaunch:
    from splinter.providers import kimi

    result = kimi.run(
        prompt,
        f"kimi/{engine.model}" if engine.model else engine.model,
        session=session,
        timeout=timeout,
        cwd=cwd,
        env=env,
        idle_timeout=idle_timeout,
        add_dirs=add_dirs,
    )
    return EngineLaunch(
        text=result.text,
        model=engine.model,
        session_id=result.session_id,
        tokens=result.tokens,
        cost=result.cost,
        cost_indeterminate=result.cost_indeterminate,
    )


def _agy_launch(
    engine: Engine,
    prompt: str,
    *,
    cwd: str | None,
    env: dict[str, str] | None,
    timeout: int | None,
    idle_timeout: float | None,
    effort: str | None,
    add_dirs: Sequence[str],
) -> EngineLaunch:
    from splinter.providers import agy

    result = agy.run(
        prompt,
        f"agy/{engine.model}" if engine.model else engine.model,
        variant=effort or engine.default_effort,
        timeout=timeout,
        cwd=cwd,
        env=env,
        idle_timeout=idle_timeout,
        add_dirs=add_dirs,
    )
    return EngineLaunch(
        text=result.text,
        model=engine.model,
        session_id=result.session_id,
        tokens=result.tokens,
        cost=result.cost,
        cost_indeterminate=result.cost_indeterminate,
    )


def _codex_launch(
    engine: Engine,
    prompt: str,
    *,
    cwd: str | None,
    env: dict[str, str] | None,
    session: str | None,
    timeout: int | None,
    idle_timeout: float | None,
    effort: str | None,
) -> EngineLaunch:
    from splinter.providers import codex

    result = codex.run(
        prompt,
        engine.model or _CODEx_DEFAULT_MODEL,
        effort=effort or engine.default_effort,
        resume=session,
        timeout=timeout,
        cwd=cwd,
        env=env,
        idle_timeout=idle_timeout,
        sandbox="workspace-write",
        network_access=False,
    )
    return EngineLaunch(
        text=result.text,
        model=engine.model or _CODEx_DEFAULT_MODEL,
        session_id=result.session_id,
        tokens=result.tokens,
        cost=result.cost,
        cost_indeterminate=result.cost_indeterminate,
    )


def _opencode_launch(
    engine: Engine,
    prompt: str,
    *,
    cwd: str | None,
    env: dict[str, str] | None,
    timeout: int | None,
    idle_timeout: float | None,
    effort: str | None,
) -> EngineLaunch:
    from splinter.providers import opencode
    from splinter.vcs.guard import opencode_permission_json

    result = opencode.run(
        prompt,
        engine.model,
        variant=effort or engine.default_effort,
        timeout=timeout,
        cwd=cwd,
        env=env,
        idle_timeout=idle_timeout,
        standalone=True,
        auto=True,
        config_content=opencode_permission_json(),
    )
    raw_sid = result.raw.get("session_id") or result.raw.get("session")
    session_id = raw_sid if isinstance(raw_sid, str) else None
    return EngineLaunch(
        text=result.text,
        model=engine.model,
        session_id=session_id,
        tokens=result.tokens,
        cost=result.cost,
        cost_indeterminate=result.cost_indeterminate,
    )


def run_engine(
    engine: Engine,
    prompt: str,
    *,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    session: str | None = None,
    timeout: int | None = None,
    idle_timeout: float | None = None,
    files: Sequence[str] = (),
    add_dirs: Sequence[str] = (),
    effort: str | None = None,
) -> EngineLaunch:
    """Run ``prompt`` on ``engine``'s provider CLI with its specific flags.

    Raises whatever the provider raises — FileNotFoundError for an absent
    binary (MISSING), RuntimeError for a non-zero exit (CRASH), and
    :class:`splinter.procreg.IdleTimeoutError` on a STALL — so the caller can
    classify via :mod:`splinter.engines.failures`.
    """
    if engine.provider == "mcode":
        return _mcode_launch(
            engine,
            prompt,
            cwd=cwd,
            env=env,
            session=session,
            timeout=timeout,
            idle_timeout=idle_timeout,
            files=files,
        )
    if engine.provider == "kimi":
        return _kimi_launch(
            engine,
            prompt,
            cwd=cwd,
            env=env,
            session=session,
            timeout=timeout,
            idle_timeout=idle_timeout,
            add_dirs=add_dirs,
        )
    if engine.provider == "agy":
        return _agy_launch(
            engine,
            prompt,
            cwd=cwd,
            env=env,
            timeout=timeout,
            idle_timeout=idle_timeout,
            effort=effort,
            add_dirs=add_dirs,
        )
    if engine.provider == "claude":
        return _claude_launch(
            engine,
            prompt,
            cwd=cwd,
            env=env,
            session=session,
            timeout=timeout,
            idle_timeout=idle_timeout,
            effort=effort,
        )
    if engine.provider == "codex":
        return _codex_launch(
            engine,
            prompt,
            cwd=cwd,
            env=env,
            session=session,
            timeout=timeout,
            idle_timeout=idle_timeout,
            effort=effort,
        )
    if engine.provider == "opencode":
        return _opencode_launch(
            engine,
            prompt,
            cwd=cwd,
            env=env,
            timeout=timeout,
            idle_timeout=idle_timeout,
            effort=effort,
        )
    raise ValueError(f"unknown engine provider {engine.provider!r} for {engine.key!r}")
