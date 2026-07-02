"""Route a single model call to the backend that owns the model id.

The rule is simple and absolute: ``opencode-go/*`` ids go to the ``opencode`` CLI,
claude aliases (``opus``/``sonnet``/``haiku``) go to ``claude -p``. Sending an
opencode model to the claude CLI 404s (and vice versa), so every ad-hoc model
call (localize, plan, eval) funnels through here instead of hardcoding a provider.

Every function accepts optional ``trace`` + ``iteration`` / ``tier`` /
``task_index`` / ``role`` keyword arguments. When ``trace`` is provided and the
provider response is billable (cost > 0 or non-empty tokens), a single
:class:`~splinter.obs.trace.RunEntry` is appended — exactly once per provider
call, with no separate stage-level logging.
"""

from __future__ import annotations

from dataclasses import replace

from splinter.models.roster import provider_for
from splinter.obs.trace import RunEntry
from splinter.providers.base import ProviderResponse
from splinter.providers.registry import get_provider


def _log_trace(
    trace: object,
    model: str,
    tokens: dict[str, int],
    cost: float,
    *,
    tier: int,
    iteration: int,
    task_index: int,
    role: str,
    cost_indeterminate: bool = False,
) -> None:
    if cost <= 0 and sum(tokens.values()) <= 0 and not cost_indeterminate:
        return
    trace.entries.append(  # type: ignore[attr-defined]
        RunEntry(
            model=model,
            tier=tier,
            iteration=iteration,
            tokens=tokens,
            cost=cost,
            latency_s=0.0,
            task=task_index,
            role=role,
            cost_indeterminate=cost_indeterminate,
        )
    )


def _log_trace_from_exc(
    trace: object,
    model: str,
    exc: BaseException,
    *,
    tier: int,
    iteration: int,
    task_index: int,
    role: str,
) -> None:
    tokens: dict[str, int] = getattr(exc, "tokens", {}) or {}
    cost: float = getattr(exc, "cost", 0.0) or 0.0
    if cost <= 0 and sum(tokens.values()) <= 0:
        return
    trace.entries.append(  # type: ignore[attr-defined]
        RunEntry(
            model=model,
            tier=tier,
            iteration=iteration,
            tokens=tokens,
            cost=cost,
            latency_s=0.0,
            task=task_index,
            role=role,
        )
    )


def _log_session(session: object, model: str, tokens: dict[str, int], cost: float) -> None:
    try:
        session.log_llm_usage(model, tokens, cost)  # type: ignore[attr-defined]
    except Exception:
        pass


def _provider_agent(*, role: str, agent: str) -> str:
    """Map pipeline role onto the provider agent hint (codex uses it for sandbox)."""
    return role if role != "run" else agent


def run_text(
    prompt: str,
    model: str,
    *,
    variant: str | None = None,
    output_format: str = "json",
    timeout: int | None = None,
    agent: str = "build",
    session: object = None,
    trace: object = None,
    iteration: int = 0,
    tier: int = 0,
    task_index: int = 0,
    role: str = "run",
) -> str:
    """Run ``prompt`` on ``model``'s backend and return the response text.

    Warn-only: when a non-OpenCode model has no synced pricing, logs a warning
    naming the model and provider but still runs (no block/abort).
    """
    from splinter.models.pricing import warn_missing_model_pricing

    warn_missing_model_pricing(model)
    provider = get_provider(provider_for(model))
    try:
        resp = provider.run(
            prompt,
            model,
            variant=variant,
            output_format=output_format,
            timeout=timeout,
            agent=_provider_agent(role=role, agent=agent),
        )
    except Exception as exc:
        if trace is not None:
            _log_trace_from_exc(
                trace, model, exc, tier=tier, iteration=iteration, task_index=task_index, role=role
            )
        raise
    if trace is not None:
        _log_trace(
            trace,
            model,
            resp.tokens,
            resp.cost,
            tier=tier,
            iteration=iteration,
            task_index=task_index,
            role=role,
            cost_indeterminate=resp.cost_indeterminate,
        )
    if session is not None:
        _log_session(session, model, resp.tokens, resp.cost)
    return resp.text


def run_text_session(
    prompt: str,
    model: str,
    *,
    variant: str | None = None,
    output_format: str = "json",
    session: str | None = None,
    timeout: int | None = None,
    agent: str = "build",
    trace: object = None,
    iteration: int = 0,
    tier: int = 0,
    task_index: int = 0,
    role: str = "run",
) -> tuple[str, str | None]:
    """Like :func:`run_text`, but resumes ``session`` and returns the (text, new
    session id). Used by the evaluator to keep one conversation across retries of
    the same runner — pass the returned id back in to continue it."""
    provider = get_provider(provider_for(model))
    try:
        resp = provider.run(
            prompt,
            model,
            variant=variant,
            output_format=output_format,
            session=session,
            timeout=timeout,
            agent=_provider_agent(role=role, agent=agent),
        )
    except Exception as exc:
        if trace is not None:
            _log_trace_from_exc(
                trace, model, exc, tier=tier, iteration=iteration, task_index=task_index, role=role
            )
        raise
    if trace is not None:
        _log_trace(
            trace,
            model,
            resp.tokens,
            resp.cost,
            tier=tier,
            iteration=iteration,
            task_index=task_index,
            role=role,
            cost_indeterminate=resp.cost_indeterminate,
        )
    sid = resp.session_id or session
    return resp.text, sid


def run_provider_session(
    prompt: str,
    model: str,
    *,
    variant: str | None = None,
    output_format: str = "json",
    session: str | None = None,
    timeout: int | None = None,
    agent: str = "build",
    trace: object = None,
    iteration: int = 0,
    tier: int = 0,
    task_index: int = 0,
    role: str = "run",
) -> tuple[ProviderResponse, str | None]:
    """Like :func:`run_text_session` but returns the full :class:`ProviderResponse`
    (with cost and token counts) alongside the session id."""
    provider = get_provider(provider_for(model))
    try:
        resp = provider.run(
            prompt,
            model,
            variant=variant,
            output_format=output_format,
            session=session,
            timeout=timeout,
            agent=_provider_agent(role=role, agent=agent),
        )
    except Exception as exc:
        if trace is not None:
            _log_trace_from_exc(
                trace, model, exc, tier=tier, iteration=iteration, task_index=task_index, role=role
            )
        raise
    if trace is not None:
        _log_trace(
            trace,
            model,
            resp.tokens,
            resp.cost,
            tier=tier,
            iteration=iteration,
            task_index=task_index,
            role=role,
            cost_indeterminate=resp.cost_indeterminate,
        )
    sid = resp.session_id or session or None
    if resp.session_id != sid:
        resp = replace(resp, session_id=sid)
    return resp, sid
