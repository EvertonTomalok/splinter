"""Route a single model call to the backend that owns the model id.

The rule is simple and absolute: ``opencode-go/*`` ids go to the ``opencode`` CLI,
claude aliases (``opus``/``sonnet``/``haiku``) go to ``claude -p``. Sending an
opencode model to the claude CLI 404s (and vice versa), so every ad-hoc model
call (localize, plan, eval) funnels through here instead of hardcoding a provider.
"""

from __future__ import annotations

from dataclasses import replace

from splinter.models.roster import provider_for
from splinter.providers.base import ProviderResponse
from splinter.providers.registry import get_provider


def run_text(
    prompt: str,
    model: str,
    *,
    variant: str | None = None,
    output_format: str = "json",
    timeout: int | None = None,
    agent: str = "build",
    session: object = None,
) -> str:
    """Run ``prompt`` on ``model``'s backend and return the response text."""
    provider = get_provider(provider_for(model))
    resp = provider.run(
        prompt, model, variant=variant, output_format=output_format, timeout=timeout, agent=agent
    )
    if session is not None:
        _log(session, model, resp.tokens, resp.cost)
    return resp.text


def _log(session: object, model: str, tokens: dict[str, int], cost: float) -> None:
    try:
        session.log_llm_usage(model, tokens, cost)  # type: ignore[attr-defined]
    except Exception:
        pass


def run_text_session(
    prompt: str,
    model: str,
    *,
    variant: str | None = None,
    output_format: str = "json",
    session: str | None = None,
    timeout: int | None = None,
    agent: str = "build",
) -> tuple[str, str | None]:
    """Like :func:`run_text`, but resumes ``session`` and returns the (text, new
    session id). Used by the evaluator to keep one conversation across retries of
    the same runner — pass the returned id back in to continue it."""
    provider = get_provider(provider_for(model))
    resp = provider.run(
        prompt, model, variant=variant, output_format=output_format, session=session,
        timeout=timeout, agent=agent,
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
) -> tuple[ProviderResponse, str | None]:
    """Like :func:`run_text_session` but returns the full :class:`ProviderResponse`
    (with cost and token counts) alongside the session id."""
    provider = get_provider(provider_for(model))
    resp = provider.run(
        prompt, model, variant=variant, output_format=output_format, session=session,
        timeout=timeout, agent=agent,
    )
    sid = resp.session_id or session or None
    if resp.session_id != sid:
        resp = replace(resp, session_id=sid)
    return resp, sid
