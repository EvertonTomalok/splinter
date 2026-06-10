"""Route a single model call to the backend that owns the model id.

The rule is simple and absolute: ``opencode-go/*`` ids go to the ``opencode`` CLI,
claude aliases (``opus``/``sonnet``/``haiku``) go to ``claude -p``. Sending an
opencode model to the claude CLI 404s (and vice versa), so every ad-hoc model
call (localize, plan, eval) funnels through here instead of hardcoding a provider.
"""

from __future__ import annotations

from splinter.models.roster import provider_for
from splinter.providers import claude_cli, opencode


def run_text(
    prompt: str,
    model: str,
    *,
    variant: str | None = None,
    output_format: str = "json",
    timeout: int | None = None,
) -> str:
    """Run ``prompt`` on ``model``'s backend and return the response text."""
    if provider_for(model) == "opencode":
        return opencode.run(
            prompt, model, variant=variant, fmt=output_format, timeout=timeout
        ).text
    return claude_cli.run(
        prompt, model, effort=variant, output_format=output_format, timeout=timeout
    ).text


def run_text_session(
    prompt: str,
    model: str,
    *,
    variant: str | None = None,
    output_format: str = "json",
    session: str | None = None,
    timeout: int | None = None,
) -> tuple[str, str | None]:
    """Like :func:`run_text`, but resumes ``session`` and returns the (text, new
    session id). Used by the evaluator to keep one conversation across retries of
    the same runner — pass the returned id back in to continue it."""
    if provider_for(model) == "opencode":
        oc = opencode.run(
            prompt, model, variant=variant, fmt=output_format,
            session=session, timeout=timeout,
        )
        sid = session or oc.raw.get("session_id") or oc.raw.get("session")
        return oc.text, sid
    cl = claude_cli.run(
        prompt, model, effort=variant, output_format=output_format,
        resume=session, timeout=timeout,
    )
    return cl.text, (cl.raw.get("_session_id") or session)
