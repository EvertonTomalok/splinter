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

import logging
import subprocess
import threading
import time
import uuid
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from splinter.models.roster import provider_for
from splinter.obs.agentic import _now_iso
from splinter.obs.trace import RunEntry
from splinter.providers.base import ProviderResponse
from splinter.providers.registry import get_provider

_log = logging.getLogger(__name__)


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
    latency_s: float = 0.0,
    ts: str = "",
) -> None:
    if cost <= 0 and sum(tokens.values()) <= 0 and not cost_indeterminate:
        return
    trace.add_entry(  # type: ignore[attr-defined]
        RunEntry(
            model=model,
            tier=tier,
            iteration=iteration,
            tokens=tokens,
            cost=cost,
            latency_s=latency_s,
            task=task_index,
            role=role,
            cost_indeterminate=cost_indeterminate,
            ts=ts,
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
    latency_s: float = 0.0,
    ts: str = "",
) -> None:
    tokens: dict[str, int] = getattr(exc, "tokens", {}) or {}
    cost: float = getattr(exc, "cost", 0.0) or 0.0
    if cost <= 0 and sum(tokens.values()) <= 0:
        return
    trace.add_entry(  # type: ignore[attr-defined]
        RunEntry(
            model=model,
            tier=tier,
            iteration=iteration,
            tokens=tokens,
            cost=cost,
            latency_s=latency_s,
            task=task_index,
            role=role,
            ts=ts,
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


_TRANSIENT_GAP_KINDS = {"rate_limit", "overload"}


def _pool_candidates(
    model: str, pool: Sequence[str] | None
) -> list[tuple[str, str | None]]:
    """Primary first, then pool siblings — entries may carry their own variant."""
    from splinter.models.roster import parse_pool_entry

    entries: list[tuple[str, str | None]] = [(model, None)]
    for raw in pool or []:
        entry_model, variant = parse_pool_entry(raw)
        if entry_model == model:
            entries[0] = (model, variant)
        if entry_model and entry_model != model:
            entries.append((entry_model, variant))
    return entries


def _call_provider(
    prompt: str,
    model: str,
    *,
    variant: str | None,
    pool: Sequence[str] | None,
    fallback_prompt: str | None = None,
    output_format: str = "json",
    session: str | None = None,
    session_id: str | None = None,
    timeout: int | None = None,
    agent: str = "build",
    cwd: str | None = None,
    trace: object = None,
    tier: int = 0,
    iteration: int = 0,
    task_index: int = 0,
    role: str = "run",
) -> tuple[ProviderResponse, str]:
    """One model call with pool fallback for EVERY pipeline step.

    On a transient provider gap (rate limit, overload) the failing model's
    breaker opens and the next healthy pool sibling runs instead — each sibling
    with its own ``model@variant`` effort. A sibling may live on another
    provider, so the provider is resolved per candidate; sessions resume only
    on the primary (a sibling's backend cannot honour a foreign session id).
    Returns ``(response, model_used)``.
    """
    from splinter.engines import runtime
    from splinter.providers.base import ProviderGapError

    candidates = _pool_candidates(model, pool)
    available = [c for c in candidates if runtime.model_available(c[0])]
    if not available:
        # Every sibling tripped and cooldown not elapsed: run the primary
        # anyway — a hot breaker must not hard-block a step with no fallback.
        available = candidates[:1]
    last_exc: Exception | None = None
    for cand_model, cand_variant in available:
        provider = get_provider(provider_for(cand_model))
        _t0 = time.monotonic()
        _ts = _now_iso()
        call_kwargs: dict[str, Any] = dict(
            variant=cand_variant or variant,
            output_format=output_format,
            session=session if cand_model == model else None,
            timeout=timeout,
            agent=agent,
            cwd=cwd,
        )
        if cand_model == model and session_id is not None:
            call_kwargs["session_id"] = session_id
        try:
            call_prompt = fallback_prompt if cand_model != model and fallback_prompt else prompt
            _log.info(
                "%s · %s%s · waiting for response",
                role,
                cand_model,
                f" · effort {call_kwargs['variant']}" if call_kwargs["variant"] else "",
                extra={"waiting_model": cand_model},
            )
            resp = provider.run(call_prompt, cand_model, **call_kwargs)
        except Exception as exc:
            last_exc = exc
            if trace is not None:
                _log_trace_from_exc(
                    trace,
                    cand_model,
                    exc,
                    tier=tier,
                    iteration=iteration,
                    task_index=task_index,
                    role=role,
                    latency_s=time.monotonic() - _t0,
                    ts=_ts,
                )
            if isinstance(exc, ProviderGapError) and exc.kind in _TRANSIENT_GAP_KINDS:
                runtime.report_model_failure(cand_model)
                _log.warning(
                    "provider gap (%s) on %s (%s) — rerouting to next pool sibling",
                    exc.kind,
                    cand_model,
                    role,
                )
                time.sleep(2)
                continue
            raise
        if trace is not None:
            _log_trace(
                trace,
                cand_model,
                resp.tokens,
                resp.cost,
                tier=tier,
                iteration=iteration,
                task_index=task_index,
                role=role,
                cost_indeterminate=resp.cost_indeterminate,
                latency_s=time.monotonic() - _t0,
                ts=_ts,
            )
        runtime.report_model_success(cand_model)
        _log.info(
            "%s completed · %s · %d characters · %d tokens · %.1fs",
            role,
            cand_model,
            len(resp.text),
            sum(resp.tokens.values()),
            time.monotonic() - _t0,
            extra={"waiting_model": cand_model},
        )
        return resp, cand_model
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"no pool candidate available for {model!r}")  # pragma: no cover


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
    pool: Sequence[str] | None = None,
) -> str:
    """Run ``prompt`` on ``model``'s backend and return the response text.

    ``pool`` is the ordered fallback list for the calling step (entries may
    carry ``@variant``); transient gaps reroute to the next healthy sibling.

    Warn-only: when a non-OpenCode model has no synced pricing, logs a warning
    naming the model and provider but still runs (no block/abort).
    """
    from splinter.models.pricing import warn_missing_model_pricing

    warn_missing_model_pricing(model)
    resp, used = _call_provider(
        prompt,
        model,
        variant=variant,
        pool=pool,
        output_format=output_format,
        timeout=timeout,
        agent=_provider_agent(role=role, agent=agent),
        trace=trace,
        tier=tier,
        iteration=iteration,
        task_index=task_index,
        role=role,
    )
    if session is not None:
        _log_session(session, used, resp.tokens, resp.cost)
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
    pool: Sequence[str] | None = None,
) -> tuple[str, str | None]:
    """Like :func:`run_text`, but resumes ``session`` and returns the (text, new
    session id). Used by the evaluator to keep one conversation across retries of
    the same runner — pass the returned id back in to continue it."""
    resp, _used = _call_provider(
        prompt,
        model,
        variant=variant,
        pool=pool,
        output_format=output_format,
        session=session,
        timeout=timeout,
        agent=_provider_agent(role=role, agent=agent),
        trace=trace,
        tier=tier,
        iteration=iteration,
        task_index=task_index,
        role=role,
    )
    sid = (resp.session_id or session) if _used == model else None
    return resp.text, sid


def run_provider_session(
    prompt: str,
    model: str,
    *,
    variant: str | None = None,
    output_format: str = "json",
    session: str | None = None,
    session_id: str | None = None,
    timeout: int | None = None,
    agent: str = "build",
    cwd: str | None = None,
    trace: object = None,
    iteration: int = 0,
    tier: int = 0,
    task_index: int = 0,
    role: str = "run",
    pool: Sequence[str] | None = None,
    fallback_prompt: str | None = None,
) -> tuple[ProviderResponse, str | None]:
    """Like :func:`run_text_session` but returns the full :class:`ProviderResponse`
    (with cost and token counts) alongside the session id.

    ``cwd`` is the working directory the backend runs in — passed through so a
    parallel task edits (and is gated on) its own git worktree, not the main repo.
    On the first turn (no ``session``, no ``session_id``) a per-task UUID is
    generated here for providers that support create-with-id, so the id is known
    from iteration 1 and reused (via the returned id) on later turns.

    ``pool`` is the ordered fallback list (entries may carry ``@variant``);
    transient gaps reroute to the next healthy sibling.
    """
    provider = get_provider(provider_for(model))
    create_id = session_id
    if (
        session is None
        and create_id is None
        and getattr(provider, "supports_session_create", False)
    ):
        create_id = str(uuid.uuid4())
    resp, _used = _call_provider(
        prompt,
        model,
        variant=variant,
        pool=pool,
        output_format=output_format,
        session=session,
        session_id=create_id,
        fallback_prompt=fallback_prompt,
        timeout=timeout,
        agent=_provider_agent(role=role, agent=agent),
        cwd=cwd,
        trace=trace,
        tier=tier,
        iteration=iteration,
        task_index=task_index,
        role=role,
    )
    sid = (resp.session_id or session or create_id) if _used == model else None
    if resp.session_id != sid:
        resp = replace(resp, session_id=sid)
    return resp, sid


def _resume_argv(provider: str, model: str, session_id: str, message: str) -> list[str]:
    """Build the resume+message argv for ``provider``. Message always trails a
    ``--`` terminator so a message starting with ``-`` isn't read as a flag."""
    if provider == "claude":
        return [
            "claude",
            "-p",
            "--model",
            model,
            "--dangerously-skip-permissions",
            "--resume",
            session_id,
            "--",
            message,
        ]
    if provider == "opencode":
        return [
            "opencode",
            "run",
            "-m",
            model,
            "--dangerously-skip-permissions",
            "-s",
            session_id,
            "--",
            message,
        ]
    if provider == "codex":
        bare = model.removeprefix("codex/")
        return [
            "codex",
            "exec",
            "resume",
            "--json",
            "--skip-git-repo-check",
            "-m",
            bare,
            "--dangerously-bypass-approvals-and-sandbox",
            "--",
            session_id,
            message,
        ]
    if provider == "cursor":
        bare = model.removeprefix("cursor/")
        cmd = ["agent", "-p", "--trust", "--resume", session_id]
        if bare and bare != "auto":
            cmd += ["--model", bare]
        cmd += ["--", message]
        return cmd
    raise ValueError(f"unknown provider '{provider}'")


def _reap(proc: subprocess.Popen[bytes]) -> None:
    """Wait on a detached, fired process so it doesn't linger as a zombie."""
    try:
        proc.wait()
    except Exception:
        _log.debug("reap of fired process %s failed", proc.pid, exc_info=True)


def fire_message_into_session(
    *,
    provider: str,
    model: str,
    session_id: str,
    message: str,
    cwd: str | None = None,
) -> None:
    """Spawn a detached provider process that resumes ``session_id`` with
    ``message`` and returns immediately without waiting for it to finish.

    The process is untracked: it is never added to :data:`splinter.procreg._active`,
    so a graceful stop/kill of the main run never touches it. A daemon thread reaps
    it once it exits so it doesn't linger as a zombie. Spawn and reap failures are
    logged, never raised — this is best-effort, fire-and-forget.
    """
    try:
        get_provider(provider)
        argv = _resume_argv(provider, model, session_id, message)
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=cwd,
            start_new_session=True,
        )
        threading.Thread(target=_reap, args=(proc,), daemon=True).start()
    except Exception:
        _log.warning(
            "failed to fire message into session %s (provider=%s)",
            session_id,
            provider,
            exc_info=True,
        )
        return
