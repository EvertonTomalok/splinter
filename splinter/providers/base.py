"""Strategy interface for model providers.

Each provider (claude CLI, opencode CLI) is a concrete :class:`ModelProvider`
that knows how to invoke its backend and normalise the response into a
:class:`ProviderResponse`. The runner selects one by name via the registry in
:mod:`splinter.providers.registry`, so adding a backend means adding a strategy —
no ``if provider == ...`` branching at the call site.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, runtime_checkable


@dataclass(frozen=True)
class ModelPrice:
    """USD per million tokens (MTok) for input, output, and cache tiers."""

    input: float
    output: float
    cache_read: float = 0.0
    cache_write: float = 0.0


@dataclass(frozen=True)
class ProviderResponse:
    """Backend-agnostic result of a single model invocation."""

    text: str
    tokens: dict[str, int] = field(default_factory=dict)
    cost: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None
    cost_indeterminate: bool = False


class ModelProvider(ABC):
    """A pluggable backend that can run a prompt against a named model."""

    name: ClassVar[str]

    @abstractmethod
    def run(
        self,
        prompt: str,
        model: str,
        *,
        variant: str | None = None,
        output_format: str = "json",
        session: str | None = None,
        timeout: int | None = None,
        agent: str = "build",
        cwd: str | None = None,
    ) -> ProviderResponse:
        """Execute ``prompt`` against ``model`` and return a normalised response.

        ``cwd`` is the working directory the backend runs in — used to isolate
        parallel tasks in their own git worktrees. ``None`` means the process cwd.
        """
        ...


@runtime_checkable
class PriceableProvider(Protocol):
    """Provider that can fetch live per-model pricing (USD/MTok)."""

    supports_pricing: ClassVar[bool]

    def fetch_pricing(self) -> dict[str, ModelPrice]:
        ...


# ── provider gap ─────────────────────────────────────────────────────────────

# (kind, regex) — evaluated in order; first match wins.
_GAP_PATTERNS: list[tuple[str, str]] = [
    ("insufficient_balance", r"insufficient.{0,10}balance|out of credit|quota exceeded|billing"),
    ("rate_limit", r"\b429\b|rate.?limit"),
    ("overload", r"\b5[02][29]\b|overload|service.?unavailable|unavailable"),
]

_RETRY_AFTER_RE = re.compile(r"retry.?after[:\s]+(\d+)", re.IGNORECASE)


class ProviderGapError(Exception):
    """Provider temporarily unable to serve — pause the run, resume later.

    Raised by provider adapters when the backend signals it cannot fulfil
    the request right now (rate-limit, overload, insufficient balance).
    The pipeline catches this, persists run state, prints guidance, and
    exits with code 2 so the operator can resume once the gap clears.
    """

    def __init__(
        self,
        kind: str,
        provider: str,
        model: str,
        original: Exception,
        retry_after: int | None = None,
    ) -> None:
        #: ``"rate_limit"`` | ``"overload"`` | ``"insufficient_balance"``
        self.kind = kind
        self.provider = provider
        self.model = model
        self.original = original
        #: Seconds to wait before retrying, when known from the response.
        self.retry_after = retry_after
        super().__init__(str(original))

    @property
    def resumable(self) -> bool:
        """False only for ``insufficient_balance`` — needs manual fix."""
        return self.kind != "insufficient_balance"

    @property
    def guidance(self) -> str:
        """Human-readable pause notice with resume instructions."""
        lines = [
            f"provider gap ({self.kind}): {self.provider}/{self.model}",
            f"  cause: {self.original}",
        ]
        if self.kind == "insufficient_balance":
            lines += [
                "  action: top up your account / check billing.",
                "  note:   run will NOT auto-resume — fix billing first.",
            ]
        else:
            wait = f"{self.retry_after}s" if self.retry_after else "~60s"
            lines += [
                f"  suggested wait: {wait}",
                "  then resume:    splinter resume",
            ]
        return "\n".join(lines)


def detect_provider_gap(
    exc: Exception,
    provider: str,
    model: str,
) -> ProviderGapError | None:
    """Return a :class:`ProviderGapError` if *exc* looks like a provider gap.

    Inspects the stringified exception message; returns ``None`` when the
    error is not a recognisable gap signal (so the caller should re-raise
    the original).
    """
    msg = str(exc)
    msg_lower = msg.lower()
    for kind, pattern in _GAP_PATTERNS:
        if re.search(pattern, msg_lower):
            retry_after: int | None = None
            m = _RETRY_AFTER_RE.search(msg)
            if m:
                try:
                    retry_after = int(m.group(1))
                except ValueError:
                    pass
            return ProviderGapError(kind, provider, model, exc, retry_after)
    return None
