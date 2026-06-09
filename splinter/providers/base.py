"""Strategy interface for model providers.

Each provider (claude CLI, opencode CLI) is a concrete :class:`ModelProvider`
that knows how to invoke its backend and normalise the response into a
:class:`ProviderResponse`. The runner selects one by name via the registry in
:mod:`splinter.providers.registry`, so adding a backend means adding a strategy —
no ``if provider == ...`` branching at the call site.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar


@dataclass(frozen=True)
class ProviderResponse:
    """Backend-agnostic result of a single model invocation."""

    text: str
    tokens: dict[str, int] = field(default_factory=dict)
    cost: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None


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
        session: str | None = None,
        timeout: int | None = None,
    ) -> ProviderResponse:
        """Execute ``prompt`` against ``model`` and return a normalised response."""
        ...
