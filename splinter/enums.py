"""Domain enumerations shared across the harness.

These are :class:`enum.StrEnum` so members compare equal to their plain-string
value (e.g. ``Decision.PASS == "PASS"``). That keeps backward compatibility with
the LLM text protocol — which speaks in bare strings — while giving the code
typo-proof constants instead of scattered string literals.
"""

from __future__ import annotations

from enum import StrEnum


class Decision(StrEnum):
    """Verdict an evaluator can return for a run."""

    PASS = "PASS"
    RETRY = "RETRY"
    ESCALATE = "ESCALATE"
    JUMP_PREMIUM = "JUMP_PREMIUM"
    ASK_USER = "ASK_USER"


class Effort(StrEnum):
    """Task difficulty buckets that map onto ladder start-tiers and variants."""

    TRIVIAL = "trivial"
    NORMAL = "normal"
    HARD = "hard"
    CRITICAL = "critical"


class Variant(StrEnum):
    """Reasoning-effort variants passed through to the provider CLIs."""

    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"  # "high+": between high and max
    MAX = "max"
    AUTO = "auto"


class FinalEvalKind(StrEnum):
    """Final eval gate execution modes."""

    ASK_USER = "ask_user"
    REVIEW = "review"
    SKILL = "skill"
    COMMAND = "command"
