"""Deterministic sessions shared by the C2 run-screen tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from splinter.memory.session import Session
from splinter.models.roster import EffortMapping, Ladder, Tier
from splinter.obs.trace import RunEntry, Trace

if TYPE_CHECKING:
    import pytest

FIXTURE_LADDER = Ladder(
    tiers=[
        Tier("easy", 0, ["flash", "flash-fallback"]),
        Tier("normal", 1, ["sonnet", "sonnet-fallback"]),
        Tier("hard", 2, ["codex", "codex-fallback"]),
        Tier("very-hard", 3, ["opus", "opus-fallback"]),
        Tier("last-resort", 4, ["opus-max", "opus-max-fallback"]),
    ],
    effort_map={"normal": EffortMapping(0, "medium")},
    eval_model="opus",
    eval_effort="high",
    planner_model="opus",
    planner_effort="high",
    localizer_recall_model="flash",
    localizer_recall_large_model="flash",
    localizer_precision_model="flash",
    tier_variants={0: "low", 1: "high", 2: "xhigh", 3: "high", 4: "max"},
)


def build_session(
    tmp_path: Path,
    monkeypatch: "pytest.MonkeyPatch",
    *,
    stories: int = 58,
    live: int = 24,
    failed: int = 9,
    asked: int = 17,
    iters_each: int = 0,
    jev: bool = False,
) -> Session:
    monkeypatch.setenv("SPLINTER_HOME", str(tmp_path))
    session = Session("ses_run_view")
    session.update_index("# Run view\n")
    session.write("loop.md", _loop(stories, live, failed, asked, iters_each))
    session.write("knowledge/plan.md", "# Plan\n\n1. Implement\n")
    session.write("knowledge/plan-9.md", "# Re-plan\n\n1. Fix exemptions\n")
    _write_artifacts(session, failed)
    _write_trace(session, stories, live, failed, asked, iters_each)
    if jev:
        for _ in range(6):
            session.log_llm_usage("jev:typesafe/jev", {"input": 1, "output": 1}, 0.0001)
    session.log_llm_usage("opus", {"input": 2, "output": 1}, 0.02)
    session.set_status(
        "running",
        tasks=stories,
        stage="run",
        task_index=live - 1,
        started_at="2026-11-08T00:00:00+00:00",
        max_iterations=5,
    )
    return session


def build_single_session(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> Session:
    session = build_session(tmp_path, monkeypatch, stories=1, live=1, failed=0, asked=0)
    session.write(
        "loop.md",
        "# Task: Standalone task\n\n## Iteration 1\n- model: flash (tier 0)\n- verdict: PASS\n",
    )
    session.set_status("completed", tasks=1, stage="done", task_index=0)
    return session


def _loop(stories: int, live: int, failed: int, asked: int, iters_each: int) -> str:
    lines: list[str] = []
    for story in range(1, stories + 1):
        lines.append(f"# Task {story}/{stories}: US-{story:03d}: Story {story}")
        lines.append("")
        for n, tier, verdict in _iterations(story, live, failed, asked, iters_each):
            lines.extend((f"## Iteration {n}", f"- model: model (tier {tier})"))
            if verdict:
                lines.append(f"- verdict: {verdict}")
            lines.append("")
    if failed:
        lines.extend(("## Blocked tasks", f"- US-{failed:03d}: iteration cap"))
    return "\n".join(lines)


def _iterations(
    story: int, live: int, failed: int, asked: int, iters_each: int
) -> list[tuple[int, int, str]]:
    if iters_each:
        return [(n, 0, "PASS") for n in range(1, iters_each + 1)]
    if story == failed:
        return [
            (1, 0, "RETRY"),
            (2, 0, "ESCALATE"),
            (3, 1, "RETRY"),
            (4, 1, "ESCALATE"),
            (5, 2, "FAIL"),
        ]
    if story == asked:
        return [(1, 0, "RETRY"), (2, 0, "ASK_USER")]
    if story == live:
        return [(1, 0, "RETRY"), (2, 0, "RETRY"), (3, 1, "")]
    if story < live:
        return [(1, 0, "PASS")]
    return []


def _write_artifacts(session: Session, failed: int) -> None:
    if not failed:
        return
    for n in range(1, 6):
        reason, fix = _REVIEWS[n - 1]
        session.write(f"runs/task-{failed}-iter-{n}.md", _run_output(n))
        session.write(
            f"knowledge/task{failed - 1}-review-iter-{n}.md",
            f"- decision: {'FAIL' if n == 5 else 'RETRY'}\n- reason: {reason}\n\n"
            f"## Corrections\n- {fix}\n",
        )


_REVIEWS = (
    ("3DS challenge never reached: fallback returns before the redirect", "await the redirect"),
    ("redirect URL built from the test host, not the merchant config", "read MERCHANT_RETURN"),
    ("tests mock the provider client; the fallback path is untested", "cover the 3DS branch"),
    ("timeout on the challenge leaves the charge pending", "cancel on timeout"),
    ("same pending charge after timeout; no cancel call reaches Stripe", "call cancel_intent"),
)


def _run_output(n: int) -> str:
    return (
        f"# Run output — iteration {n}\n- model: flash (tier 0)\n- cost: $0.0100\n"
        "- tokens: {'input': 1200, 'output': 3400}\n\n## Actions\n"
        "- 💬 Reading the fallback path before the change.\n"
        "- 🔧 Read /repo/.splinter/worktrees/US-009/src/payments/fallback.py\n"
        "- 🔧 Edit /repo/.splinter/worktrees/US-009/src/payments/fallback.py\n"
        "- 🔧 Bash python -m pytest -q tests/test_fallback.py\n"
        "- 💬 Two tests fail on the redirect; adjusting the return URL.\n"
    )


def _write_trace(
    session: Session, stories: int, live: int, failed: int, asked: int, iters_each: int
) -> None:
    trace = Trace(session)
    stamp = datetime(2026, 11, 8, tzinfo=timezone.utc)
    for story in range(1, stories + 1):
        for n, tier, verdict in _iterations(story, live, failed, asked, iters_each):
            model = ("flash", "sonnet", "codex")[min(tier, 2)]
            trace.add_entry(
                RunEntry(
                    model,
                    tier,
                    n,
                    {"input": 1, "output": 1},
                    0.01,
                    1.0,
                    story - 1,
                    "run",
                    ts=stamp.isoformat(),
                )
            )
            if verdict:
                trace.add_entry(
                    RunEntry(
                        "opus",
                        tier,
                        n,
                        {"input": 1, "output": 1},
                        0.002,
                        0.5,
                        story - 1,
                        "eval",
                        ts=stamp.isoformat(),
                    )
                )
