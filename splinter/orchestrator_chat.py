"""Hand a person's note to the orchestrator now and act on its answer."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from splinter.models.roster import Ladder
from splinter.providers.dispatch import run_text
from splinter.strategies.intervention import ORCHESTRATOR, REROUTE, TIER_UP

_ACTIONS = frozenset({ORCHESTRATOR, REROUTE, TIER_UP})
_TIMEOUT_S = 180
_PROMPT = """You are the orchestrator of a coding run. A person just wrote to you.
Answer now, in the language they wrote in. Do not use tools and do not edit files.

Stories (number | title | state | tier):
{stories}

Their message:
{note}

You may change how stories run:
- "orchestrator": you finish the story yourself
- "reroute": the next model of the same tier takes it
- "tier_up": the story moves up one tier

Reply with one JSON object and nothing else:
{{"reply": "<short answer>", "actions": [{{"story": <number>, "action": "<name>"}}]}}
Use no action unless the message asks for it."""


@dataclass(frozen=True)
class OrchestratorAnswer:
    reply: str
    actions: tuple[tuple[int, str], ...] = ()


def ask_orchestrator(note: str, stories: Sequence[Any], ladder: Ladder) -> OrchestratorAnswer:
    """The orchestrator's answer to ``note``; a failure comes back as the reply."""
    table = "\n".join(f"{s.number} | {s.title} | {s.state} | {s.tier}" for s in stories)
    prompt = _PROMPT.format(stories=table, note=note)
    try:
        raw = run_text(
            prompt,
            ladder.planner_model,
            variant=ladder.planner_effort,
            timeout=_TIMEOUT_S,
            role="orchestrator",
        )
    except Exception as exc:
        return OrchestratorAnswer(f"The orchestrator could not answer: {exc}")
    return _parse(raw, {s.number for s in stories})


def _parse(raw: str, known: set[int]) -> OrchestratorAnswer:
    try:
        data = json.loads(raw[raw.index("{") : raw.rindex("}") + 1])
    except ValueError:
        return OrchestratorAnswer(raw.strip())
    actions = tuple(pair for item in data.get("actions") or [] if (pair := _action(item, known)))
    return OrchestratorAnswer(str(data.get("reply") or "").strip(), actions)


def _action(item: Any, known: set[int]) -> tuple[int, str] | None:
    if not isinstance(item, dict):
        return None
    number, action = item.get("story"), item.get("action")
    if isinstance(number, int) and number in known and action in _ACTIONS:
        return number, str(action)
    return None
