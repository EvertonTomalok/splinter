# AGENTS.md — Splinter Development Guide

## Final Gate (run before any commit)

```bash
uv run ruff check && uv run mypy splinter && uv run pytest
```

All three must pass with zero errors.

# IMPORTANT: it's forbidden create tests that call real external models or spawn real subprocesses. This is not negotiable.
# IMPORTANT: it's forbidden call or spawn multiple agents, only sequential tasks.

## Unit Tests (pytest gate)

The pytest gate must finish in seconds. **No real external calls** — unit tests never spawn CLIs or hit live models.

When exercising the run loop (`_run_task_loop`, `DirectStrategy.execute`, etc.), mock every I/O boundary:

| Boundary | Mock target |
|---|---|
| Planner | `splinter.strategies.direct._make_plan` |
| Runner / gate / eval chain | `splinter.strategies.stages.run_task`, `run_gate`, `Evaluator.judge`, or `build_chain` |
| Provider dispatch | `splinter.providers.dispatch.run_text`, `run_text_session` |
| Subprocess | `splinter.providers.claude_cli.run_subprocess`, opencode equivalents |

If a test takes more than a second, it is probably calling `_make_plan` or `run_text` for real. Fix the mock before committing.

Real CLI/model calls belong only in manual E2E runs (`uv run splinter run …`), not in `uv run pytest`.

## Testing the Pipeline End-to-End

### Quick task (no PRD)

```bash
uv run splinter run --strategy raphael --task samples/hello-world-task.yaml
```

### Full PRD flow

```bash
uv run splinter run --strategy raphael --prd samples/hello-world-prd.md
```

### Check session state

```bash
uv run splinter analyze
```

## Project Structure

```
splinter/
  cli.py              # Entry point, argparse subcommands
  setup.py            # Environment verification
  prd.py              # Interactive PRD generation
  pipeline.py         # Orchestrates locate → plan → run → gate → eval
  analyze.py          # Session state viewer
  configure.py        # Project config writer
  memory/
    session.py        # Session dir management, markdown I/O
    knowledge.py      # Topic-scoped markdown notes (no embeddings)
  models/
    ladder.yaml       # Tier definitions, effort map, model roster
    roster.py         # Ladder dataclass + loader
  agents/
    localizer.py      # LLM-driven code search (recall + precision)
    runner.py         # Task execution, model/variant resolution
    gate.py           # Deterministic checks (ruff/mypy/pytest)
  strategies/
    base.py           # Strategy ABC, EvalVerdict
    direct.py         # Raphael: single-task loop with escalation
  providers/
    claude_cli.py     # Subprocess wrapper for `claude -p`
    opencode.py       # Subprocess wrapper for `opencode run`
  tools/
    search.py         # grep/cat/file-list through rtk
  obs/
    trace.py          # Per-run cost/token/latency tracking
tests/
samples/
  hello-world-prd.md
  hello-world-task.yaml
```

## The Loop (Direct/Raphael Strategy)

1. **Plan** — sensei (sonnet) writes implementation plan
2. **Run** — cheap model (T0 flash) implements in opencode session
3. **Gate** — deterministic checks (ruff, mypy, pytest)
4. **Eval** — cross-family judge (sonnet) returns PASS/RETRY/ESCALATE with corrections
5. **On 1st fail** — save corrections to memory, retry in **same session** with corrections
6. **On 2nd consecutive fail** — re-plan with eval history, **escalate tier**, fresh session
7. **Escalates** T0→T1→T2→T3→T4 until **opus 4.8 high**, then stops

## Code Conventions

- Python 3.11+, type hints everywhere (`from __future__ import annotations`)
- `mypy --strict` compatible: `disallow_untyped_defs = true`
- Ruff: E, F, W, I (isort)
- Line length: 100
- No comments unless asked
- Dataclasses for structured data
- Subprocess wrappers for external CLIs (claude, opencode, rtk)
- Session memory is markdown files in `.splinter/sessions/<id>/`

## Key Design Decisions

- **No embeddings/RAG** — retrieval is LLM-driven, persisted as markdown knowledge files
- **`--variant` is validated pass-through** — allowlist `{minimal, low, high, max, auto}`
- **rtk required** — all grep/cat/git/gh calls route through rtk for token savings
- **Cross-family eval** — judge runs on different model family than executor

## Runtime Edit Config — Meta-Loop

When final eval fails or pauses for user review, TUI modals let the user retune models and effort before the next round starts.

### Modals

- `_ManualValidationModal` (`tui.py`) — shown for `awaiting_validation` state. Has **Edit Config (c)** button.
- `_AskUserModal` (`tui.py`) — shown for `awaiting_user` state. Has **Edit Config (d)** button.
- `_EditConfigModal` (`tui.py`) — opened by either button. Three sections (Planner, Runner, Final Eval), each with a model `OptionList` (all providers, all models) and an effort `OptionList` (`(default)/low/medium/high/max`). Confirm dismisses with a dict; `(default)` maps to `None`; cancel → `None`.

### Session state keys

`_store_config_overrides` in `RunApp` writes six keys to `session.set_status(...)`:

| Key | Applied to |
|---|---|
| `next_planner_model` | `ladder.planner_model` |
| `next_planner_effort` | `ladder.planner_effort` |
| `next_runner_model` | all `ladder.tiers[*].models` via `rewrite_runner_tiers` |
| `next_runner_effort` | variant passed to `rewrite_runner_tiers` |
| `next_eval_model` | `ladder.eval_model` |
| `next_eval_effort` | `ladder.eval_effort` |

Empty string = "not set" (no override).

### Next-round-only semantics

`run_pipeline` reads all six keys at the start of a resume round, applies them to the ladder, then calls `session.clear_next_config()` before the round body executes. Round N+2 reverts to base unless re-edited.

`session.read_next_config()` returns only non-empty known keys. `session.clear_next_config()` sets all six to `""`.

### Per-round folder tree

Each run creates `eval-fix-{N}` under the session dir (`N` = `resume_round`):
- `eval-fix-0/final-eval.md` — final eval verbatim for round 0
- `eval-fix-0/round-eval.md` — failed eval text (if any) for round 0
- `eval-fix-1/...` — round 1, etc.

`knowledge/final-eval-{N}.md` is also kept (required by `_final_eval_rounds_md` renderer).

Each round also appends a config-used line to `events.md`:
```
round N config · planner=<model>@<effort> · runner=<model>@<variant> · eval=<model>@<effort>
```
