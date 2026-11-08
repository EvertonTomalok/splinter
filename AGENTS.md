# AGENTS.md — Splinter Development Guide

## Final Gate (run before any commit)

```bash
uv run ruff check && uv run mypy splinter && uv run pytest
```

All three must pass with zero errors.

# IMPORTANT: it's forbidden create tests that call real external models or spawn real subprocesses. This is not negotiable.
# IMPORTANT: within a single strategy execute() call, tasks may run concurrently via DAG-scheduled parallel execution (ThreadPoolExecutor). Each parallel task runs in its own git worktree. Sequential execution is the default; parallel is opt-in via --parallel flag or parallel=True in run_pipeline(). Direct/Raphael strategy is always sequential.

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
  run_snapshot.py     # Shared read-side snapshot of the session files
  run_view.py         # Pure run view model: RunView, build_run_view
  run_view_files.py   # File cache, mtime keys, review/run file paths
  run_view_agents.py  # Agents run stories and attempts from agents-run/
  run_view_rows.py    # Story, iteration, stage and ladder projections
  run_view_spend.py   # Spend aggregation (cost, time) for the rail
  task_waves.py       # PRD tasks → waves
  ui_prefs.py         # .splinter/ui.yaml preferences (questions_mode)
  configure.py        # Project config writer
  configure_chain.py  # Fallback-chain state for the configure TUI
  tui_configure.py    # configure TUI (list + detail), widgets in tui_configure_widgets.py
  tui_blockword.py    # ANSI Shadow block word with a left-to-right wipe
  tui_run_palette.py  # Design colours as Textual CSS variables
  tui_run_focus.py    # Left column: block word, sentence, stage strip, tabs, body
  tui_run_rail.py     # Right column: clock, ladder, iteration bars, heatmap, groups
  tui_run_views.py    # View registry: act, plan, gate, eval
  tui_run_views_more.py # View registry: diff, spend, tier
  tui_run_composer.py # Orchestrator composer with targets and delivery state
  tui_run_banner.py   # Needs-you banner (answer, validate, retry with guidance)
  tui_run_stop.py     # Stop bar and the stop modal
  tui_run_motion.py   # Motion clock (pulses, caret) and the toast tracker
  tui_questions_one.py # Question index in "one at a time" mode
  tui_prd_outline.py  # PRD questions screen: draft outline, nav row, spend line
  memory/
    session.py        # Session dir management, markdown I/O
    knowledge.py      # Topic-scoped markdown notes (no embeddings)
    orchestrator_notes.py # Once-per-consumer notes for the orchestrator
  models/
    ladder.yaml       # Tier definitions, effort map, model roster
    roster.py         # Ladder dataclass + loader
  agents/
    localizer.py      # LLM-driven code search (recall + precision)
    runner.py         # Task execution, model/variant resolution
    gate.py           # Deterministic checks (ruff/mypy/pytest)
    acceptance.py     # Brief acceptance validator (gate | gate+eval | gate+orchestrator)
    laws.py           # Workhorse contract, brief template, AGENT_RESULT protocol
    refiner.py        # PRD tasks → waves of briefs with deps
    brief_io.py       # Engine prompt, worktree diff, report, engine logs, saved briefs
    codegen.py        # Project generators run in a brief worktree before acceptance
    progress.py       # agents-run/stories.json + attempts.jsonl (what the run screen reads)
  engines/            # Engine CLIs for the agents strategy
    catalog.py        # Engine catalog (mcode, agy, sonnet, codex, glm, …)
    health.py         # Circuit breaker + live probe
    routing.py        # Effort → tier → model chain → engine (by model prefix)
    pool.py           # First healthy engine of a chain
    failures.py       # STALL/LIMIT/CRASH/EMPTY/MISSING classification
    launch.py         # Engine → provider invocation seam (guards, watchdog)
  strategies/
    base.py           # Strategy ABC, EvalVerdict
    direct.py         # Raphael: single-task loop with escalation
    agents.py         # plan-with-agents: refine, dispatch, merge, orchestrator takeover
    agents_brief.py   # One brief: launch, classify, codegen, acceptance, fix rounds
    agents_ctx.py     # BriefCtx: run settings, attempt log, user guidance
  providers/
    claude_cli.py     # Subprocess wrapper for `claude -p`
    opencode.py       # Subprocess wrapper for `opencode run`
    mcode.py          # Subprocess wrapper for `mcode exec`
    agy.py            # Subprocess wrapper for `agy`
  vcs/
    worktree.py       # Worktree lifecycle, --no-ff merges into integration
    diff.py           # Story diff for the `diff` view
    guard.py          # guard.sh + PATH shims (workhorses may only edit files)
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

## Model pools (every LLM step, smarter model calling)

Every step's model list is a **pool, not a singleton** — runners (ladder tiers)
and pipeline steps alike (planner, prd, eval, localizer recall/large/precision).
`models.<step>` and `models.tiers[n]` accept a single id or an ordered list of
`model[@variant]` entries; the first model whose breaker is closed wins
(`resolve_model` + `engines/runtime.py`). On a transient provider gap
(429/overload) the failing model's breaker opens and the retry **reroutes to
the next healthy sibling in the same step** — each sibling with its own effort
(`model@variant`) — inside `dispatch._call_provider`, so planner/reviewer/
localizer/prd get the same fallback as runners. Sessions resume only on the
primary; when every sibling is tripped the primary runs anyway (fail-open).

```yaml
models:
  planner: [sonnet@high, opencode/deepseek-v4-pro@medium, codex/gpt-5.5]
  eval: [opus@high, sonnet@low]
  tiers:
    - [opencode/deepseek-v4-flash-free, mcode/minimax/MiniMax-M3, claude/haiku]
```

`splinter configure` (`tui_configure.py`, pure state in `configure_chain.py`) is
list + detail: the left column lists steps, the effort tiers (EFFORT → MODEL)
and run settings with a one-line summary; the right pane shows the selected
chain as an ordered list (provider · model · effort, plus a clickable effort select) — `a` add fallback, `x`
remove, `←/→` effort, `shift+↑/↓` reorder, `enter` swap model. The picker asks
for the provider first (`tab`/`shift+tab` cycle it while typing) (claude, opencode, codex, cursor, mcode, kimi, agy) and
prefixes a typed custom id with it. Effort cycles only through the provider's
native names (`configure_chain.PROVIDER_EFFORTS`): claude low…max, codex
low…xhigh, opencode minimal…max, agy low/medium/high; mcode/kimi/cursor have no
effort knob ("—").

Model lists are live, never hard-coded only:

| source | feeds | cache |
|---|---|---|
| `opencode models`, `agent --list-models` | opencode, cursor | fetched on open |
| public pricing catalogue (litellm) | claude, codex | `.splinter/pricing.json` |
| `agy models` (effort suffix stripped) | agy | `.splinter/models.json` |
| `kimi provider list --json` | kimi | `.splinter/models.json` |
| OpenRouter `/api/v1/models?output_modalities=all` (`decisions`) | jev models | `.splinter/models.json` |
| Vercel `/v1/models` (`type: evaluation`) | jev models | `.splinter/models.json` |

mcode has no listing command, so it keeps the engine-catalog ids. "Sync models"
in the TUI (or `splinter configure --sync-prices`) refreshes both caches
(`models/live_models.py`); opening configure only reads them. The picker lists
each family newest-first. Tests block `live_models._run`/`_get_json` in
`conftest.py`.

## Changed-scope gates (`splinter gate <kind>`)

`gate_checks` run with `shell=True` in the worktree, each with its own
`timeout` (seconds, default 120). On a big repo a whole-repo `go test ./...` or
`tsc` blows that and the machine, so projects point their gates at
`splinter gate <kind>` (`agents/gate_scope.py`). It checks only what the change
touched: commits since the `splinter/*` branch was created (its oldest reflog
entry) plus uncommitted and untracked files; on any other branch only the
uncommitted ones (`SPLINTER_GATE_BASE` overrides). Nothing touched → pass.

| kind | runs |
|---|---|
| `go-fmt` / `go-vet` / `go-test` | per module (closest `go.mod`), touched packages, `GOMAXPROCS=4 GOFLAGS=-p=2` |
| `go-lint` | `golangci-lint run --concurrency=2` on touched packages; skipped without a `.golangci.*` |
| `ts-typecheck` | workspace `typecheck` script, else a `lint` script that is `tsc`, else `tsc --noEmit` |
| `ts-lint` | `eslint <touched files>` when an eslint config exists |
| `ts-test` | `vitest related --run --maxWorkers=2 <touched files>`, `bun test`, else the `test` script |
| `py-lint` / `py-type` / `py-test` | per project (closest `uv.lock`/`pyproject.toml`): `ruff check <files>`, `mypy <files>` (with a mypy config), `pytest -m "not integration"` on touched tests + the member's `tests/` |

The package runner comes from the closest lockfile (bun, pnpm, yarn, npm); bun
workspaces run tools with `bun --bun`. Worktrees get every `node_modules` up to
four levels deep symlinked from the main checkout (`vcs/worktree.py`), and
`node_modules` goes into `.git/info/exclude` — a `node_modules/` gitignore
pattern does not match a symlink, so `git add -A` would commit it.

```yaml
gate_checks:
  - {name: go-test, cmd: splinter gate go-test, when: always, language: all, timeout: 900}
  - {name: ts-lint, cmd: splinter gate ts-lint, when: always, language: all, timeout: 300}
```

## Decision model (jev) on recall/filter

`splinter.providers.decisions` asks decision models (TypeSafe **jev** and
friends) typed questions and gets calibrated probabilities back, no generated
text. The localizer uses it to re-rank recall anchors and filter target files:
one batched noul/predicate question per item, below-threshold items drop,
probabilities become anchor confidence. Fails open (any error keeps the LLM's
own answer).

Providers are adapters over two wire formats:

| provider | endpoint | wire | default model | key env |
|---|---|---|---|---|
| `openrouter` | `/api/alpha/decisions` | typesafe (`state`, keyed `noul`) | `typesafe/jev-1.14` | `OPENROUTER_API_KEY` |
| `vercel` | `ai-gateway.vercel.sh/v1/decisions` | OpenAI (`input`, `predicate` array) | `typesafe-ai/jev` | `AI_GATEWAY_API_KEY` |

`decide_chain` walks `decisions.providers` in order — primary, then fallback —
each slot with its own model and token (`api_key` literal or `api_key_env`):

```yaml
decisions:
  providers:
    - {provider: vercel, model: typesafe-ai/jev, api_key_env: AI_GATEWAY_API_KEY}
    - {provider: openrouter, model: typesafe/jev-1.14}   # key from OPENROUTER_API_KEY
  use_for: [recall, recall_large, filter]   # per-step toggles
  threshold: 0.5
```

A legacy flat `provider`/`model`/`url`/`api_key_env` config reads as one slot.

```bash
splinter configure --jev-recall --jev-filter --decisions-model typesafe/jev-1.14
```

In the configure TUI the "jev" item holds only the credentials (primary +
fallback: provider, model, token or `$VAR`, plus the cutoff); the "usar jev"
toggles live on the Recall, Recall large and Filter panes (`--jev-recall`,
`--jev-recall-large`, `--jev-filter`). The TUI copy is English.

## Agents Strategy (plan-with-agents)

`splinter run --strategy agents` (alias `splinter`) ports the plan-with-agents
orchestration: PRD tasks are refined into **waves of briefs** (`agents/refiner.py`),
each brief runs on the **model chain its effort picks** (`engines/routing.py`:
brief effort → `effort_map` start tier → that tier's ordered models; the engine
CLI comes from the model prefix) inside
its own git worktree forked from a dedicated integration branch
(`splinter/<slug>-int`); splinter commits, validates and merges each brief back
with `--no-ff`. Flow per brief:

1. **Refine** — LLM turns tasks into small briefs (goal, files, sketch,
   acceptance, deps, effort — rated per brief, empty takes the task's) grouped into waves
2. **Launch** — the first healthy model of the effort chain runs the brief in a
   guarded worktree
   (PATH shims block git write/build/test/lint; the workhorse only edits files)
3. **Classify** — STALL/LIMIT/CRASH/EMPTY/MISSING (`engines/failures.py`);
   infra failures trip that model's circuit breaker and reroute the same brief
   to the next model of the chain **without burning attempts**. When every
   model of the chain is cooling down, the brief waits for the first breaker
   to let a trial through; after `max_attempts` such rounds it goes to the
   orchestrator. A workhorse that prints `AGENT_RESULT: BLOCKED` goes to the
   orchestrator at once, with its report as the reason (no acceptance run).
4. **Generate** — `agents.codegen` steps run in the worktree outside the guard
   (workhorses may not run generators), each only when a changed file starts
   with one of its `paths`. A failing step is a fix round with its output.
5. **Acceptance** — per brief: `gate` | `gate+eval` (default) | `gate+orchestrator`
   (`agents/acceptance.py`, config `agents.acceptance_check`) on the full diff
   (file list first, every new file listed); failures return a fix brief to the
   same engine session and burn one attempt
6. **Assume** — after `max_attempts` burned attempts the orchestrator takes the
   brief over via the direct loop in the same worktree, with the brief document
   (fix rounds and the reason included) as its plan: the planner does not run
7. **Merge** — acceptance PASS → commit + `merge --no-ff` into the integration
   branch; after all waves the final gate runs there and the branch lands in the
   main tree

Progress lives in `agents-run/`: `stories.json` (the PRD stories and the story
of each brief), `attempts.jsonl` (a start and an end line per attempt: story,
attempt number, brief, engine, tier, verdict, note), `checkpoint.json` (merged
briefs) and `logs/<brief>-<n>.log` (each engine's output). The run screen reads
these (`run_view_agents.py`): a story passes when all its briefs merged and is
live while one of its attempts is open, in sequential and parallel runs alike.

### Config (`agents:` in `.splinter/config.yaml`)

Only run knobs live here — the model is never configured per engine; edit the
effort tiers (`models.tiers`) instead. Legacy `engines`/`engine_models`/
`engine_efforts` keys are dropped on the next write.

```yaml
agents:
  acceptance_check: gate+eval   # gate | gate+eval | gate+orchestrator
  stall_min: 8                  # idle watchdog: minutes without output → STALL
  max_attempts: 3               # burned attempts before the orchestrator assumes
  probe_on_start: false         # live-probe the effort-chain models before wave 1
  refine_workers: 4             # stories refined at once, at most 2 per planner-pool model
  codegen:                      # generators splinter runs before acceptance
    - {cmd: make gen, paths: ["api/"], timeout: 600}
```

```bash
splinter configure --no-interactive \
  --agents-acceptance gate+eval \
  --agents-stall-min 8 --agents-max-attempts 3 --agents-probe-on-start
```

mcode/kimi/agy catalog models show up in the configure model picker.

### Engine catalog

`splinter engines` lists the catalog (mcode, kimi, kimi-flash, mcode-flash,
agy, sonnet, claude-plan, codex, glm, opencode-minimax + reviewers
opus/fable/sonnet);
`splinter engines --probe [--only mcode,agy] [--json]` live-probes health
("Reply with the word ok only."; up = standalone `ok`). Reviewer availability is
inferred from the sonnet ping. New provider CLIs: `mcode/…`, `kimi/…` and
`agy/…` model prefixes route to `providers/mcode.py` / `providers/kimi.py` /
`providers/agy.py`.

## Code Conventions

- Python 3.11+, type hints everywhere (`from __future__ import annotations`)
- `mypy --strict` compatible: `disallow_untyped_defs = true`
- Ruff: E, F, W, I (isort)
- Line length: 100
- Small files (800 lines max), except `splinter/tui.py`: it holds the app classes
  (`RunApp`, `PrdSessionApp`) and their modals, and may stay larger. New widgets and
  views still go to their own `tui_*.py` module.
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

## Run screen (C2)

`splinter run` opens the C2 screen (`RunApp` in `splinter/tui.py`). It is built for
long runs: 60 stories, one screen, no horizontal story rail.

### Layout

| Side | Widgets |
|---|---|
| Focus column (`#focus`) | block word (`#block-word`), sentence (`#sentence`), stage strip (`#stage-strip`), view tabs (`#view-tabs`), view body (`#view-body` → `#view-scroll` + `#log`) |
| Right rail (`#rail`) | clock (`#clock`), model ladder (`#ladder`), iteration bars (`#iter-bars`), story heatmap (`#heatmap`), groups (`#groups`) |
| Bottom | orchestrator composer (`#composer`), footer bar (`#footer-bar`), stop bar (`#stop-bar`) |

The header shows `story 24/58` only. The rail groups the stories: `needs you`,
`now`, `next` (5) and a count for the rest. Every visible cell, row, tab, rung
and bar is clickable and changes the screen state.

### Views and keys

| View | Key | Reads |
|---|---|---|
| `act` | `1` | activity log, collapsible iteration blocks, escalation lines |
| `diff` | `2` / `d` | the story's diff (`vcs/diff.py`) |
| `plan` | `3` | `knowledge/plan.md`, `knowledge/plan-N.md` |
| `gate` | `4` | the gate lines of `loop.md` |
| `eval` | `5` | `knowledge/task{N}-review-iter-{M}.md` |
| `spend` | `6` / `$` | spend by cost or by time |
| `tier` | `7` / `t` | the ladder, its models and their breakers |

Other keys: `←`/`→` previous/next story, `i` instruct (focus the composer), `o` open
the latest iteration, `/` find a story by id or title, `l` logs, `L` the full-screen
trace (needs you / now / done / queued tree, selected text, live log), `g` back to
the live story, `b` toggle spend by cost/time, `r` reload the diff, `s` stop,
`ctrl+k` cowabunga, `q` quit (or stop while the run lives). A view reads session
files only — no view calls a model.

### View model

`splinter/run_view.py` holds the pure model: `RunView`, `StoryRow`,
`IterationRow`, `SpendRow`, `LadderRung` and `build_run_view`. `RunViewBuilder`
caches every session file by its mtime and size (`run_view_files.FileCache`) and
groups the trace entries by story once per build, so a 60-story session rebuilds
in well under 50 ms.

The widgets never rebuild the tree. Each widget keeps its own render keys and
re-renders only the slice that changed (`FocusColumn.renders`,
`RunRail.renders` count those repaints). A tick with no file change rebuilds
nothing.

### Composer

The composer (`splinter/tui_run_composer.py`) replaces the old guide input. It
has three targets and shows the real delivery state of every message:

| Target | Backend |
|---|---|
| `story` | `Session.queue_live_command(text, task_no)` → `pending_directive.task-N.txt` (the shared queue is `pending_directive.txt`) |
| `all agents` | `Session.broadcast_live_command(text, task_nos)` → one directive per task |
| `orchestrator only` | `Session.queue_orchestrator_note(text)` → `knowledge/orchestrator-notes.md` |

Each directive carries an id; the pipeline writes a receipt under `receipts/`
when it drains the queue, and the composer flips `queued` → `delivered` from
those receipts (`Session.receipt_exists`, `Session.directive_pending`).

`knowledge/orchestrator-notes.md` has a `## Pending` and an `## Applied`
section. A note is applied when every consumer required by the strategy read it
(`direct`: replan, final eval — `agents`: refine, acceptance, final eval);
notes left over when the run ends are marked `(run ended)`.

### Needs you and stop

A story that needs the user (`asked`, `paused`, `fail`) raises a banner in the
focus column (`tui_run_banner.py`): answer the question, validate the result or
retry with guidance. The banner acts inline; below 30 rows the modals
(`_AskUserModal`, `_ManualValidationModal`) are the fallback. `s` opens the
inline stop bar (same outcomes as `StopMenuModal`).

### Preferences and motion

`.splinter/ui.yaml` keeps the PRD and plan questions mode:
`questions_mode: one` (default) or `all`. `splinter/ui_prefs.py` reads and
writes it.

`splinter/tui_run_motion.py` owns the motion: one timer (`MotionClock`)
alternates a phase every 0.5 s and the widgets repaint only what moves — the
live heatmap cell, the live ladder rung, the stage-strip connector and the
caret after the sentence while the runner streams. The timer is never created
when `App.animation_level == "none"` (`TEXTUAL_ANIMATIONS=none`), and it stops
when the run is done, when the app is suspended and when the app unmounts.

The live iteration on `act` streams every 🔧 tool call and 💬 thought as it lands
(`tui_run_live.py`): the newest row lights up, a caret blinks after it, and the
sentence above the stage strip says what the runner does now.

The toasts are the run's own events, each of them once (`ToastTracker`):
`Gate passed · T24`, `Escalated T24 to T1`, `T17 is asking you`,
`Run complete · 58/58 · $12.4120`.

