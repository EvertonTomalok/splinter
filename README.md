# 🐀 Splinter

**One model plans. Cheaper models do the work. A judge decides when to level up.**

Splinter is a multiagent coding harness built on one stubborn idea: you should
not be paying frontier prices to write a hello world. A smart, expensive model
should act like a sensei (plan the work, pick the right student, and only step
in when the students fail). Everything else runs on cheap, fast models until
proven otherwise.

```
PRD  ->  PLAN  ->  RUN  ->  EVAL  ->  🔁 loop until the judge is happy
```

---

## The problem

Most agent setups burn a top tier model on every single step. Planning,
boilerplate, retries, evaluation... all at the same eye watering per token rate.
It works, but it is like hiring a master chef to butter your toast.

## The idea

Splitter splits the brain from the hands.

- 🧠 **The sensei** (`opus-4.8` or `sonnet` via `claude -p`) reads your PRD and
  writes the plan. Once. Up front.
- 🐢 **The students** (15+ open models via `opencode-go`) do the actual typing,
  starting from the cheapest one that can plausibly handle the task.
- ⚖️ **The judge** checks the output against acceptance criteria. If a cheap
  model flails, the judge climbs the ladder: `qwen -> sonnet -> sonnet max -> opus-4.8`.

You only pay for intelligence when the work actually demands it.

---

## Meet the squad

Four strategies, four personalities. Each has a formal name (what you pass to
`--strategy`) and a turtle alias (also accepted). Same pipeline, different attitude.

| Strategy | Turtle | Vibe |
|----------|--------|------|
| `cascade` | 🔵 **Leonardo** | Discipline. Breaks a big PRD into many tiny tasks, runs them in order, checkpoints as it goes. |
| `direct` | 🔴 **Raphael** | Attitude. One task, implement, evaluate, loop hard, escalate fast. Gets it done. |
| `adaptive` | 🟣 **Donatello** | Brains. Estimates effort per task, routes to the cheapest capable model, respects a budget. |
| `sprint` | 🟠 **Michelangelo** | Chill. Always starts on flash tier, short loops, bails up a tier the moment it stalls. |

So `--strategy cascade` and `--strategy leonardo` are the same thing.

---

## Requirements

Splinter is the conductor, not the models. You bring the two CLIs it drives:

- **[uv](https://github.com/astral-sh/uv)** the Python project manager Splinter
  is built on
- **Python 3.11+** (uv can install it for you)
- **[Claude Code](https://docs.claude.com/en/docs/claude-code/overview)** the
  `claude` CLI, authenticated, with access to `sonnet` and `opus-4.8`
- **[opencode](https://opencode.ai)** the `opencode` CLI, authenticated on the
  `opencode-go` provider
- **programming languages [optional]** python is required, but you must check if the language you're working is working, like GoLang, rust, etc.

## Setup

```bash
# 1. clone and install with uv
git clone https://github.com/evertontomalok/splinter.git
cd splinter
uv sync

# 2. authenticate the two providers (one time)
claude            # sign in to Claude Code
opencode auth login

# 3. let Splinter verify everything is wired up
uv run splinter setup
```

`splinter setup` does not just check the binaries exist, it pings each provider
for real:

```
checking providers...
  claude -p (sonnet) ..... OK
  opencode models ........ OK (14 models)
  ladder vs roster ....... OK
environment ready.
```

If a provider is missing or not authenticated, setup tells you exactly which one
and exits non zero, so you can drop it in CI too.

## Quickstart

```bash
# the direct strategy: one task, loop until it passes
uv run splinter run --strategy raphael --task task.yaml
```

```yaml
# task.yaml
description: "write a hello world in rust, compile it, run it"
acceptance: "binary compiles with exit 0 and prints something containing 'hello'"
effort: trivial          # task difficulty, sets the starting tier
reasoning_effort: auto   # how hard the model thinks, or let the agent decide
suggested_tier: 0
```

That run will: ask the sensei for a plan, hand it to a flash tier model, let it
create the folder, write the Rust, compile and execute, then let the judge
confirm the output. If the cheap model trips, Splinter quietly levels up and
tries again.

---

## The escalation ladder

Splinter never starts at the top. It earns its way there.

```
T0  flash      deepseek-v4-flash · mimo-v2.5 · qwen3.6-plus · glm-5
T1  mid        qwen3.7-plus · glm-5.1 · minimax-m2.5 · kimi-k2.5 · mimo-v2.5-pro
T2  strong     qwen3.7-max · deepseek-v4-pro · kimi-k2.6 · minimax-m2.7 · minimax-m3
T3  premium    sonnet  ->  sonnet (variant max)
T4  top        opus-4.8
```

The plan tags each task with an effort hint, so trivial work starts at T0 and a
gnarly refactor can start higher. The judge owns the climb from there.

---

## Two kinds of effort

Splinter separates **task difficulty** from **reasoning effort**.

- `effort` (task difficulty) picks the starting tier.
- `reasoning_effort` controls how hard the chosen model thinks. On open models it
  maps to `opencode run --variant <minimal|high|max>`. On Claude it maps to the
  equivalent `claude -p` effort control.

You can set it three ways: an explicit `--effort` flag wins, otherwise the
planner annotates each task, otherwise `reasoning_effort: auto` lets the agent
decide. Valid effort levels are read straight from the CLI, so when a new model
lands on `opencode-go` with different levels, Splinter adapts with no code change.

---

## 🤙 `--cowabunga` mode

By default, when the judge hits something genuinely critical (destructive
actions, architecture forks, blowing the budget) it stops and asks you, the
sensei.

Pass `--cowabunga` and the turtles stop waking the old man up. Full autonomy,
critical calls included. Powerful, occasionally chaotic, exactly as the name
suggests.

```bash
uv run splinter run --strategy raphael --prd prd.md --cowabunga
```

---

## How the judge thinks

Every loop, the evaluator returns one of five verdicts:

- ✅ `PASS` acceptance met, task done
- 🔄 `RETRY` recoverable miss, same model, try again
- ⬆️ `ESCALATE` this model cannot do it, climb one tier
- 🚀 `JUMP_PREMIUM` skip the line, go straight to sonnet max or opus-4.8
- 🙋 `ASK_USER` too important to guess, hand it to the human (unless `--cowabunga`)

Evaluation runs two ways: a written acceptance check, or a real skill/script
(think `go test` or a custom validator) plus a judgment on top.

---

## Why it is different

- **Cost aware by design.** Cheap first, expensive only when proven necessary.
  Every step logs its own token count and cost.
- **Bring your own ladder.** The tiers and escalation rules live in a single
  `ladder.yaml`. Reorder, swap models, set your own jump points.
- **Two CLIs, one brain.** Premium thinking through `claude -p`, a deep bench of
  open models through `opencode-go`, unified behind one pipeline.
- **Pick your fighter.** Long marathon or quick brawl, same harness, one flag.

---

## Status

Early and moving fast. The core loop (plan with the sensei, execute with a
student, evaluate, escalate) already runs end to end. Strategies and the
budget aware router are landing next.

Cowabunga. 🐢
