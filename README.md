# Splinter

**One model plans. Cheaper models do the work. A judge decides when to level up.**

Splinter is a multiagent coding harness. A smart, expensive model acts as sensei (plans the work, picks the right student, steps in only when students fail). Everything else runs on cheap, fast models until proven otherwise.

```
PRD → LOCATE → PLAN → RUN → GATE → EVAL → 🔁
```

1. **PRD** — you describe what needs to be built
2. **LOCATE** — cheap models find relevant code (broad recall → precise filter)
3. **PLAN** — sensei writes the plan (once per session)
4. **RUN** — student model implements
5. **GATE** — deterministic checks (lint, typecheck, test)
6. **EVAL** — judge returns PASS / RETRY / ESCALATE

On failure: same model retries with corrections. On repeated failure: escalate to a higher tier. Loop continues until pass or top of ladder.

---

## Strategies

| Strategy | Alias | Style |
|----------|-------|-------|
| `cascade` | leonardo | Breaks PRD into ordered tasks, checkpoints as it goes |
| `direct` | raphael | One task, implement, evaluate, loop hard, escalate fast |
| `adaptive` | donatello | Estimates effort per task, routes to cheapest capable model |
| `sprint` | michelangelo | Starts on flash tier, short loops, escalates on stall |

---

## Escalation ladder

```
T0  easy       glm-5.1 · kimi-k2.6
T1  moderate   deepseek-v4-pro
T2  hard       qwen3.7-max
T3  premium    sonnet → sonnet (effort max)
T4  top        opus-4.8
```

---

## Requirements

- [uv](https://github.com/astral-sh/uv)
- Python 3.11+
- [Claude Code](https://docs.claude.com/en/docs/claude-code/overview) CLI (authenticated)
- [opencode](https://opencode.ai) CLI (authenticated on `opencode-go`)

## Setup

```bash
# install
uv tool install git+https://github.com/evertontomalok/splinter.git

# authenticate providers
claude
opencode auth login

# allow non-interactive file edits
mkdir -p ~/.config/opencode
cat > ~/.config/opencode/opencode.json <<'JSON'
{"permission": {"edit": "allow"}}
JSON

# verify
splinter setup
```

---

## Commands

| Command | Description |
|---------|-------------|
| `splinter setup` | Verify environment and providers |
| `splinter prd "<description>"` | Generate a PRD interactively |
| `splinter run --prd <file>` | Run pipeline from a PRD |
| `splinter run --strategy <name> --task <file>` | Run a single task with a strategy |
| `splinter run ... --cowabunga` | Full autonomy mode (no user prompts) |
| `splinter analyze` | View state of most recent session |
| `splinter analyze --session <id>` | View state of a specific session |
| `splinter configure` | Open configuration TUI |
| `splinter configure --init-prompts` | Scaffold editable prompt templates |
| `splinter configure --init-prompts --force` | Reset prompts to defaults |
