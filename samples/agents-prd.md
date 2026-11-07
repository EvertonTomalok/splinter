---
strategy: agents
parallel: true
---

# PRD — Agents strategy smoke run

## Goal

Exercise the `agents` strategy end to end on this repository: refine the tasks
below into briefs, run them through the engine pool in guarded worktrees,
validate acceptance per brief, and merge the integration branch back.

## User Stories

### US-101: greeting module

Add `splinter_greeting.py` at the repo root exposing
`greet(name: str) -> str` that returns `hello, <name>`. Pure function, no I/O.

- [ ] `greet("splinter")` returns `hello, splinter`
- [ ] module has no imports

deps: []

### US-102: greeting CLI flag

Extend the greeting module with `main(argv: list[str]) -> int` that prints
`greet(argv[1])` when an argument is given (exit 0) and prints `usage: greeting`
to stdout (exit 1) otherwise.

- [ ] `main(["splinter"])` prints `hello, splinter` and returns 0
- [ ] `main([])` prints `usage: greeting` and returns 1

deps: [US-101]

### US-103: greeting docs

Add a `GREETING.md` note at the repo root describing both functions in one
paragraph each.

- [ ] file exists and mentions `greet` and `main`

deps: [US-101]
