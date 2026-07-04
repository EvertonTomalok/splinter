---
feature: hello-world-cascade
strategy: cascade
kind: feature
created: 2026-06-10T00:00:00Z
parallel: false
---

# Hello World (Cascade)

Two-step pipeline: a shared module, then a script that imports it.
Demonstrates `cascade` strategy with explicit task dependencies and parallelizable hints.

### US-001: Greeting module
**Description:** As a developer, I want a `greet.py` module with a `greeting()` function that returns a string so that the main script can import it.

**Splinter hints:**
- effort: trivial
- parallelizable: true

**Acceptance Criteria:**
- [ ] `greet.py` exists with a `greeting()` function
- [ ] `greeting()` returns a non-empty string containing "hello"

### US-002: Main script
**Description:** As a user, I want a `hello_cascade.py` script that imports `greet.greeting()` and prints it so that the cascade dependency is exercised end to end.

**Splinter hints:**
- effort: trivial
- deps: [US-001]
- parallelizable: false

**Acceptance Criteria:**
- [ ] `hello_cascade.py` exists and imports from `greet`
- [ ] Running `python hello_cascade.py` exits with code 0
- [ ] stdout contains "hello"
