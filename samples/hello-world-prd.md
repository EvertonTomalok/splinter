---
feature: hello-world
strategy: direct
kind: feature
created: 2026-06-09T00:00:00Z
---

# Hello World

Write a Python script that prints "hello world" to stdout and exits successfully.

### US-001: Print hello world
**Description:** As a user, I want a Python script that prints "hello world" so that I can verify the pipeline works end to end.

**Splinter hints:**
- effort: trivial
- eval_skill: run_python

**Acceptance Criteria:**
- [ ] A Python file named `hello.py` exists
- [ ] Running `python hello.py` exits with code 0
- [ ] stdout contains the string "hello"
