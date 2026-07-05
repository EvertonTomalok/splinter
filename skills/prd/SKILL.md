---
name: prd
description: "Generate a Product Requirements Document (PRD) for a new feature or bug fix, adapted for the Splinter harness. Use when planning a feature, starting a new project, fixing a bug, or when asked to create a PRD. Triggers on: create a prd, write prd for, plan this feature, requirements for, spec out, fix bug. Saves into Splinter session files."
user-invocable: true
---

Runs are sequential-only. No strategy selection. No parallel execution. One task at a time.

Multi-task strategies (cascade/adaptive/sprint) support opt-in parallel execution via `--parallel` when git worktree support is detected. Each task runs in its own worktree; results squash-merged on PASS.

# PRD Generator (Splinter)

Create detailed Product Requirements Documents that are clear, actionable, and
ready for the Splinter pipeline (`PRD -> LOCATE -> PLAN -> RUN -> GATE -> EVAL`).
Valid for features and bug fixes alike.

This skill is invoked by `uv run splinter prd`. It produces a PRD that Splinter
can consume directly: each user story maps to a Splinter task, acceptance criteria
map to gate/eval checks. Strategy is decided later in the pipeline; the PRD remains
strategy-agnostic and grounded in codebase analysis.

---

## The Job

1. Receive a feature or bug description from the user.
2. Read injected localization grounding (codebase anchors: file:func findings).
3. Ask 3-4 essential clarifying questions (with lettered options) grounded in real
   code locations.
4. Generate a structured PRD based on the answers, with critical-analysis findings
   tied to specific files/functions.
5. Save to the Splinter session: `.splinter/sessions/<session-id>/prd.md` and
   update `index.md` to point at it.

**Important:** Do NOT start implementing. Just create the PRD.

---

## Step 0: Localization (First)

Before any question or PRD text, ground your analysis in the codebase.

**Injected grounding block:** Read `.splinter/sessions/<session-id>/knowledge/localization.md`
(and per-task `localization-N.md` if present). This block contains real `file:func` anchors
discovered by the localizer.

**Hot-path read strategy:**
- **Anchors marked `relevance: high`**: Read inline using the `rtk:` tip per anchor.
  Format: `rtk read <file> | sed -n 'A,Bp'` (read lines A to B of file).
- **Anchors marked `relevance: medium` or `low`**: Pointer only; read on demand if
  a question or PRD section requires grounding.

**Critical-analysis mandate:** Before asking any question or generating any PRD section:
- Name concrete files and functions (never vague references like "the auth code").
- Flag regressions and impact: "Changing X in `file:func` will require updating
  Y in `file:func`" (cite both anchors).
- Ground every assertion in a real `file:func` anchor from localization.
- No hand-waving; no claims without anchors.

---

## Step 1: Clarifying Questions

Always ask 3-10 clarifying questions — never skip this step, even when the request
looks complete. If you would otherwise assume a default, turn that assumption into a
question and make the default one of the lettered options. Do NOT emit an
"assumptions" / "defaults" list in place of questions, and do NOT answer them
yourself. Always cover:

- **Problem/Goal:** What problem does this solve?
- **Core Functionality:** What are the key actions?
- **Scope/Boundaries:** What should it NOT do?
- **Success Criteria:** How do we know it is done?

**NEVER ask about execution strategy (cascade/direct/adaptive/sprint).** Strategy is decided later in the pipeline, not here.

### Format Questions Like This

```
1. What is the primary goal of this feature?
   A. Improve user onboarding experience
   B. Increase user retention
   C. Reduce support burden
   D. Other: [please specify]

2. What is the scope?
   A. Minimal viable version
   B. Full-featured implementation
   C. Just the backend/API
   D. Just the UI

3. Which files or systems will this change?
   A. Only frontend (app/* components)
   B. Only backend (internal/handlers/*, internal/services/*)
   C. Both frontend and backend
   D. Other (please specify)

4. What are the acceptance criteria for success?
   A. Must pass tests and lint
   B. Must not break existing features
   C. Must include user acceptance verification
   D. Other: [please specify]
```

Users respond with "1A, 2C, 3B, 4C" for quick iteration. Indent the options.

### Citing Real Findings

Each question MUST cite a real `file:func` finding from localization. Example:

```
1. localizer.localize() caches on knowledge/localization.md — should grounding
   move into prd_session.py?
   A. Yes, add grounding to _load_prd_skill()
   B. No, keep it injected as a separate prompt section
   C. Defer to Step 2 generation
   D. Other: [please specify]
```

This ensures questions target actual code anchors and improve PRD quality.

---

## Step 2: PRD Structure

Generate the PRD with these sections. The PRD MUST begin with YAML frontmatter so
Splinter can parse it.

```markdown
---
feature: [feature-name-kebab-case]
kind: [feature | bugfix]
created: [ISO date]
---
```

### 1. Introduction/Overview
Brief description of the feature and the problem it solves. Ground high-impact
design decisions in `file:func` anchors from localization.

### 2. Goals
Specific, measurable objectives (bullet list).

### 3. User Stories
Each story maps to one Splinter task. Keep each small enough for one focused
session.

**Format:**
```markdown
### US-001: [Title]
**Description:** As a [user], I want [feature] so that [benefit].

**Splinter hints:**
- effort: [trivial | normal | hard | critical]
- eval_skill: [skill name, or omit for written-criteria eval]
- deps: [US-NNN, US-MMM]          # optional: IDs of tasks this depends on
- parallelizable: [true | false]   # optional: override derived default

**Acceptance Criteria:**
- [ ] Specific verifiable criterion
- [ ] Another criterion
- [ ] Typecheck/lint passes (gate)
- [ ] **[UI stories only]** Verify in browser using dev-browser skill

**Tests:** [Backend and Cronjob only]
- [ ] {{ Discover the unit tests needed for this story }} - all must pass before
  done. Do not force any previous test to pass.
```

**Dependency and parallelism hints:**
- `deps: []` — list of US-IDs this task depends on. Task starts only after all deps PASS.
  Use `Depends on US-NNN` or `Blocked until US-NNN` anywhere in the block (also parsed).
- `parallelizable: true/false` — explicit override. When omitted, derived from deps:
  tasks with no deps are parallelizable by default; tasks with deps are not.
- Independent tasks (no deps) may run concurrently when `--parallel` is passed to `splinter run`.
- Failed task aborts only its transitive dependents; independent tasks keep running.
- **File-disjointness invariant (required for parallel):** two stories that modify
  the same file/module MUST NOT be parallel — give one a `deps` edge on the other
  so the DAG serialises them. Parallel tasks run in isolated git worktrees and are
  squash-merged independently; if they touch the same file their merges conflict.
  Only stories with **disjoint file sets** may be left dependency-free. When in
  doubt, chain them with `deps` — a serial edge is cheap, a lost merge is not.

**Important:**
- Acceptance criteria must be verifiable, not vague. "Works correctly" is bad.
  "Button shows confirmation dialog before deleting" is good. These criteria are
  what the Splinter EVAL judges against, so precision here directly improves the
  loop.
- The `effort` hint sets the starting tier in the escalation ladder.
- Criteria that a machine can check (compile, tests, lint, type) belong to the
  deterministic GATE; keep them explicit so the gate can run them.
- For any story with UI changes, include browser verification.

### 4. Functional Requirements
Numbered list of specific functionalities:
- "FR-1: The system must allow users to..."
- "FR-2: When a user clicks X, the system must..."

Be explicit and unambiguous.

### 5. Non-Goals (Out of Scope)
What this feature will NOT include. Critical for scope.

### 6. Design Considerations (Optional)
- UI/UX requirements, mockups, components to reuse.

### 7. Technical Considerations (Optional)
- Known constraints, dependencies, integration points, performance needs.
- **Impact analysis:** Name each file:func anchor that will be modified; flag
  regressions and downstream effects tied to real code locations.
- **Regression list:** For each story, list code paths that must not break,
  anchored to specific files/functions.

### 8. Success Metrics
How success is measured, including the project-wide gates:
- No lint error
- No build error (frontend and backend)
- No unit test errors [Cronjob or Backend only]

### 9. Open Questions
Remaining questions or areas needing clarification.

---

## Writing for Junior Developers (and cheap models)

The PRD reader may be a junior developer, a cheap executor model, or the planner.
Therefore:

- Be explicit and unambiguous.
- Avoid jargon or explain it.
- Provide enough detail to understand purpose and core logic.
- Number requirements for easy reference.
- Use concrete examples where helpful.

This matters extra in Splinter: cheap models do the implementation, so an
ambiguous criterion costs you escalation cycles (and tokens).

---

## Output

- **Format:** Markdown (`.md`) with the YAML frontmatter above.
- **Location:** `.splinter/sessions/<session-id>/prd.md`
- After saving, update `.splinter/sessions/<session-id>/index.md` with a one line
  pointer to the PRD.

---

## Checklist

Before saving the PRD:

- [ ] Localization grounding read (Step 0)
- [ ] Each clarifying question cites a real `file:func` anchor from localization
- [ ] Incorporated the user's answers
- [ ] Frontmatter includes `feature`, `kind`, `created`
- [ ] User stories are small, specific, and carry `effort` hints
- [ ] Acceptance criteria are verifiable (they drive the EVAL)
- [ ] Machine-checkable criteria are explicit (they drive the GATE)
- [ ] Non-goals define clear boundaries
- [ ] §7 Technical Considerations names regressions/impact with `file:func` anchors
- [ ] Saved to `.splinter/sessions/<session-id>/prd.md` and `index.md` updated
