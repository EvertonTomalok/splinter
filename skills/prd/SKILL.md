---
name: prd
description: "Generate a Product Requirements Document (PRD) for a new feature or bug fix, adapted for the Splinter harness. Use when planning a feature, starting a new project, fixing a bug, or when asked to create a PRD. Triggers on: create a prd, write prd for, plan this feature, requirements for, spec out, fix bug. Saves into Splinter session files and records which strategy (turtle) to run."
user-invocable: true
---

# PRD Generator (Splinter)

Create detailed Product Requirements Documents that are clear, actionable, and
ready for the Splinter pipeline (`PRD -> LOCATE -> PLAN -> RUN -> GATE -> EVAL`).
Valid for features and bug fixes alike.

This skill is invoked by `uv run splinter prd`. It produces a PRD that Splinter
can consume directly: each user story maps to a Splinter task, acceptance criteria
map to gate/eval checks, and the chosen strategy is recorded in the frontmatter so
`splinter run` knows which turtle to use without asking again.

---

## The Job

1. Receive a feature or bug description from the user.
2. Ask 3-5 essential clarifying questions (with lettered options). One of them is
   always the **strategy** question, unless a strategy was already passed in.
3. Generate a structured PRD based on the answers.
4. Save to the Splinter session: `.splinter/sessions/<session-id>/prd.md` and
   update `index.md` to point at it.

**Important:** Do NOT start implementing. Just create the PRD.

---

## Step 1: Clarifying Questions

Ask only critical questions where the prompt is ambiguous. Always cover:

- **Problem/Goal:** What problem does this solve?
- **Core Functionality:** What are the key actions?
- **Scope/Boundaries:** What should it NOT do?
- **Success Criteria:** How do we know it is done?
- **Strategy:** Which Splinter turtle fits this work? (skip only if `--strategy`
  was passed to `splinter prd`)

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

3. Which Splinter strategy should run this?
   A. cascade (leonardo): big PRD, many small tasks, run sequentially
   B. direct (raphael): one focused change or bug fix, loop until it passes
   C. adaptive (donatello): cost sensitive, route each task to the cheapest capable model
   D. sprint (michelangelo): trivial or batch work, fast and cheap
```

Users respond with "1A, 2C, 3B" for quick iteration. Indent the options.

### Strategy guidance

When recommending a strategy, use this rule of thumb:

- Many user stories spanning several files -> `cascade`
- Single story, bug fix, or tight focused change -> `direct`
- Budget is a hard constraint, lots of medium tasks -> `adaptive`
- Trivial or repetitive low risk work -> `sprint`

---

## Step 2: PRD Structure

Generate the PRD with these sections. The PRD MUST begin with YAML frontmatter so
Splinter can parse it.

```markdown
---
feature: [feature-name-kebab-case]
strategy: [cascade | direct | adaptive | sprint]
kind: [feature | bugfix]
created: [ISO date]
---
```

### 1. Introduction/Overview
Brief description of the feature and the problem it solves.

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

**Acceptance Criteria:**
- [ ] Specific verifiable criterion
- [ ] Another criterion
- [ ] Typecheck/lint passes (gate)
- [ ] **[UI stories only]** Verify in browser using dev-browser skill

**Tests:** [Backend and Cronjob only]
- [ ] {{ Discover the unit tests needed for this story }} - all must pass before
  done. Do not force any previous test to pass.
```

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
  pointer to the PRD and the chosen strategy.

---

## Checklist

Before saving the PRD:

- [ ] Asked clarifying questions with lettered options (including strategy, unless
      passed in)
- [ ] Incorporated the user's answers
- [ ] Frontmatter includes `feature`, `strategy`, `kind`
- [ ] User stories are small, specific, and carry `effort` hints
- [ ] Acceptance criteria are verifiable (they drive the EVAL)
- [ ] Machine-checkable criteria are explicit (they drive the GATE)
- [ ] Non-goals define clear boundaries
- [ ] Saved to `.splinter/sessions/<session-id>/prd.md` and `index.md` updated
