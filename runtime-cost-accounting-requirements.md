# Runtime Cost Accounting Requirements

## Problem
`splinter run` does not report runtime cost correctly. The visible total can diverge from the actual provider usage, and the discrepancy is most obvious with `opencode`, because that provider returns a cost for each call. The runtime accounting path must treat each provider response as one billable event and accumulate those costs across the full run.

This bug fix is about runtime accounting only. It must not change the pre-run tier estimator in `splinter/models/pricing.py`.

## Goals
- Report one correct total runtime cost for every `splinter run` session.
- Treat each `opencode` response cost as a per-call runtime cost and add it into the session total.
- Keep session traces, console output, and analysis views consistent with each other.
- Preserve existing token tracking and provider routing behavior.

## Functional Requirements
1. `splinter run` must record the cost from every successful runtime provider call.
2. The runtime total must equal the sum of all recorded runtime call costs for the session.
3. `opencode` costs must be taken from the provider response for each call, not reconstructed from token estimates or pre-run pricing tables.
4. If a run performs multiple model calls, retries, or eval-driven loops, each call cost must be included exactly once.
5. Runtime cost totals must exclude pre-run usage such as planning, localization, and PRD refinement.
6. Session trace output must display the same accumulated runtime total that the CLI reports at completion.
7. Analysis views that read session trace data must show the same runtime total as the underlying trace file.
8. When a provider does not supply a cost, the runtime accounting layer must keep the existing fallback behavior for that provider, but it must not invent a second estimate on top of the provider value.

## Acceptance Criteria
- A session with three `opencode` calls returning costs `0.0100`, `0.0200`, and `0.0300` reports a runtime total of `0.0600`.
- A mixed session with both Claude and `opencode` calls reports one total equal to the sum of all call-level costs, regardless of provider.
- The trace markdown, CLI summary, and any session analysis view all show the same total cost for the same session.
- Unit tests cover at least one `opencode` path where cost is returned per call and verified as an accumulated total.
- Unit tests do not hit live models or subprocesses.

## Non-Goals
- Do not change model routing.
- Do not change pre-run cost estimation for tier selection.
- Do not change provider authentication or session reuse semantics.
- Do not add a new pricing system or billing model.

## Technical Notes
- Likely runtime accounting touchpoints:
  - `splinter.providers.opencode._extract_cost`
  - `splinter.providers.opencode.OpencodeProvider.run`
  - `splinter.providers.dispatch.run_text`
  - `splinter.providers.dispatch.run_provider_session`
  - `splinter.agents.runner.run_task`
  - `splinter.obs.trace.log_run` and `Trace.total_cost`
  - `splinter.memory.session.log_llm_usage`
  - `splinter.pipeline` completion summary
- The provider cost returned by `opencode` should be treated as authoritative for that call.
- Session totals should be derived from accumulated runtime records, not from a single run result.

## Suggested Test Coverage
- Add or update a unit test for `splinter.providers.opencode` that returns a cost in the raw payload and verifies the parsed value.
- Add or update a trace test that records multiple runtime entries and verifies the summed total.
- Add or update a dispatch/runner test that ensures the recorded cost is propagated unchanged from provider response to runtime accounting.
- Keep tests isolated with mocks for provider calls and subprocess boundaries.
