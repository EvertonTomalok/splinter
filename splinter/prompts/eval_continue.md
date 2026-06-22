The runner applied your corrections. Judge the updated implementation against the same task and acceptance criteria as before.

{output_section}

{gate_section}

{previous_evals_section}

{user_directive_section}

A User Directive (if present) is a live instruction from the human running the loop
and OVERRIDES your default judgment on the point it addresses — honor it.

Do not ask the user to run commands locally. The loop is autonomous; return a
proper VERDICT with concrete runner corrections.

Respond in EXACTLY this format. The first line MUST be the literal word `VERDICT:`
followed by a single decision token and nothing else:

VERDICT: <PASS|RETRY|ESCALATE|JUMP_PREMIUM|ASK_USER>
REASON: <brief explanation, one line>
CORRECTIONS: <specific actionable instructions for what to fix, or 'none' if PASS>
