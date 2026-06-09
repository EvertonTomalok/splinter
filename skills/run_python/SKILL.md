---
name: run_python
description: "Run a generated Python script via uv run python, capture stdout and exit code. Use as an eval_skill for tasks that produce runnable Python code."
---

# Run Python Skill

Execute a Python file with the project interpreter and report the result.

## Usage

This skill is referenced by `eval_skill: run_python` in a task definition.
The pipeline calls it after the runner produces code output.

## Execution

```bash
uv run python <generated_file.py>
```

## Output

- **exit code**: 0 = success, non-zero = failure
- **stdout**: captured for eval judgment
- **stderr**: captured for diagnostics

## Eval Integration

The gate runs this skill before the LLM evaluator. If exit code is non-zero,
the loop RETRYs without burning an eval call. If exit 0, the stdout is passed
to the evaluator along with the acceptance criteria.
