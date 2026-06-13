.PHONY: install install-dev lint fmt fmt-check typecheck test gate clean

# ── install ──────────────────────────────────────────────────────────────────

install:
	uv sync

install-dev:
	uv sync --group dev

# ── format ───────────────────────────────────────────────────────────────────

fmt:
	uv run ruff format splinter/ tests/

fmt-check:
	uv run ruff format --check splinter/ tests/

# ── lint ─────────────────────────────────────────────────────────────────────

lint:
	uv run ruff check splinter/ tests/

lint-fix:
	uv run ruff check --fix splinter/ tests/

# ── types ────────────────────────────────────────────────────────────────────

typecheck:
	uv run mypy splinter/

# ── tests ────────────────────────────────────────────────────────────────────

test:
	uv run pytest tests/ -x -q

test-v:
	uv run pytest tests/ -v

# ── gate (CI equivalent — must all pass before push) ─────────────────────────

gate: fmt lint typecheck test

# ── clean ────────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
