"""Markdown-file prompt templates with project-level overrides.

Every prompt the harness sends a model is stored as an editable ``.md`` template.
Resolution order for a template named ``foo``:

1. ``./.splinter/prompts/foo.md`` — project override (written by ``splinter configure``)
2. the packaged default shipped in ``splinter/prompts/``

Templates are plain markdown with ``{placeholder}`` slots filled by
:func:`render`. Pass whole *sections* (header + body, built with :func:`section`)
so optional context can collapse to nothing when absent — :func:`render` strips
the blank gaps an empty section leaves behind.
"""

from __future__ import annotations

import importlib.resources
import re
from pathlib import Path

PROMPTS_PACKAGE = "splinter.prompts"

#: Template names the harness ships and that ``configure`` scaffolds.
TEMPLATE_NAMES = (
    "plan",
    "run",
    "run_fix",
    "eval",
    "localize_recall",
    "localize_precision",
)


def _override_dir() -> Path:
    return Path(".splinter") / "prompts"


def _override_path(name: str) -> Path:
    return _override_dir() / f"{name}.md"


def packaged_template(name: str) -> str:
    """Read the packaged default template text for ``name``."""
    ref = importlib.resources.files(PROMPTS_PACKAGE) / f"{name}.md"
    return ref.read_text()


def load_template(name: str) -> str:
    """Return the template text, preferring a project override over the default."""
    override = _override_path(name)
    if override.exists():
        return override.read_text()
    return packaged_template(name)


class _Blanks(dict):
    """format_map backing dict that renders any missing placeholder as empty."""

    def __missing__(self, key: str) -> str:
        return ""


def section(title: str, body: str) -> str:
    """Render a ``## {title}`` block, or an empty string when ``body`` is blank."""
    if body and body.strip():
        return f"## {title}\n{body.strip()}"
    return ""


def render(name: str, **values: str) -> str:
    """Fill template ``name`` with ``values`` and collapse blank-line gaps."""
    filled = load_template(name).format_map(_Blanks(values))
    collapsed = re.sub(r"\n{3,}", "\n\n", filled).strip()
    return collapsed + "\n"


def load_standards() -> str:
    """Return the Code Conventions section from AGENTS.md, or '' if absent.

    Resolution order:
    1. ``.splinter/AGENTS.md`` (project override)
    2. ``AGENTS.md`` at cwd (standard project root location)
    """
    for candidate in (Path(".splinter") / "AGENTS.md", Path("AGENTS.md")):
        if candidate.exists():
            text = candidate.read_text()
            match = re.search(
                r"^## Code Conventions\s*\n(.*?)(?=^## |\Z)",
                text,
                re.MULTILINE | re.DOTALL,
            )
            if match:
                return match.group(1).strip()
    return ""
