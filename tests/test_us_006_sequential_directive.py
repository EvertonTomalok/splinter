from __future__ import annotations

from pathlib import Path

import pytest

from splinter.templating import SEQUENTIAL_DIRECTIVE

# Five target files for US-006 sequential directive
FILES = [
    Path("skills/prd/SKILL.md"),
    Path("splinter/prompts/plan.md"),
    Path("splinter/prompts/eval.md"),
    Path("splinter/prompts/localize_recall.md"),
    Path("splinter/prompts/localize_precision.md"),
]


@pytest.mark.parametrize("file_path", FILES)
def test_sequential_directive_present(file_path: Path) -> None:
    """Verify SEQUENTIAL_DIRECTIVE is present in all five files."""
    content = file_path.read_text()
    assert SEQUENTIAL_DIRECTIVE.strip() in content, (
        f"Sequential directive not found in {file_path}.\n"
        f"Expected to find:\n{SEQUENTIAL_DIRECTIVE.strip()}"
    )
