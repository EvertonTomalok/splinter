"""Per-user UI preferences stored as ``.splinter/ui.yaml``.

Small key/value pairs that survive between runs — today only the questions
display mode (``one`` at a time or ``all`` at once). The file is optional: a
missing or malformed file reads as no preferences, and every getter carries
its own default so callers never branch on read errors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml

UI_PREFS_FILE = "ui.yaml"

QuestionsMode = Literal["one", "all"]


def _default_root() -> Path:
    return Path.cwd() / ".splinter"


def _prefs_path(root: Path | None) -> Path:
    base = root if root is not None else _default_root()
    return base / UI_PREFS_FILE


def read_ui_prefs(root: Path | None = None) -> dict[str, str]:
    """Read ``ui.yaml`` under ``root`` (default ``.splinter`` in the cwd).

    A missing file, unreadable file or anything that is not a string mapping
    reads as empty preferences.
    """
    try:
        data = yaml.safe_load(_prefs_path(root).read_text())
    except (OSError, yaml.YAMLError, UnicodeDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): str(value) for key, value in data.items()}


def write_ui_pref(key: str, value: str, root: Path | None = None) -> None:
    """Merge one key into ``ui.yaml`` under ``root``, creating the directory."""
    prefs = read_ui_prefs(root)
    prefs[key] = value
    base = root if root is not None else _default_root()
    base.mkdir(parents=True, exist_ok=True)
    _prefs_path(base).write_text(yaml.safe_dump(prefs))


def questions_mode(root: Path | None = None) -> QuestionsMode:
    """The saved questions display mode; ``one`` unless saved as ``all``."""
    return "all" if read_ui_prefs(root).get("questions_mode") == "all" else "one"
