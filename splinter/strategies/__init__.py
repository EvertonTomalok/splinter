"""Strategies (turtles). Importing the package registers the built-in strategies."""

from __future__ import annotations

# Imported for its registration side effect (@register on DirectStrategy).
from splinter.strategies import direct as _direct  # noqa: F401
