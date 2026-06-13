from __future__ import annotations


def feature_x(value: int | float) -> int | float:
    if not isinstance(value, (int, float)):
        raise TypeError(f"Expected a number, got {type(value).__name__}")
    if value < 0:
        raise ValueError("Expected a non-negative number")
    return value * 2
