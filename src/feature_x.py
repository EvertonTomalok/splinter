from __future__ import annotations

from numbers import Real


def feature_x(value: Real) -> Real:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("Expected a number")
    if value < 0:
        raise ValueError("Expected a non-negative number")
    return value * 2
