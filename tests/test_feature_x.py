from __future__ import annotations

import pytest

from src.feature_x import feature_x


def test_feature_x_happy_path() -> None:
    assert feature_x(5) == 10
    assert feature_x(3.5) == 7.0


def test_feature_x_edge_empty() -> None:
    assert feature_x(0) == 0


def test_feature_x_invalid_input() -> None:
    with pytest.raises(TypeError, match="Expected a number"):
        feature_x("hello")
    with pytest.raises(ValueError, match="Expected a non-negative number"):
        feature_x(-1)
