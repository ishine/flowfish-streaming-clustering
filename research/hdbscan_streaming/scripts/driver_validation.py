"""Shared validation for benchmark driver arguments."""
from __future__ import annotations

import math


def validate_positive_floats(values: list[float], field_name: str) -> list[float]:
    if not values:
        raise ValueError(f"{field_name} must contain at least one value")
    for value in values:
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{field_name} values must be finite and greater than 0")
    return values


def validate_nonnegative_ints(values: list[int], field_name: str) -> list[int]:
    if not values:
        raise ValueError(f"{field_name} must contain at least one value")
    for value in values:
        if value < 0:
            raise ValueError(f"{field_name} values must be non-negative")
    return values
