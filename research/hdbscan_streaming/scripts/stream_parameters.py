'''Shared validation for synthetic benchmark stream generators.'''
from __future__ import annotations

import math


def positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f'{name} must be an integer')
    if value < 1:
        raise ValueError(f'{name} must be at least 1')
    return value


def nonnegative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f'{name} must be an integer')
    if value < 0:
        raise ValueError(f'{name} must be at least 0')
    return value


def nonnegative_finite_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a number')
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    if value < 0:
        raise ValueError(f'{name} must be non-negative')
    return float(value)


def center_similarity_threshold(name: str, value: float) -> float:
    value = nonnegative_finite_float(name, value)
    if value > 1:
        raise ValueError(f'{name} must be at most 1')
    return value


def drift_center_similarity(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a number')
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    if value <= -1 or value > 1:
        raise ValueError(f'{name} must be in (-1, 1]')
    return float(value)
