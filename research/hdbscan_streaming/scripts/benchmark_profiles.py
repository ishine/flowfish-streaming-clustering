"""Shared benchmark argument profiles for reproduction drivers."""

from __future__ import annotations

import argparse
import math


def validate_benchmark_args(args: argparse.Namespace) -> argparse.Namespace:
    missing = object()
    positive_int_fields = (
        "cap",
        "tradeoff_full_interval",
        "tradeoff_batch_size",
        "tradeoff_max_queue",
        "min_cluster_size",
        "min_samples",
        "checkpoint_every",
        "warmup",
        "tradeoff_warmup",
    )
    for field_name in positive_int_fields:
        value = getattr(args, field_name, missing)
        if value is missing:
            raise ValueError(f"{field_name} is required")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field_name} must be an integer")
        if value < 1:
            raise ValueError(f"{field_name} must be at least 1")

    value = getattr(args, "arrival_interval_ms", missing)
    if value is missing:
        raise ValueError("arrival_interval_ms is required")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("arrival_interval_ms must be a number")
    if not math.isfinite(value) or value < 0:
        raise ValueError("arrival_interval_ms must be finite and non-negative")

    positive_float_fields = ("tradeoff_margin", "tradeoff_max_distance")
    for field_name in positive_float_fields:
        value = getattr(args, field_name, missing)
        if value is missing:
            raise ValueError(f"{field_name} is required")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field_name} must be a number")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{field_name} must be finite and greater than 0")

    return args


def async_pair_benchmark_args(
    *,
    interval_ms: float,
    warmup: int,
    full_interval: int,
    checkpoint_every: int,
    batch_size: int = 4,
    max_queue: int = 8,
    min_cluster_size: int = 3,
    min_samples: int = 2,
) -> list[str]:
    validate_benchmark_args(
        argparse.Namespace(
            cap=1,
            warmup=warmup,
            tradeoff_warmup=warmup,
            tradeoff_margin=0.02,
            tradeoff_max_distance=0.70,
            tradeoff_full_interval=full_interval,
            tradeoff_batch_size=batch_size,
            arrival_interval_ms=interval_ms,
            tradeoff_max_queue=max_queue,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            checkpoint_every=checkpoint_every,
        )
    )
    return [
        "--warmup",
        str(warmup),
        "--tradeoff-warmup",
        str(warmup),
        "--tradeoff-margin",
        "0.02",
        "--tradeoff-max-distance",
        "0.70",
        "--tradeoff-full-interval",
        str(full_interval),
        "--tradeoff-batch-size",
        str(batch_size),
        "--arrival-interval-ms",
        str(interval_ms),
        "--tradeoff-max-queue",
        str(max_queue),
        "--min-cluster-size",
        str(min_cluster_size),
        "--min-samples",
        str(min_samples),
        "--checkpoint-every",
        str(checkpoint_every),
    ]
