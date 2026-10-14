"""Shared benchmark argument profiles for reproduction drivers."""

from __future__ import annotations


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
