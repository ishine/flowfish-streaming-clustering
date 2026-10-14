"""Lightweight checks for shared benchmark profiles."""

from __future__ import annotations

from benchmark_profiles import async_pair_benchmark_args


def test_async_pair_benchmark_args_exposes_scale_knobs() -> None:
    args = async_pair_benchmark_args(
        interval_ms=5.0,
        warmup=20,
        full_interval=8,
        checkpoint_every=80,
    )

    pairs = dict(zip(args[0::2], args[1::2]))
    assert pairs["--warmup"] == "20"
    assert pairs["--tradeoff-warmup"] == "20"
    assert pairs["--tradeoff-full-interval"] == "8"
    assert pairs["--tradeoff-batch-size"] == "4"
    assert pairs["--arrival-interval-ms"] == "5.0"
    assert pairs["--tradeoff-max-queue"] == "8"
    assert pairs["--checkpoint-every"] == "80"


if __name__ == "__main__":
    test_async_pair_benchmark_args_exposes_scale_knobs()
