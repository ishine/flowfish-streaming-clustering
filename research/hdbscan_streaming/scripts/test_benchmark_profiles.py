"""Lightweight checks for shared benchmark profiles."""

from __future__ import annotations

import argparse

from benchmark_profiles import async_pair_benchmark_args, validate_benchmark_args


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


def test_validate_benchmark_args_rejects_invalid_ranges() -> None:
    valid = argparse.Namespace(
        cap=100,
        warmup=20,
        tradeoff_warmup=20,
        tradeoff_margin=0.02,
        tradeoff_max_distance=0.70,
        tradeoff_full_interval=8,
        tradeoff_batch_size=4,
        arrival_interval_ms=5.0,
        tradeoff_max_queue=8,
        min_cluster_size=3,
        min_samples=2,
        checkpoint_every=80,
    )
    assert validate_benchmark_args(valid) is valid

    invalid_batch = argparse.Namespace(**{**vars(valid), "tradeoff_batch_size": 0})
    try:
        validate_benchmark_args(invalid_batch)
    except ValueError as exc:
        assert "tradeoff_batch_size" in str(exc)
    else:
        raise AssertionError("expected invalid tradeoff_batch_size to fail")

    invalid_interval = argparse.Namespace(**{**vars(valid), "arrival_interval_ms": -1.0})
    try:
        validate_benchmark_args(invalid_interval)
    except ValueError as exc:
        assert "arrival_interval_ms" in str(exc)
    else:
        raise AssertionError("expected invalid arrival_interval_ms to fail")

    invalid_nonfinite_interval = argparse.Namespace(
        **{**vars(valid), "arrival_interval_ms": float("nan")}
    )
    try:
        validate_benchmark_args(invalid_nonfinite_interval)
    except ValueError as exc:
        assert "arrival_interval_ms" in str(exc)
        assert "finite" in str(exc)
    else:
        raise AssertionError("expected non-finite arrival_interval_ms to fail")

    invalid_nonfinite_margin = argparse.Namespace(
        **{**vars(valid), "tradeoff_margin": float("inf")}
    )
    try:
        validate_benchmark_args(invalid_nonfinite_margin)
    except ValueError as exc:
        assert "tradeoff_margin" in str(exc)
        assert "finite" in str(exc)
    else:
        raise AssertionError("expected non-finite tradeoff_margin to fail")


def test_validate_benchmark_args_reports_missing_fields() -> None:
    try:
        validate_benchmark_args(argparse.Namespace(cap=1))
    except ValueError as exc:
        assert "tradeoff_full_interval is required" in str(exc)
    else:
        raise AssertionError("expected missing benchmark field to fail")


if __name__ == "__main__":
    for name, test_func in sorted(globals().items()):
        if name.startswith("test_") and callable(test_func):
            test_func()
