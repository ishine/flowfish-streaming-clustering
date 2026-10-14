#!/usr/bin/env python3
"""Lightweight checks for benchmark driver argument boundaries."""
from __future__ import annotations

import tempfile
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

from driver_validation import validate_nonnegative_ints, validate_positive_floats
from output_paths import resolve_experiment_output_dir
from run_multiseed_validation import main as multiseed_main
from run_rate_sweep import main as rate_sweep_main


def test_resolve_experiment_output_dir_accepts_results_and_external() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        results_out = root / "results" / "run"
        external_out = root.parent / f"{root.name}_run"
        assert resolve_experiment_output_dir(results_out, root) == results_out
        assert resolve_experiment_output_dir(external_out, root) == external_out


def test_resolve_experiment_output_dir_rejects_repo_root_output() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        try:
            resolve_experiment_output_dir(root / "run", root)
        except ValueError as exc:
            assert "results/" in str(exc)
        else:
            raise AssertionError("expected repository-root --out to fail")


def test_resolve_experiment_output_dir_rejects_file_target() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        target = root / "results" / "run"
        target.parent.mkdir()
        target.write_text("not a directory", encoding="utf-8")
        try:
            resolve_experiment_output_dir(target, root)
        except ValueError as exc:
            assert "directory" in str(exc)
        else:
            raise AssertionError("expected file --out to fail")


def test_validate_driver_sequences() -> None:
    assert validate_positive_floats([0.5, 1.0], "--intervals") == [0.5, 1.0]
    assert validate_nonnegative_ints([0, 7], "--seeds") == [0, 7]

    try:
        validate_positive_floats([1.0, 0.0], "--intervals")
    except ValueError as exc:
        assert "--intervals" in str(exc)
    else:
        raise AssertionError("expected zero interval to fail")

    for candidate in (float("nan"), float("inf")):
        try:
            validate_positive_floats([candidate], "--intervals")
        except ValueError as exc:
            assert "finite" in str(exc)
        else:
            raise AssertionError(f"expected non-finite interval to fail: {candidate!r}")

    try:
        validate_nonnegative_ints([-1], "--seeds")
    except ValueError as exc:
        assert "--seeds" in str(exc)
    else:
        raise AssertionError("expected negative seed to fail")


def test_rate_sweep_cli_rejects_invalid_interval_before_subprocess() -> None:
    with tempfile.TemporaryDirectory() as tmp, redirect_stderr(StringIO()):
        try:
            rate_sweep_main(
                [
                    "--data",
                    str(Path(tmp) / "data.npz"),
                    "--out",
                    str(Path(tmp) / "results" / "rate"),
                    "--intervals",
                    "0",
                ]
            )
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError("expected invalid interval to fail")


def test_rate_sweep_cli_rejects_nonfinite_interval_before_subprocess() -> None:
    with tempfile.TemporaryDirectory() as tmp, redirect_stderr(StringIO()):
        try:
            rate_sweep_main(
                [
                    "--data",
                    str(Path(tmp) / "data.npz"),
                    "--out",
                    str(Path(tmp) / "results" / "rate"),
                    "--intervals",
                    "nan",
                ]
            )
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError("expected non-finite interval to fail")


def test_multiseed_cli_rejects_repo_root_output_before_subprocess() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    with redirect_stderr(StringIO()):
        try:
            multiseed_main(
                [
                    "--out",
                    str(repo_root / "scratch_probe_for_test"),
                    "--seeds",
                    "1",
                    "--interval-ms",
                    "5",
                ]
            )
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError("expected repository-root output to fail")


if __name__ == "__main__":
    test_resolve_experiment_output_dir_accepts_results_and_external()
    test_resolve_experiment_output_dir_rejects_repo_root_output()
    test_resolve_experiment_output_dir_rejects_file_target()
    test_validate_driver_sequences()
    test_rate_sweep_cli_rejects_invalid_interval_before_subprocess()
    test_rate_sweep_cli_rejects_nonfinite_interval_before_subprocess()
    test_multiseed_cli_rejects_repo_root_output_before_subprocess()
