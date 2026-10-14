#!/usr/bin/env python3
"""Lightweight checks for the common NPZ dataset validator."""
from __future__ import annotations

import tempfile
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

import numpy as np

from validate_dataset import main, resolve_report_path, validate_common_npz


def write_npz(path: Path, labels: list[int]) -> None:
    X = np.eye(len(labels), 3, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    np.savez(path, X=X, y=y)


def write_raw_npz(path: Path, X: np.ndarray, y: np.ndarray) -> None:
    np.savez(path, X=X, y=y)


def test_validate_common_npz_accepts_dataset() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sample.npz"
        write_npz(path, [0, 0, 1])
        report = validate_common_npz(path, min_speakers=2)
        assert report["n"] == 3
        assert report["dim"] == 3
        assert report["label_dtype"] == "int64"
        assert report["speakers"] == 2


def test_validate_common_npz_rejects_too_few_speakers() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sample.npz"
        write_npz(path, [0, 0, 0])
        try:
            validate_common_npz(path, min_speakers=2)
        except ValueError as exc:
            assert "expected at least 2 speakers" in str(exc)
        else:
            raise AssertionError("validator accepted a single-speaker dataset")


def test_validate_common_npz_rejects_non_integer_labels() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sample.npz"
        write_raw_npz(
            path,
            np.eye(3, 3, dtype=np.float32),
            np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        )
        try:
            validate_common_npz(path, min_speakers=2)
        except ValueError as exc:
            assert "integer speaker labels" in str(exc)
        else:
            raise AssertionError("validator accepted non-integer labels")


def test_validate_common_npz_rejects_non_finite_labels() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sample.npz"
        write_raw_npz(
            path,
            np.eye(3, 3, dtype=np.float32),
            np.asarray([0.0, np.nan, 1.0], dtype=np.float32),
        )
        try:
            validate_common_npz(path, min_speakers=2)
        except ValueError as exc:
            assert "y contains NaN or Inf" in str(exc)
        else:
            raise AssertionError("validator accepted non-finite labels")


def test_validate_common_npz_rejects_empty_dataset() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sample.npz"
        write_raw_npz(
            path,
            np.empty((0, 3), dtype=np.float32),
            np.asarray([], dtype=np.int64),
        )
        try:
            validate_common_npz(path, min_speakers=1)
        except ValueError as exc:
            assert "at least one row" in str(exc)
        else:
            raise AssertionError("validator accepted an empty dataset")


def test_cli_rejects_invalid_min_speakers() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sample.npz"
        write_npz(path, [0, 1])
        try:
            with redirect_stderr(StringIO()):
                main([str(path), "--min-speakers", "0"])
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError("CLI accepted an invalid min-speakers value")


def test_resolve_report_path_rejects_input_overwrite() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sample.json"
        try:
            resolve_report_path(path, path)
        except ValueError as exc:
            assert "must not overwrite" in str(exc)
        else:
            raise AssertionError("json_out accepted an input overwrite path")


def test_resolve_report_path_rejects_non_json_suffix() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sample.npz"
        write_npz(path, [0, 1])
        try:
            resolve_report_path(Path(tmp) / "report.txt", path)
        except ValueError as exc:
            assert "must end with .json" in str(exc)
        else:
            raise AssertionError("json_out accepted a non-json suffix")


def test_resolve_report_path_rejects_directory() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sample.npz"
        write_npz(path, [0, 1])
        report_dir = Path(tmp) / "report.json"
        report_dir.mkdir()
        try:
            resolve_report_path(report_dir, path)
        except ValueError as exc:
            assert "directory" in str(exc)
        else:
            raise AssertionError("json_out accepted a directory")


def test_resolve_report_path_accepts_json_artifact() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sample.npz"
        write_npz(path, [0, 1])
        report = resolve_report_path(Path(tmp) / "results" / "sample.json", path)
        assert report.name == "sample.json"


if __name__ == "__main__":
    test_validate_common_npz_accepts_dataset()
    test_validate_common_npz_rejects_too_few_speakers()
    test_validate_common_npz_rejects_non_integer_labels()
    test_validate_common_npz_rejects_non_finite_labels()
    test_validate_common_npz_rejects_empty_dataset()
    test_cli_rejects_invalid_min_speakers()
    test_resolve_report_path_rejects_input_overwrite()
    test_resolve_report_path_rejects_non_json_suffix()
    test_resolve_report_path_rejects_directory()
    test_resolve_report_path_accepts_json_artifact()
