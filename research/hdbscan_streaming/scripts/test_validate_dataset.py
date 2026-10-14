#!/usr/bin/env python3
"""Lightweight checks for the common NPZ dataset validator."""
from __future__ import annotations

import tempfile
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

import numpy as np

from validate_dataset import main, validate_common_npz


def write_npz(path: Path, labels: list[int]) -> None:
    X = np.eye(len(labels), 3, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    np.savez(path, X=X, y=y)


def test_validate_common_npz_accepts_dataset() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sample.npz"
        write_npz(path, [0, 0, 1])
        report = validate_common_npz(path, min_speakers=2)
        assert report["n"] == 3
        assert report["dim"] == 3
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


if __name__ == "__main__":
    test_validate_common_npz_accepts_dataset()
    test_validate_common_npz_rejects_too_few_speakers()
    test_cli_rejects_invalid_min_speakers()
