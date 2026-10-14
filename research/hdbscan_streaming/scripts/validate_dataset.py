#!/usr/bin/env python3
"""Validate the common X/y npz contract used by all benchmark adapters."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from output_paths import write_text_atomic


def validate_common_npz(data_path: Path, min_speakers: int) -> dict[str, object]:
    with np.load(data_path, allow_pickle=False) as payload:
        if "X" not in payload or "y" not in payload:
            raise ValueError("npz must contain X and y arrays")
        X = np.asarray(payload["X"])
        y = np.asarray(payload["y"])
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError(f"invalid shape contract: X={X.shape}, y={y.shape}")
    if X.shape[0] == 0:
        raise ValueError("dataset must contain at least one row")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or Inf")
    if not np.issubdtype(y.dtype, np.number):
        raise ValueError("y must contain numeric integer speaker labels")
    if not np.isfinite(y).all():
        raise ValueError("y contains NaN or Inf")
    if not np.equal(y, np.floor(y)).all():
        raise ValueError("y must contain integer speaker labels")

    y_labels = y.astype(np.int64, copy=False)
    speakers = int(np.unique(y_labels).size)
    if speakers < min_speakers:
        raise ValueError(f"expected at least {min_speakers} speakers, got {speakers}")
    return {
        "path": str(data_path.resolve()),
        "n": int(X.shape[0]),
        "dim": int(X.shape[1]),
        "dtype": str(X.dtype),
        "label_dtype": str(y.dtype),
        "speakers": speakers,
        "class_counts": {
            str(int(k)): int(v)
            for k, v in zip(*np.unique(y_labels, return_counts=True))
        },
        "row_norm_min": float(np.linalg.norm(X, axis=1).min()),
        "row_norm_max": float(np.linalg.norm(X, axis=1).max()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--min-speakers", type=int, default=2)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser


def resolve_report_path(path: Path, data_path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.suffix.lower() != ".json":
        raise ValueError(f"json_out must end with .json: {path}")
    if resolved.exists() and resolved.is_dir():
        raise ValueError(f"json_out is a directory, not a file: {path}")
    if resolved == data_path.expanduser().resolve():
        raise ValueError("json_out must not overwrite the input dataset")
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.min_speakers < 1:
        parser.error("min_speakers must be positive")
    try:
        json_out = (
            resolve_report_path(args.json_out, args.data)
            if args.json_out is not None
            else None
        )
    except ValueError as exc:
        parser.error(str(exc))
    result = validate_common_npz(args.data, args.min_speakers)
    result_json = json.dumps(result, indent=2, ensure_ascii=False)
    print(result_json)
    if json_out is not None:
        write_text_atomic(json_out, result_json + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
