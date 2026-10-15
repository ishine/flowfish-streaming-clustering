#!/usr/bin/env python3
"""Lightweight checks for synthetic stream output path boundaries."""
from __future__ import annotations

import tempfile
from pathlib import Path

from output_paths import resolve_generated_stream_paths
from script_test_runner import run_test_functions
from generate_stream import build_stream as build_basic_stream
from generate_drift_stream import build_stream as build_drift_stream


def test_resolve_generated_stream_paths_accepts_data_results_and_external() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data_out = root / "data" / "synthetic.npz"
        data_meta = root / "data" / "synthetic.json"
        results_out = root / "results" / "synthetic.npz"
        results_meta = root / "results" / "synthetic.json"
        external_out = root.parent / f"{root.name}_synthetic.npz"
        external_meta = root.parent / f"{root.name}_synthetic.json"

        assert resolve_generated_stream_paths(data_out, data_meta, root) == (
            data_out,
            data_meta,
        )
        assert resolve_generated_stream_paths(results_out, results_meta, root) == (
            results_out,
            results_meta,
        )
        assert resolve_generated_stream_paths(external_out, external_meta, root) == (
            external_out,
            external_meta,
        )


def test_resolve_generated_stream_paths_rejects_repo_root_output() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        try:
            resolve_generated_stream_paths(
                root / "synthetic.npz",
                root / "data" / "synthetic.json",
                root,
            )
        except ValueError as exc:
            assert "--out" in str(exc)
            assert "data/" in str(exc)
            assert "results/" in str(exc)
        else:
            raise AssertionError("expected repository-root --out to fail")


def test_resolve_generated_stream_paths_rejects_bad_suffixes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        try:
            resolve_generated_stream_paths(
                root / "data" / "synthetic.npy",
                root / "data" / "synthetic.json",
                root,
            )
        except ValueError as exc:
            assert ".npz" in str(exc)
        else:
            raise AssertionError("expected non-npz --out to fail")

        try:
            resolve_generated_stream_paths(
                root / "data" / "synthetic.npz",
                root / "data" / "synthetic.txt",
                root,
            )
        except ValueError as exc:
            assert ".json" in str(exc)
        else:
            raise AssertionError("expected non-json --meta to fail")


def test_resolve_generated_stream_paths_rejects_directories() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        out_dir = root / "data" / "synthetic.npz"
        meta_path = root / "data" / "synthetic.json"
        out_dir.mkdir(parents=True)

        try:
            resolve_generated_stream_paths(out_dir, meta_path, root)
        except ValueError as exc:
            assert "directory" in str(exc)
        else:
            raise AssertionError("expected directory --out to fail")


def test_stream_builders_reject_invalid_dimensions() -> None:
    for build, kwargs in (
        (build_basic_stream, (8, 0, 64, 0, 0.1, 0.16, 0.05)),
        (build_drift_stream, (8, 0, 64, 0, 0.2, 0.12)),
        (build_basic_stream, (8, 60, 0, 0, 0.1, 0.16, 0.05)),
        (build_drift_stream, (8, 320, True, 0, 0.2, 0.12)),
    ):
        try:
            build(*kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("expected invalid builder dimensions to fail")


def test_stream_builders_reject_non_finite_scales() -> None:
    try:
        build_basic_stream(8, 60, 64, 1, float("nan"), 0.16, 0.05)
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("expected non-finite scale to fail")


def test_drift_stream_rejects_unreachable_center_similarity() -> None:
    try:
        build_drift_stream(8, 320, 64, 1, 0.2, 0.12, max_center_similarity=-1)
    except ValueError as exc:
        assert "max_center_similarity" in str(exc)
    else:
        raise AssertionError("expected unreachable center similarity to fail")


if __name__ == "__main__":
    run_test_functions(globals())
