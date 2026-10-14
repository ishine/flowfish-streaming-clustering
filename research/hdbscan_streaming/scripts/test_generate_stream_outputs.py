#!/usr/bin/env python3
"""Lightweight checks for synthetic stream output path boundaries."""
from __future__ import annotations

import tempfile
from pathlib import Path

from output_paths import resolve_generated_stream_paths


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


if __name__ == "__main__":
    for name, test_func in sorted(globals().items()):
        if name.startswith("test_") and callable(test_func):
            test_func()
