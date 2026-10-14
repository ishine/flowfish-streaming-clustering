#!/usr/bin/env python3
"""Lightweight checks for audio manifest path boundaries."""
from __future__ import annotations

import tempfile
from pathlib import Path

from audio_to_embeddings import resolve_manifest_wav_path
from script_test_runner import run_test_functions


def test_resolve_manifest_wav_path_accepts_relative_paths() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        path = resolve_manifest_wav_path(root, "speaker/file.wav", 0)
        assert path == root / "speaker" / "file.wav"


def test_resolve_manifest_wav_path_rejects_escape_paths() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        invalid_paths = ("../outside.wav", "/tmp/outside.wav", "speaker\\file.wav", "")
        for index, row_path in enumerate(invalid_paths):
            try:
                resolve_manifest_wav_path(root, row_path, index)
            except ValueError as exc:
                assert "raw_root" in str(exc)
            else:
                raise AssertionError(f"accepted invalid manifest path: {row_path!r}")


if __name__ == "__main__":
    run_test_functions(globals())
