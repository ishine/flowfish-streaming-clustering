"""Output path validation helpers for benchmark data generators."""
from __future__ import annotations

from pathlib import Path


RESEARCH_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_OUTPUT_DIRS = ("data", "results")


def path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def resolve_generated_stream_paths(
    out_path: Path,
    meta_path: Path,
    research_root: Path = RESEARCH_ROOT,
) -> tuple[Path, Path]:
    resolved_out = out_path.expanduser().resolve(strict=False)
    resolved_meta = meta_path.expanduser().resolve(strict=False)

    if out_path.exists() and out_path.is_dir():
        raise ValueError("--out must be an .npz file path, not a directory")
    if meta_path.exists() and meta_path.is_dir():
        raise ValueError("--meta must be a .json file path, not a directory")
    if resolved_out.suffix.lower() != ".npz":
        raise ValueError("--out must use a .npz suffix")
    if resolved_meta.suffix.lower() != ".json":
        raise ValueError("--meta must use a .json suffix")
    if resolved_out == resolved_meta:
        raise ValueError("--out and --meta must be different paths")

    resolved_root = research_root.expanduser().resolve(strict=False)
    allowed_dirs = [
        (resolved_root / dirname).resolve(strict=False)
        for dirname in REPOSITORY_OUTPUT_DIRS
    ]
    for label, path in (("--out", resolved_out), ("--meta", resolved_meta)):
        if path_is_relative_to(path, resolved_root) and not any(
            path_is_relative_to(path, directory) for directory in allowed_dirs
        ):
            allowed = ", ".join(f"{dirname}/" for dirname in REPOSITORY_OUTPUT_DIRS)
            raise ValueError(f"{label} inside the repository must be under {allowed}")

    return resolved_out, resolved_meta
