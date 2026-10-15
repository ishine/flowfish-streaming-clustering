#!/usr/bin/env python3
"""Generate a deterministic interleaved speaker-embedding stream."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from output_paths import resolve_generated_stream_paths
from stream_parameters import nonnegative_finite_float, nonnegative_int, positive_int


RESEARCH_ROOT = Path(__file__).resolve().parents[1]


def metadata_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(RESEARCH_ROOT))
    except ValueError:
        return str(path)


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x), 1e-12)


def build_stream(
    speakers: int,
    points_per_speaker: int,
    dim: int,
    seed: int,
    mode_scale: float,
    noise_scale: float,
    outlier_rate: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    positive_int('speakers', speakers)
    positive_int('points_per_speaker', points_per_speaker)
    positive_int('dim', dim)
    nonnegative_int('seed', seed)
    nonnegative_finite_float('mode_scale', mode_scale)
    nonnegative_finite_float('noise_scale', noise_scale)
    nonnegative_finite_float('outlier_rate', outlier_rate)
    if outlier_rate > 1:
        raise ValueError('outlier_rate must be at most 1')

    rng = np.random.default_rng(seed)
    centers: list[np.ndarray] = []
    while len(centers) < speakers:
        candidate = unit(rng.normal(size=dim))
        if all(float(np.dot(candidate, c)) < 0.45 for c in centers):
            centers.append(candidate)

    rows: list[np.ndarray] = []
    labels: list[int] = []
    for sid, center in enumerate(centers):
        mode = unit(rng.normal(size=dim))
        for _ in range(points_per_speaker):
            x = center + mode_scale * mode + noise_scale * rng.normal(size=dim)
            if rng.random() < outlier_rate:
                x = 0.35 * center + 0.65 * rng.normal(size=dim)
            rows.append(unit(x).astype(np.float32))
            labels.append(sid)

    X = np.asarray(rows, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    order = rng.permutation(len(y))
    X, y = X[order], y[order]
    meta = {
        "seed": seed,
        "speakers": speakers,
        "points_per_speaker": points_per_speaker,
        "dim": dim,
        "mode_scale": mode_scale,
        "noise_scale": noise_scale,
        "outlier_rate": outlier_rate,
        "order": "seeded random interleaving",
        "embedding_geometry": "unit-normalized Euclidean, cosine-related",
    }
    return X, y, meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--meta", type=Path, required=True)
    p.add_argument("--speakers", type=lambda value: positive_int('speakers', int(value)), default=8)
    p.add_argument("--points-per-speaker", type=lambda value: positive_int('points_per_speaker', int(value)), default=60)
    p.add_argument("--dim", type=lambda value: positive_int('dim', int(value)), default=64)
    p.add_argument("--seed", type=lambda value: nonnegative_int('seed', int(value)), default=20260910)
    p.add_argument("--mode-scale", type=lambda value: nonnegative_finite_float('mode_scale', float(value)), default=0.10)
    p.add_argument("--noise-scale", type=lambda value: nonnegative_finite_float('noise_scale', float(value)), default=0.16)
    p.add_argument("--outlier-rate", type=lambda value: nonnegative_finite_float('outlier_rate', float(value)), default=0.05)
    args = p.parse_args()
    try:
        out_path, meta_path = resolve_generated_stream_paths(args.out, args.meta)
    except ValueError as exc:
        p.error(str(exc))

    X, y, meta = build_stream(
        args.speakers,
        args.points_per_speaker,
        args.dim,
        args.seed,
        args.mode_scale,
        args.noise_scale,
        args.outlier_rate,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=X, y=y)
    meta = {**meta, "n": int(len(y)), "out": metadata_path(out_path)}
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
