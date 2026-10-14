#!/usr/bin/env python3
"""Run paired multi-seed validation for the proposed async method and FISHDBC."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from benchmark_profiles import async_pair_benchmark_args
from driver_validation import validate_nonnegative_ints, validate_positive_floats
from output_paths import resolve_experiment_output_dir


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[20260910, 20260911, 20260912])
    p.add_argument("--interval-ms", type=float, default=5.0)
    args = p.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        out_dir = resolve_experiment_output_dir(args.out, root)
        seeds = validate_nonnegative_ints(args.seeds, "--seeds")
        interval_ms = validate_positive_floats([args.interval_ms], "--interval-ms")[0]
    except ValueError as exc:
        p.error(str(exc))
    py = os.environ.get("PYTHON") or sys.executable
    sys.path.insert(0, str(root / "scripts"))
    from generate_stream import build_stream

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        data = out_dir / f"data_seed_{seed}.npz"
        X, y, meta = build_stream(8, 40, 64, seed, 0.10, 0.16, 0.05)
        np.savez_compressed(data, X=X, y=y)
        for method in ("adaptive_fishdbc_async", "fishdbc"):
            run = out_dir / f"seed_{seed}" / method
            common = async_pair_benchmark_args(
                interval_ms=interval_ms,
                warmup=20,
                full_interval=8,
                checkpoint_every=80,
            )
            subprocess.run([py, str(root / "scripts/run_benchmark.py"), "--data", str(data),
                            "--out", str(run), "--methods", method, *common], cwd=root, check=True,
                           stdout=subprocess.DEVNULL)
            summary = json.loads((run / method / "summary.json").read_text())
            final = summary["final"]
            rows.append({"seed": seed, "method": method, "interval_ms": interval_ms,
                         "decision_mean_ms": summary["steady_mean_step_ms"],
                         "decision_p95_ms": summary["steady_p95_step_ms"],
                         "maintenance_ms_total": summary.get("maintenance_ms_total", 0.0),
                         "max_queue_depth": summary.get("max_queue_depth", 0),
                         "recompute_count": summary.get("recompute_count", 0),
                         "acc": final["cluster_accuracy_noise_singletons"],
                         "ari": final["ari"], "nmi": final["nmi"],
                         "false_merge_rate": final["false_merge_rate"]})
    out_csv = out_dir / "multiseed.csv"
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {}
    for method in ("adaptive_fishdbc_async", "fishdbc"):
        subset = [r for r in rows if r["method"] == method]
        summary[method] = {key: {"mean": float(np.mean([r[key] for r in subset])),
                                 "std": float(np.std([r[key] for r in subset]))}
                          for key in ("decision_mean_ms", "decision_p95_ms", "acc", "ari", "nmi", "false_merge_rate", "max_queue_depth")}
    (out_dir / "multiseed_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": "ok", "rows": len(rows), "csv": str(out_csv)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
