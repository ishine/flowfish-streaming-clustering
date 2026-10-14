#!/usr/bin/env python3
"""Evaluate async Adaptive-FISHDBC under several arrival rates."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from benchmark_profiles import async_pair_benchmark_args


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--intervals", type=float, nargs="+", default=[1, 5, 10, 20])
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    py = os.environ.get("PYTHON") or sys.executable
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for interval in args.intervals:
        common = async_pair_benchmark_args(
            interval_ms=interval,
            warmup=20,
            full_interval=8,
            checkpoint_every=80,
        )
        for method in ("adaptive_fishdbc_async", "fishdbc"):
            name = f"{method}_i{interval:g}"
            run_dir = args.out / name
            subprocess.run([py, str(root / "scripts/run_benchmark.py"), "--data", str(args.data),
                            "--out", str(run_dir), "--methods", method, *common], cwd=root, check=True,
                           stdout=subprocess.DEVNULL)
            item = json.loads((run_dir / method / "summary.json").read_text())
            final = item["final"]
            rows.append({"interval_ms": interval, "method": method,
                         "decision_mean_ms": item["steady_mean_step_ms"],
                         "decision_p95_ms": item["steady_p95_step_ms"],
                         "maintenance_ms_total": item.get("maintenance_ms_total", 0.0),
                         "maintenance_ms_max": item.get("maintenance_ms_max", 0.0),
                         "max_pending": item.get("max_pending", 0),
                         "max_queue_depth": item.get("max_queue_depth", 0),
                         "recompute_count": item.get("recompute_count", 0),
                         "acc": final["cluster_accuracy_noise_singletons"],
                         "ari": final["ari"], "nmi": final["nmi"],
                         "false_merge_rate": final["false_merge_rate"]})
    out = args.out / "rate_sweep.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (args.out / "rate_sweep.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps({"status": "ok", "rows": len(rows), "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
