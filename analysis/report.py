"""Aggregate an experiment's per-seed metrics into one CSV (mean and 95 % seed-bootstrap CI per cell and metric).

    python -m analysis.report results/<experiment>            # -> results/<experiment>/summary.csv
    python -m analysis.report results/<experiment> --paired A B  # paired difference A - B per metric (same seeds)
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

METRICS = ("ever_collided", "safe_settled_5s_b0.30", "completeness_settled_5s_b0.30", "safe_settled_5s_b0.00",
           "completeness_settled_5s_b0.00", "safe_T", "completeness_T", "collision_pair_steps",
           "cbf_intervention_rate", "min_sep_overall_m")
N_BOOT = 10_000


def boot(x):
    x = np.asarray(x, float)
    b = x[np.random.default_rng(0).integers(0, len(x), (N_BOOT, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def load(cell_dir):
    return {m["seed"]: m for m in (json.load(open(p)) for p in glob.glob(os.path.join(cell_dir, "metrics_seed*.json")))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dir")
    ap.add_argument("--paired", nargs=2, metavar=("A", "B"))
    a = ap.parse_args()
    if a.paired:
        A, B = (load(os.path.join(a.exp_dir, c)) for c in a.paired)
        seeds = sorted(set(A) & set(B))
        for m in METRICS:
            if all(m in A[s] and m in B[s] for s in seeds) and seeds:
                mu, lo, hi = boot([A[s][m] - B[s][m] for s in seeds])
                print(f"{m:34s} {mu:+.3f} [{lo:+.3f}, {hi:+.3f}]  n={len(seeds)}")
        return
    rows = ["cell,metric,mean,ci_lo,ci_hi,n_seeds"]
    for d in sorted(glob.glob(os.path.join(a.exp_dir, "*/"))):
        R = load(d)
        for m in METRICS:
            v = [r[m] for r in R.values() if m in r]
            if v:
                rows.append(f"{os.path.basename(d.rstrip('/'))},{m},{','.join(f'{x:.6g}' for x in boot(v))},{len(v)}")
    out = os.path.join(a.exp_dir, "summary.csv")
    open(out, "w").write("\n".join(rows) + "\n")
    print(f"{len(rows) - 1} rows -> {out}")


if __name__ == "__main__":
    main()
