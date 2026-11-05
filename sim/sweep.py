"""Run an experiment file cell by cell (skips seeds already done, so it can be restarted at any time).

    python -m sim.sweep configs/exp/crossover.yaml              # all cells, all seeds
    python -m sim.sweep configs/exp/crossover.yaml --list       # show the cells
    python -m sim.sweep configs/exp/crossover.yaml --only N100 --seeds 0 1

Output: results/<experiment name>/<cell>/{traj,metrics}_seed*.json|npz and config.yaml (the resolved config).
Cells run in-process one after another; the TDMA slot and noise scale are module globals, so give each experiment
with different t_slot values its own process (this script does that automatically by re-launching per cell).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

import yaml

from sim.config import load_experiment
from sim.setup import ROOT

DEFAULT_WEIGHTS = os.path.join(ROOT, "weights", "predictor_v3.npz")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--only", default=None, help="regex on cell names")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--cell", default=None, help=argparse.SUPPRESS)   # internal: run exactly this cell
    a = ap.parse_args(argv)
    exp, cells = load_experiment(a.experiment)
    names = [n for n in cells if (a.only is None or re.search(a.only, n))]
    if a.list:
        for n in names:
            print(n, cells[n]["label"])
        print(f"{len(names)} cells x {len(a.seeds or exp['seeds'])} seeds")
        return
    seeds = a.seeds or exp["seeds"]
    datagen = bool(exp.get("datagen", False))
    if a.cell is None:      # one subprocess per cell: module globals (t_slot, noise scale) never leak across cells
        for n in names:
            cmd = [sys.executable, "-m", "sim.sweep", a.experiment, "--cell", n, "--weights", a.weights,
                   "--seeds", *map(str, seeds)]
            r = subprocess.run(cmd, cwd=ROOT)
            if r.returncode:
                print(f"[sweep] cell {n} failed (exit {r.returncode}); continuing", flush=True)
        return
    c = cells[a.cell]
    out_dir = os.path.join(ROOT, "results", exp["name"], a.cell)
    os.makedirs(out_dir, exist_ok=True)
    yaml.safe_dump({"arm": c["label"], **c["cfg"]}, open(os.path.join(out_dir, "config.yaml"), "w"), sort_keys=False)
    from sim.run import run_cell
    weights = a.weights if ("learned" in c["arm"] or c["cfg"]["cbf"].get("margin_mode") == "sigma") else None
    run_cell(c["cfg"], c["arm"], seeds, out_dir, weights=weights, datagen=datagen)


if __name__ == "__main__":
    main()
