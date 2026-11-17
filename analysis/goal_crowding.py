"""Goal-region crowding report for every cell of an experiment (reads traj_seed*.npz, writes nothing).

Per cell, mean over seeds:
  arrive p50/p90/max  first time inside r_goal [s] (median / 90th pct / last agent)
  queue@T             agents in the approach band (0 < r - r_goal <= 3 m) at T s: waiting outside the disc
  edge@T / core@T     agents in the edge band (-2 m < r - r_goal <= 0) / deeper than 2 m inside, at T s
  contact_in          pair-steps closer than D_COLLISION (0.30 m) with both agents inside the disc (goal crowding)
  contact_out         the same for all other pairs (transit / gate)
  ever                ever_collided (from metrics)
  r_final p50         median final distance to the centre / r_goal (0 = all at the centre, 1 = all at the edge)
  nn_final p50        median final nearest-neighbour distance [m] (how packed the arrived crowd is)

    python -m analysis.goal_crowding results/<experiment> [--at 12 20]
"""
import argparse
import glob
import json
import os

import numpy as np

D_COLL = 0.30


def cell_stats(path, at):
    d = np.load(path)
    m = json.load(open(path.replace("traj_seed", "metrics_seed").replace(".npz", ".json")))
    pos = np.concatenate([d["pos"], d["pos_final"][None]])
    g, rg = d["goal"], float(d["r_goal"])
    dt = float(d["t"][1] - d["t"][0]) if "t" in d.files else 0.01
    r = np.linalg.norm(pos - g, axis=-1)
    e = r - rg
    inside = r <= rg
    ever_in = inside.any(0)
    first = np.argmax(inside, axis=0)[ever_in] * dt
    c_in = c_out = 0
    for k in range(pos.shape[0]):
        D = np.linalg.norm(pos[k][:, None] - pos[k][None], axis=-1)
        iu = np.triu_indices(D.shape[0], 1)
        close = D[iu] < D_COLL
        both_in = inside[k][iu[0]] & inside[k][iu[1]]
        c_in += int((close & both_in).sum())
        c_out += int((close & ~both_in).sum())
    out = {"arr_p50": np.percentile(first, 50) if len(first) else np.nan,
           "arr_p90": np.percentile(first, 90) if len(first) else np.nan,
           "arr_max": first.max() if len(first) else np.nan,
           "contact_in": c_in, "contact_out": c_out, "ever": m["ever_collided"],
           "settled": m["completeness_settled_5s_b0.30"], "r_final": np.median(r[-1] / rg)}
    Df = np.linalg.norm(pos[-1][:, None] - pos[-1][None], axis=-1)
    np.fill_diagonal(Df, np.inf)
    out["nn_final"] = np.median(Df.min(1))
    for T in at:
        k = min(int(round(T / dt)), pos.shape[0] - 1)
        out[f"queue@{T:g}"] = int(((e[k] > 0) & (e[k] <= 3)).sum())
        out[f"edge@{T:g}"] = int(((e[k] <= 0) & (e[k] > -2)).sum())
        out[f"core@{T:g}"] = int((e[k] <= -2).sum())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp_dir")
    ap.add_argument("--at", type=float, nargs="+", default=[12.0, 20.0])
    a = ap.parse_args()
    rows = []
    for cell in sorted(glob.glob(os.path.join(a.exp_dir, "*/"))):
        stats = [cell_stats(p, a.at) for p in sorted(glob.glob(os.path.join(cell, "traj_seed*.npz")))]
        if not stats:
            continue
        mean = {k: float(np.nanmean([s[k] for s in stats])) for k in stats[0]}
        rows.append((os.path.basename(cell.rstrip("/")), len(stats), mean))
    keys = list(rows[0][2]) if rows else []
    print(f"{'cell':34s} {'n':>2s} " + " ".join(f"{k:>11s}" for k in keys))
    for name, n, mean in rows:
        print(f"{name:34s} {n:2d} " + " ".join(f"{mean[k]:11.2f}" for k in keys))


if __name__ == "__main__":
    main()
