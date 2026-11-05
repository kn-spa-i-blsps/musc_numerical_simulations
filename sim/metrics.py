"""Per-agent outcome metrics from the TRUE trajectory.

ever_collided : fraction of agents whose true distance to any other agent was ever < D_COLLISION (whole run).
settled       : agent inside r_goal + band for >= 90 % of the last W seconds.
safe_settled  : settled and never collided.
completeness_T / safe_T : the same at the final instant only (kept for reference; it over-counts last-second arrivals).
"""
from __future__ import annotations

import numpy as np

from control.params import D_COLLISION

DT = 0.01
SETTLE_FRAC = 0.90
WINDOWS = (3.0, 5.0, 10.0)
BANDS = (0.0, 0.30)
HEADLINE = "safe_settled_5s_b0.30"


def collisions(pos_all):
    """pos_all (T+1, N, 2) -> (pair_steps over the first T steps, collided (N,) bool)."""
    T1, N, _ = pos_all.shape
    iu, ju = np.triu_indices(N, 1)
    collided = np.zeros(N, bool)
    pair_steps = 0
    for t0 in range(0, T1, 1000):
        c = np.linalg.norm(pos_all[t0:t0 + 1000, iu] - pos_all[t0:t0 + 1000, ju], axis=-1) < D_COLLISION
        pair_steps += int(c[: max(0, min(1000, T1 - 1 - t0))].sum())
        hit = c.any(0)
        collided[iu[hit]] = True
        collided[ju[hit]] = True
    return pair_steps, collided


def outcome_metrics(pos, pos_final, goal, r_goal):
    pos_all = np.concatenate([np.asarray(pos, np.float64), np.asarray(pos_final, np.float64)[None]])
    pair_steps, coll = collisions(pos_all)
    dist = np.linalg.norm(pos_all - np.asarray(goal), axis=-1)
    arr_T = dist[-1] <= r_goal
    out = {"ever_collided": float(coll.mean()), "collision_pair_steps": pair_steps,
           "completeness_T": float(arr_T.mean()), "safe_T": float((arr_T & ~coll).mean())}
    for b in BANDS:
        inside = dist <= r_goal + b
        for W in WINDOWS:
            st = inside[-(int(round(W / DT)) + 1):].mean(0) >= SETTLE_FRAC
            out[f"completeness_settled_{W:g}s_b{b:.2f}"] = float(st.mean())
            out[f"safe_settled_{W:g}s_b{b:.2f}"] = float((st & ~coll).mean())
    return out, coll
