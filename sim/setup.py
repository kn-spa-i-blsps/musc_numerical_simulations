"""Scenario construction and the cached offline MFG solve."""
from __future__ import annotations

import hashlib
import json
import os

from control.params import horizon_for_N, horizon_for_path
from env.scenarios import LayoutA, SingleGateA
from mfg.equilibrium import MacroCost, load_macro, save_macro, solve_macro

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.environ.get("MUSC_CACHE", os.path.join(ROOT, "results", "cache"))

_SCENARIO_KW = ("goal_gap_mult", "goal_spacing", "goal_clearance", "goal_attractor_shift",
                "goal_geometry_spacing", "gate_width")


def build_scenario(cfg, N):
    """Scenario for N agents. scenario.geometry_N sizes the geometry for that N while N stays the agent count,
    so one arena is shared by every N. scenario.layout selects a LayoutA variant (absent = single centred gate)."""
    s = cfg["scenario"]
    geom_N = s.get("geometry_N")
    kw = dict(n_agents=None if geom_N is None else N, w_goal=s["w_goal"], w_hold=s["w_hold"],
              goal_softness=s["goal_softness"], obs_penalty=s["obs_penalty"], wall_thickness=s["wall_thickness"],
              edge_softness=s["edge_softness"], goal_shape=s.get("goal_shape", "conical"),
              **{k: s.get(k) for k in _SCENARIO_KW})
    n_geom = N if geom_N is None else geom_N
    if s.get("layout") is not None:
        return LayoutA(n_geom, layout=s["layout"], pillar_side=s.get("pillar_side", 2.0), **kw)
    return SingleGateA(n_geom, **kw)


def horizon_for(cfg, sc):
    """horizon: path -> from the path length to the goal; otherwise the default N-based horizon."""
    if cfg.get("horizon") == "path":
        return horizon_for_path(sc.path_length_to_goal())
    return horizon_for_N(sc.N)


def get_macro(cfg, sc, grid, cache_dir=CACHE_DIR):
    """Offline MFG equilibrium (Phi, m*), cached on disk under a hash of everything it depends on."""
    mc = cfg["macro_cost"]
    cost = MacroCost(mc["c_w"], mc["w_bar"], mc["m_bar"], mc["f_cap"])
    fp = cfg["fp"]
    tag = {"N": sc.N, "scenario": cfg["scenario"], "cost": cost.__dict__, "fp": fp, "step": cfg["grid_step"]}
    if "horizon" in cfg:
        tag["horizon"] = [cfg["horizon"], grid.horizon]
    terminal = mc.get("terminal")
    if terminal is not None:
        tag["terminal"] = terminal
    tag = json.dumps(tag, sort_keys=True)
    key = f"macro_N{sc.N}_cw{mc['c_w']:g}_{hashlib.sha1(tag.encode()).hexdigest()[:10]}"
    path = os.path.join(cache_dir, key + ".npz")
    if os.path.exists(path):
        return load_macro(path), cost, path
    os.makedirs(cache_dir, exist_ok=True)
    print(f"[macro] solving {key} ...", flush=True)
    sol = solve_macro(sc, grid, cost, damping=fp["damping"], max_iters=fp["max_iters"],
                      tol=fp["tol"], burn_in=fp["burn_in"], terminal=terminal)
    save_macro(path, sol)
    with open(path + ".json", "w") as f:
        f.write(tag)
    return sol, cost, path
