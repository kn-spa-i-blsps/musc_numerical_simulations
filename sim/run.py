"""Closed-loop rollouts of one cell (one fully resolved config + one arm) for a list of seeds.

Per seed, into <out_dir>/:
  traj_seed{s}.npz     full rollout log (true positions every step, estimates, CBF stats, ...) - the raw archive
  metrics_seed{s}.json in-run metrics + outcome metrics (ever_collided, settled, ...) + provenance
datagen=True instead saves only the true position history (pos_seed{s}.npz), used to train the predictor.

Arms: real | noise_only | fresh | cv | cvls | smith | learned | split:<mfg arm>/<cbf arm>.
"""
from __future__ import annotations

import copy
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from env.base import Grid
from env.rollout import make_rollout, summarize
from sim.metrics import outcome_metrics
from sim.setup import ROOT, build_scenario, get_macro, horizon_for


def _load_npz(path):
    return {k: jnp.asarray(v) for k, v in np.load(path).items()}


def apply_globals(cfg):
    """Settings read as module globals at trace time: matmul precision, TDMA slot, sensor-noise scale.
    Must run before anything is traced. Returns their effective values (logged with every result)."""
    import compensation.perception as pc
    import control.constants as const
    if cfg.get("matmul_precision"):
        jax.config.update("jax_default_matmul_precision", cfg["matmul_precision"])
    if cfg.get("t_slot") is not None:
        const.T_SLOT = pc.T_SLOT = float(cfg["t_slot"])
    if cfg.get("sigma_scale") is not None:
        pc.SIGMA_R, pc.SIGMA_THETA = pc.SIGMA_R * float(cfg["sigma_scale"]), pc.SIGMA_THETA * float(cfg["sigma_scale"])
    return {"matmul_precision": jax.config.jax_default_matmul_precision, "t_slot": pc.T_SLOT,
            "sigma_r": pc.SIGMA_R, "sigma_theta": pc.SIGMA_THETA}


def prepare(cfg):
    N = cfg["N"]
    sc = build_scenario(cfg, N)
    grid = Grid.build(sc.arena, cfg["grid_step"], horizon_for(cfg, sc))
    sol, cost, cache_path = get_macro(cfg, sc, grid)
    rcfg = dict(dt_exec=cfg["dt_exec"], n_steps=int(round(grid.horizon / cfg["dt_exec"])),
                nu=float(cfg.get("nu", 0.0)), cbf_on=cfg.get("cbf_on", True), c_w=cost.c_w, w_bar=cost.w_bar,
                m_bar=cost.m_bar, c_dens=cfg["control"]["c_dens"], cbf=copy.deepcopy(cfg["cbf"]),
                hold_on_arrival=cfg["control"].get("hold_on_arrival", False), log_xhat=cfg.get("log_xhat", False),
                sigma_weights=cfg.get("sigma_weights"))
    fields = {k: sol[k] for k in ("phi_grad", "m_star", "m_star_grad")}
    fields.update(dt=sol["dt"], step=sol["step"])
    return sc, sol, rcfg, fields, cache_path


def make_arm_rollout(arm, sc, rcfg, sol, weights=None):
    kw = {}
    if rcfg["cbf"].get("margin_mode") == "sigma":       # the error model's input needs the predictor whatever the arm
        kw["sigma_params"] = _load_npz(os.path.join(ROOT, rcfg["sigma_weights"]))
        kw["predictor_params"] = _load_npz(weights)
    parts = arm.split(":", 1)[1].split("/") if arm.startswith("split:") else [arm]
    if "smith" in parts:
        kw["field_params"] = dict(phi_grad_field=jnp.asarray(sol["phi_grad"]), grid_step=float(sol["step"]),
                                  field_dt=float(sol["dt"]))
    if "learned" in parts:
        kw["predictor_params"] = _load_npz(weights)
    return make_rollout(arm, sc, rcfg, **kw)


def run_cell(cfg, arm, seeds, out_dir, weights=None, datagen=False, skip_existing=True):
    g = apply_globals(cfg)
    os.makedirs(out_dir, exist_ok=True)
    marker = "pos_seed{}.npz" if datagen else "metrics_seed{}.json"
    seeds = [s for s in seeds if not (skip_existing and os.path.exists(os.path.join(out_dir, marker.format(s))))]
    if not seeds:
        return
    sc, sol, rcfg, fields, cache_path = prepare(cfg)
    p_loss = float(cfg.get("p_loss", 0.0))
    import compensation.perception as pc
    run = make_arm_rollout(arm, sc, rcfg, sol, weights)
    for seed in seeds:
        if p_loss > 0.0:   # packet-loss pattern per seed: globals are read at trace time -> re-trace
            pc.P_LOSS, pc.LOSS_SALT = p_loss, 1000 + seed
            run = make_arm_rollout(arm, sc, rcfg, sol, weights)
        t0 = time.time()
        logs = jax.device_get(run(jax.random.PRNGKey(seed), fields))
        wall = time.time() - t0
        if datagen:
            np.savez_compressed(os.path.join(out_dir, f"pos_seed{seed}.npz"),
                                pos=np.asarray(logs["pos"]), pos_final=np.asarray(logs["pos_final"]))
            print(f"[datagen {arm} seed={seed}] {wall:.0f}s", flush=True)
            continue
        metrics, series = summarize(logs, sc, cfg["dt_exec"], c_w=1.0)
        out, _ = outcome_metrics(logs["pos"], logs["pos_final"], sc.goal, sc.r_goal)
        out["outcome_check"] = (out["collision_pair_steps"] == metrics["collision_pair_steps"]
                                and abs(out["completeness_T"] - metrics["completeness_final"]) < 1e-9)
        metrics.update(out, seed=seed, arm=arm, N=cfg["N"], wall_seconds=wall,
                       macro_cache=os.path.basename(cache_path), weights=weights, p_loss=p_loss, nu=rcfg["nu"], **g)
        np.savez_compressed(os.path.join(out_dir, f"traj_seed{seed}.npz"),
                            **{k: np.asarray(v) for k, v in logs.items()},
                            t=series["t"], arrived_frac=series["arrived_frac"],
                            goal=np.asarray(sc.goal), r_goal=np.float64(sc.r_goal))
        json.dump(metrics, open(os.path.join(out_dir, f"metrics_seed{seed}.json"), "w"), indent=1, default=float)
        print(f"[{arm} N={cfg['N']} seed={seed}] {wall:.0f}s ever={out['ever_collided']:.3f} "
              f"safe_settled={out['safe_settled_5s_b0.30']:.3f} compl_T={out['completeness_T']:.3f}", flush=True)
