"""Fixed-point check of a cached MFG solve: is m* (the Cesaro mean) actually an equilibrium?

The FP log's `res` is damping * ||Phi[m] - m|| / ||m|| for the running iterate m; the Cesaro residual only
shows that the average stopped moving (it shrinks like 1/count regardless). The test that matters is
whether the crowd reproduces m* when every agent best-responds to m*:

    m_br = rollout(m0, policy*),   policy* = HJB best response to m*   (the field the agents use: phi_grad)
    rel_L2 = ||m_br - m*|| / ||m*||                (same norm as `res`; compare with res / damping)
    TV(t)  = 0.5 * sum_x |m_br(t) - m*(t)| h^2 / N (fraction of agents "in the wrong place" at time t, in [0, 1])

Reads only the cache npz (+ the cell's resolved config.yaml for the grid); writes nothing. Never solves:
if the cache file is missing it stops. Memory: m*, policy* and m_br on the device (~10 GiB on the
campaign grids) -> run when the GPU is free, or with JAX_PLATFORMS=cpu (slower).

    python -m analysis.fixed_point_check results/<exp>/<cell>/config.yaml <cache>/macro_N..._<hash>.npz
"""
import argparse
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from env.base import Grid
from mfg.fictitious_play import _rollout_density
from sim.setup import build_scenario, horizon_for


@jax.jit
def _compare(m_br, m_star, cell_area):
    """Global relative L2 and per-time total-variation fraction, without keeping a full difference array."""
    diff = m_br - m_star
    rel_l2 = jnp.linalg.norm(diff) / (jnp.linalg.norm(m_star) + 1e-12)
    mass = jnp.sum(m_star, axis=(1, 2)) * cell_area                 # agents at each time (should stay ~N)
    tv = 0.5 * jnp.sum(jnp.abs(diff), axis=(1, 2)) * cell_area / mass
    mass_br = jnp.sum(m_br, axis=(1, 2)) * cell_area
    return rel_l2, tv, mass, mass_br


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="resolved config.yaml of a cell that used this solve")
    ap.add_argument("cache", help="macro_N*_*.npz from the cache directory")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    sc = build_scenario(cfg, cfg["N"])
    grid = Grid.build(sc.arena, cfg["grid_step"], horizon_for(cfg, sc))

    tag_path = a.cache + ".json"                                    # sanity: the cache belongs to this config
    if os.path.exists(tag_path):
        tag = json.load(open(tag_path))
        if tag["N"] != sc.N or tag["step"] != cfg["grid_step"] or tag["scenario"] != cfg["scenario"]:
            raise SystemExit(f"cache {a.cache} was solved for a different N/grid/scenario than {a.config}")

    d = np.load(a.cache)                                            # lazy: only the arrays we index are read
    m_star = jnp.asarray(d["m_star"])
    policy = jnp.asarray(d["phi_grad"])
    if m_star.shape != (grid.n_steps + 1, grid.nx, grid.nz):
        raise SystemExit(f"grid mismatch: cache {m_star.shape} vs config {(grid.n_steps + 1, grid.nx, grid.nz)}")

    m0 = sc.initial_density(grid)
    nu = float(cfg["fp"].get("nu", 0.0))                           # same dynamics as the solve
    m_br = _rollout_density(m0, policy, grid, nu)
    del policy
    rel_l2, tv, mass, mass_br = jax.device_get(_compare(m_br, m_star, grid.step ** 2))

    fp = cfg["fp"]
    res = np.asarray(d["residual_history"])
    cres = np.asarray(d["cesaro_residual_history"])
    tail = res[-100:]
    t = np.arange(len(tv)) * grid.dt
    print(f"cache        {os.path.basename(a.cache)}   N={sc.N}  grid {grid.nx}x{grid.nz}x{grid.n_steps + 1}"
          f"  horizon {grid.horizon:.1f}s  solve {float(d['solve_seconds']):.0f}s")
    print(f"solver       iters={int(d['iters'])}  converged_last_iterate={bool(d['converged_last_iterate'])}"
          f"  cesaro_converged={bool(d['cesaro_converged'])}  nan={bool(d['nan'])}"
          f"  max_F(m*)={float(d['max_F_at_mstar']):.1f} (f_cap {cfg['macro_cost']['f_cap']})")
    print(f"res tail     last 100 iters: mean {tail.mean():.3e}  min {tail.min():.3e}  max {tail.max():.3e}"
          f"  -> ||Phi[m]-m||/||m|| ~ {tail.mean() / fp['damping']:.2f} for the running iterate")
    if len(cres):
        print(f"cesaro_res   last {cres[-1]:.3e}  (x count={len(cres) + 1}: {cres[-1] * (len(cres) + 1):.2f};"
              f" ~const => only the 1/count shrinkage)")
    print(f"mass         m*: {mass.min():.2f}..{mass.max():.2f}   m_br: {mass_br.min():.2f}..{mass_br.max():.2f}"
          f"   (N={sc.N})")
    print(f"FIXED POINT  rel_L2 ||Phi[m*]-m*||/||m*|| = {float(rel_l2):.4f}")
    q = [0.1, 0.25, 0.5, 0.75, 1.0]
    idx = [min(len(tv) - 1, int(round(f * (len(tv) - 1)))) for f in q]
    print("             TV fraction of agents misplaced: max {:.3f} at t={:.1f}s, mean {:.3f};  ".format(
        tv.max(), t[tv.argmax()], tv.mean())
        + "  ".join(f"t={t[i]:.0f}s:{tv[i]:.3f}" for i in idx))


if __name__ == "__main__":
    main()
