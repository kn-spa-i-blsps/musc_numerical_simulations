"""Offline macro solve: running cost
    f(t,x,m) = V_goal(x) + V_obs(x) + W(m) + w_bar * B(m),   W(m) = c*m^2,  B(m) = relu(m - m_bar)^2
(both monotone in m). m* = Cesaro mean of FP iterates (ported mfg.cesaro, unmodified); the routing
field Phi is the value of ONE final HJB solve against m* (so Phi and m* are mutually consistent —
the ported cesaro returns the last iterate's policy, not the policy against the averaged m*).

Phi is the isolated "global oracle router" module: the controller only consumes
`phi_grad` (= -grad Phi) and m*, so a learned field can replace this module in Project B.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from env.base import Grid, grid_gradient
from mfg.cesaro import solve_equilibrium_cesaro
from mfg.hjb import CostParams, solve_hjb


@dataclass(frozen=True)
class MacroCost:
    c_w: float       # congestion W(m) = c_w * m^2 ; c_w=0 is the "W off" control
    w_bar: float     # soft collision barrier weight
    m_bar: float     # barrier onset density, agents/m^2
    f_cap: float     # solver safeguard: F <- min(F, f_cap). Non-decreasing, so monotonicity is kept;
                     # bounds |grad V| (explicit-scheme CFL) on transient FP iterates. Must stay
                     # inactive at the equilibrium (checked: max F(m*) reported in diagnostics).

    def total(self, m):
        m = jnp.maximum(m, 0.0)
        f = self.c_w * m ** 2 + self.w_bar * jnp.maximum(m - self.m_bar, 0.0) ** 2
        return jnp.minimum(f, self.f_cap)


def solve_macro(scenario, grid: Grid, cost: MacroCost, *, damping: float, max_iters: int,
                tol: float, burn_in: int, terminal: str | None = None) -> dict:
    v_goal, v_obs = scenario.potentials(grid)
    v_term = None
    if terminal == "goal":                   # terminal cost g = goal potential (at T if time-dependent)
        v_term = v_goal[-1] if v_goal.ndim == 3 else v_goal
    elif terminal is not None:
        raise ValueError(f"unknown terminal cost {terminal!r}")
    cp = CostParams(V_goal=v_goal, V_obs=v_obs, W_fn=cost.total, V_terminal=v_term)
    t0 = time.time()
    m_star, _, diag = solve_equilibrium_cesaro(scenario, cp, grid, damping=damping,
                                               max_iters=max_iters, tol=tol, burn_in=burn_in)
    value, policy = solve_hjb(m_star, cp, grid)
    m_star_grad = jax.vmap(lambda f: grid_gradient(f, grid.step))(m_star)
    return {
        "phi0": value[0], "phi_grad": policy, "m_star": m_star, "m_star_grad": m_star_grad,
        "dt": grid.dt, "step": grid.step,
        "residual_history": np.asarray(diag.residual_history),
        "cesaro_residual_history": np.asarray(diag.cesaro_residual_history),
        "converged_last_iterate": bool(diag.converged),
        "cesaro_converged": bool(diag.cesaro_converged),
        "iters": int(diag.iters), "solve_seconds": time.time() - t0,
        "max_F_at_mstar": float(jnp.max(cost.total(m_star))),
        "nan": bool(jnp.any(jnp.isnan(m_star)) or jnp.any(jnp.isnan(policy))),
    }


def save_macro(path: str, sol: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **{k: np.asarray(v) for k, v in sol.items()})


def load_macro(path: str) -> dict:
    d = np.load(path)
    out = {k: d[k] for k in d.files}
    for k in ("phi0", "phi_grad", "m_star", "m_star_grad"):
        out[k] = jnp.asarray(out[k])
    for k in ("dt", "step", "solve_seconds", "max_F_at_mstar"):
        out[k] = float(out[k])
    for k in ("converged_last_iterate", "cesaro_converged", "nan"):
        out[k] = bool(out[k])
    out["iters"] = int(out["iters"])
    return out
