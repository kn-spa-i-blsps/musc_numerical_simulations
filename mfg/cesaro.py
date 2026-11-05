"""PART OO: Cesaro (uniform time-average) fictitious-play solve, implemented in CALLING code only.
Cardaliaguet & Hadikhanloo (2017): fictitious play converges in the time-average for monotone MFG
even when the raw iterate cycles. Reuses the UNEDITED mfg.hjb.solve_hjb and
mfg.fictitious_play._rollout_density -- the same primitives mfg.fictitious_play.solve_equilibrium
itself calls -- rather than mfg.fictitious_play.solve_equilibrium's orchestration wrapper (which
only returns the last iterate, with no hook for accumulating an average). mfg/hjb.py and
mfg/fictitious_play.py are imported, never modified. No extra iterations beyond the existing
budget: averaging happens over the SAME max_iters sweep, from burn_in onward. Commit nothing.
"""
from typing import NamedTuple

import jax.numpy as jnp

from mfg.hjb import solve_hjb
from mfg.fictitious_play import _rollout_density


class CesaroDiagnostics(NamedTuple):
    iters: int
    converged: bool                      # last-iterate convergence (same definition as before)
    residual_history: list               # last-iterate residual per iteration (unchanged metric)
    cesaro_residual_history: list        # the AVERAGE's own stability, from burn_in+1 onward
    cesaro_converged: bool                # last cesaro_residual_history entry < tol


def solve_equilibrium_cesaro(scenario, cost_params, grid, *, damping, max_iters, tol, burn_in):
    """Same loop as mfg.fictitious_play.solve_equilibrium (damped mixing: m_next = (1-damping)*m +
    damping*m_new_raw), but ALSO accumulates the Cesaro mean of m over [burn_in, max_iters) and
    returns that average as m_star instead of the last iterate. If tol is reached on the last
    iterate before max_iters, the loop still continues to max_iters so the average has its full
    window (the FP cost here is already paid; stopping early would truncate the average)."""
    m0 = scenario.initial_density(grid)
    m = jnp.broadcast_to(m0, (grid.n_steps + 1, grid.nx, grid.nz))

    residuals = []
    cesaro_residuals = []
    m_sum = None
    count = 0
    cesaro_mean = None
    policy = None
    converged = False
    it = 0

    for it in range(1, max_iters + 1):
        _, policy = solve_hjb(m, cost_params, grid)
        m_new_raw = _rollout_density(m0, policy, grid)
        m_next = (1 - damping) * m + damping * m_new_raw
        residual = float(jnp.linalg.norm(m_next - m) / (jnp.linalg.norm(m) + 1e-12))
        residuals.append(residual)
        m = m_next
        if residual < tol:
            converged = True

        if it > burn_in:
            if m_sum is None:
                m_sum = jnp.zeros_like(m)
            m_sum = m_sum + m
            count += 1
            new_cesaro_mean = m_sum / count
            if cesaro_mean is not None:
                cesaro_res = float(jnp.linalg.norm(new_cesaro_mean - cesaro_mean) /
                                    (jnp.linalg.norm(cesaro_mean) + 1e-12))
                cesaro_residuals.append(cesaro_res)
            cesaro_mean = new_cesaro_mean

    cesaro_converged = bool(len(cesaro_residuals) > 0 and cesaro_residuals[-1] < tol)
    diagnostics = CesaroDiagnostics(
        iters=it, converged=converged, residual_history=residuals,
        cesaro_residual_history=cesaro_residuals, cesaro_converged=cesaro_converged,
    )
    return cesaro_mean, policy, diagnostics
