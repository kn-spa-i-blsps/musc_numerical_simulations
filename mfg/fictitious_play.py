"""Pętla FP: solve_hjb -> push_forward -> mieszanie z tłumieniem, aż m się ustabilizuje.

LICZONE RAZ, nie per-N. To jest 'równowaga MFG'. N wchodzi dopiero w rollout/.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from env.base import Grid, Scenario
from mfg.hjb import CostParams, solve_hjb
from mfg.transport import push_forward_core


class FPDiagnostics(NamedTuple):
    iters: int
    residual_history: list[float]
    converged: bool


@jax.jit
def _scan_density_forward(m0: jnp.ndarray, policy_steps: jnp.ndarray, h: float, dt: float) -> jnp.ndarray:
    """Zdefiniowane RAZ na poziomie modułu — jax.jit reużywa skompilowany program XLA
    przez wszystkie iteracje fictitious_play (patrz analogiczny komentarz w mfg/hjb.py)."""

    def step(m: jnp.ndarray, policy_k: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        m_next = push_forward_core(m, policy_k[..., 0], policy_k[..., 1], h, dt)
        return m_next, m_next

    _, m_stack = jax.lax.scan(step, m0, policy_steps)
    return m_stack


def _rollout_density(m0: jnp.ndarray, policy: jnp.ndarray, grid: Grid) -> jnp.ndarray:
    """Niesie m0 w przód pod całym polem policy -> trajektoria (n_steps+1, nx, nz)."""
    m_stack = _scan_density_forward(m0, policy[: grid.n_steps], grid.step, grid.dt)
    return jnp.concatenate([m0[None], m_stack], axis=0)


def solve_equilibrium(
    scenario: Scenario,
    cost_params: CostParams,
    grid: Grid,
    *,
    damping: float = 0.5,
    max_iters: int = 200,
    tol: float = 1e-4,
) -> tuple[jnp.ndarray, jnp.ndarray, FPDiagnostics]:
    """HJB wstecz przeciw m^k -> transport w przód -> tłumiona mieszanka, aż m się ustabilizuje.

    Zwraca (m_star, policy_star, diagnostics). m_star/policy_star mają kształt
    (n_steps+1, nx, nz) / (n_steps+1, nx, nz, 2) — pełna trajektoria równowagi na horyzoncie.
    """
    m0 = scenario.initial_density(grid)
    m = jnp.broadcast_to(m0, (grid.n_steps + 1, grid.nx, grid.nz))

    residuals: list[float] = []
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
            break

    diagnostics = FPDiagnostics(iters=it, residual_history=residuals, converged=converged)
    return m, policy, diagnostics
