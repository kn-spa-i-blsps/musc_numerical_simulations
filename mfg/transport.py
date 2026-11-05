"""FPK/continuity: niesie gęstość m w przód pod policy. Jeden krok (finite-volume upwind)."""
from __future__ import annotations

import jax.numpy as jnp

from env.base import Grid


def push_forward_core(m: jnp.ndarray, ux: jnp.ndarray, uz: jnp.ndarray, h: float, dt: float) -> jnp.ndarray:
    """Czysta funkcja (bez obiektu Grid) — żeby dało się ją scanować/jitować (np. w
    mfg/fictitious_play.py) bez przeciągania Grid przez granicę jax.jit."""
    ux_face = 0.5 * (ux + jnp.roll(ux, -1, axis=0))
    m_face_x = jnp.where(ux_face >= 0, m, jnp.roll(m, -1, axis=0))
    flux_x = (ux_face * m_face_x).at[-1, :].set(0.0)
    flux_x_left = jnp.roll(flux_x, 1, axis=0).at[0, :].set(0.0)

    uz_face = 0.5 * (uz + jnp.roll(uz, -1, axis=1))
    m_face_z = jnp.where(uz_face >= 0, m, jnp.roll(m, -1, axis=1))
    flux_z = (uz_face * m_face_z).at[:, -1].set(0.0)
    flux_z_left = jnp.roll(flux_z, 1, axis=1).at[:, 0].set(0.0)

    divergence = (flux_x - flux_x_left) / h + (flux_z - flux_z_left) / h
    m_new = m - dt * divergence
    return jnp.maximum(m_new, 0.0)


def push_forward(m: jnp.ndarray, policy: jnp.ndarray, grid: Grid, dt: float) -> jnp.ndarray:
    """dm/dt + div(m*u) = 0, jeden krok Eulera w przód, donor-cell upwind na ścianach komórek.

    m: (nx, nz) gęstość w chwili t. policy: (nx, nz, 2) pole u*(t,x) w tej samej chwili.
    No-flux na brzegu areny (bez odpływu masy poza domenę).
    """
    ux, uz = policy[..., 0], policy[..., 1]
    return push_forward_core(m, ux, uz, grid.step, dt)
