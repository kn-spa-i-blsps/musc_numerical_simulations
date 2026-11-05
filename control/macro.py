"""Macro layer: nominal control from the offline MFG solution plus a local density
term driven by the (possibly delay-compensated) neighbour estimate x_hat.

    u_nom_i = -grad Phi(t, x_i)                                              [routing, oracle field]
              - C_DENS * grad_{x_i} [ c_w * relu(m_hat_i - m*(t,x_i))^2      [congestion in EXCESS of
                                      + w_bar * relu(m_hat_i - m_bar)^2 ]     the equilibrium; soft barrier]

Density is emergent: there is no attraction towards m* (the excess term is one-sided, zero whenever
the realized local density is at or below the equilibrium's). The local goal observation
(range, bearing) is computed from the agent's own position, independently of Phi;
the CBF uses it for goal-biased priority.

Hold-on-arrival (optional): an agent inside r_goal whose nearest ESTIMATED neighbour is at
least d_hold away gets u_nom = 0; otherwise u_nom is unchanged (the density term can still separate).
"""
import jax.numpy as jnp

from control.law import local_density_and_grad, sample_field_bilinear


def sample_time_field(field, positions, t_sim, dt, step, bilinear=True):
    """field: (T+1, nx, nz[, 2]); nearest time index, bilinear in space."""
    t_idx = jnp.clip(jnp.round(t_sim / dt).astype(int), 0, field.shape[0] - 1)
    f_t = field[t_idx]
    return sample_field_bilinear(f_t, positions, step)


def goal_observation(positions, goal):
    rel = jnp.asarray(goal)[None, :] - positions
    rng = jnp.linalg.norm(rel, axis=-1)
    bearing = jnp.arctan2(rel[:, 1], rel[:, 0])
    return rng, bearing


def nominal_control(positions, x_hat, t_sim, N, macro, *, c_w, w_bar, m_bar, c_dens, hold=None):
    """macro: dict with phi_grad (T+1,nx,nz,2), m_star (T+1,nx,nz), m_star_grad (T+1,nx,nz,2),
    dt, step. hold: None or (goal (2,), r_goal, d_hold) for hold-on-arrival. Returns (u_nom (N,2), diag)."""
    dt, step = macro["dt"], macro["step"]
    route = sample_time_field(macro["phi_grad"], positions, t_sim, dt, step)
    m_hat, grad_m_hat = local_density_and_grad(positions, x_hat, N)
    m_star = sample_time_field(macro["m_star"], positions, t_sim, dt, step)
    grad_m_star = sample_time_field(macro["m_star_grad"], positions, t_sim, dt, step)
    excess = jnp.maximum(m_hat - m_star, 0.0)
    over = jnp.maximum(m_hat - m_bar, 0.0)
    grad_local = (2.0 * c_w * excess[:, None] * (grad_m_hat - grad_m_star)
                  + 2.0 * w_bar * over[:, None] * grad_m_hat)
    u = route - c_dens * grad_local
    held = jnp.zeros(positions.shape[0], bool)
    if hold is not None:
        goal, r_goal, d_hold = hold
        d_hat = jnp.linalg.norm(positions[:, None] - x_hat, axis=-1)
        d_hat = jnp.where(jnp.eye(N, dtype=bool), jnp.inf, d_hat)
        held = (jnp.linalg.norm(positions - jnp.asarray(goal)[None], axis=-1) <= r_goal) & \
               (jnp.min(d_hat, axis=1) >= d_hold)
        u = jnp.where(held[:, None], 0.0, u)
    return u, {"m_hat": m_hat, "m_star": m_star, "route_mag": jnp.linalg.norm(route, axis=-1),
               "local_mag": c_dens * jnp.linalg.norm(grad_local, axis=-1), "held": held}
