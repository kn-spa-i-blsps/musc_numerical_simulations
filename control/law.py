"""Physical control law:
  u_i = -grad Phi(x_i)                               [routing, undelayed]
        - c_dens * grad (m_hat_i - m*(x_i,t))^2        [congestion: deviation from MFG target]
        + BoundedBarrierForce(x_i; x_hat neighbours)    [collision avoidance, saturating < R_BARRIER,
                                                          already oriented away from neighbours]

Control/avoidance uses ESTIMATED neighbour positions x_hat (from perception.py); the density
target m* and routing field Phi come from the EXISTING, unedited solver (equilibrium.py). All
metrics are computed elsewhere (rollout.py) from TRUE positions only.

The avoidance term is a BOUNDED, saturating force, constructed directly (not autodiffed from a
1/d-type potential): per-neighbour-pair magnitude is capped at F_BARRIER_MAX, an ABSOLUTE constant
(FROZEN, PART U) -- NOT a multiple of V_MAX (that coupling was a confound caught by external
review: raising V_MAX for de-saturation was silently also raising avoidance strength). Never
explodes, regardless of how close a perceived obstacle is. Justification for bounding at all: a
real actuator has bounded authority regardless of how close a perceived obstacle is; a repulsive
potential that can demand 1e5 m/s near-field is not a model of anything physical.
"""
import jax
import jax.numpy as jnp

from control.constants import H_BANDWIDTH, R_SENSE, R_BARRIER, D_COLLISION, C_DENS, F_BARRIER_MAX


def local_density_and_grad(positions: jnp.ndarray, x_hat: jnp.ndarray, N: int,
                            h: float = H_BANDWIDTH, R: float = R_SENSE):
    """positions: (N,2) TRUE positions (probe points -- each agent at its own location).
    x_hat: (N,N,2), x_hat[i,j] = i's estimate of j. Leave-one-out. Returns (m_hat (N,), units
    agents/m^2; grad_m_hat (N,2), the gradient of m_hat_i w.r.t. x_i)."""
    diff = positions[:, None, :] - x_hat          # (N,N,2): x_i - x_hat_j
    d2 = jnp.sum(diff ** 2, axis=-1)               # (N,N)
    within_R = (d2 < R * R).astype(jnp.float32)
    self_mask = 1.0 - jnp.eye(N)
    mask = within_R * self_mask                    # (N,N)
    kernel = (1.0 / (2 * jnp.pi * h ** 2)) * jnp.exp(-d2 / (2 * h ** 2))  # (N,N), area-normalized
    m_hat = jnp.sum(kernel * mask, axis=1)          # (N,)
    grad_kernel = kernel[..., None] * (-diff / h ** 2)  # (N,N,2): d/dx_i of each term
    grad_m_hat = jnp.sum(grad_kernel * mask[..., None], axis=1)  # (N,2)
    return m_hat, grad_m_hat


def sample_field_nearest(field: jnp.ndarray, positions: jnp.ndarray, step: float) -> jnp.ndarray:
    nx, nz = field.shape[0], field.shape[1]
    ix = jnp.clip(jnp.round(positions[:, 0] / step).astype(int), 0, nx - 1)
    iz = jnp.clip(jnp.round(positions[:, 1] / step).astype(int), 0, nz - 1)
    return field[ix, iz]


def sample_field_bilinear(field: jnp.ndarray, positions: jnp.ndarray, step: float) -> jnp.ndarray:
    """Bilinear interpolation. USED ONLY for m_star_grad in sample_m_star_and_grad below (per
    PART G: scope restricted to the m* gradient only -- phi_grad and m_star_val stay nearest-cell,
    untouched). field: (nx,nz) or (nx,nz,2)."""
    nx, nz = field.shape[0], field.shape[1]
    fx = jnp.clip(positions[:, 0] / step, 0.0, nx - 1.0 - 1e-6)
    fz = jnp.clip(positions[:, 1] / step, 0.0, nz - 1.0 - 1e-6)
    ix0 = jnp.floor(fx).astype(int)
    iz0 = jnp.floor(fz).astype(int)
    ix1 = jnp.minimum(ix0 + 1, nx - 1)
    iz1 = jnp.minimum(iz0 + 1, nz - 1)
    tx = (fx - ix0)[..., None] if field.ndim == 3 else (fx - ix0)
    tz = (fz - iz0)[..., None] if field.ndim == 3 else (fz - iz0)
    f00, f10 = field[ix0, iz0], field[ix1, iz0]
    f01, f11 = field[ix0, iz1], field[ix1, iz1]
    f0 = f00 * (1 - tx) + f10 * tx
    f1 = f01 * (1 - tx) + f11 * tx
    return f0 * (1 - tz) + f1 * tz


def sample_m_star_and_grad(positions: jnp.ndarray, t_sim: float, m_star: jnp.ndarray,
                            m_star_grad_field: jnp.ndarray, m_star_dt: float, grid_step: float):
    """Time-indexed sample of the precomputed m* trajectory and its spatial gradient at each
    agent's current position. m_star/m_star_grad_field: (n_steps+1, nx, nz[, 2]). Value: nearest-
    cell (unchanged). Gradient: BILINEAR (PART G fix -- nearest-cell on a gradient field is
    piecewise-constant with jumps at every cell boundary; scope restricted to this one field)."""
    n_steps = m_star.shape[0] - 1
    t_idx = jnp.clip(jnp.round(t_sim / m_star_dt).astype(int), 0, n_steps)
    m_star_t = m_star[t_idx]
    m_star_grad_t = m_star_grad_field[t_idx]
    m_star_val = sample_field_nearest(m_star_t, positions, grid_step)
    m_star_grad_val = sample_field_bilinear(m_star_grad_t, positions, grid_step)
    return m_star_val, m_star_grad_val


def bounded_barrier_force_all(positions: jnp.ndarray, x_hat: jnp.ndarray, N: int,
                               f_max: float = F_BARRIER_MAX) -> jnp.ndarray:
    """Returns (N,2): the bounded avoidance force on every agent, directly constructed (not
    autodiffed from a singular potential). Per-pair magnitude: 0 at d=R_BARRIER, ramps up as d
    shrinks, SATURATES (tanh) at F_BARRIER_MAX as d -> 0 -- never explodes, regardless of how
    close (or how many) perceived neighbours are. f_max is an ABSOLUTE constant (FROZEN, PART U),
    with NO dependence on V_MAX anywhere in this function. Direction: away from the neighbour,
    evaluated at the TRUE agent position x_i against its ESTIMATED neighbour positions x_hat."""
    diff = positions[:, None, :] - x_hat          # (N,N,2): x_i - x_hat_j, away from neighbour
    d2 = jnp.sum(diff ** 2, axis=-1)
    d = jnp.sqrt(d2 + 1e-9)
    d_safe = jnp.maximum(d, 0.05)
    raw_strength = jnp.where(d < R_BARRIER, (1.0 / d_safe - 1.0 / R_BARRIER), 0.0)  # >=0
    bounded_strength = f_max * jnp.tanh(raw_strength / f_max)                        # in [0, f_max)
    self_mask = 1.0 - jnp.eye(N)
    direction = diff / d_safe[..., None]           # (N,N,2) unit vectors
    force_per_pair = bounded_strength[..., None] * direction * self_mask[..., None]
    return jnp.sum(force_per_pair, axis=1)          # (N,2), summed over neighbours


def control_law(positions: jnp.ndarray, x_hat: jnp.ndarray, t_sim: float, N: int,
                 phi_grad_field: jnp.ndarray, m_star: jnp.ndarray, m_star_grad_field: jnp.ndarray,
                 m_star_dt: float, grid_step: float, c_dens: float = C_DENS,
                 f_max: float = F_BARRIER_MAX):
    """Full control law. Returns (u: (N,2), diagnostics dict with the three force components'
    norms, for calibration/auditing). V_MAX does NOT appear anywhere in this function or in
    bounded_barrier_force_all (FROZEN, PART U) -- it is applied only downstream, once, as the
    final actuator clip on the returned u (by the caller, not here)."""
    phi_grad = sample_field_nearest(phi_grad_field, positions, grid_step)  # == -grad Phi(x_i)

    m_hat, grad_m_hat = local_density_and_grad(positions, x_hat, N)
    m_star_val, m_star_grad_val = sample_m_star_and_grad(
        positions, t_sim, m_star, m_star_grad_field, m_star_dt, grid_step)
    dens_dev = m_hat - m_star_val
    grad_dens_term = 2.0 * dens_dev[:, None] * (grad_m_hat - m_star_grad_val)  # (N,2)

    barrier_force = bounded_barrier_force_all(positions, x_hat, N, f_max=f_max)

    u = phi_grad - c_dens * grad_dens_term + barrier_force

    diag = {
        "mag_phi": jnp.linalg.norm(phi_grad, axis=-1),
        "mag_dens": c_dens * jnp.linalg.norm(grad_dens_term, axis=-1),
        "mag_col": jnp.linalg.norm(barrier_force, axis=-1),
        "m_hat": m_hat, "m_star": m_star_val,
    }
    return u, diag
