"""Initial positions are a quantization of the MFG plan m*(·, 0), not a uniform fill of the start box.

The nominal law is the offline plan plus a small online correction:

    u = -grad Phi(t, x) - C_DENS * grad penalty(m_hat - m*)

The correction is the difference between the realized density and the original plan. It stays
small when that difference averages out, and it is relatively larger at small N and on the
boundary of the support. A uniform lattice puts too much mass on that boundary (where the
smoothed m0 tapers), so the correction is systematic at t = 0 and the swarm spreads at once.
An i.i.d. draw from m0 has the right margins but Poisson clumps, and those clumps make the
same correction fire locally.

Centroidal Voronoi quantization of m*(·, 0) matches the plan and keeps agents spaced. Pairs
closer than D_COLLISION — only a safety net, the quantizer already sits well above it for the
densities used here — are projected apart. The floor is the collision diameter only: the CBF
safe distance is an experiment knob and is not always packable inside the start support, so
enforcing it here would push agents back off the plan. The CBF still sees neighbour estimates
and does the separation filtering.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from control.params import D_COLLISION

D_MIN_INIT = D_COLLISION


def positions_matching_density(key, density, arena, n: int, d_min: float = D_MIN_INIT,
                               lloyd_iters: int = 25, separate_passes: int = 4) -> jnp.ndarray:
    """Centroidal Voronoi quantization of `density` into `n` points, then a hard-core projection.

    Seeds are N distinct nodes drawn without replacement with probability proportional to the
    density (Gumbel-top-k). Lloyd iteration moves each seed to the centre of mass of its cell,
    which removes Poisson clumps while preserving the density. `d_min` is only a floor.

    density: (nx, nz) on the nodal grid linspace(0, W, nx) x linspace(0, H, nz), the same
    nodes Grid.build uses. Total mass need not equal n; only the shape of the density matters.
    `n` is a Python int (static). Returns (n, 2).
    """
    density = jnp.asarray(density)
    nx, nz = density.shape
    if n > nx * nz:
        raise ValueError(f"cannot place {n} agents on a {nx}x{nz} density grid")
    w_box, h_box = float(arena[0]), float(arena[1])
    xs = jnp.linspace(0.0, w_box, nx)
    zs = jnp.linspace(0.0, h_box, nz)
    x, z = jnp.meshgrid(xs, zs, indexing="ij")
    sites = jnp.stack([x, z], axis=-1).reshape(-1, 2)
    w = jnp.maximum(density.reshape(-1), 0.0)
    w = w / jnp.maximum(jnp.sum(w), 1e-12)

    key, gk, jk = jax.random.split(key, 3)
    u = jax.random.uniform(gk, w.shape, minval=1e-6, maxval=1.0 - 1e-6)
    gumbel = -jnp.log(-jnp.log(u))
    idx = jax.lax.top_k(jnp.log(w + 1e-20) + gumbel, n)[1]
    pos = sites[idx]

    dx = w_box / max(nx - 1, 1)
    dz = h_box / max(nz - 1, 1)
    jitter = jax.random.uniform(jk, (n, 2), minval=-0.5, maxval=0.5) * jnp.array([dx, dz])
    pos = _clip(pos + jitter, w_box, h_box)
    if n >= 2:
        pos = _lloyd(pos, sites, w.astype(pos.dtype), lloyd_iters)
        if d_min > 0.0:
            pos = _separate(pos, float(d_min), separate_passes, w_box, h_box)
    elif n == 1:
        pos = jnp.sum(sites * w[:, None], axis=0, keepdims=True)
    return pos


def _lloyd(pos, sites, w, iters):
    """Discrete Lloyd: each point moves to the mass centre of the nodes closest to it."""
    n = pos.shape[0]

    def step(pos, _):
        d2 = jnp.sum((sites[:, None, :] - pos[None, :, :]) ** 2, axis=-1)
        owner = jnp.argmin(d2, axis=1)
        sum_w = jnp.zeros(n, dtype=pos.dtype).at[owner].add(w)
        sum_x = jnp.zeros((n, 2), dtype=pos.dtype).at[owner].add(w[:, None] * sites)
        centre = sum_x / jnp.maximum(sum_w[:, None], jnp.asarray(1e-12, dtype=pos.dtype))
        alive = sum_w[:, None] > jnp.asarray(1e-8, dtype=pos.dtype)
        return jnp.where(alive, centre, pos), None

    return jax.lax.scan(step, pos, None, length=iters)[0]


def _clip(pos, w_box, h_box):
    return jnp.clip(pos, jnp.zeros(2), jnp.array([w_box, h_box]))


def _separate(pos, d_min, passes, w_box, h_box):
    """Pairwise projection onto distance >= d_min, staying inside the arena.

    An unconstrained pair keeps its midpoint. If the wall blocks one agent, the partner
    takes up the rest of the gap. A pair the box cannot fit is left at the best in-box
    separation this pass can reach.
    """
    n = pos.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    pairs = jnp.asarray(np.stack([iu, ju], axis=1))
    dt = pos.dtype
    d_min = jnp.asarray(d_min, dtype=dt)
    lo = jnp.zeros(2, dtype=dt)
    hi = jnp.asarray([w_box, h_box], dtype=dt)
    zero = jnp.asarray(0.0, dtype=dt)
    eps = jnp.asarray(1e-12, dtype=dt)
    small = jnp.asarray(1e-6, dtype=dt)

    def one_pass(pos, _):
        def body(pos, ij):
            i, j = ij
            a, b = pos[i], pos[j]
            dvec = a - b
            dist = jnp.sqrt(jnp.sum(dvec * dvec) + eps)
            direction = dvec / jnp.maximum(dist, small)
            need = dist < d_min
            shift = 0.5 * jnp.maximum(d_min - dist, zero) * direction
            a1, b1 = a + shift, b - shift
            inside = jnp.all((a1 >= lo) & (a1 <= hi) & (b1 >= lo) & (b1 <= hi))
            a_pin, b_pin = jnp.clip(a1, lo, hi), jnp.clip(b1, lo, hi)
            # Partner absorbs the gap the wall took: a - b stays parallel to direction.
            b_take = jnp.clip(a_pin - d_min * direction, lo, hi)
            a_take = jnp.clip(b_pin + d_min * direction, lo, hi)
            sep_b = jnp.linalg.norm(a_pin - b_take)
            sep_a = jnp.linalg.norm(a_take - b_pin)
            sep_c = jnp.linalg.norm(a_pin - b_pin)
            use_b = (~inside) & (sep_b >= sep_a) & (sep_b >= sep_c)
            use_a = (~inside) & (~use_b) & (sep_a > sep_c)
            a_f = jnp.where(inside, a1, a_pin)
            b_f = jnp.where(inside, b1, b_pin)
            a_f = jnp.where(use_b, a_pin, jnp.where(use_a, a_take, a_f))
            b_f = jnp.where(use_b, b_take, jnp.where(use_a, b_pin, b_f))
            pos = pos.at[i].set(jnp.where(need, a_f, a))
            pos = pos.at[j].set(jnp.where(need, b_f, b))
            return pos, None

        return jax.lax.scan(body, pos, pairs)[0], None

    return jax.lax.scan(one_pass, pos, None, length=passes)[0]
