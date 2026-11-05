"""Micro layer: per-agent, decentralized CBF-QP on the realized control, from local
sensing only (own position, neighbour estimates x_hat, known static walls). Single-integrator
dynamics x' = u.

    min_{u,s}  1/2 |u - u_nom|^2 + 1/2 rho |s|^2 + slack_lin * sum(s)   (L1 part: exact penalty)
    s.t.       2 (x_i - x_hat_j) . u + s_j >= -a_ij * gamma * h_ij,   h_ij = |x_i - x_hat_j|^2 - d_safe^2
               2 (x_i - q_w) . u + s_w    >= -gamma * h_w,           h_w  = |x_i - q_w|^2 - r_wall^2
               arena half-planes (linear h), |u_x|,|u_z| <= V_MAX/sqrt(2),  s >= 0

Neighbour pairs use reciprocal responsibility (each agent of a pair guarantees its share a_ij of the
barrier rate; a_ij + a_ji = 1). Liveness: goal-biased priority — the agent nearer the goal gets the
larger share a_ij (more freedom), the one behind yields. Fallback: soft slack s (never an emergency
stop) — a QP whose hard version is infeasible returns the minimum-violation control. For h=|d|^2-r^2
the Euler step only adds dt^2|du|^2 >= 0 to h, so the linear constraint is conservative in discrete
time. Solvers (prm.qp_solver): "exact_hard" (default) -- lexicographic: walls/arena rows get their own
slack (zero unless the geometry alone is infeasible), neighbour rows share one slack, so an infeasible
neighbour set never loosens a wall; "exact" (legacy, smoke attempt 3) -- ONE slack shared by all rows
(the P1 wall-penetration bug); "boxosqp" -- the per-row-slack QP above via jaxopt (reference).
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxopt import BoxOSQP

BIG = 1e6
LOOSE = 1e3   # never-binding bound (|u| <= u_box ~ 5.7); keeps the OSQP data well scaled


class CBFParams(NamedTuple):
    d_safe: float      # pair safe distance (D_COLLISION + margin)
    r_wall: float      # wall clearance (wall half-thickness + airframe radius)
    r_body: float      # arena-boundary clearance
    gamma: float       # class-K gain, 1/s
    rho: float         # slack quadratic weight
    slack_lin: float   # slack linear (exact-penalty) weight: s = 0 whenever the hard QP is feasible
    k_nb: int          # neighbours per QP (nearest by estimated distance)
    r_sense: float
    prio_gain: float   # 1/m, goal-biased priority sharpness
    prio_min: float    # share clip [prio_min, 1-prio_min]
    u_box: float       # per-axis velocity bound
    tol: float
    maxiter: int
    qp_solver: str = "exact_hard"  # "exact_hard" (default; walls/boundaries own slack) |
                                   # "exact" (legacy: one slack shared by ALL rows) | "boxosqp" (reference)
    margin_k: float | None = None  # None = constant d_safe (unchanged); else staleness-aware per-pair
                                   # d_safe_ij = min(D_COLLISION + margin_k * v_hat_ij * age_j, margin_d_max)
    margin_d_max: float = 1.5      # cap of the adaptive per-pair d_safe
    margin_mode: str = "staleness"  # "staleness" (above) | "closing": d_safe_ij =
                                    # max(margin_floor, D_COLLISION + margin_k * max(0, v_closing_ij) * margin_h), no cap
    margin_floor: float = 0.55     # floor
    margin_h: float = 0.3          # look-ahead horizon (s)
    margin_q: float = 1.0          # "sigma": conformal multiplier q_N(alpha) on the error model's exp(log|e|);
                                   # d_safe_ij = max(margin_floor, D_COLLISION + margin_k * margin_q * sigma_ij), no cap
    margin_q_edges: tuple | None = None  # sigma_bin: log-sigma bin edges (4); q = margin_q_vals[bin] per pair
    margin_q_vals: tuple | None = None   # (5); None = the scalar margin_q
    margin_cap: float | None = None      # sigma mode: cap on d_safe_ij (and sigma_c <= (cap - D_COLLISION)/k); None = no cap


def _closest_on_segments(p, segs):
    a, b = segs[:, 0], segs[:, 1]
    ab = b - a
    t = jnp.clip(jnp.sum((p[None] - a) * ab, -1) / jnp.maximum(jnp.sum(ab * ab, -1), 1e-12), 0.0, 1.0)
    return a + t[:, None] * ab                                         # (S,2)


def build_rows(p_i, x_hat_i, self_idx, goal, arena, segs, prm: CBFParams, d_safe_i=None):
    """Constraint rows (a (C,2), lower (C,)) for agent i: k_nb neighbours, S walls, 4 boundaries.
    d_safe_i (N,): per-neighbour safe distance; None = the constant prm.d_safe."""
    if x_hat_i.shape[0] < prm.k_nb:                                    # pad with never-valid rows
        pad = prm.k_nb - x_hat_i.shape[0]
        x_hat_i = jnp.concatenate([x_hat_i, jnp.full((pad, 2), BIG)], 0)
        if d_safe_i is not None:
            d_safe_i = jnp.concatenate([d_safe_i, jnp.full((pad,), prm.d_safe)], 0)
    N = x_hat_i.shape[0]
    rel = p_i[None] - x_hat_i                                          # (N,2)
    d2 = jnp.sum(rel ** 2, -1)
    d2 = jnp.where(jnp.arange(N) == self_idx, jnp.inf, d2)
    neg, idx = jax.lax.top_k(-d2, prm.k_nb)
    d2_k = -neg
    rel_k = rel[idx]
    valid = d2_k < prm.r_sense ** 2                                    # (inf / padded -> False)
    d_safe_k = prm.d_safe if d_safe_i is None else d_safe_i[idx]
    d_safe_sq = d_safe_k ** 2
    if d_safe_i is not None and prm.margin_mode == "sigma":
        # rows at the floor use the constant path's square (python float64 0.7**2 -> f32 0.49), not
        # the f32 product 0.7f*0.7f = 0.48999998 (1 ulp lower), so k=0 is bit-identical to fixed d_safe = floor.
        d_safe_sq = jnp.where(d_safe_k == jnp.float32(prm.margin_floor), jnp.float32(prm.margin_floor ** 2), d_safe_sq)
    h = d2_k - d_safe_sq
    r_i = jnp.linalg.norm(goal - p_i)
    r_j = jnp.linalg.norm(goal[None] - x_hat_i[idx], axis=-1)
    share = jax.nn.sigmoid(prm.prio_gain * (r_j - r_i))
    share = jnp.clip(share, prm.prio_min, 1.0 - prm.prio_min)
    share = jnp.where(h >= 0, share, 0.5)
    a_nb = 2.0 * rel_k
    lo_nb = jnp.where(valid, -share * prm.gamma * h, -BIG)

    q = _closest_on_segments(p_i, segs)
    rel_w = p_i[None] - q
    h_w = jnp.sum(rel_w ** 2, -1) - prm.r_wall ** 2
    a_w = 2.0 * rel_w
    lo_w = -prm.gamma * h_w

    W, H = arena
    r = prm.r_body
    a_b = jnp.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    h_b = jnp.array([p_i[0] - r, W - r - p_i[0], p_i[1] - r, H - r - p_i[1]])
    lo_b = -prm.gamma * h_b

    A = jnp.concatenate([a_nb, a_w, a_b], 0)
    lo = jnp.concatenate([lo_nb, lo_w, lo_b], 0)
    active = jnp.concatenate([valid, jnp.ones(a_w.shape[0] + 4, bool)])
    # row normalization a.u >= lo  <=>  (a/|a|).u >= lo/|a|  (same feasible set, unit-scaled data)
    na = jnp.maximum(jnp.linalg.norm(A, axis=-1), 1e-9)
    A = jnp.where(active[:, None], A / na[:, None], 0.0)
    lo = jnp.where(active, jnp.clip(lo / na, -LOOSE, LOOSE), -LOOSE)
    return A, lo, idx, valid, jnp.broadcast_to(d_safe_k, idx.shape)


def _qp_matrices(A_c, lo_c, u_nom, prm: CBFParams):
    C = A_c.shape[0]
    nv = 2 + C
    Q = jnp.diag(jnp.concatenate([jnp.ones(2), prm.rho * jnp.ones(C)]))
    c = jnp.concatenate([-u_nom, prm.slack_lin * jnp.ones(C)])
    A = jnp.concatenate([
        jnp.concatenate([A_c, jnp.eye(C)], 1),                         # CBF rows with slack
        jnp.concatenate([jnp.eye(2), jnp.zeros((2, C))], 1),           # velocity box
        jnp.concatenate([jnp.zeros((C, 2)), jnp.eye(C)], 1),           # s >= 0
    ], 0)
    lower = jnp.concatenate([lo_c, -prm.u_box * jnp.ones(2), jnp.zeros(C)])
    upper = jnp.concatenate([LOOSE * jnp.ones(C), prm.u_box * jnp.ones(2), LOOSE * jnp.ones(C)])
    return Q, c, A, lower, upper, nv


def solve_one(A_c, lo_c, u_nom, init, prm: CBFParams):
    Q, c, A, lower, upper, _ = _qp_matrices(A_c, lo_c, u_nom, prm)
    solver = BoxOSQP(tol=prm.tol, maxiter=prm.maxiter, check_primal_dual_infeasability=False)
    sol = solver.run(init_params=init, params_obj=(Q, c), params_eq=A, params_ineq=(lower, upper))
    return sol.params, sol.state.iter_num


def cbf_filter(positions, x_hat, u_nom, goal, arena, segs, init, prm: CBFParams, d_safe_pair=None):
    """positions (N,2), x_hat (N,N,2), u_nom (N,2). init: batched BoxOSQP KKTSolution or None.
    d_safe_pair (N,N): per-pair safe distance d_safe[i, j]; None = the constant prm.d_safe.
    Returns (u (N,2), info dict, kkt for warm start)."""
    N = positions.shape[0]
    goal = jnp.asarray(goal)
    segs = jnp.asarray(segs)
    u_nom_box = jnp.clip(u_nom, -prm.u_box, prm.u_box)
    if d_safe_pair is None:
        rows = jax.vmap(lambda p, xh, i: build_rows(p, xh, i, goal, arena, segs, prm))(
            positions, x_hat, jnp.arange(N))
    else:
        rows = jax.vmap(lambda p, xh, i, ds: build_rows(p, xh, i, goal, arena, segs, prm, ds))(
            positions, x_hat, jnp.arange(N), d_safe_pair)
    A_c, lo_c, nb_idx = rows[0], rows[1], rows[2]
    if prm.qp_solver == "exact_hard":
        u, s, iters = jax.vmap(lambda a, l, un: solve_exact_hard(a, l, un, prm, prm.k_nb))(
            A_c, lo_c, u_nom_box)
        kkt = init
    elif prm.qp_solver == "exact":
        u, t_star, n_feas = jax.vmap(lambda a, l, un: solve_exact(a, l, un, prm))(A_c, lo_c, u_nom_box)
        kkt = init
        s = jnp.broadcast_to(t_star[:, None], lo_c.shape)
        iters = n_feas
    else:
        kkt, iters = jax.vmap(lambda a, l, un, ini: solve_one(a, l, un, ini, prm))(A_c, lo_c, u_nom_box, init)
        z = kkt.primal[0]
        u = jnp.clip(z[:, :2], -prm.u_box, prm.u_box)
        s = jnp.maximum(z[:, 2:], 0.0)
    residual = jnp.einsum("ncd,nd->nc", A_c, u) + s - lo_c            # >= 0 when satisfied
    info = {"slack_max": jnp.max(s, axis=1),
            "slack_hard": jnp.max(s[:, prm.k_nb:], axis=1),          # walls + arena rows
            "intervention": jnp.linalg.norm(u - u_nom_box, axis=-1),
            "qp_iters": iters,
            "row_violation": jnp.maximum(-jnp.min(residual, axis=1), 0.0),
            "nb_idx": nb_idx, "nb_valid": rows[3], "nb_d_safe": rows[4]}
    return u, info, kkt


def init_warm_start(N, prm: CBFParams, n_walls: int):
    """Zero-initialized batched KKTSolution matching cbf_filter's QP shape."""
    from jaxopt.base import KKTSolution
    C = prm.k_nb + n_walls + 4
    nv, nr = 2 + C, C + 2 + C
    return KKTSolution(primal=(jnp.zeros((N, nv)), jnp.zeros((N, nr))), dual_eq=jnp.zeros((N, nr)),
                       dual_ineq=(jnp.zeros((N, nr)), jnp.zeros((N, nr))))


# ---------------------------------------------------------------------------------------------
# Exact 2-D solver (default). BoxOSQP did not converge on the smoke QPs (47% of agent-steps hit
# maxiter=2000; still ~1e-2 constraint residual at 20000 iterations), so the QP is solved exactly
# by enumeration, which is cheap because u is 2-D:
#   phase 1: t* = min_{u in box} max_k (lo_k - a_k.u)_+   (uniform slack; 0 iff the hard QP is
#            feasible) -- candidates: equal-violation triples, pairs on a box face, box vertices.
#   phase 2: u = argmin |u - u_nom| s.t. a_k.u >= lo_k - t*, u in box -- exact projection onto a
#            convex polygon: candidates u_nom, projections onto every line, pairwise vertices.
# The fallback is therefore soft-slack with the violation (uniform over rows) minimized; never a stop.
# ---------------------------------------------------------------------------------------------
import itertools

import numpy as _np

DET_EPS = 1e-6


def _combos(n, r):
    return jnp.asarray(_np.array(list(itertools.combinations(range(n), r)), dtype=_np.int32))


def _finite(c):
    """Sanitize enumerated candidates: non-finite / huge -> sentinel far outside the box (never
    feasible, never NaN-producing: a NaN candidate would otherwise win jnp.argmin)."""
    ok = jnp.all(jnp.isfinite(c) & (jnp.abs(c) < LOOSE), axis=-1, keepdims=True)
    return jnp.where(ok, c, LOOSE)


def solve_exact(A_c, lo_c, u_nom, prm: CBFParams):
    C = A_c.shape[0]
    active = jnp.any(A_c != 0.0, axis=-1)
    ub = prm.u_box
    box_a = jnp.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    box_b = -ub * jnp.ones(4)

    def viol(u):  # (M,2) -> (M,)
        v = lo_c[None] - u @ A_c.T
        return jnp.max(jnp.where(active[None], v, -jnp.inf), axis=1, initial=0.0)

    # ---- phase 1 ----
    tri = _combos(C, 3)
    M3 = jnp.concatenate([A_c[tri], jnp.ones(tri.shape + (1,))], -1)         # (T,3,3)
    ok3 = (jnp.abs(jnp.linalg.det(M3)) > DET_EPS) & jnp.all(active[tri], -1)
    sol3 = jnp.linalg.solve(jnp.where(ok3[:, None, None], M3, jnp.eye(3)), lo_c[tri][..., None])[..., 0]
    pr = _combos(C, 2)
    cands = [_finite(jnp.where(ok3[:, None], sol3[:, :2], 0.0))]
    for f in range(4):
        M2 = jnp.concatenate([jnp.concatenate([A_c[pr], jnp.ones(pr.shape + (1,))], -1),
                              jnp.broadcast_to(jnp.concatenate([box_a[f], jnp.zeros(1)])[None, None],
                                               (pr.shape[0], 1, 3))], 1)    # (P,3,3)
        rhs = jnp.concatenate([lo_c[pr], jnp.full((pr.shape[0], 1), box_b[f])], 1)
        okf = (jnp.abs(jnp.linalg.det(M2)) > DET_EPS) & jnp.all(active[pr], -1)
        s = jnp.linalg.solve(jnp.where(okf[:, None, None], M2, jnp.eye(3)), rhs[..., None])[..., 0]
        cands.append(_finite(jnp.where(okf[:, None], s[:, :2], 0.0)))
    cands.append(jnp.array([[ub, ub], [ub, -ub], [-ub, ub], [-ub, -ub]]))
    cands.append(jnp.clip(u_nom, -ub, ub)[None])
    U1 = jnp.clip(jnp.concatenate(cands, 0), -ub, ub)
    t_star = jnp.min(viol(U1))

    # ---- phase 2 ----
    lo_r = lo_c - t_star - 1e-6 * (t_star > 0)
    La = jnp.concatenate([A_c, box_a], 0)
    Lb = jnp.concatenate([jnp.where(active, lo_r, -LOOSE), box_b], 0)
    Lact = jnp.concatenate([active, jnp.ones(4, bool)])
    proj = u_nom[None] + (Lb - La @ u_nom)[:, None] * La                    # |a| = 1 rows
    pp = _combos(C + 4, 2)
    M = La[pp]                                                              # (P,2,2)
    okv = (jnp.abs(jnp.linalg.det(M)) > DET_EPS) & jnp.all(Lact[pp], -1)
    vert = jnp.linalg.solve(jnp.where(okv[:, None, None], M, jnp.eye(2)), Lb[pp][..., None])[..., 0]
    U2 = _finite(jnp.concatenate([u_nom[None], jnp.where(Lact[:, None], proj, LOOSE),
                                  jnp.where(okv[:, None], vert, LOOSE)], 0))
    tol = 1e-5
    feas = (jnp.all((U2 @ La.T - Lb[None] >= -tol) | ~Lact[None], axis=1))
    dist = jnp.where(feas, jnp.sum((U2 - u_nom[None]) ** 2, -1), jnp.inf)
    best = jnp.argmin(dist)
    u = jnp.where(jnp.isfinite(dist[best]), U2[best], U1[jnp.argmin(viol(U1))])
    bad = ~jnp.all(jnp.isfinite(u))                       # last line of defence; logged
    u = jnp.where(bad, jnp.clip(u_nom, -ub, ub), u)
    return u, t_star, jnp.where(bad, -1, jnp.sum(feas))


# ---------------------------------------------------------------------------------------------
# "exact_hard" (P1 fix): static-geometry rows (walls + arena boundaries) get their OWN slack, never
# shared with the neighbour rows. Lexicographic, exact by vertex enumeration in (u, t):
#   phase 0: t_h = min_{u in box} max_hard (lo - a.u)_+           (0 unless geometry alone infeasible)
#   phase 1: t_s = min max_nb (lo - a.u)_+ s.t. u in box, hard rows relaxed by t_h only
#   phase 2: u = argmin |u - u_nom| s.t. nb rows relaxed by t_s, hard rows by t_h, u in box
# An infeasible neighbour set therefore can never loosen a wall (smoke: shared t* reached 994).
# LP vertices: 3 active lines with a t-column on the minimized rows (triples) or t = 0 active
# (2-D vertices of two lines, pairs); box faces are lines of the must-hold set.
# ---------------------------------------------------------------------------------------------
_BOX_A = jnp.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])


def _min_max_violation(La, Lb, Lact, minimize, must, u_nom, ub):
    """min_u max_{minimize rows} (Lb - La.u)_+ s.t. must rows hold. Returns (t*, u at t*)."""
    L = La.shape[0]
    tri = _combos(L, 3)
    tcol = minimize.astype(La.dtype)
    M3 = jnp.concatenate([La[tri], tcol[tri][..., None]], -1)
    ok3 = (jnp.abs(jnp.linalg.det(M3)) > DET_EPS) & jnp.all(Lact[tri], -1)
    s3 = jnp.linalg.solve(jnp.where(ok3[:, None, None], M3, jnp.eye(3)), Lb[tri][..., None])[..., 0]
    pr = _combos(L, 2)
    M2 = La[pr]
    ok2 = (jnp.abs(jnp.linalg.det(M2)) > DET_EPS) & jnp.all(Lact[pr], -1)
    s2 = jnp.linalg.solve(jnp.where(ok2[:, None, None], M2, jnp.eye(2)), Lb[pr][..., None])[..., 0]
    U = jnp.concatenate([_finite(jnp.where(ok3[:, None], s3[:, :2], LOOSE)),
                         _finite(jnp.where(ok2[:, None], s2, LOOSE)),
                         jnp.array([[ub, ub], [ub, -ub], [-ub, ub], [-ub, -ub]]),
                         jnp.clip(u_nom, -ub, ub)[None]], 0)
    res = U @ La.T - Lb[None]                                           # >= 0 when satisfied
    t = jnp.max(jnp.where((minimize & Lact)[None], -res, 0.0), axis=1, initial=0.0)
    ok = jnp.all((res >= -1e-5) | ~(must & Lact)[None], axis=1)
    t = jnp.where(ok, t, jnp.inf)
    b = jnp.argmin(t)
    return t[b], U[b]


def solve_exact_hard(A_c, lo_c, u_nom, prm: CBFParams, n_soft: int):
    """Rows [0, n_soft) are neighbour rows (soft, shared slack); the rest are hard static rows.
    Returns (u, per-row slack (C,), n_feasible_candidates or -1 on non-finite fallback)."""
    C = A_c.shape[0]
    ub = prm.u_box
    soft = jnp.arange(C) < n_soft
    active = jnp.any(A_c != 0.0, axis=-1)
    La = jnp.concatenate([A_c, _BOX_A], 0)
    Lact = jnp.concatenate([active, jnp.ones(4, bool)])
    is_box = jnp.arange(C + 4) >= C
    hard = jnp.concatenate([~soft, jnp.zeros(4, bool)])
    nb = jnp.concatenate([soft, jnp.zeros(4, bool)])
    Lb0 = jnp.concatenate([lo_c, -ub * jnp.ones(4)])

    t_h, _ = _min_max_violation(La, Lb0, Lact, hard, is_box, u_nom, ub)            # phase 0
    t_h = jnp.where(jnp.isfinite(t_h), t_h, LOOSE)
    Lb1 = Lb0 - jnp.where(hard, t_h + 1e-6 * (t_h > 0), 0.0)
    t_s, u1 = _min_max_violation(La, Lb1, Lact, nb, is_box | hard, u_nom, ub)       # phase 1
    t_s = jnp.where(jnp.isfinite(t_s), t_s, LOOSE)

    Lb = Lb1 - jnp.where(nb, t_s + 1e-6 * (t_s > 0), 0.0)                           # phase 2
    Lb = jnp.where(Lact, Lb, -LOOSE)
    proj = u_nom[None] + (Lb - La @ u_nom)[:, None] * La
    pp = _combos(C + 4, 2)
    M = La[pp]
    okv = (jnp.abs(jnp.linalg.det(M)) > DET_EPS) & jnp.all(Lact[pp], -1)
    vert = jnp.linalg.solve(jnp.where(okv[:, None, None], M, jnp.eye(2)), Lb[pp][..., None])[..., 0]
    U2 = _finite(jnp.concatenate([u_nom[None], jnp.where(Lact[:, None], proj, LOOSE),
                                  jnp.where(okv[:, None], vert, LOOSE)], 0))
    feas = jnp.all((U2 @ La.T - Lb[None] >= -1e-5) | ~Lact[None], axis=1)
    dist = jnp.where(feas, jnp.sum((U2 - u_nom[None]) ** 2, -1), jnp.inf)
    best = jnp.argmin(dist)
    u = jnp.where(jnp.isfinite(dist[best]), U2[best], u1)
    bad = ~jnp.all(jnp.isfinite(u))
    u = jnp.where(bad, jnp.clip(u_nom, -ub, ub), u)
    s = jnp.where(soft, t_s, t_h)
    return u, s, jnp.where(bad, -1, jnp.sum(feas))
