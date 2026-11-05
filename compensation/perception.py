"""Per-drone neighbour perception: TDMA staggered age + UWB range noise + vision bearing noise.
Five conditions: fresh (oracle), real (deployable), cv/pred (real + per-neighbour constant-
velocity forward extrapolation by that neighbour's own age), learned (a frozen,
pre-trained small MLP consuming the SAME 4-broadcast stale/noisy observation window as 'learned'
training data generation, no privileged velocity), smith (a principled
model-based predictor that forward-integrates a neighbour's last stale position under the shared
controller's FIELD-BASED routing+congestion velocity, omitting the unobservable neighbour-barrier
term). No ground truth is used for anything the real hardware wouldn't have. Named 'cv' going
forward: this is a fixed constant-velocity COMPENSATOR, not a learned
predictor -- 'pred' is kept as an accepted synonym only for backward compatibility with older
scripts (gate_and_sweep.py, smoke_test.py, sweep.py) that still pass that string. 'learned' and
'smith' are deliberately distinct condition names so no two compensators ever collide.
"""
import math

import jax
import jax.numpy as jnp

from control.constants import T_SLOT, SIGMA_R, SIGMA_THETA, t_frame, V_MAX
from control.law import sample_field_nearest

K_WINDOW_LEARNED = 4  # must match phase2b_datagen.py's K_WINDOW exactly


def tdma_age(t_sim: float, N: int) -> jnp.ndarray:
    """age[j] = time since neighbour-in-slot-j's last broadcast, at simulation time t_sim.
    Sawtooth per neighbour, offset by slot phase j*T_SLOT; mean over j = N*T_SLOT/2, max ->
    N*T_SLOT, matching the brief's (N-k)*t_slot approximation exactly at frame-aligned instants
    and correctly (continuously) at every other instant."""
    T_frame = t_frame(N)
    j = jnp.arange(N)
    age = jnp.mod(t_sim - j * T_SLOT, T_frame)
    if P_LOSS > 0.0:   # packet loss (module globals set by the runner; P_LOSS = 0 -> unchanged expression)
        f0 = jnp.floor((t_sim - j * T_SLOT) / T_frame).astype(jnp.int32)       # j's latest scheduled frame
        m = jnp.arange(LOSS_MAX_FRAMES)[:, None]
        dropped = (_loss_uniform(f0[None, :] - m, j[None, :]) < P_LOSS) & (f0[None, :] - m >= 0)
        extra = jnp.sum(jnp.cumprod(dropped.astype(jnp.int32), axis=0), axis=0)  # consecutive drops back from f0
        age = age + jnp.minimum(extra, LOSS_MAX_FRAMES - 1) * T_frame
    return age


# TDMA packet loss. Each scheduled broadcast (frame f, slot j) is lost independently with probability P_LOSS,
# decided by a deterministic integer hash of (f, j, LOSS_SALT) so every caller of tdma_age within one rollout agrees;
# age_j = time since j's last SUCCESSFUL broadcast (capped at LOSS_MAX_FRAMES - 1 extra frames). Older observations in
# cv's / the learned window keep their one-frame spacing behind that broadcast (stated simplification).
P_LOSS = 0.0
LOSS_SALT = 0
LOSS_MAX_FRAMES = 8


def _loss_uniform(f, j):
    h = (f.astype(jnp.uint32) * jnp.uint32(73856093)) ^ (j.astype(jnp.uint32) * jnp.uint32(19349663)) \
        ^ (jnp.uint32(LOSS_SALT) * jnp.uint32(83492791))
    h = h ^ (h >> 16); h = h * jnp.uint32(0x7FEB352D); h = h ^ (h >> 15); h = h * jnp.uint32(0x846CA68B); h = h ^ (h >> 16)
    return h.astype(jnp.float32) / jnp.float32(4294967296.0)


def _noisy_rel_vec(pos_from: jnp.ndarray, pos_to: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
    """TWR-style range+bearing estimate of the vector from pos_from to pos_to, both evaluated at
    the SAME past instant (the ranging-exchange time), corrupted by independent range/bearing
    sensor noise. Shapes broadcast; last axis is (x,z)."""
    rel = pos_to - pos_from
    true_range = jnp.linalg.norm(rel, axis=-1)
    true_bearing = jnp.arctan2(rel[..., 1], rel[..., 0])
    k1, k2 = jax.random.split(key)
    noisy_range = true_range + SIGMA_R * jax.random.normal(k1, true_range.shape)
    noisy_bearing = true_bearing + SIGMA_THETA * jax.random.normal(k2, true_bearing.shape)
    return jnp.stack([noisy_range * jnp.cos(noisy_bearing), noisy_range * jnp.sin(noisy_bearing)], axis=-1)


def _mlp_forward(predictor_params: dict, X: jnp.ndarray) -> jnp.ndarray:
    """predictor_params: {"W1","b1","W2","b2","W3","b3"} frozen weights
    2 hidden layers x64 ReLU, shared across neighbour pairs). X: (..., 20) -> (..., 2)."""
    h1 = jax.nn.relu(X @ predictor_params["W1"] + predictor_params["b1"])
    h2 = jax.nn.relu(h1 @ predictor_params["W2"] + predictor_params["b2"])
    return h2 @ predictor_params["W3"] + predictor_params["b3"]


def learned_features_dense(history, k_now, dt_exec, N, key, *, rel_now_ij, H_then, age, k_then,
                            T_frame, T_frame_steps, positions_now):
    """Builds the (N_i, N_j, 5*K_WINDOW_LEARNED) feature tensor for condition='learned', reusing
    the m=0 block already computed by perceive() (rel_now_ij, H_then, etc.) and extending it back
    K_WINDOW_LEARNED-1 more TDMA frames. Feature order per m-block: (range_m, bearing_m, age_m,
    ego_disp_m_x, ego_disp_m_z); blocks concatenated m=0..K_WINDOW_LEARNED-1 -- MUST match
    phase2b_datagen.py's extract_windows() feature order exactly (cross-checked, see
    phase2b_crosscheck_features.py)."""
    range_0 = jnp.linalg.norm(rel_now_ij, axis=-1)
    bearing_0 = jnp.arctan2(rel_now_ij[..., 1], rel_now_ij[..., 0])
    H_then_ij = jnp.transpose(H_then, (1, 0, 2))               # (N_i, N_j, 2) = i's pos at k_then[j]
    ego_disp_0 = positions_now[:, None, :] - H_then_ij          # (N_i, N_j, 2)
    age_ij_base = jnp.broadcast_to(age[None, :], (N, N))        # age[j], same for every i
    feat_blocks = [jnp.concatenate(
        [range_0[..., None], bearing_0[..., None], age_ij_base[..., None], ego_disp_0], axis=-1)]

    k_then_prev = k_then
    for m in range(1, K_WINDOW_LEARNED):
        k_then_m = jnp.maximum(k_then_prev - T_frame_steps, 0)   # (N_j,)
        H_then_m = history[k_then_m]                             # (N_j, N_i, 2)
        j_pos_then_m = H_then_m[jnp.arange(N), jnp.arange(N)]
        true_rel_m = j_pos_then_m[:, None, :] - H_then_m          # (N_j, N_i, 2)

        key, k_m_noise = jax.random.split(key)
        rel_m = _noisy_rel_vec(jnp.zeros_like(true_rel_m), true_rel_m, k_m_noise)
        rel_m_ij = jnp.transpose(rel_m, (1, 0, 2))                # (N_i, N_j, 2)
        range_m = jnp.linalg.norm(rel_m_ij, axis=-1)
        bearing_m = jnp.arctan2(rel_m_ij[..., 1], rel_m_ij[..., 0])

        H_then_m_ij = jnp.transpose(H_then_m, (1, 0, 2))          # (N_i, N_j, 2)
        ego_disp_m = positions_now[:, None, :] - H_then_m_ij
        age_m_ij = age_ij_base + m * T_frame

        feat_blocks.append(jnp.concatenate(
            [range_m[..., None], bearing_m[..., None], age_m_ij[..., None], ego_disp_m], axis=-1))
        k_then_prev = k_then_m

    X_full = jnp.concatenate(feat_blocks, axis=-1)  # (N_i, N_j, 5*K_WINDOW_LEARNED) = (N,N,20)
    return X_full, key


def learned_features(history: jnp.ndarray, k_now: int, dt_exec: float, N: int,
                      key: jax.Array) -> jnp.ndarray:
    """Standalone convenience wrapper (recomputes age/k_then/rel_now_ij from scratch) -- used for
    cross-checking against phase2b_datagen.py's offline feature extraction, and available for any
    caller that wants the raw 20-dim feature tensor without the MLP forward pass."""
    positions_now = history[k_now]
    T_frame = t_frame(N)
    T_frame_steps = max(1, int(round(T_frame / dt_exec)))
    t_sim = k_now * dt_exec
    age = tdma_age(t_sim, N)
    age_steps = jnp.round(age / dt_exec).astype(jnp.int32)
    k_then = jnp.maximum(k_now - age_steps, 0)
    H_then = history[k_then]
    j_pos_then = H_then[jnp.arange(N), jnp.arange(N)]
    true_rel_now = j_pos_then[:, None, :] - H_then
    key, k_now_noise = jax.random.split(key)
    rel_now = _noisy_rel_vec(jnp.zeros_like(true_rel_now), true_rel_now, k_now_noise)
    rel_now_ij = jnp.transpose(rel_now, (1, 0, 2))
    X_full, _ = learned_features_dense(history, k_now, dt_exec, N, key, rel_now_ij=rel_now_ij,
                                        H_then=H_then, age=age, k_then=k_then, T_frame=T_frame,
                                        T_frame_steps=T_frame_steps, positions_now=positions_now)
    return X_full


def _sample_timefield_nearest(field: jnp.ndarray, t_idx: jnp.ndarray, positions_flat: jnp.ndarray,
                               step: float) -> jnp.ndarray:
    """Nearest-cell sample of a TIME-INDEXED field (field: (T+1,nx,nz[,2])) at a DIFFERENT time
    index per sample (t_idx: (M,)) -- used by Smith's forward integration, where different
    neighbour-slots are integrating from different historical start times simultaneously. A single
    fused advanced-index on (time,x,z) avoids ever materializing a per-sample (nx,nz) slice."""
    nx, nz = field.shape[1], field.shape[2]
    ix = jnp.clip(jnp.round(positions_flat[:, 0] / step).astype(int), 0, nx - 1)
    iz = jnp.clip(jnp.round(positions_flat[:, 1] / step).astype(int), 0, nz - 1)
    return field[t_idx, ix, iz]  # (M,) or (M,2)


def smith_predict(positions_now: jnp.ndarray, x_hat_real: jnp.ndarray, age: jnp.ndarray, N: int,
                   t_sim: float, dt_exec: float, *, phi_grad_field, grid_step,
                   m_star=None, m_star_grad_field=None, m_star_dt=None, c_dens=None,
                   field_dt=None) -> jnp.ndarray:
    """Forward-integrates each neighbour's last stale (noisy) position under the controller's
    ROUTING FIELD ONLY, -grad Phi(x) (a static, agent-independent field, well-defined at ANY
    position). CORRECTED: the original
    version also included a congestion term that held m_hat fixed at agent i's own, ALREADY-
    COMPUTED, POSITION-INDEPENDENT scalar density estimate while re-evaluating only m*'s spatial
    gradient at the neighbour's moving position -- ill-posed (a spatial gradient evaluated against
    a density value unrelated to that position), and because that SAME frozen scalar was shared
    across ALL of i's neighbours, produced CORRELATED (not independent) errors across the whole
    neighbour cloud. This surfaced as a severe, reproducible closed-loop pathology (confirmed via
    a targeted diagnostic: the ego's own AGGREGATE local-density-estimate error under the old
    smith was far larger than under 'real', despite smith's raw PER-NEIGHBOUR position accuracy
    being fine) that survived an unrelated, separately-fixed V_MAX-clipping bug. The congestion
    (and, as before, the barrier) term is therefore dropped entirely, not patched: both require a
    local density estimate that is only well-defined at an agent's OWN sensed position, and Smith
    has no principled way to evaluate either at a neighbour's different, unobserved location.
    m_star/m_star_grad_field/m_star_dt/c_dens are accepted (unused) kwargs ONLY so existing callers'
    field_params dicts (built for the full control law) don't need to change shape.

    NEAREST-cell field sampling (a stated simplification for this auxiliary estimator; distinct
    from the bilinear requirement on the ego's OWN m*-gradient sampling in
    control.sample_m_star_and_grad, which is unaffected -- moot now anyway since m* is unused here).
    Integrates over the sim's own dt_exec for age_steps = round(age/dt_exec) steps per neighbour-
    slot (same rounding convention and residual bound as the cv branch) -- a FIXED-length
    jax.lax.scan over the worst-case step count, masking per-(i,j) pairs once their own budget
    (age_steps[j]) is exhausted, so variable per-neighbour integration length is still a single
    static-shape, JIT-traceable computation. V_MAX-clipped per step, same convention as the real
    controller (this clip was already necessary and correct for the routing-only velocity too)."""
    age_steps = jnp.round(age / dt_exec).astype(jnp.int32)               # (N_j,)
    T_frame = t_frame(N)
    max_steps = int(math.ceil(T_frame / dt_exec)) + 1  # static python int (N is static under jit)

    time_indexed = phi_grad_field.ndim == 4   # (T+1,nx,nz,2): time-dependent Phi

    def velocity_flat(x_flat, t_j):
        if not time_indexed:
            return sample_field_nearest(phi_grad_field, x_flat, grid_step)  # (M,2) -- routing only
        # routing field at each neighbour-slot's own integration instant (nearest time / cell)
        t_idx = jnp.clip(jnp.round(t_j / field_dt).astype(int), 0, phi_grad_field.shape[0] - 1)
        t_flat = jnp.broadcast_to(t_idx[None, :], (N, N)).reshape(-1)
        return _sample_timefield_nearest(phi_grad_field, t_flat, x_flat, grid_step)

    def scan_body(carry, s):
        x_ij, t_j = carry  # x_ij: (N_i,N_j,2); t_j: (N_j,) current integration time per slot j (unused
                            # now that velocity is time-independent, kept for interface symmetry/future use)
        active = (s < age_steps).astype(jnp.float32)  # (N_j,)
        x_flat = x_ij.reshape(-1, 2)
        v_flat = velocity_flat(x_flat, t_j)
        v_norm = jnp.linalg.norm(v_flat, axis=-1, keepdims=True)
        v_flat = v_flat * jnp.minimum(1.0, V_MAX / jnp.maximum(v_norm, 1e-9))  # same clip as real dynamics
        v_ij = v_flat.reshape(N, N, 2)
        active_ij = jnp.broadcast_to(active[None, :, None], (N, N, 1))
        x_ij_new = x_ij + dt_exec * v_ij * active_ij
        t_j_new = t_j + dt_exec
        return (x_ij_new, t_j_new), None

    init_carry = (x_hat_real, t_sim - age)
    (x_final, _), _ = jax.lax.scan(scan_body, init_carry, jnp.arange(max_steps))
    return x_final


def staleness_and_rel_velocity(history: jnp.ndarray, k_now: int, dt_exec: float, N: int,
                               key: jax.Array) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Adaptive CBF margin inputs: (age (N_j,), v_hat (N_i, N_j, 2)). age[j] = time since neighbour j's last TDMA
    broadcast (tdma_age, the schedule perceive() uses); v_hat[i, j] = finite difference of i's last two heard
    relative observations of j over one frame, (rel_now - rel_prev) / T_frame. Called with the SAME key as perceive():
    the key splits reproduce perceive()'s draws exactly, so rel_now is the m=0 observation every arm uses and rel_prev
    is cv's previous-frame observation = the m=1 block of the learned window. Read-only: x_hat is unaffected."""
    T_frame = t_frame(N)
    T_frame_steps = max(1, int(round(T_frame / dt_exec)))
    age = tdma_age(k_now * dt_exec, N)
    age_steps = jnp.round(age / dt_exec).astype(jnp.int32)
    k_then = jnp.maximum(k_now - age_steps, 0)
    H_then = history[k_then]
    true_rel_now = H_then[jnp.arange(N), jnp.arange(N)][:, None, :] - H_then
    key, k_now_noise = jax.random.split(key)
    rel_now = _noisy_rel_vec(jnp.zeros_like(true_rel_now), true_rel_now, k_now_noise)
    k_prev = jnp.maximum(k_then - T_frame_steps, 0)
    H_prev = history[k_prev]
    true_rel_prev = H_prev[jnp.arange(N), jnp.arange(N)][:, None, :] - H_prev
    key, k_prev_noise = jax.random.split(key)
    rel_prev = _noisy_rel_vec(jnp.zeros_like(true_rel_prev), true_rel_prev, k_prev_noise)
    v_hat = jnp.transpose(rel_now - rel_prev, (1, 0, 2)) / T_frame     # (N_i, N_j, 2)
    return age, v_hat


def learned_window_and_pred(history: jnp.ndarray, k_now: int, dt_exec: float, N: int, key: jax.Array,
                            predictor_params: dict) -> tuple[jnp.ndarray, jnp.ndarray]:
    """sigma-CBF inputs: (X_full (N_i,N_j,20), F (N_i,N_j,2)) = the learned window and the frozen predictor's
    relative-position output, recomputed with perceive()'s exact key splits (same key -> the same m=0..3 draws every
    arm received). Read-only: x_hat is unaffected. Used for the error model's input whatever arm feeds x_hat."""
    positions_now = history[k_now]
    T_frame = t_frame(N)
    T_frame_steps = max(1, int(round(T_frame / dt_exec)))
    age = tdma_age(k_now * dt_exec, N)
    k_then = jnp.maximum(k_now - jnp.round(age / dt_exec).astype(jnp.int32), 0)
    H_then = history[k_then]
    true_rel_now = H_then[jnp.arange(N), jnp.arange(N)][:, None, :] - H_then
    key, k_now_noise = jax.random.split(key)
    rel_now_ij = jnp.transpose(_noisy_rel_vec(jnp.zeros_like(true_rel_now), true_rel_now, k_now_noise), (1, 0, 2))
    X_full, _ = learned_features_dense(history, k_now, dt_exec, N, key, rel_now_ij=rel_now_ij, H_then=H_then,
                                        age=age, k_then=k_then, T_frame=T_frame, T_frame_steps=T_frame_steps,
                                        positions_now=positions_now)
    return X_full, _mlp_forward(predictor_params, X_full)


def neighbour_abs_velocity(history: jnp.ndarray, k_now: int, dt_exec: float, N: int,
                           key: jax.Array) -> jnp.ndarray:
    """Closure-gated margin inputs: v_hat (N_i, N_j, 2) = finite difference of i's last two HEARD ABSOLUTE positions
    of j over one frame: ((p_i(k_then_j) + rel_now) - (p_i(k_prev_j) + rel_prev)) / T_frame, i.e. the relative
    estimator plus i's own (known) displacement between the two broadcast instants. Same key splits as perceive() ->
    rel_now / rel_prev are the observations every arm already received. Read-only."""
    T_frame = t_frame(N)
    _, v_rel = staleness_and_rel_velocity(history, k_now, dt_exec, N, key)
    T_frame_steps = max(1, int(round(T_frame / dt_exec)))
    age = tdma_age(k_now * dt_exec, N)
    k_then = jnp.maximum(k_now - jnp.round(age / dt_exec).astype(jnp.int32), 0)
    k_prev = jnp.maximum(k_then - T_frame_steps, 0)
    own_disp = jnp.transpose(history[k_then] - history[k_prev], (1, 0, 2))   # (N_i, N_j, 2): i's own move, j's instants
    return v_rel + own_disp / T_frame


def perceive(condition: str, history: jnp.ndarray, k_now: int, dt_exec: float, N: int,
             key: jax.Array, predictor_params: dict = None, field_params: dict = None) -> jnp.ndarray:
    """Returns x_hat: (N, N, 2), x_hat[i, j] = agent i's estimate of agent j's CURRENT position.
    history: (n_steps, N, 2) true positions, all agents, maintained by the caller's scan carry.
    Diagonal (i==j) is unspecified/unused -- callers self-exclude (leave-one-out)."""
    positions_now = history[k_now]  # (N, 2), true, current

    if condition == "fresh":
        return jnp.broadcast_to(positions_now[None, :, :], (N, N, 2))

    T_frame = t_frame(N)
    T_frame_steps = max(1, int(round(T_frame / dt_exec)))
    t_sim = k_now * dt_exec
    age = tdma_age(t_sim, N)                              # (N,) -- age of neighbour-slot j
    if condition == "noise_only":                         # DIAGNOSTIC: perfect delay compensation,
        age = jnp.zeros_like(age)                         # sensor noise kept (same key use as real)
    age_steps = jnp.round(age / dt_exec).astype(jnp.int32)  # (N,)
    k_then = jnp.maximum(k_now - age_steps, 0)             # (N,) -- per-neighbour-j lookup step

    H_then = history[k_then]                               # (N_j, N_i, 2) = history[k_then[j], i]
    j_pos_then = H_then[jnp.arange(N), jnp.arange(N)]        # (N_j, 2) = history[k_then[j], j]
    true_rel_now = j_pos_then[:, None, :] - H_then           # (N_j, N_i, 2): j's pos - i's pos, both at j's broadcast time

    key, k_now_noise = jax.random.split(key)
    rel_now = _noisy_rel_vec(jnp.zeros_like(true_rel_now), true_rel_now, k_now_noise)  # (N_j,N_i,2)
    rel_now_ij = jnp.transpose(rel_now, (1, 0, 2))           # (N_i, N_j, 2)
    x_hat_real = positions_now[:, None, :] + rel_now_ij       # (N, N, 2)

    if condition in ("real", "noise_only"):
        return x_hat_real

    if condition in ("cv", "pred"):
        # cv (formerly 'pred', kept as a synonym): estimate each neighbour's velocity from TWO
        # independently-noisy historical observations (this broadcast's and the previous frame's),
        # then extrapolate by that neighbour's own age. No ground truth used beyond what 'real'
        # already uses.
        k_prev = jnp.maximum(k_then - T_frame_steps, 0)          # (N_j,)
        H_prev = history[k_prev]                                 # (N_j, N_i, 2)
        j_pos_prev = H_prev[jnp.arange(N), jnp.arange(N)]
        true_rel_prev = j_pos_prev[:, None, :] - H_prev           # (N_j, N_i, 2)

        key, k_prev_noise = jax.random.split(key)
        rel_prev = _noisy_rel_vec(jnp.zeros_like(true_rel_prev), true_rel_prev, k_prev_noise)
        rel_prev_ij = jnp.transpose(rel_prev, (1, 0, 2))          # (N_i, N_j, 2)

        vel_est_ij = (rel_now_ij - rel_prev_ij) / T_frame          # (N_i, N_j, 2)
        age_ij = jnp.broadcast_to(age[None, :, None], (N, N, 1))   # age[j], same for every i
        x_hat_pred = x_hat_real + vel_est_ij * age_ij
        return x_hat_pred

    if condition == "learned":
        # learned: fixed, pre-trained MLP consuming the SAME 4-broadcast
        # stale/noisy window as phase2b_datagen.py's training data. No privileged velocity
        # broadcast; strictly a superset of cv's 2-sample input.
        if predictor_params is None:
            raise ValueError("condition='learned' requires predictor_params")
        X_full, _ = learned_features_dense(history, k_now, dt_exec, N, key,
                                            rel_now_ij=rel_now_ij, H_then=H_then, age=age,
                                            k_then=k_then, T_frame=T_frame, T_frame_steps=T_frame_steps,
                                            positions_now=positions_now)
        rel_pred_ij = _mlp_forward(predictor_params, X_full)  # (N_i, N_j, 2)
        return positions_now[:, None, :] + rel_pred_ij

    if condition == "cvls":
        # classical constant-velocity least-squares fit through the SAME 4-broadcast window the learned arm
        # receives (same draws). Per block m: j's position relative to i's CURRENT position at j's broadcast instant
        # = rel_m - ego_disp_m, at time tau_m = -age_m; x_hat = the fitted line's intercept at tau = 0.
        X_full, _ = learned_features_dense(history, k_now, dt_exec, N, key,
                                            rel_now_ij=rel_now_ij, H_then=H_then, age=age,
                                            k_then=k_then, T_frame=T_frame, T_frame_steps=T_frame_steps,
                                            positions_now=positions_now)
        B = X_full.reshape(N, N, K_WINDOW_LEARNED, 5)
        p = jnp.stack([B[..., 0] * jnp.cos(B[..., 1]), B[..., 0] * jnp.sin(B[..., 1])], -1) - B[..., 3:5]
        tau = -B[..., 2:3]                                                       # (N,N,K,1)
        tc, pc = tau - tau.mean(2, keepdims=True), p - p.mean(2, keepdims=True)
        slope = jnp.sum(tc * pc, 2) / jnp.maximum(jnp.sum(tc * tc, 2), 1e-12)   # (N,N,2)
        return positions_now[:, None, :] + p.mean(2) - slope * tau.mean(2)

    if condition == "smith":
        # smith: principled model-based compensation. Forward-
        # integrates x_hat_real (the same stale/noisy m=0 estimate 'real' uses) under agent i's
        # own field-based velocity model. No privileged velocity broadcast, no finite-differenced
        # velocity (unlike cv) -- see smith_predict's docstring for the full derivation.
        if field_params is None:
            raise ValueError("condition='smith' requires field_params")
        return smith_predict(positions_now, x_hat_real, age, N, t_sim, dt_exec, **field_params)

    raise ValueError(f"unknown condition {condition!r}")
