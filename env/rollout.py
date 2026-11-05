"""Closed-loop rollout: perception (ported, any ladder arm) -> macro nominal control ->
CBF-QP filter -> single-integrator step. The SAME scenario object (hence the same arena) that fed
the offline solve feeds the rollout. Control uses estimated neighbours x_hat; every metric uses TRUE
positions only. FULL-horizon logging (no settled-only window).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from compensation.perception import (learned_window_and_pred, neighbour_abs_velocity, perceive,
                                     staleness_and_rel_velocity)
from control.cbf import CBFParams, cbf_filter, init_warm_start
from control.law import local_density_and_grad
from control.macro import nominal_control
from control.params import D_COLLISION, V_MAX, t_frame


def make_rollout(condition, scenario, cfg, *, predictor_params=None, field_params=None, sigma_params=None):
    """Returns run(key, macro) -> logs; macro = {phi_grad, m_star, m_star_grad} (passed to jit as
    arguments) + python floats dt, step. cfg keys: dt_exec, n_steps, nu, cbf_on, c_w, w_bar, m_bar, c_dens, cbf (dict of CBFParams);
    optional: hold_on_arrival (bool; d_hold = cbf d_safe), log_xhat (bool: log x_hat - own
    position per ordered pair as float32 (float16 in early runs), the CBF neighbour indices and the hold flag)."""
    N = scenario.N
    W, H = scenario.arena
    arena_arr = jnp.array([W, H])
    goal = jnp.asarray(scenario.goal)
    segs = jnp.asarray(scenario.wall_segments())
    prm = CBFParams(**cfg["cbf"])
    dt_exec, n_steps, nu = cfg["dt_exec"], cfg["n_steps"], cfg["nu"]
    cbf_on = cfg["cbf_on"]
    hold = (goal, scenario.r_goal, prm.d_safe) if cfg.get("hold_on_arrival", False) else None
    log_xhat = cfg.get("log_xhat", False)
    adaptive = prm.margin_k is not None                                # staleness-aware margin
    closing = adaptive and prm.margin_mode == "closing"                # closure-gated margin
    sig_mode = adaptive and prm.margin_mode == "sigma"                 # conformal sigma-CBF margin
    if sig_mode and (sigma_params is None or predictor_params is None):
        raise ValueError("margin_mode='sigma' requires sigma_params and predictor_params")
    iu = jnp.triu_indices(N, 1)
    # layer split: condition "split:<mfg>/<cbf>" feeds the MFG nominal law (density term, hold) with one
    # perception arm and the CBF (+ adaptive margin, x_hat logs) with another; both from the same key -> same m=0
    # observation. Any other condition string = unchanged single-x_hat path.
    split = condition.split(":", 1)[1].split("/") if condition.startswith("split:") else None

    def step_fn(carry, k, macro):
        positions, history, key, warm = carry
        history = history.at[k].set(positions)
        key, pk = jax.random.split(key)
        if split is None:
            x_hat = x_hat_mfg = perceive(condition, history, k, dt_exec, N, pk,
                                         predictor_params=predictor_params, field_params=field_params)
        else:
            x_hat_mfg, x_hat = (perceive(c, history, k, dt_exec, N, pk, predictor_params=predictor_params,
                                         field_params=field_params) for c in split)
        t_sim = k * dt_exec
        u_nom, mdiag = nominal_control(positions, x_hat_mfg, t_sim, N, macro, c_w=cfg["c_w"],
                                       w_bar=cfg["w_bar"], m_bar=cfg["m_bar"], c_dens=cfg["c_dens"],
                                       hold=hold)
        nrm = jnp.linalg.norm(u_nom, axis=-1, keepdims=True)
        u_nom = u_nom * jnp.minimum(1.0, V_MAX / jnp.maximum(nrm, 1e-9))
        d_safe_pair = v_closing = sigma_c = None
        if sig_mode:   # sigma_ij = q_N(alpha) * exp(g(Z_ij)), g = frozen error model of v3 (train/sigma.py)
            hb, pkb = jax.lax.optimization_barrier((history, pk))
            Xw, Fw = learned_window_and_pred(hb, k, dt_exec, N, pkb, predictor_params)
            slog = sigma_logerr(sigma_params, Xw, Fw, N)
            q = (prm.margin_q if prm.margin_q_edges is None else   # sigma_bin: Mondrian (N, sigma-quintile) q
                 jnp.asarray(prm.margin_q_vals)[jnp.searchsorted(jnp.asarray(prm.margin_q_edges), slog, side="right")])
            sigma_c = q * jnp.exp(slog)
            if prm.margin_cap is not None:   # cap sigma_c and d_safe_ij at d_max; None = no cap
                sigma_c = jnp.minimum(sigma_c, (prm.margin_cap - D_COLLISION) / prm.margin_k)
            d_safe_pair = jnp.maximum(prm.margin_floor, D_COLLISION + prm.margin_k * sigma_c)
            if prm.margin_cap is not None:
                d_safe_pair = jnp.minimum(d_safe_pair, prm.margin_cap)
        elif adaptive and closing:   # d_safe_ij = max(floor, D_COLLISION + k * max(0, v_closing_ij) * h)
            hb, pkb = jax.lax.optimization_barrier((history, pk))
            v_hat_j = neighbour_abs_velocity(hb, k, dt_exec, N, pkb)            # (N_i,N_j,2) heard abs velocity of j
            v_i = jnp.where(k > 0, history[k] - history[jnp.maximum(k - 1, 0)], 0.0) / dt_exec   # own velocity (N,2)
            e = x_hat - positions[:, None]                                       # direction i -> j (the CBF's x_hat)
            e = e / jnp.maximum(jnp.linalg.norm(e, axis=-1, keepdims=True), 1e-9)
            v_closing = jnp.maximum(jnp.sum((v_i[:, None] - v_hat_j) * e, -1), 0.0)
            d_safe_pair = jnp.maximum(prm.margin_floor, D_COLLISION + prm.margin_k * v_closing * prm.margin_h)
        elif adaptive:   # d_safe_ij = min(D_COLLISION + k * |v_hat_ij| * age_j, d_max); same key -> same draws as perceive
            hb, pkb = jax.lax.optimization_barrier((history, pk))   # identity; keeps XLA from fusing into perceive()
            age, v_hat = staleness_and_rel_velocity(hb, k, dt_exec, N, pkb)
            sigma = jnp.linalg.norm(v_hat, axis=-1) * age[None, :]
            d_safe_pair = jnp.minimum(D_COLLISION + prm.margin_k * sigma, prm.margin_d_max)
        if cbf_on:
            u, info, warm = cbf_filter(positions, x_hat, u_nom, goal, (W, H), segs, warm, prm, d_safe_pair)
        else:
            u = jnp.clip(u_nom, -prm.u_box, prm.u_box)
            z = jnp.zeros(N)
            info = {"slack_max": z, "slack_hard": z, "intervention": z, "qp_iters": jnp.zeros(N, jnp.int32),
                    "row_violation": z, "nb_idx": jnp.zeros((N, prm.k_nb), jnp.int32)}
        key, nk = jax.random.split(key)
        noise = jax.random.normal(nk, positions.shape) * jnp.sqrt(2.0 * nu * dt_exec)
        positions_new = jnp.clip(positions + dt_exec * u + noise, jnp.zeros(2), arena_arr)

        dfull = jnp.linalg.norm(positions[:, None] - positions[None], axis=-1)   # true
        d = dfull[iu] if N > 1 else jnp.full((1,), jnp.inf)                     # unique pairs
        nn = jnp.min(jnp.where(jnp.eye(N, dtype=bool), jnp.inf, dfull), axis=1)  # per-agent NN dist
        rho_true, _ = local_density_and_grad(positions, jnp.broadcast_to(positions[None], (N, N, 2)), N)
        # estimation error of x_hat (TRUE positions), safety-relevant pairs (true d < 1 m) and all pairs
        e2 = jnp.sum((x_hat - positions[None]) ** 2, -1)
        offd = ~jnp.eye(N, dtype=bool)
        near = offd & (dfull < 1.0)
        xhat_err_near = jnp.sum(jnp.where(near, e2, 0.0)) / jnp.maximum(jnp.sum(near), 1)
        xhat_err_all = jnp.sum(jnp.where(offd, e2, 0.0)) / (N * (N - 1))
        y = {"pos": positions, "u": u, "u_nom": u_nom,
             "n_coll_pairs": jnp.sum(d < D_COLLISION), "min_sep": jnp.min(d),
             "p5_sep_pairs": jnp.percentile(nn, 5.0),
             "slack_max": info["slack_max"], "slack_hard": info["slack_hard"], "intervention": info["intervention"],
             "qp_iters": info["qp_iters"], "row_violation": info["row_violation"],
             "m_hat": mdiag["m_hat"], "m_star_at": mdiag["m_star"], "rho_true": rho_true,
             "xhat_err_near": xhat_err_near, "xhat_err_all": xhat_err_all,
             "n_near": jnp.sum(near).astype(jnp.int32)}   # overnight: near-pair count (logging only) -> pair-weighted MSE
        if adaptive and cbf_on:   # realized margin on the valid neighbour rows: exact sum/count + 0.01 m histogram
            ds, ok = info["nb_d_safe"], info["nb_valid"]
            b = jnp.clip(jnp.round((ds - D_COLLISION) / DSAFE_BIN).astype(jnp.int32), 0, DSAFE_NBINS - 1)
            y["d_safe_hist"] = jnp.zeros(DSAFE_NBINS, jnp.int32).at[b.reshape(-1)].add(ok.reshape(-1).astype(jnp.int32))
            y["d_safe_sum"] = jnp.sum(jnp.where(ok, ds, 0.0))
            y["d_safe_n"] = jnp.sum(ok)
            if sig_mode:  # closed-loop coverage of sigma_c on the CBF rows (all valid / true distance < 2 m)
                err = jnp.take_along_axis(jnp.linalg.norm(x_hat - positions[None], axis=-1), info["nb_idx"], axis=1)
                sg = jnp.take_along_axis(sigma_c, info["nb_idx"], axis=1)
                near2 = ok & (jnp.take_along_axis(dfull, info["nb_idx"], axis=1) < 2.0)
                y["sig_cov_n"], y["sig_cov_hit"] = jnp.sum(ok), jnp.sum(ok & (err <= sg))
                y["sig_cov_n2"], y["sig_cov_hit2"] = jnp.sum(near2), jnp.sum(near2 & (err <= sg))
                y["sig_sum"] = jnp.sum(jnp.where(ok, sg, 0.0))
                bind = ok & (ds > prm.margin_floor)   # rows where the sigma margin, not the floor, sets d_safe
                y["sig_cov_nb"], y["sig_cov_hitb"] = jnp.sum(bind), jnp.sum(bind & (err <= sg))
            if closing:   # v_closing on the same valid neighbour rows, 0.05 m/s histogram + exact sum
                vc = jnp.take_along_axis(v_closing, info["nb_idx"], axis=1)
                bv = jnp.clip(jnp.round(vc / VCLOSE_BIN).astype(jnp.int32), 0, VCLOSE_NBINS - 1)
                y["v_closing_hist"] = jnp.zeros(VCLOSE_NBINS, jnp.int32).at[bv.reshape(-1)].add(
                    ok.reshape(-1).astype(jnp.int32))
                y["v_closing_sum"] = jnp.sum(jnp.where(ok, vc, 0.0))
        if log_xhat:
            y["xhat_rel"] = (x_hat - positions[:, None]).astype(jnp.float32)   # (N,N,2) estimate - own pos (float32)
            y["cbf_nb_idx"] = info["nb_idx"].astype(jnp.int8)
            y["held"] = mdiag["held"]
        return (positions_new, history, key, warm), y

    def run_arrays(key, arrays, dt, step):
        macro = dict(arrays, dt=dt, step=step)
        key, k0 = jax.random.split(key)
        pos0 = scenario.initial_positions(k0)
        n_walls = segs.shape[0]
        carry0 = (pos0, jnp.zeros((n_steps, N, 2)), key, init_warm_start(N, prm, n_walls))
        (pos_T, _, _, _), logs = jax.lax.scan(lambda c, k: step_fn(c, k, macro), carry0,
                                              jnp.arange(n_steps))
        logs = dict(logs)
        logs["pos_final"] = pos_T
        return logs

    jitted = jax.jit(run_arrays, static_argnames=("dt", "step"))

    def run(key, macro):
        arrays = {k: macro[k] for k in ("phi_grad", "m_star", "m_star_grad")}
        return jitted(key, arrays, dt=float(macro["dt"]), step=float(macro["step"]))

    return run


def sigma_logerr(sp, X, F, N):
    """The error model g of train.sigma (recipe D of the G-pred gate) on the (N_i, N_j) pair
    tensors: Z = X (20) + F (2) + (|F|, stale range, |F - stale|, T_frame) -> predicted log|e| [log m]."""
    stale = jnp.stack([X[..., 0] * jnp.cos(X[..., 1]), X[..., 0] * jnp.sin(X[..., 1])], -1)
    der = jnp.stack([jnp.linalg.norm(F, axis=-1), X[..., 0], jnp.linalg.norm(F - stale, axis=-1),
                     jnp.full(X.shape[:-1], t_frame(N))], -1)
    Z = (jnp.concatenate([X, F, der], -1) - sp["mu"]) / sp["sd"]
    n_layers = sum(1 for k in sp if k.startswith("W"))
    for li in range(n_layers - 1):
        Z = jax.nn.relu(Z @ sp[f"W{li}"] + sp[f"b{li}"])
    o = Z @ sp[f"W{n_layers - 1}"] + sp[f"b{n_layers - 1}"]
    return o[..., 0] * sp["lsd"] + sp["lmu"]


DSAFE_BIN, DSAFE_NBINS = 0.01, 1001   # realized-d_safe histogram: bins D_COLLISION + 0.01 b, b = 0..1000
                                      # (widened so that closing margins up to ~10 m fit)
VCLOSE_BIN, VCLOSE_NBINS = 0.05, 341  # v_closing histogram: bins 0.05 b m/s, b = 0..340 (top bin = clip)


def dsafe_stats(logs):
    """Mean / p50 / p95 of the realized adaptive d_safe over all valid neighbour rows x agents x steps
    (mean exact; percentiles at the 0.01 m bin resolution). None for constant-d_safe runs."""
    if "d_safe_hist" not in logs:
        return {"d_safe_mean": None, "d_safe_p50": None, "d_safe_p95": None}
    h = np.asarray(logs["d_safe_hist"]).sum(0).astype(np.float64)
    c = np.cumsum(h) / h.sum()
    centres = D_COLLISION + DSAFE_BIN * np.arange(DSAFE_NBINS)
    return {"d_safe_mean": float(np.sum(np.asarray(logs["d_safe_sum"], np.float64)) / np.sum(logs["d_safe_n"])),
            "d_safe_p50": float(centres[np.searchsorted(c, 0.50)]), "d_safe_p95": float(centres[np.searchsorted(c, 0.95)]),
            **vclosing_stats(logs), **sigma_cov_stats(logs)}


def sigma_cov_stats(logs):
    """Closed-loop coverage P(|x_hat - x| <= sigma_c) on the CBF's valid neighbour rows (all; true distance
    < 2 m) and the mean sigma_c there. Empty dict for runs without the sigma margin."""
    if "sig_cov_n" not in logs:
        return {}
    n, n2 = (int(np.sum(np.asarray(logs[k], np.int64))) for k in ("sig_cov_n", "sig_cov_n2"))
    return {"sigma_cov_rows": float(np.sum(np.asarray(logs["sig_cov_hit"], np.int64)) / max(n, 1)),
            "sigma_cov_rows_near2m": float(np.sum(np.asarray(logs["sig_cov_hit2"], np.int64)) / max(n2, 1)),
            "sigma_cov_n_near2m": n2, "sigma_bind_frac": int(np.sum(np.asarray(logs["sig_cov_nb"], np.int64))) / max(n, 1),
            "sigma_cov_binding": float(np.sum(np.asarray(logs["sig_cov_hitb"], np.int64))
                                       / max(int(np.sum(np.asarray(logs["sig_cov_nb"], np.int64))), 1)),
            "sigma_mean_rows_m": float(np.sum(np.asarray(logs["sig_sum"], np.float64)) / max(n, 1))}


def vclosing_stats(logs):
    """Mean / p50 / p95 of v_closing over the valid neighbour rows (mean exact; percentiles at 0.05 m/s) + the
    summed histogram. Empty for runs without the closing margin."""
    if "v_closing_hist" not in logs:
        return {}
    h = np.asarray(logs["v_closing_hist"]).sum(0).astype(np.int64)
    c = np.cumsum(h) / h.sum()
    centres = VCLOSE_BIN * np.arange(VCLOSE_NBINS)
    return {"v_closing_mean": float(np.sum(np.asarray(logs["v_closing_sum"], np.float64)) / np.sum(logs["d_safe_n"])),
            "v_closing_p50": float(centres[np.searchsorted(c, 0.50)]),
            "v_closing_p95": float(centres[np.searchsorted(c, 0.95)]),
            "v_closing_frac_zero_bin": float(h[0] / h.sum()), "v_closing_hist": h.tolist(),
            "v_closing_hist_bin_mps": VCLOSE_BIN}


def summarize(logs, scenario, dt_exec, c_w=1.0):
    """Per-run metrics from TRUE positions. All numpy."""
    pos = np.asarray(logs["pos"])                                     # (T,N,2), state at step start
    pos = np.concatenate([pos, np.asarray(logs["pos_final"])[None]], 0)
    g = np.asarray(scenario.goal)
    r = np.linalg.norm(pos - g[None, None], axis=-1)                   # (T+1,N)
    arrived = r <= scenario.r_goal
    frac = arrived.mean(1)
    t = np.arange(pos.shape[0]) * dt_exec
    ever = arrived.any(0)
    first = np.where(ever, arrived.argmax(0), -1)
    crossed = pos[..., 0] > scenario.wall_x
    coll = np.asarray(logs["n_coll_pairs"])
    interv = np.asarray(logs["intervention"])
    slack = np.asarray(logs["slack_max"])
    p5 = np.asarray(logs["p5_sep_pairs"])
    out = {
        "nan_position_entries": int(np.isnan(pos).sum()),
        "qp_nan_fallbacks": int((np.asarray(logs["qp_iters"]) < 0).sum()),
        "completeness_final": float(frac[-1]),
        "arrived_frac_max": float(frac.max()),
        "t_first_90pct": float(t[np.argmax(frac >= 0.9)]) if (frac >= 0.9).any() else None,
        "makespan_s": float(t[first].max()) if ever.all() else None,
        "mean_time_to_goal_s": float(t[first[ever]].mean()) if ever.any() else None,
        "frac_ever_arrived": float(ever.mean()),
        "frac_left_goal_after_arrival": float(np.mean([(~arrived[f:, i]).any() for i, f in enumerate(first) if f >= 0])) if ever.any() else None,
        "crossed_wall_final": float(crossed[-1].mean()),
        "collision_pair_steps": int(coll.sum()),
        "collision_steps": int((coll > 0).sum()),
        "min_sep_overall_m": float(np.asarray(logs["min_sep"]).min()),
        "p5_sep_worst_step_m": float(p5.min()),
        "p5_sep_median_over_time_m": float(np.median(p5)),
        "cbf_intervention_rate": float((interv > 0.01).mean()),
        "cbf_intervention_mean_mps": float(interv.mean()),
        "slack_active_rate": float((slack > 1e-4).mean()),
        "slack_max": float(slack.max()),
        "slack_hard_max": float(np.asarray(logs["slack_hard"]).max()) if "slack_hard" in logs else None,
        "qp_aux_mean": float(np.asarray(logs["qp_iters"]).mean()),
        "qp_aux_max": int(np.asarray(logs["qp_iters"]).max()),
        "row_violation_max": float(np.asarray(logs["row_violation"]).max()),
        "mean_speed_mps": float(np.linalg.norm(np.asarray(logs["u"]), axis=-1).mean()),
        "closedloop_xhat_mse_near": float(np.mean(logs["xhat_err_near"])) if "xhat_err_near" in logs else None,
        "closedloop_xhat_mse_all": float(np.mean(logs["xhat_err_all"])) if "xhat_err_all" in logs else None,
        # pair-weighted (every near pair-step counts once; steps without near pairs do not enter as 0)
        "closedloop_xhat_mse_near_pairw": (float(np.sum(np.asarray(logs["xhat_err_near"], np.float64) * logs["n_near"])
                                                 / max(int(np.sum(logs["n_near"])), 1)) if "n_near" in logs else None),
        "near_pair_steps": int(np.sum(logs["n_near"])) if "n_near" in logs else None,
        "integral_W_true": float(np.sum(np.mean(c_w * np.asarray(logs["rho_true"]) ** 2, 1)) * dt_exec),
        **dsafe_stats(logs),
    }
    return out, {"t": t, "arrived_frac": frac, "r_goal_dist": r, "first_arrival_idx": first}
