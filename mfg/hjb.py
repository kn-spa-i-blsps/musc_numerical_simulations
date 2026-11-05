"""HJB wstecz przeciw danemu polu m. Pierwszy rząd, 2D. Zwraca value i policy u*(x).

Forward-backward, nie APF: ten solver liczy przeciw m_field dostarczonemu
z zewnątrz (poprzednia iteracja fictitious_play, tj. przyszłe/skądinąd ustalone m),
nigdy przeciw m bieżącemu w jednym przebiegu.

Hamiltonian z L(x,u,m) = 1/2|u|^2 + V_goal + V_obs + W(m): H(p) = min_u[1/2|u|^2 + p.u]
= -1/2|p|^2, więc HJB wstecz to dV/dt - 1/2|grad V|^2 + f(t,x) = 0, V(T,.)=0, f = V_goal+V_obs+W(m).
Rozwiązywane jawnym schematem Lax-Friedrichs, z LOKALNĄ dyssypacją (alpha = |p| policzone
per-punkt z jednostronnych różnic, nie stały odgórny współczynnik) — V_obs bywa stromy
(miękki, ale wysoki barrier), więc stały h/dt nie daje dość dyssypacji i schemat wybucha.
"""
from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from env.base import Grid, grid_gradient


class CostParams(NamedTuple):
    V_goal: jnp.ndarray  # (nx, nz) lub (n_steps+1, nx, nz) — lokalne, nieopóźnione
    V_obs: jnp.ndarray  # (nx, nz) — lokalne, nieopóźnione
    W_fn: Callable[[jnp.ndarray], jnp.ndarray]  # crowd-aversion, monotoniczna w m
    V_terminal: jnp.ndarray | None = None  # terminal cost g(x), V(T,.) = g. None = V(T,.) = 0 (old graph)


def _one_sided_diffs(v: jnp.ndarray, h: float) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    dxp = (jnp.roll(v, -1, axis=0) - v) / h
    dxp = dxp.at[-1, :].set(0.0)
    dxm = (v - jnp.roll(v, 1, axis=0)) / h
    dxm = dxm.at[0, :].set(0.0)
    dzp = (jnp.roll(v, -1, axis=1) - v) / h
    dzp = dzp.at[:, -1].set(0.0)
    dzm = (v - jnp.roll(v, 1, axis=1)) / h
    dzm = dzm.at[:, 0].set(0.0)
    return dxp, dxm, dzp, dzm


@jax.jit
def _scan_hjb_backward(running_rev: jnp.ndarray, h: float, dt: float) -> jnp.ndarray:
    """Zdefiniowane RAZ na poziomie modułu (nie wewnątrz solve_hjb): jax.jit cache'uje
    skompilowany program XLA po kształcie wejścia i reużywa go przez wszystkie iteracje
    fictitious_play, zamiast rekompilować przy każdym wywołaniu solve_hjb."""

    def step(v_next: jnp.ndarray, f_k: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        dxp, dxm, dzp, dzm = _one_sided_diffs(v_next, h)
        h_central = 0.125 * ((dxp + dxm) ** 2 + (dzp + dzm) ** 2)
        alpha_x = jnp.maximum(jnp.abs(dxp), jnp.abs(dxm))
        alpha_z = jnp.maximum(jnp.abs(dzp), jnp.abs(dzm))
        dissipation = 0.5 * alpha_x * (dxp - dxm) + 0.5 * alpha_z * (dzp - dzm)
        h_num = h_central - dissipation
        v_k = v_next + dt * (f_k - h_num)
        return v_k, v_k

    v_terminal = jnp.zeros_like(running_rev[0])
    _, v_stack_rev = jax.lax.scan(step, v_terminal, running_rev)
    return v_stack_rev


@jax.jit
def _scan_hjb_backward_from(running_rev: jnp.ndarray, v_terminal: jnp.ndarray, h: float, dt: float) -> jnp.ndarray:
    """Same scheme as _scan_hjb_backward, started from V(T,.) = g instead of 0. Kept separate so the
    terminal-cost-0 path (V_terminal None) traces exactly the old graph."""

    def step(v_next: jnp.ndarray, f_k: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        dxp, dxm, dzp, dzm = _one_sided_diffs(v_next, h)
        h_central = 0.125 * ((dxp + dxm) ** 2 + (dzp + dzm) ** 2)
        alpha_x = jnp.maximum(jnp.abs(dxp), jnp.abs(dxm))
        alpha_z = jnp.maximum(jnp.abs(dzp), jnp.abs(dzm))
        dissipation = 0.5 * alpha_x * (dxp - dxm) + 0.5 * alpha_z * (dzp - dzm)
        h_num = h_central - dissipation
        v_k = v_next + dt * (f_k - h_num)
        return v_k, v_k

    _, v_stack_rev = jax.lax.scan(step, v_terminal, running_rev)
    return v_stack_rev


def solve_hjb(
    m_field: jnp.ndarray, cost_params: CostParams, grid: Grid
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """m_field: (n_steps+1, nx, nz), gęstość na całym horyzoncie (dana z zewnątrz).

    Zwraca (value, policy): value (n_steps+1, nx, nz) z value[0] == V(t=0);
    policy (n_steps+1, nx, nz, 2), u*(t,x) = -grad V(t,x).
    """
    h, dt = grid.step, grid.dt
    # V_goal: (nx, nz) statyczne albo (n_steps+1, nx, nz) zależne od czasu (PersistentGate).
    v_goal = cost_params.V_goal if cost_params.V_goal.ndim == 3 else cost_params.V_goal[None]
    running = v_goal + cost_params.V_obs[None] + cost_params.W_fn(m_field)

    running_rev = running[:-1][::-1]  # f_{n_steps-1}, ..., f_0 — kolejność cofania w czasie
    if cost_params.V_terminal is None:
        v_stack_rev = _scan_hjb_backward(running_rev, h, dt)
        v_terminal = jnp.zeros_like(running[0])
    else:                                    # V(T,.) = g(x)
        v_terminal = jnp.broadcast_to(cost_params.V_terminal, running[0].shape).astype(running.dtype)
        v_stack_rev = _scan_hjb_backward_from(running_rev, v_terminal, h, dt)
    value = jnp.concatenate([v_stack_rev[::-1], v_terminal[None]], axis=0)

    policy = -jax.vmap(lambda f: grid_gradient(f, h))(value)
    return value, policy
