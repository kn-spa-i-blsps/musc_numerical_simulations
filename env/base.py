"""Protokół Scenario + siatka przestrzenno-czasowa współdzielona przez mfg/."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import jax.numpy as jnp

# CFL-safe ratio dt/step dla jawnego schematu Lax-Friedrichs w mfg/hjb.py. Mały margines,
# bo lokalna dyssypacja skaluje się z |grad V|, który bywa duży blisko stromego V_obs/W(m).
_CFL = 0.02


def grid_gradient(field: jnp.ndarray, h: float) -> jnp.ndarray:
    """Wolna funkcja (nie metoda) — żeby dało się ją jitować/scanować bez przeciągania
    całego obiektu Grid (niehashowalny pytree z tablicami) przez granicę jax.jit."""
    gx = (jnp.roll(field, -1, axis=0) - jnp.roll(field, 1, axis=0)) / (2 * h)
    gx = gx.at[0, :].set((field[1, :] - field[0, :]) / h)
    gx = gx.at[-1, :].set((field[-1, :] - field[-2, :]) / h)
    gz = (jnp.roll(field, -1, axis=1) - jnp.roll(field, 1, axis=1)) / (2 * h)
    gz = gz.at[:, 0].set((field[:, 1] - field[:, 0]) / h)
    gz = gz.at[:, -1].set((field[:, -1] - field[:, -2]) / h)
    return jnp.stack([gx, gz], axis=-1)


@dataclass(frozen=True)
class Grid:
    step: float
    arena: tuple[float, float]
    horizon: float
    xs: jnp.ndarray
    zs: jnp.ndarray
    dt: float
    n_steps: int

    @property
    def nx(self) -> int:
        return self.xs.shape[0]

    @property
    def nz(self) -> int:
        return self.zs.shape[0]

    @classmethod
    def build(cls, arena: tuple[float, float], step: float, horizon: float) -> "Grid":
        w, h = arena
        nx = int(round(w / step)) + 1
        nz = int(round(h / step)) + 1
        xs = jnp.linspace(0.0, w, nx)
        zs = jnp.linspace(0.0, h, nz)
        dt = _CFL * step
        n_steps = max(1, int(jnp.ceil(horizon / dt)))
        return cls(step=step, arena=arena, horizon=horizon, xs=xs, zs=zs, dt=dt, n_steps=n_steps)

    def meshgrid(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        return jnp.meshgrid(self.xs, self.zs, indexing="ij")

    def gradient(self, field: jnp.ndarray) -> jnp.ndarray:
        """Centralna różnica, jednostronna na brzegu. field: (nx, nz) -> (nx, nz, 2)."""
        return grid_gradient(field, self.step)

    def sample_nearest(self, field: jnp.ndarray, positions: jnp.ndarray) -> jnp.ndarray:
        """Nearest-cell lookup. positions: (n, 2) -> wartości pola w tych punktach."""
        ix = jnp.clip(jnp.round(positions[:, 0] / self.step).astype(int), 0, self.nx - 1)
        iz = jnp.clip(jnp.round(positions[:, 1] / self.step).astype(int), 0, self.nz - 1)
        return field[ix, iz]


class Scenario(Protocol):
    def potentials(self, grid: Grid) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Zwraca (V_goal, V_obs). V_obs: (nx, nz). V_goal: (nx, nz) albo, dla celu zmiennego w
        czasie (PersistentGate), (n_steps+1, nx, nz). Lokalne, nieopóźnione."""
        ...

    def domain(self) -> tuple[float, float]:
        """Rozmiar areny (W, H)."""
        ...

    def initial_density(self, grid: Grid) -> jnp.ndarray:
        """m0(x,z), kształt (nx, nz), znormalizowane do sum(m0)*step**2 == total_mass."""
        ...
