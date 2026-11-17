"""Viscous (second-order) MFG option: fp.nu > 0."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_platform_name", "cpu")

from env.base import Grid
from env.scenarios import SingleGateA
from mfg.equilibrium import MacroCost, solve_macro
from mfg.hjb import _scan_hjb_backward_visc
from mfg.transport import diffuse_core


def test_diffuse_mass_positivity_and_spread_rate():
    """Mass is exact, m stays >= 0 at the stability limit, and a point mass spreads with per-axis
    variance 2*nu*t (the explicit scheme is an exact discrete random walk away from the walls)."""
    h, nu = 0.2, 0.5
    dt = 0.25 * h ** 2 / nu                                  # r = 1/4, the stability limit
    m = jnp.zeros((81, 81)).at[40, 40].set(1.0 / h ** 2)    # unit mass in the centre cell
    n = 40
    for _ in range(n):
        m = diffuse_core(m, h, dt, nu)
    assert float(m.min()) >= 0.0
    assert abs(float(m.sum()) * h ** 2 - 1.0) < 1e-5
    x = (jnp.arange(81) - 40) * h
    var_x = float(jnp.sum(m.sum(1) * x ** 2) * h ** 2)
    assert var_x == pytest.approx(2 * nu * n * dt, rel=1e-4)


def test_diffuse_no_flux_walls():
    """Mass next to the arena edge stays in the arena."""
    m = jnp.zeros((10, 10)).at[0, 0].set(1.0)
    for _ in range(200):
        m = diffuse_core(m, 0.2, 0.004, 1.0)
    assert abs(float(m.sum()) - 1.0) < 1e-5 and float(m.min()) >= 0.0


def test_hjb_laplacian_smooths_backward():
    """With f = 0 and a terminal bump, the viscous HJB run backward must spread the bump (sign check:
    a wrong sign would sharpen it and blow up)."""
    h, nu, dt = 0.2, 0.3, 0.004
    xs = (jnp.arange(61) - 30) * h
    g = 0.05 * jnp.exp(-(xs[:, None] ** 2 + xs[None, :] ** 2) / 0.5)
    v = _scan_hjb_backward_visc(jnp.zeros((500, 61, 61)), g, h, dt, nu)
    v0 = v[-1]                                               # V at t = 0 (scan output is reversed in time)
    assert bool(jnp.all(jnp.isfinite(v0)))
    assert float(v0.max()) < float(g.max())                  # peak lowered
    assert float(v0[30, 38]) > float(g[30, 38])              # flank raised


def test_solve_macro_viscous_small():
    """End to end on a tiny grid: no NaN, mass conserved, and the stability guard rejects a too-large nu."""
    sc = SingleGateA(20, w_goal=2.0, w_hold=8.0)
    grid = Grid.build(sc.arena, 1.0, 20.0)
    cost = MacroCost(1.0, 0.0, 5.0, 300.0)
    sol = solve_macro(sc, grid, cost, damping=0.1, max_iters=6, tol=1e-3, burn_in=2, nu=0.5)
    assert not sol["nan"]
    mass = np.asarray(sol["m_star"].sum(axis=(1, 2))) * grid.step ** 2
    assert np.allclose(mass, mass[0], rtol=1e-4)
    with pytest.raises(ValueError):
        solve_macro(sc, grid, cost, damping=0.1, max_iters=2, tol=1e-3, burn_in=1, nu=1e3)
