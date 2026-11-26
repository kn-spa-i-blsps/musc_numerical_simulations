"""Density-matched start (scenario.start_spacing): the start box shrinks so the hex spacing is the same for every N."""
import math

import jax
import numpy as np
import pytest

jax.config.update("jax_platform_name", "cpu")

from env.base import Grid
from env.scenarios import LayoutA, SingleGateA

KW = dict(w_goal=2.0, w_hold=8.0, goal_gap_mult=3.0, goal_spacing=1.05, goal_clearance=1.18,
          goal_geometry_spacing=1.05)


def _default_spacing(sc):
    x0, x1, z0, z1 = sc.start_box
    return math.sqrt(2.0 * (x1 - x0) * (z1 - z0) / (math.sqrt(3.0) * sc.N))


def test_default_is_unchanged():
    """start_spacing=None: same box, same initial positions, same initial density (bit-identical)."""
    a, b = SingleGateA(230, n_agents=50, **KW), SingleGateA(230, n_agents=50, start_spacing=None, **KW)
    assert a.start_box == b.start_box
    key = jax.random.PRNGKey(3)
    assert np.array_equal(np.asarray(a.initial_positions(key)), np.asarray(b.initial_positions(key)))
    grid = Grid.build(a.arena, 0.2, 10.0)
    assert np.array_equal(np.asarray(a.initial_density(grid)), np.asarray(b.initial_density(grid)))


@pytest.mark.parametrize("N", [25, 50])
def test_spacing_matched(N):
    """The box has area N*(sqrt3/2)*s^2, keeps the wall gap and the default aspect ratio, is centred on the gate,
    and the lattice spacing (hence the minimum initial distance up to jitter) is s."""
    s = _default_spacing(SingleGateA(230, n_agents=100, **KW))
    d, sc = SingleGateA(230, n_agents=N, **KW), SingleGateA(230, n_agents=N, start_spacing=s, **KW)
    x0, x1, z0, z1 = sc.start_box
    dx0, dx1, dz0, dz1 = d.start_box
    assert math.isclose((x1 - x0) * (z1 - z0), N * math.sqrt(3.0) / 2.0 * s ** 2, rel_tol=1e-9)
    assert x1 == dx1
    assert math.isclose((x1 - x0) / (z1 - z0), (dx1 - dx0) / (dz1 - dz0), rel_tol=1e-9)
    assert math.isclose((z0 + z1) / 2.0, sc.gate_z, abs_tol=1e-9)
    assert math.isclose(_default_spacing(sc), s, rel_tol=1e-9)
    p = np.asarray(sc.initial_positions(jax.random.PRNGKey(0)))
    D = np.linalg.norm(p[:, None] - p[None], axis=-1) + np.eye(N) * 1e9
    assert D.min() >= 0.7 * s                      # lattice s, jitter +-0.1 s per coordinate
    assert np.all((p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 1] >= z0) & (p[:, 1] <= z1))
    grid = Grid.build(sc.arena, 0.2, 10.0)
    assert math.isclose(float(np.asarray(sc.initial_density(grid)).sum()) * 0.2 ** 2, N, rel_tol=1e-5)
    assert sc.path_length_to_goal() < d.path_length_to_goal()


def test_reference_N_reproduces_default_box_and_never_enlarges():
    d = SingleGateA(230, n_agents=100, **KW)
    s = _default_spacing(d)
    np.testing.assert_allclose(SingleGateA(230, n_agents=100, start_spacing=s, **KW).start_box, d.start_box, atol=1e-9)
    assert SingleGateA(230, n_agents=100, start_spacing=2 * s, **KW).start_box == d.start_box


@pytest.mark.parametrize("layout", ["two_gates", "pillar"])
def test_layouts_accept_it(layout):
    s = _default_spacing(SingleGateA(230, n_agents=100, **KW))
    sc = LayoutA(230, layout=layout, n_agents=25, start_spacing=s, **KW)
    assert math.isclose(_default_spacing(sc), s, rel_tol=1e-9)
