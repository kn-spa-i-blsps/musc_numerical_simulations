"""Goal-region options: flat core in V_goal (scenario.goal_core_frac) and the hold radius (control.hold_radius_frac)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_platform_name", "cpu")

from control.macro import nominal_control
from env.base import Grid
from env.scenarios import LayoutA, SingleGateA

KW = dict(w_goal=2.0, w_hold=8.0, goal_gap_mult=3.0, goal_spacing=1.05, goal_clearance=1.18,
          goal_geometry_spacing=1.05, n_agents=50)


def _radial_profile(sc, grid):
    v_goal, _ = sc.potentials(grid)
    x, z = grid.meshgrid()
    r = np.asarray(jnp.sqrt((x - sc.goal[0]) ** 2 + (z - sc.goal[1]) ** 2))
    return np.asarray(v_goal), r


def test_default_is_unchanged():
    """goal_core_frac=None must give exactly the old potential."""
    grid = Grid.build(SingleGateA(230, **KW).arena, 0.2, 10.0)
    a, _ = _radial_profile(SingleGateA(230, **KW), grid)
    b, _ = _radial_profile(SingleGateA(230, goal_core_frac=None, **KW), grid)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("cls,extra", [(SingleGateA, {}), (LayoutA, {"layout": "two_gates"}),
                                       (LayoutA, {"layout": "pillar"})])
def test_core_is_flat_and_cone_outside(cls, extra):
    """Inside c*r_goal V_goal is constant; between the core and r_goal it rises with slope w_goal (cone);
    the tanh hold step outside r_goal is untouched. All three geometries."""
    c = 0.75
    sc0, sc = cls(230, **KW, **extra), cls(230, goal_core_frac=c, **KW, **extra)
    grid = Grid.build(sc.arena, 0.2, 10.0)
    v, r = _radial_profile(sc, grid)
    v0, _ = _radial_profile(sc0, grid)
    core = r < c * sc.r_goal - 0.3
    assert np.ptp(v[core]) < 1e-3                                     # flat core
    band = (r > c * sc.r_goal + 0.5) & (r < sc.r_goal - 1.0)          # cone band inside the disc
    slope = np.polyfit(r[band], v[band], 1)[0]
    assert slope == pytest.approx(2.0, rel=0.02)
    far = r > sc.r_goal + 2.0                                         # outside: old field minus a constant shift
    assert np.ptp((v0 - v)[far]) == pytest.approx(0.0, abs=0.05)


def test_core_validation():
    with pytest.raises(ValueError):
        SingleGateA(230, goal_core_frac=1.2, **KW)
    with pytest.raises(ValueError):
        SingleGateA(230, goal_core_frac=0.5, goal_shape="flat", **KW)


def test_hold_radius():
    """An isolated agent is held only inside the hold radius."""
    goal = jnp.array([0.0, 0.0])
    N = 3
    pos = jnp.array([[3.0, 0.0], [20.0, 0.0], [-20.0, 0.0]])          # agent 0 at r=3, others far away
    x_hat = jnp.broadcast_to(pos[None], (N, N, 2))
    T, n = 4, 41
    macro = {"phi_grad": jnp.ones((T, n, n, 2)), "m_star": jnp.zeros((T, n, n)),
             "m_star_grad": jnp.zeros((T, n, n, 2)), "dt": 1.0, "step": 1.0}
    kw = dict(c_w=1.0, w_bar=0.0, m_bar=10.0, c_dens=0.0)
    _, d_in = nominal_control(pos, x_hat, 0.0, N, macro, hold=(goal, 5.0, 0.7), **kw)    # r_hold 5 > 3
    _, d_out = nominal_control(pos, x_hat, 0.0, N, macro, hold=(goal, 2.0, 0.7), **kw)   # r_hold 2 < 3
    assert bool(d_in["held"][0]) and not bool(d_out["held"][0])
