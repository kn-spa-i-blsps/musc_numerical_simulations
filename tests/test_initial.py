"""Initial positions are a separated sample of the MFG initial density."""
import os

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from control.params import D_COLLISION
from env.base import Grid
from env.initial import positions_matching_density
from sim.config import deep_merge
from sim.setup import ROOT, build_scenario


def _min_dist(pos):
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def _scenario(n):
    base = yaml.safe_load(open(os.path.join(ROOT, "configs", "base.yaml")))
    geo = yaml.safe_load(open(os.path.join(ROOT, "configs", "geometry", "single_gate.yaml")))
    return build_scenario(deep_merge(base, geo), n)


def test_sampler_follows_a_one_sided_density():
    density = jnp.zeros((30, 30))
    density = density.at[2:14, 2:28].set(1.0)
    pos = np.asarray(positions_matching_density(
        jax.random.PRNGKey(0), density, (3.0, 3.0), n=20, d_min=D_COLLISION))
    assert pos.shape == (20, 2)
    assert _min_dist(pos) >= D_COLLISION - 1e-3
    assert np.all(pos[:, 0] >= -1e-6) and np.all(pos[:, 0] <= 3.0 + 1e-6)
    assert np.mean(pos[:, 0] < 1.5) > 0.9


def test_separation_when_nodes_are_closer_than_the_floor():
    pos = np.asarray(positions_matching_density(
        jax.random.PRNGKey(1), jnp.ones((12, 12)), (1.5, 1.5), n=8, d_min=0.45))
    assert _min_dist(pos) >= 0.45 - 1e-3


def test_scenario_initial_positions_match_m0():
    sc = _scenario(25)
    grid = Grid.build(sc.arena, 0.2, 1.0)
    m0 = sc.initial_density(grid)
    pos = np.asarray(sc.initial_positions(jax.random.PRNGKey(2), m0))
    assert pos.shape == (25, 2)
    assert _min_dist(pos) >= D_COLLISION - 1e-3
    assert np.all(pos[:, 0] >= -1e-5) and np.all(pos[:, 0] <= sc.arena[0] + 1e-5)
    assert np.all(pos[:, 1] >= -1e-5) and np.all(pos[:, 1] <= sc.arena[1] + 1e-5)

    xs = np.linspace(0.0, sc.arena[0], m0.shape[0])
    zs = np.linspace(0.0, sc.arena[1], m0.shape[1])
    mass = np.asarray(m0) * grid.step ** 2
    com = np.array([np.sum(mass * xs[:, None]), np.sum(mass * zs[None, :])]) / mass.sum()
    assert np.linalg.norm(pos.mean(0) - com) < 1.5
    # Quantizer, not a Poisson draw: nearest neighbours sit on the density's spacing, far above collision.
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    assert float(np.median(d.min(1))) > 1.0


def test_dense_initial_positions_stay_separated_and_track_the_rim():
    sc = _scenario(200)
    grid = Grid.build(sc.arena, 0.2, 1.0)
    m0 = sc.initial_density(grid)
    pos = np.asarray(sc.initial_positions(jax.random.PRNGKey(3), m0))
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    nn = d.min(1)
    assert float(nn.min()) >= D_COLLISION - 1e-3
    assert float(np.median(nn)) > 0.5
    x0, x1, z0, z1 = sc.start_box
    band = 0.6
    on_edge = ((pos[:, 0] < x0 + band) | (pos[:, 0] > x1 - band)
               | (pos[:, 1] < z0 + band) | (pos[:, 1] > z1 - band))
    xs = np.linspace(0.0, sc.arena[0], m0.shape[0])
    zs = np.linspace(0.0, sc.arena[1], m0.shape[1])
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    edge = (X < x0 + band) | (X > x1 - band) | (Z < z0 + band) | (Z > z1 - band)
    mass = np.asarray(m0) * grid.step ** 2
    assert abs(float(on_edge.mean()) - float(mass[edge].sum() / mass.sum())) < 0.08
