import glob
import os

import numpy as np

from sim.config import load_experiment
from sim.metrics import outcome_metrics
from sim.setup import ROOT, build_scenario


def test_example_experiment_resolves():
    exp, cells = load_experiment(os.path.join(ROOT, "configs", "exp", "example.yaml"))
    assert "N100_learned" in cells and cells["N100_real_d0.55"]["cfg"]["cbf"]["d_safe"] == 0.55


def test_every_geometry_builds():
    from sim.config import deep_merge
    import yaml
    base = yaml.safe_load(open(os.path.join(ROOT, "configs", "base.yaml")))
    for g in glob.glob(os.path.join(ROOT, "configs", "geometry", "*.yaml")):
        cfg = deep_merge(base, yaml.safe_load(open(g)))
        sc = build_scenario(cfg, 25)
        assert sc.r_goal > 0


def test_outcome_metrics_toy():
    goal, r = (0.0, 0.0), 1.0
    T = 1000
    pos = np.zeros((T, 2, 2)); pos[:, 0] = [0.2, 0.0]; pos[:, 1] = [5.0, 0.0]      # agent 0 settled, 1 far away
    m, coll = outcome_metrics(pos, pos[-1], goal, r)
    assert m["ever_collided"] == 0.0 and m["completeness_settled_5s_b0.00"] == 0.5
    pos[10, 1] = [0.3, 0.0]                                                         # one close pass -> both collided
    m, coll = outcome_metrics(pos, pos[-1], goal, r)
    assert m["ever_collided"] == 1.0 and m["safe_settled_5s_b0.00"] == 0.0
