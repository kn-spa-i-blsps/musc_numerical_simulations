"""Experiment files -> resolved cells.

An experiment YAML names a geometry, seeds and a list of blocks. Each block is a Cartesian product over its keys;
dotted keys address nested config entries (cbf.d_safe, scenario.gate_width). `arm` is the perception arm;
`variant` picks a named set of overrides from the experiment's `variants:` map (e.g. controller ablations):

    name: crossover
    geometry: single_gate
    seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    blocks:
      - {N: [25, 50, 100, 150, 200], arm: [real], cbf.d_safe: [0.55, 0.70]}
      - {N: [25, 50, 100, 150, 200], arm: [learned, noise_only]}

Config = configs/base.yaml <- configs/geometry/<geometry>.yaml <- experiment `set:` <- block values.
"""
from __future__ import annotations

import copy
import itertools
import os

import yaml

from sim.setup import ROOT

CONFIGS = os.path.join(ROOT, "configs")


def deep_merge(a, b):
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def set_dotted(cfg, key, value):
    d = cfg
    *path, last = key.split(".")
    for p in path:
        d = d.setdefault(p, {})
    d[last] = value


def _tag(key, v):
    if key == "N":
        return f"N{int(v):03d}"
    if key == "variant":
        return str(v)
    if key == "arm":
        return str(v).replace(":", "_").replace("/", "-")
    if key == "cbf.d_safe":
        return f"d{float(v):.2f}"
    if key == "t_slot":
        return f"ts{1000 * float(v):.2f}ms"
    if key == "scenario.gate_width":
        return f"g{float(v):.3f}"
    return f"{key.split('.')[-1]}{v}"


def load_experiment(path):
    exp = yaml.safe_load(open(path))
    base = yaml.safe_load(open(os.path.join(CONFIGS, "base.yaml")))
    geo = yaml.safe_load(open(os.path.join(CONFIGS, "geometry", exp["geometry"] + ".yaml")))
    root = deep_merge(deep_merge(base, geo), exp.get("set"))
    cells = {}
    for block in exp["blocks"]:
        keys = list(block)
        for combo in itertools.product(*(block[k] if isinstance(block[k], list) else [block[k]] for k in keys)):
            cfg, arm = copy.deepcopy(root), "real"
            for k, v in zip(keys, combo):
                if k == "arm":
                    arm = v
                elif k == "variant":
                    for vk, vv in exp["variants"][v].items():
                        set_dotted(cfg, vk, vv)
                else:
                    set_dotted(cfg, k, v)
            name = "_".join(_tag(k, v) for k, v in zip(keys, combo))
            run_arm = arm
            if name in cells:
                raise ValueError(f"duplicate cell {name} in {path}")
            cells[name] = {"cfg": cfg, "arm": run_arm, "label": arm}
    return exp, cells
