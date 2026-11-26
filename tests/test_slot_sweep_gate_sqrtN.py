"""Config-level checks for the packing-collapse controls (gate √N, start density, geom ∝ N).

These do not run rollouts. They lock the experimental design: shared T_frame grid and the
intended geometry overrides.
"""
from __future__ import annotations

import math
import os

from control.params import gate_width_for_N
from sim.config import load_experiment
from sim.setup import ROOT

G0 = 2.309401076758503
N_REF = 25
T_FRAME_MS = (75, 100, 125, 150, 175, 200, 250, 300)
START_SPACING = 1.051


def _gate(N: int) -> float:
    return G0 * math.sqrt(N / N_REF)


def _cells(name: str):
    path = os.path.join(ROOT, "configs", "exp", name)
    return load_experiment(path)


def _by_N(cells):
    out = {}
    for cell in cells.values():
        N = cell["cfg"]["N"]
        out.setdefault(N, []).append(cell)
    return out


def test_gate_sqrtN_widths_and_shared_t_frame():
    exp, cells = _cells("slot_sweep_gate_sqrtN.yaml")
    assert exp["name"] == "slot_sweep_gate_sqrtN"
    assert len(cells) == 24  # 3 N x 8 slots
    grouped = _by_N(cells)
    assert set(grouped) == {25, 50, 100}
    for N, group in grouped.items():
        assert len(group) == 8
        frames = sorted(round(1000 * N * c["cfg"]["t_slot"]) for c in group)
        assert frames == list(T_FRAME_MS)
        for c in group:
            assert c["arm"] == "real"
            assert abs(c["cfg"]["scenario"]["gate_width"] - _gate(N)) < 1e-9


def test_gate_sqrtN_density_adds_matched_start():
    exp, cells = _cells("slot_sweep_gate_sqrtN_density.yaml")
    assert exp["name"] == "slot_sweep_gate_sqrtN_density"
    assert len(cells) == 24
    for c in cells.values():
        N = c["cfg"]["N"]
        assert c["cfg"]["scenario"]["start_spacing"] == START_SPACING
        assert abs(c["cfg"]["scenario"]["gate_width"] - _gate(N)) < 1e-9


def test_gate_sqrtN_widens_with_N_relative_to_baseline():
    """N=25 keeps the original single_gate gate; larger N gets a wider opening."""
    assert abs(_gate(25) - G0) < 1e-12
    assert _gate(50) > G0
    assert abs(_gate(100) / G0 - 2.0) < 1e-12


def test_built_scenario_receives_scaled_gate():
    """Scenario object must see the override (enters MFG geometry / CBF wall gap)."""
    _, cells = _cells("slot_sweep_gate_sqrtN.yaml")
    from sim.setup import build_scenario
    for N in (25, 50, 100):
        cfg = next(c["cfg"] for c in cells.values() if c["cfg"]["N"] == N)
        sc = build_scenario(cfg, N)
        assert abs(sc.gate_width - _gate(N)) < 1e-9


def test_geom_scaleN_self_similar_arena_and_gate():
    """geometry_N = N and gate = gate_width_for_N(N); start spacing ~constant across N."""
    exp, cells = _cells("slot_sweep_geom_scaleN.yaml")
    assert exp["name"] == "slot_sweep_geom_scaleN"
    assert len(cells) == 24
    from sim.setup import build_scenario
    spacings = {}
    arenas = {}
    for N, group in _by_N(cells).items():
        frames = sorted(round(1000 * N * c["cfg"]["t_slot"]) for c in group)
        assert frames == list(T_FRAME_MS)
        cfg = group[0]["cfg"]
        assert cfg["scenario"]["geometry_N"] == N
        assert abs(cfg["scenario"]["gate_width"] - gate_width_for_N(N)) < 1e-9
        assert cfg["scenario"].get("start_spacing") is None
        sc = build_scenario(cfg, N)
        assert sc.N_geom == N and sc.N == N
        assert abs(sc.gate_width - gate_width_for_N(N)) < 1e-9
        x0, x1, z0, z1 = sc.start_box
        area = (x1 - x0) * (z1 - z0)
        spacings[N] = math.sqrt(area / (N * (math.sqrt(3.0) / 2.0)))
        arenas[N] = sc.arena[0] * sc.arena[1]
    # Self-similar: start hex spacing nearly flat; arena area grows ~linearly with N
    assert max(spacings.values()) / min(spacings.values()) < 1.05
    assert arenas[100] / arenas[25] > 3.0  # order-of-N growth (goal chamber included)
