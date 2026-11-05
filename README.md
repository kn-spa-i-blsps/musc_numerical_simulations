# MUSC — MFG + CBF swarm under TDMA delay

Decentralized drone swarm: an offline mean-field-game (MFG) routing field plus a per-agent CBF-QP safety filter.
Agents see neighbours through noisy UWB range / bearing, broadcast over TDMA (T_frame = N * t_slot), and may
compensate the stale neighbour estimates.

## Layout
```
mfg/           offline MFG solve (HJB, transport, fictitious play, Cesaro)
env/           grid, scenarios (SingleGateA, LayoutA), closed-loop rollout
control/       constants and parameters, macro control law, CBF-QP
compensation/  perception + TDMA model and the compensation arms
sim/           setup (scenario + cached MFG solve), run (one cell), sweep (an experiment file), metrics
analysis/      report: per-seed metrics -> summary.csv with bootstrap CIs, paired differences
configs/       base.yaml, geometry/*.yaml, exp/*.yaml
weights/       predictor_v3.npz (learned arm, pooled over N = 25..200)
```

## Arms
`real` (stale + noisy), `noise_only` (noise, no delay), `fresh` (exact positions), `cv`, `cvls`, `smith`,
`learned`, and `split:<MFG arm>/<CBF arm>` (different estimate per layer).

## Run
```bash
python -m venv .venv && . .venv/bin/activate && pip install -e . && pip install "jax[cuda12]"
python -m sim.sweep configs/exp/example.yaml --list
python -m sim.sweep configs/exp/example.yaml            # restartable: finished seeds are skipped
python -m analysis.report results/example
python -m analysis.report results/example --paired N100_learned N100_real_d0.70
pytest
```
An experiment file = geometry + seeds + blocks (Cartesian products of config keys); see `sim/config.py`.
MFG solves are cached in `results/cache` (set `MUSC_CACHE` to move it).

## Results written per seed
`results/<experiment>/<cell>/`: `config.yaml` (resolved config), `traj_seed*.npz` (full rollout log incl. true
positions every step — the raw archive) and `metrics_seed*.json`:
- `ever_collided` — fraction of agents ever closer than 0.30 m to another (whole run)
- `safe_settled_{3,5,10}s_b{0.00,0.30}` / `completeness_settled_…` — inside goal radius + band for ≥ 90 % of the
  last W s (and never collided); headline: 5 s, 0.30 m band
- `safe_T`, `completeness_T` — at the final instant only (over-counts last-second arrivals)
- in-run CBF / separation statistics and provenance (precision, t_slot, noise, cache key)
