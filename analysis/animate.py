"""Animate one or more cells side by side from their saved trajectories (traj_seed<k>.npz). Writes nothing else.

Per panel: arena walls, goal disc (solid), flat core if the scenario has one (dashed), agents as 0.30 m discs:
  grey  = not yet arrived      green = inside the goal disc
  red   = in contact NOW (centre distance < 0.30 m)      black ring = has collided at some point (ever_collided)
Title: cell, time, arrived / N, ever-collided / N.

    pip install matplotlib                     # once (Pillow comes with it; .mp4 also needs ffmpeg)
    python -m analysis.animate results/crowding_test/N100_real_cone_hold results/crowding_test/N100_real_core75 \
        --seed 0 --out ~/musc_det/hold_vs_core.gif --tmax 40 --speed 2
"""
import argparse
import os

import numpy as np
import yaml

from sim.setup import build_scenario

D_COLL = 0.30


def load(cell, seed):
    d = np.load(os.path.join(cell, f"traj_seed{seed}.npz"))
    cfg = yaml.safe_load(open(os.path.join(cell, "config.yaml")))
    sc = build_scenario(cfg, cfg["N"])
    pos = np.concatenate([d["pos"], d["pos_final"][None]])
    contact = np.zeros(pos.shape[:2], bool)
    for k in range(pos.shape[0]):
        D = np.linalg.norm(pos[k][:, None] - pos[k][None], axis=-1)
        np.fill_diagonal(D, np.inf)
        contact[k] = D.min(1) < D_COLL
    ever = np.logical_or.accumulate(contact, axis=0)
    inside = np.linalg.norm(pos - np.asarray(sc.goal), axis=-1) <= sc.r_goal
    return {"name": os.path.basename(cell.rstrip("/")), "sc": sc, "pos": pos, "contact": contact,
            "ever": ever, "inside": inside, "dt": cfg["dt_exec"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cells", nargs="+", help="one or more result cell directories (shown side by side)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="animation.gif", help=".gif (Pillow) or .mp4 (ffmpeg)")
    ap.add_argument("--tmax", type=float, default=40.0, help="simulated seconds to show")
    ap.add_argument("--speed", type=float, default=2.0, help="simulated seconds per real second")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--dpi", type=int, default=80)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.collections import EllipseCollection
    from matplotlib.patches import Circle

    runs = [load(c, a.seed) for c in a.cells]
    dt = runs[0]["dt"]
    stride = max(1, int(round(a.speed / (a.fps * dt))))
    n_frames = min(int(a.tmax / dt), min(r["pos"].shape[0] for r in runs) - 1) // stride + 1

    W, H = runs[0]["sc"].arena
    fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs) * W / max(W, H) * 1.0, 5.2 * H / max(W, H) + 0.6),
                             squeeze=False)
    artists = []
    for ax, r in zip(axes[0], runs):
        sc = r["sc"]
        for (x0, z0), (x1, z1) in sc.wall_segments():
            ax.plot([x0, x1], [z0, z1], color="k", lw=2.5, solid_capstyle="butt")
        ax.add_patch(Circle(sc.goal, sc.r_goal, fill=False, color="tab:green", lw=1.5))
        if getattr(sc, "goal_core", None):
            ax.add_patch(Circle(sc.goal, sc.goal_core, fill=False, color="tab:green", lw=1.0, ls="--"))
        ax.set_xlim(0, sc.arena[0]); ax.set_ylim(0, sc.arena[1]); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        N = r["pos"].shape[1]
        agents = EllipseCollection(np.full(N, D_COLL), np.full(N, D_COLL), np.zeros(N), units="xy",
                                   offsets=r["pos"][0], offset_transform=ax.transData, linewidths=0.8)
        ax.add_collection(agents)
        title = ax.set_title("", fontsize=9)
        artists.append((r, agents, title))
    fig.tight_layout()

    def frame(i):
        k = i * stride
        out = []
        for r, agents, title in artists:
            col = np.where(r["contact"][k], "red", np.where(r["inside"][k], "tab:green", "0.55"))
            agents.set_offsets(r["pos"][k])
            agents.set_facecolor(col)
            agents.set_edgecolor(np.where(r["ever"][k], "k", "none"))
            N = r["pos"].shape[1]
            title.set_text(f"{r['name']}  seed {a.seed}\n t = {k * dt:5.1f} s   arrived {int(r['inside'][k].sum())}/{N}"
                           f"   ever collided {int(r['ever'][k].sum())}/{N}")
            out += [agents, title]
        return out

    anim = animation.FuncAnimation(fig, frame, frames=n_frames, interval=1000 / a.fps, blit=False)
    out = os.path.expanduser(a.out)
    writer = animation.PillowWriter(fps=a.fps) if out.endswith(".gif") else animation.FFMpegWriter(fps=a.fps, bitrate=2400)
    anim.save(out, writer=writer, dpi=a.dpi)
    print(f"wrote {out}: {n_frames} frames, {n_frames / a.fps:.1f} s at {a.speed:g}x")


if __name__ == "__main__":
    main()
