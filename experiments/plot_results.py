"""Report figures (PDF, sized for a two-column IEEE-style paper: 3.5 in wide).

  python experiments/plot_results.py e1     # tracking error bars + time series
  python experiments/plot_results.py lnn    # LNN loss curve, learned vs true gravity / torque
  python experiments/plot_results.py ppo    # PPO learning curve (reward, success rate)
  python experiments/plot_results.py e2     # success rate / peak force vs hole error, force profiles
  python experiments/plot_results.py all
Also writes LaTeX tables (results/tables/*.tex) with the key numbers.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import COLORS, LABELS, RESULTS, mpl_style, read_csv  # noqa: E402

FIG = RESULTS / "figures"
TAB = RESULTS / "tables"
W = 3.5  # inches, one column


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=200)
    print("wrote", FIG / f"{name}.pdf")


# --------------------------------------------------------------------------------- E1
def plot_e1():
    plt = mpl_style()
    rows = read_csv(RESULTS / "e1" / "metrics.csv")
    trajs = list(dict.fromkeys(r["trajectory"] for r in rows))
    conds = list(dict.fromkeys(r["condition"] for r in rows))
    ctrls = [c for c in ["pd", "ct", "lnn"] if any(r["controller"] == c for r in rows)]

    fig, axes = plt.subplots(1, len(conds), figsize=(W * len(conds) / 1.4, 2.0), sharey=True, squeeze=False)
    for ax, cond in zip(axes[0], conds):
        x = np.arange(len(trajs))
        wbar = 0.8 / len(ctrls)
        for j, c in enumerate(ctrls):
            v = [next((r["rmse_ee_mm"] for r in rows if r["trajectory"] == t and r["condition"] == cond
                       and r["controller"] == c), np.nan) for t in trajs]
            ax.bar(x + (j - (len(ctrls) - 1) / 2) * wbar, v, wbar, color=COLORS[c], label=LABELS[c])
        ax.set_xticks(x, [t.replace("_", "\n") for t in trajs])
        ax.set_yscale("log")
        ax.set_title(cond)
    axes[0, 0].set_ylabel("EE RMSE [mm]")
    axes[0, -1].legend(loc="upper left", frameon=False)
    _save(fig, "e1_rmse_bars")

    for tname in trajs:
        fig, axes = plt.subplots(2, 1, figsize=(W, 2.6), sharex=True)
        for ax, cond in zip(axes, conds):
            for c in ctrls:
                f = RESULTS / "e1" / f"{tname}_{cond}_{c}.npz"
                if not f.exists():
                    continue
                d = np.load(f)
                e = 1000 * np.linalg.norm(d["ee"] - d["ee_d"], axis=1)
                ax.plot(d["t_ee"], e, color=COLORS[c], label=LABELS[c])
            ax.set_yscale("log")
            ax.set_ylabel(f"{cond}\n|e_EE| [mm]")
        axes[-1].set_xlabel("t [s]")
        axes[0].legend(frameon=False, ncol=1)
        _save(fig, f"e1_timeseries_{tname}")

    TAB.mkdir(parents=True, exist_ok=True)
    with open(TAB / "e1.tex", "w") as f:
        f.write("% EE RMSE [mm] (max [mm]) per trajectory and condition\n")
        f.write("\\begin{tabular}{ll" + "r" * len(ctrls) + "}\n\\toprule\n")
        f.write("Trajectory & Condition & " + " & ".join(c.upper() for c in ctrls) + "\\\\\n\\midrule\n")
        for t in trajs:
            for cond in conds:
                cells = []
                for c in ctrls:
                    r = next((r for r in rows if r["trajectory"] == t and r["condition"] == cond and r["controller"] == c), None)
                    cells.append("--" if r is None else f"{r['rmse_ee_mm']:.2f} ({r['max_ee_mm']:.1f})")
                f.write(f"{t.replace('_', ' ')} & {cond} & " + " & ".join(cells) + "\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print("wrote", TAB / "e1.tex")


# -------------------------------------------------------------------------------- LNN
def plot_lnn():
    plt = mpl_style()
    d = RESULTS / "lnn"
    curve = read_csv(d / "loss_curve.csv")
    fig, axes = plt.subplots(1, 3, figsize=(2 * W, 1.8))
    ep = [r["epoch"] for r in curve]
    axes[0].semilogy(ep, [r["train_loss"] for r in curve], label="train")
    axes[0].semilogy(ep, [r["val_loss"] for r in curve], label="validation")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("normalised next-state loss")
    axes[0].legend(frameon=False)
    s = np.load(d / "lnn_eval_samples.npz")
    for j, mk in zip(range(3), "o^s"):
        axes[1].scatter(s["g_true"][:, j], s["g_pred"][:, j], s=2, marker=mk, label=f"joint {j + 1}")
    lim = [s["g_true"][:, :3].min(), s["g_true"][:, :3].max()]
    axes[1].plot(lim, lim, "k--", lw=0.6)
    axes[1].set_xlabel("true $g_i(q)$ [Nm]")
    axes[1].set_ylabel("learned $\\partial V_L/\\partial q_i$ [Nm]")
    axes[1].legend(frameon=False, markerscale=3)
    for j in range(3):
        axes[2].scatter(s["tau"][:, j], s["tau_pred"][:, j], s=2, label=f"joint {j + 1}")
    lim = [s["tau"][:, :3].min(), s["tau"][:, :3].max()]
    axes[2].plot(lim, lim, "k--", lw=0.6)
    axes[2].set_xlabel("measured $\\tau_i$ [Nm]")
    axes[2].set_ylabel("LNN inverse dynamics [Nm]")
    fig.tight_layout()
    _save(fig, "lnn_model_quality")


# -------------------------------------------------------------------------------- PPO
def plot_ppo():
    plt = mpl_style()
    runs = sorted(glob.glob(str(RESULTS / "ppo" / "seed*")))
    fig, axes = plt.subplots(1, 2, figsize=(W * 1.6, 1.8))
    for r in runs:
        ev = Path(r) / "eval" / "evaluations.npz"
        if not ev.exists():
            continue
        e = np.load(ev)
        axes[0].plot(e["timesteps"] / 1e6, e["results"].mean(1), label=Path(r).name)
        if "successes" in e:
            axes[1].plot(e["timesteps"] / 1e6, 100 * e["successes"].mean(1), label=Path(r).name)
    axes[0].set_xlabel("env steps [M]")
    axes[0].set_ylabel("eval return")
    axes[1].set_xlabel("env steps [M]")
    axes[1].set_ylabel("eval success [%]")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    _save(fig, "ppo_learning_curve")


# --------------------------------------------------------------------------------- E2
def plot_e2():
    plt = mpl_style()
    files = sorted(glob.glob(str(RESULTS / "e2" / "trials*.csv")))
    # merge all trial files; a later file overrides the same (controller, offset, trial)
    merged = {}
    for f in sorted(files, key=lambda p: Path(p).stat().st_mtime):
        for r in read_csv(f):
            merged[(r["controller"], r["offset_mm"], r["trial"])] = r
    rows = list(merged.values())
    ctrls = [c for c in ["impedance", "impedance_stiff", "pid_scripted", "ppo"] if any(r["controller"] == c for r in rows)]
    offs = sorted({r["offset_mm"] for r in rows})

    def stat(c, o, key, fn):
        v = np.array([r[key] for r in rows if r["controller"] == c and r["offset_mm"] == o], float)
        return fn(v) if len(v) else np.nan

    fig, axes = plt.subplots(1, 3, figsize=(2 * W, 1.8))
    for c in ctrls:
        sr = [100 * stat(c, o, "success", np.mean) for o in offs]
        fp = [stat(c, o, "f_peak", np.median) for o in offs]
        ts = [stat(c, o, "t_success", np.nanmedian) for o in offs]
        kw = dict(color=COLORS[c], marker="o", ms=3, label=LABELS[c])
        axes[0].plot(offs, sr, **kw)
        axes[1].plot(offs, fp, **kw)
        axes[2].plot(offs, ts, **kw)
    axes[0].set_ylabel("success rate [%]")
    axes[1].set_ylabel("median peak force [N]")
    axes[1].set_yscale("log")
    axes[2].set_ylabel("median insertion time [s]")
    for ax in axes:
        ax.set_xlabel("hole position error [mm]")
    axes[0].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    _save(fig, "e2_summary")

    # example force / depth profiles at one representative offset
    o = 3.0 if 3.0 in offs else offs[len(offs) // 2]
    fig, axes = plt.subplots(2, 1, figsize=(W, 2.6), sharex=True)
    for c in ctrls:
        f = RESULTS / "e2" / "logs" / f"{c}_off{o:g}mm.npz"
        if not f.exists():
            continue
        d = np.load(f)
        axes[0].plot(d["t"], np.linalg.norm(d["f_contact"], axis=1), color=COLORS[c], label=LABELS[c])
        axes[1].plot(d["t"], 1000 * d["depth"], color=COLORS[c])
    axes[0].set_ylabel("|F_contact| [N]")
    axes[1].set_ylabel("tip depth below\nhole top [mm]")
    axes[1].set_xlabel("t [s]")
    axes[0].set_title(f"hole error {o:g} mm, trial 0")
    axes[0].legend(frameon=False)
    _save(fig, "e2_example_profiles")

    TAB.mkdir(parents=True, exist_ok=True)
    with open(TAB / "e2.tex", "w") as f:
        f.write("% success rate [%] / median peak force [N] per hole-position error\n")
        f.write("\\begin{tabular}{l" + "r" * len(offs) + "}\n\\toprule\n")
        f.write("Controller & " + " & ".join(f"{o:g} mm" for o in offs) + "\\\\\n\\midrule\n")
        for c in ctrls:
            cells = [f"{100 * stat(c, o, 'success', np.mean):.0f} / {stat(c, o, 'f_peak', np.median):.0f}" for o in offs]
            f.write(f"{LABELS[c]} & " + " & ".join(cells) + "\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print("wrote", TAB / "e2.tex")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in [("e1", plot_e1), ("lnn", plot_lnn), ("ppo", plot_ppo), ("e2", plot_e2)]:
        if what in (name, "all"):
            try:
                fn()
            except FileNotFoundError as e:
                print(f"skip {name}: {e}")
