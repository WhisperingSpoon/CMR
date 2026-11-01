"""Shared definitions for the experiments (trajectories, paths, plotting style)."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import sys

import numpy as np

# Make the repository importable even if `pip install -e .` was skipped or failed
# (e.g. a cluster venv): cmr_control and ur_simulation live in the repo root.
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cmr_control.dynamics import RigidBodyModel  # noqa: E402
from cmr_control.learning.peg_env import QUAT_DOWN
from cmr_control.sim import Q_HOME
from cmr_control.trajectories import MultiSine, QuinticWaypoints, min_link_height

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("CMR_RESULTS", ROOT / "results"))
DATA = Path(os.environ.get("CMR_DATA", ROOT / "data"))

# Free-motion workspace: tool pointing down, centred 45 cm in front of the base, 35 cm high.
P_CENTER = np.array([0.0, 0.45, 0.35])
PICK_PLACE_POINTS = np.array([
    [0.00, 0.45, 0.35],
    [0.25, 0.40, 0.25],
    [0.25, 0.55, 0.45],
    [-0.20, 0.50, 0.30],
    [-0.20, 0.35, 0.45],
    [0.00, 0.45, 0.35],
])
# Excitation amplitudes per joint [rad] (sum over harmonics); shared by training data and tests.
EXCITATION_AMP = (0.6, 0.4, 0.4, 0.6, 0.6, 0.8)
MIN_CLEARANCE = 0.08  # m, minimum height of any link frame above the floor


def q_center(model: RigidBodyModel | None = None):
    own = model is None
    model = model or RigidBodyModel(tool="none")
    q = model.ik(P_CENTER, QUAT_DOWN, Q_HOME)
    if own:
        model.close()
    return q


def pick_and_place(model: RigidBodyModel, segment_time=1.0, hold=0.3):
    """Task-space waypoints -> joint waypoints (DLS IK, chained seeds) -> quintic segments."""
    qs, seed = [], Q_HOME
    for p in PICK_PLACE_POINTS:
        seed = model.ik(p, QUAT_DOWN, seed)
        qs.append(seed)
    return QuinticWaypoints(np.array(qs), segment_time, hold=hold)


def safe_multisine(model, rng, q0, duration, fmax, fmin=0.05, amp=EXCITATION_AMP, max_tries=200):
    """Random multi-sine around q0 whose links stay MIN_CLEARANCE above the floor."""
    for _ in range(max_tries):
        traj = MultiSine.random(rng, q0, duration, amp=amp, fmax=fmax, fmin=fmin)
        t = np.arange(0, duration, 0.05)
        Q = np.array([traj(ti)[0] for ti in t])
        if min_link_height(model, Q, stride=1) > MIN_CLEARANCE:
            return traj
    raise RuntimeError("could not find a floor-safe excitation trajectory")


def test_trajectories(model):
    """The three free-motion test trajectories of experiment E1 (fixed seeds, not used in training)."""
    q0 = q_center(model)
    return {
        "pick_place": pick_and_place(model),
        "sine_in_dist": safe_multisine(model, np.random.default_rng(424242), q0, 20.0, fmax=0.5),
        "sine_fast": safe_multisine(model, np.random.default_rng(737373), q0, 20.0, fmax=1.0, fmin=0.5),
    }


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if v in ("True", "False"):
                r[k] = 1.0 if v == "True" else 0.0
                continue
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    def _default(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return str(x)

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_default)


def mpl_style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 8, "axes.titlesize": 8,
        "axes.labelsize": 8, "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3, "lines.linewidth": 1.0,
    })
    return plt


# consistent colours per controller across all figures
COLORS = {
    "pd": "#7f7f7f", "ct": "#1f77b4", "lnn": "#d62728",
    "impedance": "#1f77b4", "impedance_stiff": "#9ecae1", "pid_scripted": "#7f7f7f", "ppo": "#d62728",
}
LABELS = {
    "pd": "PD (model-free baseline)", "ct": "Computed torque (MB)", "lnn": "LNN + Eq. 25 (MF)",
    "impedance": "Cartesian impedance (MB)", "impedance_stiff": "Stiff impedance (MB baseline)",
    "pid_scripted": "Scripted PID (MF baseline)", "ppo": "PPO (MF)",
}
