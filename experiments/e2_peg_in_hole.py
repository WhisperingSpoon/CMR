"""Experiment E2 -- peg-in-hole under hole-position uncertainty.

For every hole-position error r in --offsets (mm) and every trial i, the scene is generated
from seed = 1000 * offset_index + i: error direction, start pose and therefore the task are
identical for all controllers. Friction is fixed (--friction) during evaluation.

Controllers: impedance, impedance_stiff, pid_scripted, ppo (needs --ppo-model)

Metrics per trial: success, time to success, peak contact force (1 kHz), mean contact force
while in contact, final insertion depth and lateral error. Output: results/e2/trials*.csv and
example logs (first trial of every controller/offset).

Examples
  python experiments/e2_peg_in_hole.py --controllers impedance impedance_stiff pid_scripted
  python experiments/e2_peg_in_hole.py --ppo-model results/ppo/seed0/best_model.zip
  python experiments/e2_peg_in_hole.py --controllers impedance --offsets 3 --trials 1 --gui
  # split over a Slurm array: --shard $SLURM_ARRAY_TASK_ID --n-shards 8
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, write_csv  # noqa: E402

from cmr_control.controllers import peg_strategies as ps  # noqa: E402
from cmr_control.dynamics import RigidBodyModel  # noqa: E402
from cmr_control.learning.peg_env import PegInHoleEnv  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controllers", nargs="+", default=["impedance", "impedance_stiff", "pid_scripted", "ppo"],
                    choices=["impedance", "impedance_stiff", "pid_scripted", "ppo"])
    ap.add_argument("--offsets", nargs="+", type=float, default=[0, 1, 2, 3, 4, 5, 6], help="mm")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--friction", type=float, default=0.5)
    ap.add_argument("--ppo-model", type=Path, default=RESULTS / "ppo" / "seed0" / "best_model.zip")
    ap.add_argument("--model-mass-scale", type=float, default=1.0,
                    help="scale all link masses of the impedance controller's model (model-error study)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--out", type=Path, default=RESULTS / "e2")
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "logs").mkdir(exist_ok=True)

    env = PegInHoleEnv(gui=args.gui)   # plant first (pybullet client 0)
    model = RigidBodyModel(tool="peg", link_mass_scale=args.model_mass_scale)

    policy = None
    if "ppo" in args.controllers:
        if not args.ppo_model.exists():
            sys.exit(f"PPO model {args.ppo_model} not found (train_ppo.py) -- or drop 'ppo' from --controllers")
        from stable_baselines3 import PPO
        policy = PPO.load(args.ppo_model, device="cpu")

    jobs = [(oi, r, i) for oi, r in enumerate(args.offsets) for i in range(args.trials)]
    jobs = jobs[args.shard::args.n_shards]
    rows = []
    for oi, r_mm, i in jobs:
        seed = 1000 * oi + i
        opts = {"offset": r_mm / 1000.0, "friction": args.friction}
        for c in args.controllers:
            t0 = time.time()
            if c == "impedance":
                m, log = ps.run_impedance(env, seed, opts, model, ps.IMPEDANCE_COMPLIANT)
            elif c == "impedance_stiff":
                m, log = ps.run_impedance(env, seed, opts, model, ps.IMPEDANCE_STIFF)
            elif c == "pid_scripted":
                m, log = ps.run_pid_scripted(env, seed, opts)
            else:
                m, log = ps.run_policy(env, seed, opts, policy)
            rows.append({"controller": c, "offset_mm": r_mm, "trial": i, "seed": seed, **m})
            if i == 0:
                np.savez_compressed(args.out / "logs" / f"{c}_off{r_mm:g}mm.npz", **log)
            print(f"{c:15s} offset {r_mm:3.1f} mm trial {i:2d}: success {int(m['success'])} "
                  f"t {m['t_success']:.2f}s  Fpeak {m['f_peak']:7.1f} N  depth {1000 * m['final_depth']:5.1f} mm "
                  f"[{time.time() - t0:.1f}s]", flush=True)
        # one file per controller set (and shard) so separate runs never overwrite each other
        tag = "-".join(args.controllers) + (f"_shard{args.shard}" if args.n_shards > 1 else "")
        write_csv(args.out / f"trials_{tag}.csv", rows)
    model.close()
    env.close()


if __name__ == "__main__":
    main()
