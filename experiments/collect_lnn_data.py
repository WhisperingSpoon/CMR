"""Collect training data for the Lagrangian NN (experiment E1, model-free branch).

Following Liu et al. (2024, Sec. 4.1): the robot tracks random multi-sine joint references
("different input signals ... encompass a significant portion of the workspace"). Tracking uses
the *model-free* joint PID only (no URDF dynamics), and the applied torques, joint positions
and velocities are logged at 1 kHz with zero-order-hold torques.

Each output file holds one trajectory:  q[k], dq[k], tau[k] (applied during [k, k+1)).

Examples
--------
  python experiments/collect_lnn_data.py --n-traj 70 --duration 20 --seed 0
  # Slurm array: 10 tasks x 7 trajectories, distinct seeds
  python experiments/collect_lnn_data.py --n-traj 7 --seed $((1000 + SLURM_ARRAY_TASK_ID))
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, q_center, safe_multisine  # noqa: E402

from cmr_control.controllers.pd import KD_DEFAULT, KI_DEFAULT, KP_DEFAULT, JointPID  # noqa: E402
from cmr_control.dynamics import RigidBodyModel  # noqa: E402
from cmr_control.sim import Plant  # noqa: E402


def collect_one(plant, traj, pid, noise_std=0.0, rng=None):
    q0, _, _ = traj(0.0)
    plant.set_q(q0)
    pid.reset()
    n = int(traj.duration / plant.dt)
    Q, dQ, T, Qr = np.zeros((n, 6)), np.zeros((n, 6)), np.zeros((n, 6)), np.zeros((n, 6))
    for k in range(n):
        q, dq = plant.state()
        q_r, dq_r, _ = traj(k * plant.dt)
        q_meas = q + (rng.normal(0, noise_std, 6) if noise_std > 0 else 0.0)
        tau = plant.apply_torque(pid(q_meas, dq, q_r, dq_r))
        Q[k], dQ[k], T[k], Qr[k] = q, dq, tau, q_r
        plant.step()
    return Q, dQ, T, Qr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-traj", type=int, default=70)
    ap.add_argument("--duration", type=float, default=20.0, help="seconds per trajectory")
    ap.add_argument("--fmax", type=float, default=0.5, help="max excitation frequency [Hz]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise-std", type=float, default=0.0, help="encoder noise [rad] seen by the PID")
    ap.add_argument("--out", type=Path, default=DATA / "lnn")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    plant = Plant(tool="none")              # ground truth (created first: pybullet client 0)
    kin = RigidBodyModel(tool="none")       # kinematics only: IK of the centre + floor check
    q0 = q_center(kin)
    pid = JointPID(KP_DEFAULT, KD_DEFAULT, KI_DEFAULT, i_limit=np.full(6, 0.2), dt=plant.dt)

    rng = np.random.default_rng(args.seed)
    for i in range(args.n_traj):
        t0 = time.time()
        traj = safe_multisine(kin, rng, q0, args.duration, fmax=args.fmax)
        Q, dQ, T, Qr = collect_one(plant, traj, pid, args.noise_std, rng)
        name = args.out / f"traj_s{args.seed}_{i:03d}.npz"
        np.savez_compressed(name, q=Q, dq=dQ, tau=T, q_ref=Qr, dt=plant.dt,
                            amps=traj.a, freqs=traj.w / (2 * np.pi), phases=traj.phi)
        sat = np.mean(np.abs(T) >= plant.tau_max - 1e-9)
        print(f"[{i + 1}/{args.n_traj}] {name.name}: track RMSE {np.sqrt(np.mean((Qr - Q) ** 2)):.4f} rad, "
              f"|tau|max {np.abs(T).max(0).round(1)}, saturated {100 * sat:.2f}%, {time.time() - t0:.1f}s",
              flush=True)
    kin.close()
    plant.close()


if __name__ == "__main__":
    main()
