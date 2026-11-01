"""Experiment E1 -- free-motion trajectory tracking.

Controllers
  pd   independent-joint PD, no model                            (model-free baseline)
  ct   computed torque with the nominal URDF model                 (model-based, free motion)
  lnn  learned Lagrangian model + PD, Liu et al. (2024) Eq. 25     (model-free, free motion)
       (PD gains identical to `pd`, so the difference isolates the learned feed-forward)

Trajectories (fixed seeds, never used for training)
  pick_place    quintic segments through 6 task-space waypoints (IK), 1 s per segment
  sine_in_dist  multi-sine, 0.05-0.5 Hz  (same distribution as the LNN training data)
  sine_fast     multi-sine, 0.5-1.0 Hz   (outside the training distribution)

Conditions
  nominal   plant = nominal model
  payload   plant carries an extra mass (default 2 kg, 5 cm below tool0) that neither the
            CT model nor the LNN training data contain

Output: results/e1/metrics.csv and one .npz log per run (+ plots via plot_e1.py)

Examples
  python experiments/e1_free_motion.py --lnn results/lnn/lnn_best.pt
  python experiments/e1_free_motion.py --controllers pd ct --conditions nominal   # no LNN yet
  python experiments/e1_free_motion.py --controllers ct --trajectories pick_place --gui
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, test_trajectories, write_csv  # noqa: E402

from cmr_control.controllers.computed_torque import ComputedTorque  # noqa: E402
from cmr_control.controllers.lnn_controller import LNNTrackingController  # noqa: E402
from cmr_control.controllers.pd import KD_DEFAULT, KP_DEFAULT, JointPID  # noqa: E402
from cmr_control.dynamics import RigidBodyModel  # noqa: E402
from cmr_control.sim import Plant  # noqa: E402
from cmr_control.trajectories import sample  # noqa: E402


def run(plant, kin, traj, controller, name, lnn_ctrl=None):
    dt = plant.dt
    t, Qd, dQd, ddQd = sample(traj, dt)
    plant.set_q(Qd[0], dQd[0])
    if name == "lnn":
        lnn_ctrl.precompute(Qd, dQd, ddQd)
    n = len(t)
    Q, dQ, TAU = np.zeros((n, 6)), np.zeros((n, 6)), np.zeros((n, 6))
    EE, EEd = np.zeros((n, 3)), np.zeros((n, 3))
    for k in range(n):
        q, dq = plant.state()
        if name == "pd":
            tau = controller(q, dq, Qd[k], dQd[k])
        elif name == "ct":
            tau = controller(q, dq, Qd[k], dQd[k], ddQd[k])
        elif name == "lnn":
            tau = lnn_ctrl(k, q, dq, Qd[k], dQd[k])
        TAU[k] = plant.apply_torque(tau)
        Q[k], dQ[k] = q, dq
        if k % 10 == 0:  # task-space error every 10 ms (FK of measured and desired joints)
            EE[k] = plant.ee_pose()[0]
            EEd[k] = kin.fk(Qd[k])[0]
        plant.step()
        if not np.all(np.isfinite(q)) or np.abs(q - Qd[k]).max() > 1.5:
            print(f"   ! {name} diverged at t={t[k]:.2f}s", flush=True)
            Q, dQ, TAU, EE, EEd, t, Qd = (x[:k + 1] for x in (Q, dQ, TAU, EE, EEd, t, Qd))
            break
    m = np.arange(0, len(t), 10)
    e = Qd - Q
    e_ee = np.linalg.norm(EE[m] - EEd[m], axis=1)
    metrics = {
        "rmse_q_rad": float(np.sqrt(np.mean(e ** 2))),
        "max_q_err_rad": float(np.abs(e).max()),
        "rmse_ee_mm": float(1000 * np.sqrt(np.mean(e_ee ** 2))),
        "max_ee_mm": float(1000 * e_ee.max()),
        "rms_tau_Nm": float(np.sqrt(np.mean(TAU ** 2))),
        "saturation_pct": float(100 * np.mean(np.abs(TAU) >= plant.tau_max - 1e-9)),
        "completed": bool(len(t) == len(sample(traj, dt)[0])),
        **{f"rmse_q{j + 1}": float(np.sqrt(np.mean(e[:, j] ** 2))) for j in range(6)},
    }
    log = dict(t=t, q=Q, q_d=Qd, dq=dQ, tau=TAU, ee=EE[m], ee_d=EEd[m], t_ee=t[m])
    return metrics, log


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controllers", nargs="+", default=["pd", "ct", "lnn"], choices=["pd", "ct", "lnn"])
    ap.add_argument("--trajectories", nargs="+", default=["pick_place", "sine_in_dist", "sine_fast"])
    ap.add_argument("--conditions", nargs="+", default=["nominal", "payload"], choices=["nominal", "payload"])
    ap.add_argument("--payload-mass", type=float, default=2.0)
    ap.add_argument("--lnn", type=Path, default=RESULTS / "lnn" / "lnn_best.pt")
    ap.add_argument("--ct-omega", type=float, default=25.0, help="CT closed-loop bandwidth [rad/s]")
    ap.add_argument("--out", type=Path, default=RESULTS / "e1")
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    lnn_ctrl = None
    if "lnn" in args.controllers:
        if not args.lnn.exists():
            sys.exit(f"LNN checkpoint {args.lnn} not found (train_lnn.py) -- or drop 'lnn' from --controllers")
        from cmr_control.learning.lnn import load_checkpoint
        lnn, _ = load_checkpoint(args.lnn)
        lnn_ctrl = LNNTrackingController(lnn.double(), KP_DEFAULT, KD_DEFAULT)

    rows = []
    for cond in args.conditions:
        m_pay = args.payload_mass if cond == "payload" else 0.0
        plant = Plant(tool="payload" if cond == "payload" else "none", payload_mass=m_pay, gui=args.gui)
        plant.realtime = args.gui
        nominal = RigidBodyModel(tool="none")     # what the model-based controller believes
        trajs = test_trajectories(nominal)
        ctrls = {"pd": JointPID(KP_DEFAULT, KD_DEFAULT), "ct": ComputedTorque(nominal, omega=[args.ct_omega] * 6)}
        for tname in args.trajectories:
            for cname in args.controllers:
                t0 = time.time()
                metrics, log = run(plant, nominal, trajs[tname], ctrls.get(cname), cname, lnn_ctrl)
                row = {"trajectory": tname, "condition": cond, "controller": cname, **metrics}
                rows.append(row)
                np.savez_compressed(args.out / f"{tname}_{cond}_{cname}.npz", **log)
                print(f"{tname:13s} {cond:8s} {cname:4s}  RMSE q {metrics['rmse_q_rad']:.4f} rad  "
                      f"EE {metrics['rmse_ee_mm']:7.2f} mm (max {metrics['max_ee_mm']:7.2f})  "
                      f"sat {metrics['saturation_pct']:.1f}%  [{time.time() - t0:.0f}s]", flush=True)
                write_csv(args.out / "metrics.csv", rows)
        nominal.close()
        plant.close()


if __name__ == "__main__":
    main()
