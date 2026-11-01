"""Train the Lagrangian neural network (Liu et al. 2024) on data from collect_lnn_data.py.

Loss: next-state prediction through RK4 integration of the learned Euler-Lagrange dynamics
(Liu et al. 2024, Eq. 8), normalised per joint. Validation is on held-out *trajectories*.

Note on w_q (weight of the position term of Eq. 8): pybullet integrates with semi-implicit
Euler, so the logged position increments differ from an exact (RK4) integration by ~a dt^2/2,
which is the same order as the information the position term carries at dt = 1 ms. The
default therefore uses the velocity term only (w_q = 0); set --w-q 1 to use Eq. 8 verbatim.

After training, the learned model is compared with the ground-truth URDF model
(gravity vector, mass matrix) -- this is evaluation only, the training never sees it.

Example
-------
  python experiments/train_lnn.py --data data/lnn --epochs 200 --device cuda --out results/lnn
"""
from __future__ import annotations

import argparse
import glob
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, RESULTS, save_json, write_csv  # noqa: E402

from cmr_control.learning.lnn import LagrangianNN, next_state_loss, save_checkpoint  # noqa: E402


def load_pairs(files, stride):
    """(q_k, dq_k, tau_k, q_{k+1}-q_k, dq_{k+1}-dq_k) with differences computed in float64."""
    out = [[] for _ in range(5)]
    dt = None
    for f in files:
        d = np.load(f)
        dt = float(d["dt"])
        q, dq, tau = d["q"], d["dq"], d["tau"]
        idx = np.arange(0, len(q) - 1, stride)
        for lst, arr in zip(out, (q[idx], dq[idx], tau[idx], q[idx + 1] - q[idx], dq[idx + 1] - dq[idx])):
            lst.append(arr)
    return [np.concatenate(x) for x in out], dt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA / "lnn")
    ap.add_argument("--out", type=Path, default=RESULTS / "lnn")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--stride", type=int, default=5, help="use every n-th time step as a training pair")
    ap.add_argument("--w-q", type=float, default=0.0)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--max-train-traj", type=int, default=0, help="0 = all (for data-efficiency studies)")
    ap.add_argument("--no-damping", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    args.out.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(args.data / "*.npz")))
    if not files:
        sys.exit(f"no data in {args.data}; run collect_lnn_data.py first")
    rng.shuffle(files)
    n_val = max(1, int(round(args.val_frac * len(files))))
    val_files, train_files = files[:n_val], files[n_val:]
    if args.max_train_traj:
        train_files = train_files[: args.max_train_traj]
    print(f"{len(train_files)} train / {len(val_files)} val trajectories, device={device}", flush=True)

    (tr, dt), (va, _) = load_pairs(train_files, args.stride), load_pairs(val_files, args.stride)
    acc_scale = np.maximum((tr[4] / dt).std(0), 1e-3)
    tau_scale = float(np.abs(tr[2]).mean())

    to_t = lambda arrs: [torch.as_tensor(a, dtype=torch.float32, device=device) for a in arrs]
    tr_t, va_t = to_t(tr), to_t(va)
    acc_t = torch.as_tensor(acc_scale, dtype=torch.float32, device=device)

    arch = dict(width=args.width, depth=args.depth, eps_m=1e-5, v_scale=max(tau_scale, 1.0), m_scale=1.0,
                d_scale=0.1, learn_damping=not args.no_damping)
    model = LagrangianNN(**arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n = len(tr_t[0])
    steps_per_epoch = math.ceil(n / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch)

    def evaluate(data):
        model.eval()
        tot, m = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(data[0]), 8192):
                b = [x[i:i + 8192] for x in data]
                l, _, _ = next_state_loss(model, b, dt, acc_t, args.w_q)
                tot += l.item() * len(b[0])
                m += len(b[0])
        model.train()
        return tot / m

    curve, best = [], np.inf
    meta = {"arch": arch, "dt": dt, "acc_scale": acc_scale.tolist(), "args": {k: str(v) for k, v in vars(args).items()},
            "train_files": [Path(f).name for f in train_files], "val_files": [Path(f).name for f in val_files]}
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        perm = torch.randperm(n, device=device)
        run = 0.0
        for i in range(steps_per_epoch):
            idx = perm[i * args.batch:(i + 1) * args.batch]
            loss, _, _ = next_state_loss(model, [x[idx] for x in tr_t], dt, acc_t, args.w_q)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            sched.step()
            run += loss.item()
        val = evaluate(va_t)
        curve.append({"epoch": ep, "train_loss": run / steps_per_epoch, "val_loss": val, "lr": sched.get_last_lr()[0]})
        if val < best:
            best = val
            save_checkpoint(args.out / "lnn_best.pt", model, meta)
        print(f"epoch {ep:4d}  train {run / steps_per_epoch:.4e}  val {val:.4e}  ({time.time() - t0:.1f}s)", flush=True)
        write_csv(args.out / "loss_curve.csv", curve)
    save_checkpoint(args.out / "lnn_last.pt", model, meta)

    # ------------------------------------------------------------- evaluation
    from cmr_control.learning.lnn import load_checkpoint
    model, _ = load_checkpoint(args.out / "lnn_best.pt")
    model = model.double()
    q, dq, tau, _, d_dq = (torch.as_tensor(x, dtype=torch.float64) for x in va)
    ddq = d_dq / dt
    sel = torch.as_tensor(rng.choice(len(q), size=min(5000, len(q)), replace=False))
    with torch.no_grad():
        acc_pred = model.forward_dynamics(q[sel], dq[sel], tau[sel])
        tau_pred = model.inverse_dynamics(q[sel], dq[sel], ddq[sel])
        g_pred = model.gravity(q[sel]).numpy()
        M_pred = model.mass_matrix(q[sel]).numpy()

    from cmr_control.dynamics import RigidBodyModel  # ground truth, evaluation only
    truth = RigidBodyModel(tool="none")
    qs = q[sel].numpy()
    g_true = np.array([truth.gravity(x) for x in qs])
    M_true = np.array([truth.mass_matrix(x) for x in qs])
    truth.close()

    rmse = lambda a, b: np.sqrt(np.mean((a - b) ** 2, axis=0))
    metrics = {
        "best_val_loss": best,
        "acc_rmse_per_joint": rmse(acc_pred.numpy(), ddq[sel].numpy()).tolist(),
        "acc_std_per_joint": ddq[sel].numpy().std(0).tolist(),
        "tau_rmse_per_joint": rmse(tau_pred.numpy(), tau[sel].numpy()).tolist(),
        "tau_std_per_joint": tau[sel].numpy().std(0).tolist(),
        "gravity_rmse_per_joint": rmse(g_pred, g_true).tolist(),
        "gravity_std_per_joint": g_true.std(0).tolist(),
        "mass_matrix_rel_fro_err_median": float(np.median(
            np.linalg.norm(M_pred - M_true, axis=(1, 2)) / np.linalg.norm(M_true, axis=(1, 2)))),
        "mass_diag_rel_err_median": np.median(np.abs(np.diagonal(M_pred, 0, 1, 2) - np.diagonal(M_true, 0, 1, 2))
                                              / np.diagonal(M_true, 0, 1, 2), axis=0).tolist(),
    }
    save_json(args.out / "lnn_eval.json", metrics)
    np.savez_compressed(args.out / "lnn_eval_samples.npz", q=qs, g_pred=g_pred, g_true=g_true,
                        M_pred=M_pred, M_true=M_true, tau=tau[sel].numpy(), tau_pred=tau_pred.numpy())
    for k, v in metrics.items():
        print(f"{k}: {np.round(v, 4) if isinstance(v, list) else round(v, 5)}")


if __name__ == "__main__":
    main()
