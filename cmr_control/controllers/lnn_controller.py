"""PINN-based tracking controller, Liu et al. (2024), Eq. 25, with A_L = I:

    tau = M_L(q_r) ddq_r + C_L(q_r, dq_r) dq_r + D_L(q_r) dq_r + G_L(q_r)
          + K_P (q_r - q) + K_D (dq_r - dq)

All model terms are evaluated on the *reference*, so the feed-forward is computed for the whole
trajectory in one batched pass before the rollout (no network call inside the 1 kHz loop).
Only the learned Lagrangian model is used; no analytical/URDF dynamics.
"""
from __future__ import annotations

import numpy as np
import torch

from cmr_control.controllers.pd import KD_DEFAULT, KP_DEFAULT


class LNNTrackingController:
    def __init__(self, lnn, kp=KP_DEFAULT, kd=KD_DEFAULT, device="cpu", chunk=4096):
        self.lnn = lnn.to(device)
        self.kp = np.asarray(kp, float)
        self.kd = np.asarray(kd, float)
        self.device = device
        self.chunk = chunk
        self.ff = None

    def precompute(self, Q, dQ, ddQ):
        out = []
        dtype = next(self.lnn.parameters()).dtype
        with torch.no_grad():
            for i in range(0, len(Q), self.chunk):
                sl = slice(i, i + self.chunk)
                t = lambda x: torch.as_tensor(x[sl], dtype=dtype, device=self.device)
                out.append(self.lnn.inverse_dynamics(t(Q), t(dQ), t(ddQ)).cpu().numpy())
        self.ff = np.concatenate(out).astype(float)
        return self.ff

    def __call__(self, k, q, dq, q_r, dq_r):
        return self.ff[k] + self.kp * (q_r - q) + self.kd * (dq_r - dq)
