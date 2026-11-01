"""Computed-torque (inverse-dynamics) joint-space trajectory tracking.

    tau = M_hat(q) [ ddq_d + Kd (dq_d - dq) + Kp (q_d - q) ] + c_hat(q, dq) + g_hat(q)

With a perfect model the closed loop is the decoupled linear system
    dde + Kd de + Kp e = 0,
chosen critically damped: Kp = w^2, Kd = 2 zeta w (Siciliano et al. 2009, Sec. 8.5.2;
Spong et al. 2006, Ch. 8). M_hat, c_hat, g_hat come from a :class:`RigidBodyModel`, which may
differ from the plant (robustness experiments).
"""
from __future__ import annotations

import numpy as np


class ComputedTorque:
    def __init__(self, model, omega=(25, 25, 25, 25, 25, 25), zeta: float = 1.0):
        self.model = model
        w = np.asarray(omega, float)
        self.kp = w**2
        self.kd = 2 * zeta * w

    def __call__(self, q, dq, q_d, dq_d, ddq_d):
        a = ddq_d + self.kd * (dq_d - dq) + self.kp * (q_d - q)
        return self.model.mass_matrix(q) @ a + self.model.bias(q, dq)
