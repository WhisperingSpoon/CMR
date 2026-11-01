"""Independent-joint PID (model-free).

tau = Kp (q_d - q) + Kd (dq_d - dq) + Ki int(q_d - q) [+ tau_ff]

No dynamic model is used: gravity is rejected only by the integral action (or not at all
when Ki = 0). Used as (i) the baseline for the free-motion experiment, (ii) the excitation
controller that generates LNN training data, (iii) the low-level servo of the PPO agent.
"""
from __future__ import annotations

import numpy as np

# Torque-level gains for the simulated UR7e at 1 kHz.
# Joint-space inertias at the home pose: M_ii ~ [0.54, 1.18, 0.87, 0.025, 0.0064, 0.00043] kg m^2.
# With explicit (sampled) torque feedback the derivative gain must satisfy Kd*dt/M_ii < 2;
# all gains below keep Kd*dt/M_ii <= 0.5 and damping ratios ~1-2.
KP_DEFAULT = np.array([600.0, 900.0, 600.0, 60.0, 30.0, 4.0])
KD_DEFAULT = np.array([60.0, 90.0, 50.0, 4.0, 1.5, 0.15])
KI_DEFAULT = 2.0 * KP_DEFAULT


class JointPID:
    def __init__(self, kp=KP_DEFAULT, kd=KD_DEFAULT, ki=None, i_limit=None, dt=1e-3):
        self.kp = np.asarray(kp, float)
        self.kd = np.asarray(kd, float)
        self.ki = np.zeros(6) if ki is None else np.asarray(ki, float)
        self.i_limit = np.full(6, np.inf) if i_limit is None else np.asarray(i_limit, float)
        self.dt = dt
        self.integral = np.zeros(6)

    def reset(self):
        self.integral[:] = 0.0

    def __call__(self, q, dq, q_d, dq_d=None, tau_ff=None):
        dq_d = np.zeros(6) if dq_d is None else dq_d
        e = q_d - q
        self.integral = np.clip(self.integral + e * self.dt, -self.i_limit, self.i_limit)
        tau = self.kp * e + self.kd * (dq_d - dq) + self.ki * self.integral
        if tau_ff is not None:
            tau = tau + tau_ff
        return tau
