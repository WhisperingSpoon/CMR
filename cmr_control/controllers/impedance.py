"""Cartesian impedance control (model-based, no force measurement).

Task-space error  e = [p_d - p ; e_O(Q_d, Q)]  (e_O: quaternion error, Siciliano Eq. 3.91),
               de = [v_d ; w_d] - J(q) dq.

Control law (operational-space form, Khatib 1987; Hogan 1985; Ott 2008):
    F   = Lambda(q) (a_d - J'(q,dq) dq) + K e + D de
    tau = J(q)^T F + c_hat(q, dq) + g_hat(q),     Lambda = (J M^-1 J^T)^-1

With a perfect model the closed loop in contact is the mass-spring-damper
    Lambda(q) dde + D de + K e = F_ext,
i.e. the end-effector behaves as a programmable spring K around the virtual set-point; the
steady-state contact force is K e. The damping is set per axis with a damping ratio
zeta: D_ii = 2 zeta sqrt(K_ii Lambda_ii).
"""
from __future__ import annotations

import numpy as np

from cmr_control.rotations import quat_error


class CartesianImpedance:
    def __init__(self, model, k_trans=(500, 500, 1500), k_rot=(40, 40, 40), zeta: float = 1.0,
                 inertia_shaping: bool = True, sing_eps: float = 1e-4):
        self.model = model
        self.K = np.diag(np.concatenate([k_trans, k_rot]).astype(float))
        self.zeta = zeta
        self.inertia_shaping = inertia_shaping
        self.sing_eps = sing_eps

    def set_stiffness(self, k_trans=None, k_rot=None):
        k = np.diag(self.K).copy()
        if k_trans is not None:
            k[:3] = k_trans
        if k_rot is not None:
            k[3:] = k_rot
        self.K = np.diag(k)

    def __call__(self, q, dq, p_d, quat_d, v_d=None, w_d=None, a_d=None):
        m = self.model
        v_d = np.zeros(3) if v_d is None else v_d
        w_d = np.zeros(3) if w_d is None else w_d
        a_d = np.zeros(6) if a_d is None else a_d

        p, quat = m.fk(q)
        J = m.jacobian(q)
        M = m.mass_matrix(q)
        twist = J @ dq
        e = np.concatenate([p_d - p, quat_error(quat_d, quat)])
        de = np.concatenate([v_d, w_d]) - twist

        Minv = np.linalg.inv(M)
        Lam = np.linalg.inv(J @ Minv @ J.T + self.sing_eps * np.eye(6))
        D = np.diag(2 * self.zeta * np.sqrt(np.diag(self.K) * np.clip(np.diag(Lam), 1e-6, None)))

        F = self.K @ e + D @ de
        if self.inertia_shaping:
            F = F + Lam @ (a_d - m.jdot_dq(q, dq))
        tau = J.T @ F + m.bias(q, dq)
        return tau, {"e": e, "p": p, "quat": quat, "F_cmd": F}
