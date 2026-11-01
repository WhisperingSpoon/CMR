"""Rigid-body model of the UR7e used by the *model-based* controllers.

The model lives in its own pybullet DIRECT client, loaded from a (possibly different) URDF.
This separates "what the controller believes" from "what the plant is", so the experiments
can introduce model error (an unmodelled payload, wrong link masses) without touching the
plant. With identical URDFs the quantities equal the course ``PyBulletRobotModel`` ones.

Equations of motion:  M(q) q'' + c(q, q') + g(q) = tau
"""
from __future__ import annotations

import numpy as np
import pybullet as p
import pybullet_utils.bullet_client as bc

from cmr_control.sim import ARM_JOINT_NAMES, joint_and_link_indices
from cmr_control.urdf_tools import make_urdf


class RigidBodyModel:
    def __init__(self, tool: str = "none", payload_mass: float = 0.0, link_mass_scale: float = 1.0,
                 ee_link_name: str | None = None):
        urdf = make_urdf(tool=tool, payload_mass=payload_mass, link_mass_scale=link_mass_scale)
        self.pc = bc.BulletClient(connection_mode=p.DIRECT)
        self.pc.setGravity(0, 0, -9.81)
        self.id = self.pc.loadURDF(urdf, useFixedBase=True)
        jidx, lidx = joint_and_link_indices(self.pc, self.id)
        self.arm = [jidx[n] for n in ARM_JOINT_NAMES]
        movable = [j for j in range(self.pc.getNumJoints(self.id))
                   if self.pc.getJointInfo(self.id, j)[2] != p.JOINT_FIXED]
        assert movable == self.arm, "model assumes the 6 arm joints are the only DOFs"
        if ee_link_name is None:
            ee_link_name = "peg_tip" if tool == "peg" else "tool0"
        self.ee_link = lidx[ee_link_name]
        self._q_set = None
        # links that move with the arm (child of shoulder_lift or further down the chain);
        # excludes the fixed base frames ("base", "base_link", ...) that sit at z = 0
        self.moving_links = [j for j in range(self.pc.getNumJoints(self.id)) if self._downstream_of(j, self.arm[1])]

    def _downstream_of(self, link, joint):
        while link >= 0:
            if link == joint:
                return True
            link = self.pc.getJointInfo(self.id, link)[16]  # parent link index
        return False

    # ------------------------------------------------------------- dynamics
    def mass_matrix(self, q) -> np.ndarray:
        return np.array(self.pc.calculateMassMatrix(self.id, list(map(float, q))))

    def bias(self, q, dq) -> np.ndarray:
        """c(q, dq) + g(q)."""
        return np.array(self.pc.calculateInverseDynamics(self.id, list(map(float, q)), list(map(float, dq)), [0.0] * 6))

    def gravity(self, q) -> np.ndarray:
        return np.array(self.pc.calculateInverseDynamics(self.id, list(map(float, q)), [0.0] * 6, [0.0] * 6))

    def inverse_dynamics(self, q, dq, ddq) -> np.ndarray:
        return np.array(self.pc.calculateInverseDynamics(self.id, list(map(float, q)), list(map(float, dq)),
                                                         list(map(float, ddq))))

    def forward_dynamics(self, q, dq, tau) -> np.ndarray:
        return np.linalg.solve(self.mass_matrix(q), np.asarray(tau) - self.bias(q, dq))

    # ------------------------------------------------------------ kinematics
    def _set(self, q):
        q = np.asarray(q, float)
        if self._q_set is None or not np.array_equal(q, self._q_set):
            for j, qi in zip(self.arm, q):
                self.pc.resetJointState(self.id, j, qi)
            self._q_set = q.copy()

    def fk(self, q, link: int | None = None):
        self._set(q)
        ls = self.pc.getLinkState(self.id, self.ee_link if link is None else link, computeForwardKinematics=True)
        return np.array(ls[4]), np.array(ls[5])

    def link_heights(self, q) -> np.ndarray:
        """z of every moving link frame (for floor-collision checks of planned trajectories)."""
        self._set(q)
        return np.array([self.pc.getLinkState(self.id, j, computeForwardKinematics=True)[4][2]
                         for j in self.moving_links])

    def jacobian(self, q) -> np.ndarray:
        """6x6 geometric Jacobian [v; w] of the end-effector frame origin, world frame."""
        self._set(q)
        ls = self.pc.getLinkState(self.id, self.ee_link, computeForwardKinematics=True)
        # calculateJacobian expects a point in the link's inertial (COM) frame.
        # Link frame origin expressed in the COM frame = -R_in^T p_in.
        R_in = np.array(self.pc.getMatrixFromQuaternion(ls[3])).reshape(3, 3)
        local = -R_in.T @ np.array(ls[2])
        z = [0.0] * 6
        Jv, Jw = self.pc.calculateJacobian(self.id, self.ee_link, local.tolist(), list(map(float, q)), z, z)
        return np.vstack((np.array(Jv), np.array(Jw)))

    def jdot_dq(self, q, dq, eps: float = 1e-6) -> np.ndarray:
        """J'(q, dq) dq by a directional finite difference along dq."""
        q = np.asarray(q, float)
        dq = np.asarray(dq, float)
        return (self.jacobian(q + eps * dq) - self.jacobian(q)) @ dq / eps

    def ik(self, pos, quat, q_seed, iters: int = 100, damping: float = 1e-3, tol: float = 1e-7) -> np.ndarray:
        """Damped-least-squares IK on the full 6-D pose error (kinematics only).

        q <- q + J^T (J J^T + lambda^2 I)^-1 [p_d - p ; e_O(Q_d, Q)]
        (Siciliano et al. 2009, Sec. 3.7). pybullet's calculateInverseKinematics was not used
        because it does not reliably enforce the yaw about the tool axis.
        """
        from cmr_control.rotations import quat_error

        q = np.array(q_seed, float)
        pos, quat = np.asarray(pos, float), np.asarray(quat, float)
        for _ in range(iters):
            p, Q = self.fk(q)
            e = np.concatenate([pos - p, quat_error(quat, Q)])
            if e @ e < tol**2:
                break
            J = self.jacobian(q)
            q = q + J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), e)
        return q

    def close(self):
        self.pc.disconnect()
