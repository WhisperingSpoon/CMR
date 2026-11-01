"""Thin, fast plant wrapper around the course ``UR7e`` class.

The course class is kept as the single place where the robot is loaded (URDF, torque mode,
joint damping disabled). This wrapper only adds:
  * headless operation (pybullet DIRECT) for Slurm,
  * a configurable tool (peg / payload) through a generated URDF,
  * fast state access through the robot's own BulletClient,
  * torque saturation and a step counter.

The *plant* is ground truth. Controllers must not query its dynamics; model-based
controllers use :class:`cmr_control.dynamics.RigidBodyModel` instead, which can be given a
deliberately wrong model.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from ur_simulation.classic_control.robots.ur7e import UR7e
from cmr_control.urdf_tools import make_urdf

ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
TAU_MAX = UR7e.MAX_JOINT_TORQUES[:6].astype(float)  # [150,150,150,28,28,28] Nm
Q_HOME = np.array([1.57, -1.7, 2.4, -1.57, -1.57, -1.57])  # course neutral pose (tool pointing down)


def joint_and_link_indices(pc, body_id):
    """Map joint names and child-link names to pybullet indices."""
    joints, links = {}, {}
    for j in range(pc.getNumJoints(body_id)):
        info = pc.getJointInfo(body_id, j)
        joints[info[1].decode()] = j
        links[info[12].decode()] = j
    return joints, links


class Plant:
    """Simulated UR7e (ground truth).

    Args:
        tool: "none", "payload" or "peg" (see :mod:`cmr_control.urdf_tools`).
        payload_mass: mass of the *real* payload (unknown to controllers unless told).
        gui: open the pybullet GUI (local debugging only).
        q0: initial joint configuration.
        dt: physics/control period [s] (course default: 1 kHz).
        floor: add the ground plane at z=0.
    """

    def __init__(
        self,
        tool: str = "none",
        payload_mass: float = 0.0,
        gui: bool = False,
        q0: Optional[np.ndarray] = None,
        dt: float = 1e-3,
        floor: bool = True,
        tau_max: np.ndarray = TAU_MAX,
    ):
        self.tool = tool
        q0 = Q_HOME if q0 is None else np.asarray(q0, float)
        urdf = make_urdf(tool=tool, payload_mass=payload_mass)
        self.robot = UR7e(
            block_gripper=True,
            neutral_joints=q0,
            n_substeps=1,
            step_frequency=int(round(1.0 / dt)),
            render_mode="human" if gui else "rgb_array",
            urdf_path=urdf,
        )
        self.sim = self.robot.sim
        self.pc = self.sim.physics_client
        self.id = self.robot.robot_id
        self.dt = dt
        self.t = 0.0
        self.tau_max = np.asarray(tau_max, float)

        jidx, lidx = joint_and_link_indices(self.pc, self.id)
        self.arm = [jidx[n] for n in ARM_JOINT_NAMES]
        self.ee_link = lidx["peg_tip"] if tool == "peg" else lidx["tool0"]
        self.peg_link = lidx.get("peg")
        self.peg_joint = jidx.get("tool0-peg")

        # Re-apply torque mode through this client (the course helper uses the global client 0).
        self.pc.setJointMotorControlArray(
            self.id, self.arm, self.pc.VELOCITY_CONTROL, forces=[0.0] * 6
        )
        for j in self.arm:
            self.pc.changeDynamics(self.id, j, linearDamping=0, angularDamping=0, jointDamping=0)
        if self.peg_joint is not None:
            self.pc.enableJointForceTorqueSensor(self.id, self.peg_joint, True)
        if floor:
            self.sim.create_plane(0.0)
        self.last_tau = np.zeros(6)
        self.realtime = gui  # sleep dt per step when watching in the GUI

    # ------------------------------------------------------------------ state
    def q(self) -> np.ndarray:
        return np.array([s[0] for s in self.pc.getJointStates(self.id, self.arm)])

    def dq(self) -> np.ndarray:
        return np.array([s[1] for s in self.pc.getJointStates(self.id, self.arm)])

    def state(self):
        st = self.pc.getJointStates(self.id, self.arm)
        return np.array([s[0] for s in st]), np.array([s[1] for s in st])

    def set_q(self, q, dq=None):
        dq = np.zeros(6) if dq is None else dq
        for j, qi, dqi in zip(self.arm, q, dq):
            self.pc.resetJointState(self.id, j, qi, dqi)

    def ee_pose(self):
        """World position and quaternion (x, y, z, w) of the end-effector frame."""
        ls = self.pc.getLinkState(self.id, self.ee_link, computeForwardKinematics=True)
        return np.array(ls[4]), np.array(ls[5])

    def ee_velocity(self):
        ls = self.pc.getLinkState(self.id, self.ee_link, computeLinkVelocity=True, computeForwardKinematics=True)
        return np.array(ls[6]), np.array(ls[7])

    def wrench_sensor(self) -> np.ndarray:
        """Reaction wrench at the tool0-peg joint (simulated wrist F/T sensor), joint frame."""
        if self.peg_joint is None:
            return np.zeros(6)
        return np.array(self.pc.getJointState(self.id, self.peg_joint)[2])

    # ---------------------------------------------------------------- actuation
    def apply_torque(self, tau) -> np.ndarray:
        tau = np.clip(np.asarray(tau, float), -self.tau_max, self.tau_max)
        self.pc.setJointMotorControlArray(self.id, self.arm, self.pc.TORQUE_CONTROL, forces=tau.tolist())
        self.last_tau = tau
        return tau

    def step(self, n: int = 1):
        for _ in range(n):
            self.pc.stepSimulation()
        self.t += n * self.dt
        if self.realtime:
            time.sleep(n * self.dt)

    def close(self):
        self.sim.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
