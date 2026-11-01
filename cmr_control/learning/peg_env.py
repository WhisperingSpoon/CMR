"""Peg-in-hole environment (Gymnasium) for the simulated UR7e.

Task: insert a 20 mm square peg held rigidly at tool0 into a chamfered square hole
(1 mm total clearance, 30 mm deep). The *true* hole is displaced from the *nominal* (believed)
hole position by an unknown planar offset of radius ``offset`` (perception / calibration error).
All controllers only receive the nominal position.

The same class is used by
  * the PPO agent (model-free): ``reset`` / ``step(action)`` at ``control_hz``; an action is a
    bounded Cartesian increment of a virtual peg-tip target, mapped to joint targets by
    kinematic IK and tracked by the model-free joint PID at 1 kHz (no dynamics model);
  * the model-based controllers: they call ``reset`` and then drive ``env.plant`` directly at
    1 kHz, using ``env.task_state()`` for success / force bookkeeping.

Ground-truth contact forces (pybullet contact solver) are used only for metrics and reward;
the agent observes the simulated wrist F/T sensor (reaction at the tool0-peg joint).
"""
from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from cmr_control.controllers.pd import JointPID
from cmr_control.dynamics import RigidBodyModel
from cmr_control.scene import HoleSpec, build_fixture, contact_force
from cmr_control.sim import Q_HOME, Plant

# Peg pointing straight down (tool0 z-axis = world -z), yaw aligned with the hole edges.
QUAT_DOWN = np.array([1.0, 0.0, 0.0, 0.0])

# Servo gains of the low-level joint PID used by the RL agent (and the scripted baseline).
# Kd*dt/M_ii <= 0.5 on every joint (wrist 3 inertia is only 4.3e-4 kg m^2).
SERVO_KP = np.array([3000.0, 3000.0, 2000.0, 300.0, 150.0, 20.0])
SERVO_KD = np.array([150.0, 150.0, 100.0, 10.0, 3.0, 0.2])
SERVO_KI = 10.0 * SERVO_KP  # integral time Kp/Ki = 0.1 s: removes gravity sag within ~0.4 s


class PegInHoleEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        max_offset: float = 0.005,
        control_hz: float = 20.0,
        episode_seconds: float = 10.0,
        step_size: float = 0.002,
        start_height: tuple = (0.015, 0.035),
        start_jitter: float = 0.003,
        friction_range: tuple = (0.3, 0.8),
        force_limit: float = 150.0,
        gui: bool = False,
        hole: Optional[HoleSpec] = None,
        success_margin: float = 0.005,
    ):
        super().__init__()
        self.hole = hole or HoleSpec()
        self.max_offset = max_offset
        self.dt = 1e-3
        self.substeps = int(round(1.0 / (control_hz * self.dt)))
        self.max_steps = int(round(episode_seconds * control_hz))
        self.step_size = step_size
        self.start_height = start_height
        self.start_jitter = start_jitter
        self.friction_range = friction_range
        self.force_limit = force_limit
        self.success_margin = success_margin

        self.plant = Plant(tool="peg", gui=gui, dt=self.dt)
        self.pc = self.plant.pc
        self.fixture = build_fixture(self.pc, self.hole)
        self.servo = JointPID(SERVO_KP, SERVO_KD, SERVO_KI, i_limit=np.array([0.05] * 6), dt=self.dt)
        # Kinematic model only (FK / Jacobian for IK). Its dynamics are never used by the
        # model-free agent; model-based controllers create their own RigidBodyModel.
        self.kin = RigidBodyModel(tool="peg")
        self.q_above = self.kin.ik(self.hole.top + np.array([0, 0, 0.04]), QUAT_DOWN, Q_HOME)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(18,), dtype=np.float32)

        self.nominal_bottom = self.hole.bottom.copy()
        self.true_bottom = self.hole.bottom.copy()
        self.ft_bias = np.zeros(6)
        self.prev_action = np.zeros(3)
        self.k = 0

    # ------------------------------------------------------------------ helpers
    def ik(self, p, quat=QUAT_DOWN, q_seed=None):
        """Joint target for a peg-tip pose (kinematic DLS IK, seeded near the current pose)."""
        return self.kin.ik(p, quat, self.q_above if q_seed is None else q_seed, iters=50)

    def task_state(self):
        p_tip, quat = self.plant.ee_pose()
        f = contact_force(self.pc, self.plant.id, self.fixture)
        lateral = np.linalg.norm(p_tip[:2] - self.true_bottom[:2])
        depth = self.true_bottom[2] + self.hole.depth - p_tip[2]  # >0 when the tip is below the hole top
        success = bool(p_tip[2] <= self.true_bottom[2] + self.success_margin and lateral < self.hole.a + 1e-3)
        return {"p_tip": p_tip, "quat": quat, "f_contact": f, "f_norm": float(np.linalg.norm(f)),
                "lateral_err": lateral, "depth": depth, "success": success}

    def _obs(self):
        p_tip, _ = self.plant.ee_pose()
        v_tip, _ = self.plant.ee_velocity()
        ft = (self.plant.wrench_sensor() - self.ft_bias)
        ft_scaled = np.clip(ft / np.array([20, 20, 20, 2, 2, 2]), -5, 5)
        rel = (p_tip - self.nominal_bottom) / 0.02
        defl = (self.p_cmd - p_tip) / 0.01
        return np.concatenate([rel, v_tip / 0.05, ft_scaled, defl, self.prev_action]).astype(np.float32)

    # --------------------------------------------------------------------- API
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        rng = self.np_random
        options = options or {}

        r = options.get("offset", rng.uniform(0.0, self.max_offset))
        ang = options.get("angle", rng.uniform(0, 2 * np.pi))
        offset = r * np.array([np.cos(ang), np.sin(ang)])
        mu = options.get("friction", rng.uniform(*self.friction_range))
        self.true_bottom = self.nominal_bottom + np.array([offset[0], offset[1], 0.0])
        self.pc.resetBasePositionAndOrientation(self.fixture, self.true_bottom.tolist(), [0, 0, 0, 1])
        self.pc.changeDynamics(self.fixture, -1, lateralFriction=mu)
        self.pc.changeDynamics(self.plant.id, self.plant.peg_link, lateralFriction=mu)

        h = options.get("start_height", rng.uniform(*self.start_height))
        jit = options.get("start_xy", rng.uniform(-self.start_jitter, self.start_jitter, size=2))
        p0 = self.nominal_bottom + np.array([jit[0], jit[1], self.hole.depth + h])
        q0 = self.kin.ik(p0, QUAT_DOWN, self.q_above)
        self.plant.set_q(q0)
        self.p_cmd = p0.copy()
        self.q_cmd = q0.copy()

        # let the servo settle (integral action takes up gravity), then bias the F/T sensor
        self.servo.reset()
        for _ in range(800):
            q, dq = self.plant.state()
            self.plant.apply_torque(self.servo(q, dq, self.q_cmd))
            self.plant.step()
        self.plant.t = 0.0
        self.ft_bias = self.plant.wrench_sensor().copy()
        self.prev_action = np.zeros(3)
        self.k = 0
        self.info0 = {"offset": offset, "friction": mu, "start": p0}
        return self._obs(), dict(self.info0)

    def servo_to(self, q_cmd, n_steps):
        """Track q_cmd with the joint PID for n_steps physics steps; returns peak contact force."""
        fmax = 0.0
        for _ in range(n_steps):
            q, dq = self.plant.state()
            self.plant.apply_torque(self.servo(q, dq, q_cmd))
            self.plant.step()
            fmax = max(fmax, float(np.linalg.norm(contact_force(self.pc, self.plant.id, self.fixture))))
        return fmax

    def step(self, action):
        a = np.clip(np.asarray(action, float), -1, 1)
        lo = self.nominal_bottom + np.array([-0.012, -0.012, -0.004])
        hi = self.nominal_bottom + np.array([0.012, 0.012, self.hole.depth + 0.05])
        self.p_cmd = np.clip(self.p_cmd + self.step_size * a, lo, hi)
        self.q_cmd = self.ik(self.p_cmd, q_seed=self.plant.q())
        fmax = self.servo_to(self.q_cmd, self.substeps)
        self.k += 1
        smooth = float(np.sum(np.square(a - self.prev_action)))
        self.prev_action = a

        st = self.task_state()
        dist = np.linalg.norm(st["p_tip"] - self.true_bottom)
        reward = -dist / 0.03                                   # dense: distance to hole bottom
        reward -= 0.5 * max(0.0, fmax - 30.0) / self.force_limit  # excessive-force penalty
        reward -= 0.05 * smooth                                 # action smoothness
        terminated = False
        if st["success"]:
            reward += 50.0 + 10.0 * (1.0 - self.k / self.max_steps)  # bonus, larger if faster
            terminated = True
        elif fmax > self.force_limit:
            reward -= 20.0
            terminated = True
        truncated = self.k >= self.max_steps
        info = {**st, "f_peak_step": fmax, "is_success": st["success"]}
        return self._obs(), float(reward), terminated, truncated, info

    def close(self):
        self.kin.close()
        self.plant.close()
