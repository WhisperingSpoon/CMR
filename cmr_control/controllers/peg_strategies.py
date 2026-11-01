"""Peg-in-hole episode runners: one per controller, all on the same :class:`PegInHoleEnv`.

Every runner starts from ``env.reset(seed, options)`` (identical scene, hole error and start
pose for a given seed) and returns the same log/metrics, so results are directly comparable.

  impedance        model-based, Cartesian impedance at 1 kHz, compliant laterally   (MB, contact)
  impedance_stiff  same law with high stiffness = model-based position control     (baseline)
  pid_scripted     scripted straight descent through the model-free joint PID servo (baseline)
  ppo              learned PPO policy through the same servo                        (MF, contact)
"""
from __future__ import annotations

import numpy as np

from cmr_control.controllers.impedance import CartesianImpedance
from cmr_control.dynamics import RigidBodyModel
from cmr_control.learning.peg_env import QUAT_DOWN, PegInHoleEnv
from cmr_control.scene import contact_force
from cmr_control.trajectories import quintic_s

LOG_EVERY = 10  # log every 10 ms


def _new_log():
    return {"t": [], "p_tip": [], "f_contact": [], "depth": [], "lateral_err": []}


def _append(log, t, st):
    log["t"].append(t)
    log["p_tip"].append(st["p_tip"])
    log["f_contact"].append(st["f_contact"])
    log["depth"].append(st["depth"])
    log["lateral_err"].append(st["lateral_err"])


def summarize(log, success, t_success, extra=None):
    f = np.linalg.norm(np.array(log["f_contact"]), axis=1) if log["f_contact"] else np.zeros(1)
    in_contact = f > 0.5
    out = {
        "success": bool(success),
        "t_success": float(t_success) if success else np.nan,
        "f_peak": float(f.max()),
        "f_mean_contact": float(f[in_contact].mean()) if in_contact.any() else 0.0,
        "final_depth": float(log["depth"][-1]) if log["depth"] else np.nan,
        "final_lateral_err": float(log["lateral_err"][-1]) if log["lateral_err"] else np.nan,
    }
    if extra:
        out.update(extra)
    return out


def to_arrays(log):
    return {k: np.asarray(v) for k, v in log.items()}


# --------------------------------------------------------------------------- impedance
IMPEDANCE_COMPLIANT = dict(k_trans=(300.0, 300.0, 1000.0), k_rot=(15.0, 15.0, 15.0))
IMPEDANCE_STIFF = dict(k_trans=(8000.0, 8000.0, 8000.0), k_rot=(150.0, 150.0, 40.0))  # yaw limited: wrist-3 inertia 4e-4 kg m^2


def run_impedance(env: PegInHoleEnv, seed, options, model: RigidBodyModel, gains=IMPEDANCE_COMPLIANT,
                  t_align=1.0, v_descent=0.02, push=0.010, t_max=10.0):
    """Align over the *nominal* hole, then move the virtual set-point down at v_descent to
    ``push`` m below the nominal bottom. The contact force builds up as K_z * (set-point
    penetration); the low lateral stiffness lets the chamfer guide the peg into the true hole.
    """
    env.reset(seed=seed, options=options)
    ctrl = CartesianImpedance(model, zeta=1.0, **gains)
    plant = env.plant
    p_start, _ = plant.ee_pose()
    p_above = env.nominal_bottom + np.array([0.0, 0.0, p_start[2] - env.nominal_bottom[2]])
    z_goal = env.nominal_bottom[2] - push
    t_desc = (p_above[2] - z_goal) / v_descent

    log, success, t_succ, f_peak = _new_log(), False, np.nan, 0.0
    n = int(t_max / env.dt)
    for k in range(n):
        t = k * env.dt
        if t < t_align:
            s, ds, _ = quintic_s(t / t_align)
            p_d = p_start + s * (p_above - p_start)
            v_d = ds / t_align * (p_above - p_start)
        else:
            tau_d = min(t - t_align, t_desc)
            p_d = p_above - np.array([0, 0, v_descent * tau_d])
            v_d = np.array([0, 0, -v_descent]) if t - t_align < t_desc else np.zeros(3)
        q, dq = plant.state()
        tau, _ = ctrl(q, dq, p_d, QUAT_DOWN, v_d=v_d)
        plant.apply_torque(tau)
        plant.step()
        f_peak = max(f_peak, float(np.linalg.norm(contact_force(env.pc, plant.id, env.fixture))))
        if k % LOG_EVERY == 0:
            st = env.task_state()
            _append(log, t, st)
            if st["success"]:
                success, t_succ = True, t
                break
    return summarize(log, success, t_succ, {"f_peak": f_peak}), to_arrays(log)


# ---------------------------------------------------------------- scripted PID (model-free)
def run_pid_scripted(env: PegInHoleEnv, seed, options, t_max=10.0):
    """Model-free position control: the virtual tip target moves to the nominal hole and straight
    down (same path as the impedance controller) and is tracked by the stiff joint PID servo."""
    env.reset(seed=seed, options=options)
    log, success, t_succ, f_peak = _new_log(), False, np.nan, 0.0
    goal = env.nominal_bottom - np.array([0, 0, 0.004])
    for k in range(env.max_steps):
        d = goal - env.p_cmd
        a = np.clip(d / env.step_size, -1, 1)
        a[2] = -0.5 if np.linalg.norm(d[:2]) < 1e-3 else 0.0  # align first, then descend at 2 cm/s
        _, _, term, trunc, info = env.step(a)
        t = env.k / env.max_steps * t_max
        _append(log, t, info)
        f_peak = max(f_peak, info["f_peak_step"])
        if info["success"]:
            success, t_succ = True, t
        if term or trunc:
            break
    return summarize(log, success, t_succ, {"f_peak": f_peak}), to_arrays(log)


# ---------------------------------------------------------------------------- PPO policy
def run_policy(env: PegInHoleEnv, seed, options, policy, t_max=10.0):
    """Roll out a trained SB3 policy deterministically."""
    obs, _ = env.reset(seed=seed, options=options)
    log, success, t_succ, f_peak = _new_log(), False, np.nan, 0.0
    for k in range(env.max_steps):
        a, _ = policy.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(a)
        t = env.k / env.max_steps * t_max
        _append(log, t, info)
        f_peak = max(f_peak, info["f_peak_step"])
        if info["success"]:
            success, t_succ = True, t
        if term or trunc:
            break
    return summarize(log, success, t_succ, {"f_peak": f_peak}), to_arrays(log)
