"""Reference trajectories. Each returns (q, dq, ddq) at time t (arrays of shape (6,))."""
from __future__ import annotations

import numpy as np


def quintic_s(tau):
    """Rest-to-rest quintic time scaling s(tau) on tau in [0, 1], with ds, dds."""
    tau = np.clip(tau, 0.0, 1.0)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    ds = 30 * tau**2 - 60 * tau**3 + 30 * tau**4
    dds = 60 * tau - 180 * tau**2 + 120 * tau**3
    return s, ds, dds


class QuinticWaypoints:
    """Joint-space rest-to-rest quintic segments through waypoints (Siciliano et al. 2009, Sec. 4.2)."""

    def __init__(self, waypoints, segment_time: float | list, hold: float = 0.0):
        self.W = np.asarray(waypoints, float)
        n = len(self.W) - 1
        self.T = np.full(n, segment_time, float) if np.isscalar(segment_time) else np.asarray(segment_time, float)
        self.hold = hold
        self.t0 = np.concatenate(([0.0], np.cumsum(self.T + hold)))
        self.duration = float(self.t0[-1])

    def __call__(self, t):
        k = int(np.searchsorted(self.t0, t, side="right") - 1)
        k = min(max(k, 0), len(self.T) - 1)
        T = self.T[k]
        tau = (t - self.t0[k]) / T
        s, ds, dds = quintic_s(tau)
        d = self.W[k + 1] - self.W[k]
        if tau >= 1.0:
            return self.W[k + 1].copy(), np.zeros(6), np.zeros(6)
        return self.W[k] + s * d, ds / T * d, dds / T**2 * d


class MultiSine:
    """q(t) = q0 + w(t) * sum_i a_i (sin(w_i t + phi_i) - sin(phi_i)), with a quintic fade-in w(t).

    The fade-in makes q, dq, ddq continuous at t=0 starting from rest at q0.
    Used both for exciting the robot (LNN data, as in Liu et al. 2024) and as a test trajectory.
    """

    def __init__(self, q0, amps, freqs, phases, duration: float, ramp: float = 1.0):
        self.q0 = np.asarray(q0, float)
        self.a = np.asarray(amps, float)       # (K, 6)
        self.w = 2 * np.pi * np.asarray(freqs, float)  # (K, 6)
        self.phi = np.asarray(phases, float)   # (K, 6)
        self.duration = duration
        self.ramp = ramp

    def __call__(self, t):
        a, w, phi = self.a, self.w, self.phi
        f = (a * (np.sin(w * t + phi) - np.sin(phi))).sum(0)
        df = (a * w * np.cos(w * t + phi)).sum(0)
        ddf = (-a * w**2 * np.sin(w * t + phi)).sum(0)
        r, dr, ddr = quintic_s(t / self.ramp)
        dr, ddr = dr / self.ramp, ddr / self.ramp**2
        return self.q0 + r * f, dr * f + r * df, ddr * f + 2 * dr * df + r * ddf

    @staticmethod
    def random(rng, q0, duration, n_harmonics=3, amp=(0.3, 0.3, 0.3, 0.5, 0.5, 0.8), fmax=0.5, fmin=0.05,
               ramp=1.0):
        amp = np.asarray(amp, float)
        A = rng.uniform(0.3, 1.0, size=(n_harmonics, 6)) * amp / n_harmonics
        F = rng.uniform(fmin, fmax, size=(n_harmonics, 6))
        P = rng.uniform(0, 2 * np.pi, size=(n_harmonics, 6))
        return MultiSine(q0, A, F, P, duration, ramp)


def sample(traj, dt):
    t = np.arange(0.0, traj.duration + 1e-12, dt)
    Q, dQ, ddQ = map(np.array, zip(*(traj(ti) for ti in t)))
    return t, Q, dQ, ddQ


def min_link_height(model, Q, stride=20):
    """Lowest height of any moving link frame along a sampled joint trajectory (floor check)."""
    return min(float(model.link_heights(q).min()) for q in Q[::stride])
