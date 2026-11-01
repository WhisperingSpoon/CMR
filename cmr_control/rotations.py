"""Quaternion helpers. pybullet convention: q = (x, y, z, w)."""
import numpy as np


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def quat_error(q_d, q_e):
    """Orientation error e_O (world frame) such that e_O -> 0 when q_e -> q_d.

    Siciliano et al. (2009), Eq. (3.91):  e_O = eta_e eps_d - eta_d eps_e - S(eps_d) eps_e.
    """
    eps_d, eta_d = np.asarray(q_d[:3]), q_d[3]
    eps_e, eta_e = np.asarray(q_e[:3]), q_e[3]
    if eta_d * eta_e + eps_d @ eps_e < 0:  # take the shortest rotation
        eps_e, eta_e = -eps_e, -eta_e
    return eta_e * eps_d - eta_d * eps_e - skew(eps_d) @ eps_e


def quat_angle(q1, q2):
    d = abs(float(np.dot(q1, q2)))
    return 2 * np.arccos(min(1.0, d))
