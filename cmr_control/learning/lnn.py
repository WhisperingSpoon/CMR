"""Lagrangian neural network for the UR7e (model-free, physics-informed).

Method (taken from the literature):
  * Lagrangian structure L = 1/2 dq^T M(q) dq - V(q), mass matrix through a Cholesky factor
    with a positive diagonal (DeLaN, Lutter, Ritter & Peters, ICLR 2019; Lutter & Peters,
    IJRR 2023).
  * Non-conservative extension with a learned PSD damping matrix D(q) = L_D L_D^T, next-state
    prediction by RK4 integration of the learned forward dynamics and a next-state MSE loss
    (Liu et al., "Physics-Informed Neural Networks to Model and Control Robots", Adv. Intell.
    Syst. 2024, Sec. 3.2, Eq. 8).
  * Tracking controller (Liu et al. 2024, Eq. 25).

Our choices / modifications (to be stated in the report):
  * PyTorch + torch.func instead of JAX/Haiku.
  * Input matrix A(q) fixed to the identity: the UR7e is fully actuated and directly torque
    driven, so A-NN of Liu et al. is not needed.
  * Inputs are [sin q, cos q] (all joints revolute).
  * The two next-state error terms are normalised by dt (resp. dt^2/2) and the per-joint
    acceleration scale so both are O(1) at 200 Hz.

Euler-Lagrange equations of the learned model:
    M(q) ddq + c(q, dq) + g(q) + D(q) dq = tau,
    c(q, dq) = M'(q, dq) dq - 1/2 d/dq (dq^T M(q) dq),   g = dV/dq.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.func import jacfwd, vmap

N = 6


def mlp(n_in, n_out, width, depth, act=nn.Softplus):
    layers, d = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), act()]
        d = width
    layers.append(nn.Linear(d, n_out))
    return nn.Sequential(*layers)


def _tril_from_vec(v, n, diag_fn, diag_shift):
    """Lower-triangular matrix from a vector of n(n+1)/2 entries (first n: diagonal)."""
    L = torch.zeros(n, n, dtype=v.dtype, device=v.device)
    idx = torch.tril_indices(n, n, offset=-1, device=v.device)
    L = L.index_put((idx[0], idx[1]), v[n:])
    return L + torch.diag(diag_fn(v[:n]) + diag_shift)


class LagrangianNN(nn.Module):
    def __init__(self, width=128, depth=2, eps_m=1e-3, v_scale=50.0, m_scale=1.0, d_scale=0.1,
                 learn_damping=True):
        super().__init__()
        n_in = 2 * N
        n_tri = N * (N + 1) // 2
        self.mass_net = mlp(n_in, n_tri, width, depth)
        self.pot_net = mlp(n_in, 1, width, depth)
        self.damp_net = mlp(n_in, n_tri, width // 2, depth) if learn_damping else None
        self.eps_m = eps_m
        # fixed output scales (hyper-parameters, saved in the checkpoint)
        self.register_buffer("v_scale", torch.tensor(float(v_scale)))
        self.register_buffer("m_scale", torch.tensor(float(m_scale)))
        self.register_buffer("d_scale", torch.tensor(float(d_scale)))

    # ---------------------------------------------------- per-sample functions
    @staticmethod
    def feat(q):
        return torch.cat([torch.sin(q), torch.cos(q)], -1)

    def M(self, q):
        L = _tril_from_vec(self.mass_net(self.feat(q)), N, nn.functional.softplus, 0.0)
        L = L * torch.sqrt(self.m_scale)
        return L @ L.T + self.eps_m * torch.eye(N, dtype=q.dtype, device=q.device)

    def V(self, q):
        return self.v_scale * self.pot_net(self.feat(q))[0]

    def D(self, q):
        if self.damp_net is None:
            return torch.zeros(N, N, dtype=q.dtype, device=q.device)
        L = _tril_from_vec(self.damp_net(self.feat(q)), N, nn.functional.softplus, 0.0)
        return self.d_scale * (L @ L.T)

    def terms(self, q, dq):
        """M, c, g, D for one sample."""
        M = self.M(q)
        dM = jacfwd(self.M)(q)                      # (N, N, N): dM[i, j, k] = dM_ij / dq_k
        Mdot = torch.einsum("ijk,k->ij", dM, dq)
        c = Mdot @ dq - 0.5 * torch.einsum("i,ijk,j->k", dq, dM, dq)
        g = jacfwd(self.V)(q)
        return M, c, g, self.D(q)

    def _fwd(self, q, dq, tau):
        M, c, g, D = self.terms(q, dq)
        return torch.linalg.solve(M, tau - c - g - D @ dq)

    def _inv(self, q, dq, ddq):
        M, c, g, D = self.terms(q, dq)
        return M @ ddq + c + g + D @ dq

    # ---------------------------------------------------------- batched API
    def forward_dynamics(self, q, dq, tau):
        return vmap(self._fwd)(q, dq, tau)

    def inverse_dynamics(self, q, dq, ddq):
        return vmap(self._inv)(q, dq, ddq)

    def gravity(self, q):
        return vmap(jacfwd(self.V))(q)

    def mass_matrix(self, q):
        return vmap(self.M)(q)

    def rk4_increment(self, q, dq, tau, dt):
        """RK4 increments (dq_inc, ddq_inc) over dt with zero-order-hold torque."""
        f = self.forward_dynamics
        k1v, k1a = dq, f(q, dq, tau)
        k2v, k2a = dq + 0.5 * dt * k1a, f(q + 0.5 * dt * k1v, dq + 0.5 * dt * k1a, tau)
        k3v, k3a = dq + 0.5 * dt * k2a, f(q + 0.5 * dt * k2v, dq + 0.5 * dt * k2a, tau)
        k4v, k4a = dq + dt * k3a, f(q + dt * k3v, dq + dt * k3a, tau)
        dq_inc = dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        ddq_inc = dt / 6 * (k1a + 2 * k2a + 2 * k3a + k4a)
        return dq_inc, ddq_inc


def next_state_loss(model, batch, dt, acc_scale, w_q=1.0):
    """Liu et al. (2024) Eq. 8, normalised:  || dq_{k+1} - dq_hat ||^2 + w_q || q_{k+1} - q_hat ||^2."""
    q, dq, tau, d_q, d_dq = batch  # d_q = q_{k+1}-q_k, d_dq = dq_{k+1}-dq_k (computed in float64)
    pq, pdq = model.rk4_increment(q, dq, tau, dt)
    l_v = (((pdq - d_dq) / (dt * acc_scale)) ** 2).mean()
    l_q = (((pq - d_q) / (0.5 * dt * dt * acc_scale)) ** 2).mean()
    return l_v + w_q * l_q, l_v.detach(), l_q.detach()


def save_checkpoint(path, model, meta):
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_checkpoint(path, map_location="cpu"):
    ck = torch.load(path, map_location=map_location, weights_only=False)
    meta = ck["meta"]
    model = LagrangianNN(**meta["arch"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, meta
