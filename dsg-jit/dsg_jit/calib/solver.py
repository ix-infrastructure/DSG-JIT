# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the Business Source License 1.1.
"""
Sparse Levenberg-Marquardt for visual-inertial bundle adjustment with calibration (self-calibration).

Per-factor residuals and Jacobians come from :mod:`dsg_jit.calib.factors` (vmapped ``jax.jacfwd``); the
normal equations are assembled with ``scipy.sparse``. Map points are eliminated with the Schur complement
(their Hessian is block-diagonal 3x3), leaving a reduced system over keyframes and calibration that is
solved densely when small and with sparse LU otherwise.

The same reduced system gives the *marginal* Fisher information of the calibration parameters, i.e. what
the data say about theta after every keyframe state and map point has been optimized out. Its spectrum is
the identifiability analysis of Phase 1.2; its inverse is the Cramer-Rao bound on the estimate.

Variable layout: ``[keyframes (15 each) | free theta | points (3 each)]``.

Gauge. Visual-inertial BA is invariant to a translation and a rotation about gravity (4 DoF); without the
IMU, to any rigid motion (6 DoF). Those directions of the first keyframe are removed from the system
exactly (not with a stiff prior, which would ruin the conditioning, and not all six with the IMU, which
would pin gravity to ORB-SLAM3's possibly miscalibrated estimate).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from dsg_jit.calib import factors as F
from dsg_jit.calib.export import Problem

jax.config.update("jax_enable_x64", True)

HUBER_MONO = np.sqrt(5.991)     # ORB-SLAM3's thresholds (chi2 95 %, 2 and 3 dof)
HUBER_STEREO = np.sqrt(7.815)
LAM_MAX = 1e2                  # damping beyond this without descent = converged (round-off floor)
# IMU intervals whose whitened cost at the exported states exceeds IMU_GATE x the median are dropped (zero
# information). After an ORB-SLAM3 tracking reset a few short intervals join keyframes across a pose/velocity
# jump (whitened cost up to ~1e11 against a median of ~1e3); they dominate every IMU parameter, and neither
# Huber nor Cauchy weighting removed their influence (a 30 ms offset on MH03 was estimated as 0 either way,
# 20 ms with the gate). ORB-SLAM3's noise model is ~30x overconfident, so the gate is relative to the median,
# not a chi2 quantile.
IMU_GATE = 100.0
TIKHONOV = 1e-8                 # keeps unconstrained directions (e.g. velocity of a visual-only map) finite


def _sqrt_info(C: np.ndarray) -> np.ndarray:
    """L with L^T L = C^-1 for a batch of covariances (robust to the float32 round-off of ORB-SLAM3's C)."""
    C = 0.5 * (C + np.swapaxes(C, -1, -2))
    w, V = np.linalg.eigh(C)
    w = np.clip(w, w.max(-1, keepdims=True) * 1e-12, None)
    return np.swapaxes(V / np.sqrt(w)[..., None, :], -1, -2)


def _so3_exp_np(w: np.ndarray) -> np.ndarray:
    th = np.linalg.norm(w, axis=-1)[..., None, None]
    K = np.zeros(w.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -w[..., 2], w[..., 1], -w[..., 0]
    K = K - np.swapaxes(K, -1, -2)
    ths = np.where(th < 1e-9, 1.0, th)
    A = np.where(th < 1e-9, 1.0, np.sin(th) / ths)
    B = np.where(th < 1e-9, 0.5, (1 - np.cos(th)) / ths ** 2)
    return np.eye(3) + A * K + B * K @ K


@dataclass
class Result:
    theta: np.ndarray
    free: list
    cost0: float
    cost: float
    iterations: int
    seconds: float
    info_theta: np.ndarray | None = None     # marginal Fisher information on the free theta


class CalibBA:
    """Visual-inertial BA over one exported map with a chosen set of free calibration parameters.

    :param free: names of theta groups to estimate (keys of :data:`factors.THETA_GROUPS`), or ``[]`` for
        plain BA with the calibration ORB-SLAM3 used.
    :param use_imu: include IMU and bias random-walk factors (only meaningful for an IMU-initialized map).
    """

    def __init__(self, P: Problem, free=(), use_imu: bool = True, imu_gate: float = IMU_GATE):
        self.P = P
        self.set_free(free)
        self.use_imu = use_imu and len(P.imu_i) > 0
        self.Rwb, self.pwb = P.Rwb.copy(), P.pwb.copy()
        self.v, self.bias, self.X = P.v.copy(), P.bias.copy(), P.X.copy()
        self.theta = np.zeros(F.N_THETA)
        self.cam = jnp.array([P.fx, P.fy, P.cx, P.cy, P.bf])
        self.Rbc0, self.tbc0 = jnp.asarray(P.Tbc[:3, :3]), jnp.asarray(P.Tbc[:3, 3])
        self.obs_w = 1.0 / np.sqrt(P.obs_sigma2)
        self.huber_k = np.where(P.obs_ur >= 0, HUBER_STEREO, HUBER_MONO)
        if self.use_imu:
            self.L9 = _sqrt_info(P.imu_C[:, :9, :9])
            self.L6 = _sqrt_info(P.imu_C[:, 9:15, 9:15])
            self.imu_args = tuple(jnp.asarray(x) for x in (P.win_a, P.win_w, P.win_s, P.win_dt, self.L9))
        self.K, self.Np = P.n_kf, P.n_mp
        self.imu_gated = np.zeros(len(P.imu_i), bool)
        if self.use_imu and imu_gate:
            ri, _, _, _ = self._imu(False)
            e2 = np.sum(np.asarray(ri) ** 2, 1)
            self.imu_gated = e2 > imu_gate * np.median(e2)
            self.L9[self.imu_gated] = 0.0
            self.L6[self.imu_gated] = 0.0
            self.imu_args = self.imu_args[:4] + (jnp.asarray(self.L9),)

    def set_free(self, groups):
        self.free = sorted({k for g in groups for k in F.THETA_GROUPS[g]})

    # ---------------------------------------------------------------- residuals and Jacobians
    def _reproj(self, jac: bool):
        P = self.P
        k, m = P.obs_kf, P.obs_mp
        args = (jnp.zeros((len(k), 6)), jnp.zeros((len(k), 3)), jnp.zeros(F.N_THETA), jnp.asarray(self.Rwb[k]),
                jnp.asarray(self.pwb[k]), jnp.asarray(self.X[m]), jnp.asarray(self.theta), jnp.asarray(P.obs_uv),
                jnp.asarray(P.obs_ur), self.cam, self.Rbc0, self.tbc0)
        r = np.asarray(F.reprojection_r(*args)) * self.obs_w[:, None]
        if not jac:
            return r, None
        Jp, JX, Jt = (np.asarray(x) * self.obs_w[:, None, None] for x in F.reprojection_J(*args))
        return r, (Jp, JX, Jt)

    def _imu(self, jac: bool):
        P = self.P
        i, j = P.imu_i, P.imu_j
        args = (jnp.zeros((len(i), 15)), jnp.zeros((len(i), 9)), jnp.zeros(F.N_THETA),
                jnp.asarray(self.Rwb[i]), jnp.asarray(self.pwb[i]), jnp.asarray(self.v[i]), jnp.asarray(self.bias[i]),
                jnp.asarray(self.Rwb[j]), jnp.asarray(self.pwb[j]), jnp.asarray(self.v[j]), jnp.asarray(self.theta),
                jnp.asarray(P.t[i]), jnp.asarray(P.t[j])) + self.imu_args
        r = np.asarray(F.imu_r(*args))
        J = tuple(np.asarray(x) for x in F.imu_J(*args)) if jac else None
        bargs = (jnp.zeros((len(i), 6)), jnp.zeros((len(i), 6)), jnp.asarray(self.bias[i]),
                 jnp.asarray(self.bias[j]), jnp.asarray(self.L6))
        rb = np.asarray(F.bias_r(*bargs))
        Jb = tuple(np.asarray(x) for x in F.bias_J(*bargs)) if jac else None
        return r, J, rb, Jb

    def _gauge_basis(self):
        """Basis change on the first keyframe's rotation perturbation whose last axis is yaw (rotation about
        world z), and the indices removed from the system."""
        if not self.use_imu:
            return np.eye(3), np.arange(6)
        a = self.Rwb[0].T @ np.array([0.0, 0.0, 1.0])
        Q, _ = np.linalg.qr(np.column_stack([a, np.eye(3)]))
        Q = np.roll(Q[:, :3], -1, axis=1)          # last column = +-a
        return Q, np.array([2, 3, 4, 5])

    def evaluate(self):
        """Total robust cost and its parts."""
        r, _ = self._reproj(False)
        e2 = np.sum(r ** 2, 1)
        k = self.huber_k
        rho = np.where(e2 <= k * k, e2, 2 * k * np.sqrt(e2) - k * k)
        parts = {"reproj": float(rho.sum())}
        if self.use_imu:
            ri, _, rb, _ = self._imu(False)
            parts["imu"], parts["bias"] = float(np.sum(ri ** 2)), float(np.sum(rb ** 2))
        return sum(parts.values()), parts

    # ---------------------------------------------------------------- normal equations
    def _assemble(self, free):
        """Robustly reweighted Gauss-Newton system H dx = -g over [kf | free theta | points]."""
        K, Np, nf = self.K, self.Np, len(free)
        c_th, c_pt = 15 * K, 15 * K + nf
        n = c_pt + 3 * Np
        rows, cols, vals, res = [], [], [], []
        row0 = 0

        def block(r0, nr, c0, J):
            N, _, nc = J.shape
            rr = np.broadcast_to(r0[:, None, None] + np.arange(nr)[None, :, None], J.shape)
            cc = np.broadcast_to(c0[:, None, None] + np.arange(nc)[None, None, :], J.shape)
            rows.append(rr.ravel()); cols.append(cc.ravel()); vals.append(J.ravel())

        # reprojection, Huber-reweighted per observation
        r, (Jp, JX, Jt) = self._reproj(True)
        e = np.linalg.norm(r, axis=1)
        sw = np.sqrt(np.where(e <= self.huber_k, 1.0, self.huber_k / np.maximum(e, 1e-12)))
        r, Jp, JX, Jt = r * sw[:, None], Jp * sw[:, None, None], JX * sw[:, None, None], Jt * sw[:, None, None]
        N = len(r)
        r0 = row0 + 3 * np.arange(N)
        block(r0, 3, 15 * self.P.obs_kf, Jp)
        block(r0, 3, c_pt + 3 * self.P.obs_mp, JX)
        if nf:
            block(r0, 3, np.full(N, c_th), Jt[:, :, free])
        res.append(r.ravel()); row0 += 3 * N

        if self.use_imu:
            ri, (Ji, Jj, Jt), rb, (Jbi, Jbj) = self._imu(True)
            M = len(ri)
            r0 = row0 + 9 * np.arange(M)
            block(r0, 9, 15 * self.P.imu_i, Ji)
            block(r0, 9, 15 * self.P.imu_j, Jj)
            if nf:
                block(r0, 9, np.full(M, c_th), Jt[:, :, free])
            res.append(ri.ravel()); row0 += 9 * M
            r0 = row0 + 6 * np.arange(M)
            block(r0, 6, 15 * self.P.imu_i + 9, Jbi)
            block(r0, 6, 15 * self.P.imu_j + 9, Jbj)
            res.append(rb.ravel()); row0 += 6 * M

        J = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(row0, n))
        rv = np.concatenate(res)
        # gauge: rotate the first keyframe's rotation block so its last axis is yaw, then remove the gauge dims
        Q, fixed = self._gauge_basis()
        T = sp.eye(n, format="lil")
        T[0:3, 0:3] = Q
        T = T.tocsr()
        J = (J @ T).tocsc()
        keep = np.ones(n)
        keep[fixed] = 0.0
        Mk = sp.diags(keep)
        H = (Mk @ (J.T @ J) @ Mk + sp.diags(1.0 - keep) + TIKHONOV * sp.eye(n)).tocsr()
        g = keep * (J.T @ rv)
        self._Q = Q
        return H, g

    @staticmethod
    def _schur(H, g, c_pt, lam=0.0):
        """Eliminate points. Returns the reduced (dense or sparse) system and a back-substitution closure."""
        if lam:
            H = H + lam * sp.diags(H.diagonal())
        A, B, D = H[:c_pt, :c_pt], H[:c_pt, c_pt:], H[c_pt:, c_pt:]
        Np = D.shape[0] // 3
        Dc = D.tocoo()
        blk = np.zeros((Np, 3, 3))
        same = Dc.row // 3 == Dc.col // 3
        np.add.at(blk, (Dc.row[same] // 3, Dc.row[same] % 3, Dc.col[same] % 3), Dc.data[same])
        Dinv_b = np.linalg.inv(blk)
        ii = (3 * np.arange(Np)[:, None, None] + np.arange(3)[None, :, None]).repeat(3, 2)
        jj = (3 * np.arange(Np)[:, None, None] + np.arange(3)[None, None, :]).repeat(3, 1)
        Dinv = sp.csr_matrix((Dinv_b.ravel(), (ii.ravel(), jj.ravel())), shape=D.shape)
        BDi = (B @ Dinv).tocsr()
        S = A - BDi @ B.T
        gc, gp = g[:c_pt], g[c_pt:]
        rhs = -(gc - BDi @ gp)

        def back(dc):
            return Dinv @ (-gp - B.T @ dc)

        return S, rhs, back

    @staticmethod
    def _solve_reduced(S, rhs):
        if S.shape[0] <= 12000:
            Sd = S.toarray()
            try:
                c = scipy.linalg.cho_factor(Sd, lower=True, check_finite=False)
                return scipy.linalg.cho_solve(c, rhs, check_finite=False)
            except np.linalg.LinAlgError:
                return scipy.linalg.solve(Sd, rhs, assume_a="sym")
        return spla.splu(S.tocsc()).solve(rhs)

    def _apply(self, dx, free):
        K, nf = self.K, len(free)
        dk = dx[:15 * K].reshape(K, 15)
        dR = _so3_exp_np(dk[:, 0:3])
        self.pwb = self.pwb + np.einsum("kij,kj->ki", self.Rwb, dk[:, 3:6])
        self.Rwb = self.Rwb @ dR
        self.v = self.v + dk[:, 6:9]
        self.bias = self.bias + dk[:, 9:15]
        if nf:
            self.theta[free] += dx[15 * K:15 * K + nf]
        self.X = self.X + dx[15 * K + nf:].reshape(-1, 3)

    def _snapshot(self):
        return tuple(x.copy() for x in (self.Rwb, self.pwb, self.v, self.bias, self.X, self.theta))

    def _restore(self, s):
        self.Rwb, self.pwb, self.v, self.bias, self.X, self.theta = (x.copy() for x in s)

    def solve(self, max_iter: int = 30, tol: float = 1e-6, lam: float = 1e-4, verbose: bool = False) -> Result:
        """Levenberg-Marquardt over every state and the free calibration parameters."""
        t0 = time.time()
        free = self.free
        cost0, _ = self.evaluate()
        cost, it = cost0, 0
        for it in range(1, max_iter + 1):
            H, g = self._assemble(free)
            c_pt = 15 * self.K + len(free)
            # Try the undamped Gauss-Newton step first: Marquardt damping lam * diag(H) acts hardest on the
            # stiffest directions (bias random walk, diag ~1e12), so even a tiny lam stalls bias and gravity.
            lam = 0.0 if lam <= 1e-6 else lam
            while True:
                S, rhs, back = self._schur(H, g, c_pt, lam)
                dc = self._solve_reduced(S, rhs)
                dx = np.concatenate([dc, back(dc)])
                dx[0:3] = self._Q @ dx[0:3]
                snap = self._snapshot()
                self._apply(dx, free)
                new, parts = self.evaluate()
                if np.isfinite(new) and new < cost:
                    lam = lam / 3
                    break
                self._restore(snap)
                lam = max(lam * 10, 1e-6)
                if lam > LAM_MAX:          # no descent left beyond round-off: converged
                    break
            rel = (cost - new) / max(cost, 1e-12) if new < cost else 0.0
            if verbose:
                th = ", ".join(f"{F.THETA_NAMES[k]} {self.theta[k]:+.5g}" for k in free)
                print(f"  it {it:2d} cost {new:.6g} (rel {rel:.2e}, lam {lam:.1e}) {th}", flush=True)
            cost = min(cost, new)
            if lam > LAM_MAX or rel < tol:
                break
        return Result(theta=self.theta.copy(), free=list(free), cost0=cost0, cost=cost, iterations=it,
                      seconds=time.time() - t0)

    def marginal_information(self, groups=None) -> tuple[np.ndarray, list]:
        """Fisher information of theta (all groups by default) with every state and point marginalized,
        linearized at the current estimate. Returns ``(I, theta indices)``."""
        saved = self.free
        self.set_free(groups or F.IDENT_GROUPS)
        free = self.free
        H, g = self._assemble(free)
        c_pt = 15 * self.K + len(free)
        S, _, _ = self._schur(H, g, c_pt)
        nk = 15 * self.K
        S = S.tocsc()
        Scc, Sct, Stt = S[:nk, :nk], S[:nk, nk:].toarray(), S[nk:, nk:].toarray()
        if nk <= 12000:
            X = scipy.linalg.cho_solve(scipy.linalg.cho_factor(Scc.toarray(), lower=True), Sct)
        else:
            X = spla.splu(Scc).solve(Sct)
        I = Stt - Sct.T @ X
        I -= TIKHONOV * np.eye(len(free))
        self.free = saved
        return 0.5 * (I + I.T), free
