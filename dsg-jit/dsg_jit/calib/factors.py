# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the Business Source License 1.1.
"""
Residuals for visual-inertial bundle adjustment with calibration parameters.

Every residual is written as a function of *local perturbations* around the current state, so its Jacobian
is ``jax.jacfwd`` at zero. Perturbations:

* keyframe (15): ``[phi(3), rho(3), dv(3), dbg(3), dba(3)]`` with ``R = R0 Exp(phi)``, ``p = p0 + R0 rho``
* map point (3): ``X = X0 + dX``
* calibration (``N_THETA``): additive on ``theta`` (layout in :data:`THETA_NAMES`)

Calibration model. ``theta`` corrects what ORB-SLAM3 assumed:

* ``Tbc = (Exp(theta_rbc) Rbc0, tbc0 + theta_tbc)`` -- camera-IMU extrinsics, body frame
* gyro and accel intrinsics ``meas = M true`` with ``M = I + diag(s) + offdiag(m)``, so
  ``true = M^-1 (meas - bias)``; off-diagonal order ``(01, 02, 10, 12, 20, 21)`` for both sensors.
  Gauge: a common rotation of both sensors is indistinguishable from a rotation of ``Tbc``, so ``rbc``,
  the full ``mg`` and the full ``ma`` cannot be free together. With ``rbc`` free use ``ma_ut`` (the upper
  triangle of ``M_a``: the accelerometer defines the body frame, as in Kalibr); with ``rbc`` fixed the full
  ``ma`` is identifiable.
* ``td``: the IMU sample stamped ``t`` was taken at camera time ``t - td``; the interval between keyframes
  at ``t_i, t_j`` integrates the samples stamped in ``[t_i + td, t_j + td)``, with partial first and last
  samples, so the residual is differentiable in ``td``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dsg_jit.core.math3d import so3_exp, so3_log

GRAVITY = jnp.array([0.0, 0.0, -9.81])

THETA_NAMES = (
    ["rbc_x", "rbc_y", "rbc_z", "tbc_x", "tbc_y", "tbc_z"]
    + ["sg_x", "sg_y", "sg_z"] + [f"mg_{k}" for k in ("01", "02", "10", "12", "20", "21")]
    + ["sa_x", "sa_y", "sa_z"] + [f"ma_{k}" for k in ("01", "02", "10", "12", "20", "21")]
    + ["td"]
)
N_THETA = len(THETA_NAMES)
THETA_GROUPS = {   # name -> theta indices
    "rbc": [0, 1, 2], "tbc": [3, 4, 5], "sg": [6, 7, 8], "mg": list(range(9, 15)),
    "sa": [15, 16, 17], "ma": list(range(18, 24)), "ma_ut": [18, 19, 21], "td": [24],
}
IDENT_GROUPS = ["rbc", "tbc", "sg", "mg", "sa", "ma_ut", "td"]   # every parameter, gauge-free


def _intrinsics(s: jnp.ndarray, m: jnp.ndarray) -> jnp.ndarray:
    M = jnp.eye(3) + jnp.diag(s)
    return M.at[(0, 0, 1, 1, 2, 2), (1, 2, 0, 2, 0, 1)].add(m)


def extrinsics(theta: jnp.ndarray, Rbc0: jnp.ndarray, tbc0: jnp.ndarray):
    return so3_exp(theta[0:3]) @ Rbc0, tbc0 + theta[3:6]


def reprojection(dpose, dX, dth, Rwb0, pwb0, X0, th0, uv, ur, cam, Rbc0, tbc0):
    """Rectified stereo reprojection ``[u, v, u_right]`` minus the observation, in pixels.

    ``cam = (fx, fy, cx, cy, bf)``. The ``u_right`` row is zero for a monocular observation (``ur < 0``).
    """
    fx, fy, cx, cy, bf = cam
    R = Rwb0 @ so3_exp(dpose[0:3])
    p = pwb0 + Rwb0 @ dpose[3:6]
    Rbc, tbc = extrinsics(th0 + dth, Rbc0, tbc0)
    Rwc, pwc = R @ Rbc, p + R @ tbc
    Pc = Rwc.T @ (X0 + dX - pwc)
    z = Pc[2]
    u = fx * Pc[0] / z + cx
    v = fy * Pc[1] / z + cy
    stereo = ur >= 0
    return jnp.stack([u - uv[0], v - uv[1], jnp.where(stereo, u - bf / z - ur, 0.0)])


def integrate(a, w, s, dt, T0, T1, bias, th):
    """Re-integrate IMU samples over ``[T0, T1)`` (IMU time base) with bias and intrinsics ``th``.

    Same recursion as ORB-SLAM3's ``Preintegrated::IntegrateNewMeasurement`` (dP, dV with the previous dR,
    then dR). Returns ``dR, dV, dP``.
    """
    Mg = _intrinsics(th[6:9], th[9:15])
    Ma = _intrinsics(th[15:18], th[18:24])
    dte = jnp.clip(jnp.minimum(s + dt, T1) - jnp.maximum(s, T0), 0.0, None) * (dt > 0)

    def step(carry, x):
        dR, dV, dP = carry
        ak, wk, tk = x
        acc = jnp.linalg.solve(Ma, ak - bias[3:6])
        gyr = jnp.linalg.solve(Mg, wk - bias[0:3])
        dP = dP + dV * tk + 0.5 * (dR @ acc) * tk * tk
        dV = dV + (dR @ acc) * tk
        dR = dR @ so3_exp(gyr * tk)
        return (dR, dV, dP), None

    (dR, dV, dP), _ = jax.lax.scan(step, (jnp.eye(3), jnp.zeros(3), jnp.zeros(3)), (a, w, dte))
    return dR, dV, dP


def imu(dki, dkj, dth, Ri0, pi0, vi0, bi0, Rj0, pj0, vj0, th0, ti, tj, a, w, s, dt, L9):
    """Whitened 9-vector IMU residual (rotation, velocity, position), body frame of keyframe i.

    The preintegration is recomputed from the raw samples at the current bias of keyframe i (as ORB-SLAM3's
    inertial edge evaluates it), so no first-order bias correction is needed.
    """
    Ri = Ri0 @ so3_exp(dki[0:3])
    pi = pi0 + Ri0 @ dki[3:6]
    vi = vi0 + dki[6:9]
    bi = bi0 + dki[9:15]
    Rj = Rj0 @ so3_exp(dkj[0:3])
    pj = pj0 + Rj0 @ dkj[3:6]
    vj = vj0 + dkj[6:9]
    th = th0 + dth
    td = th[24]
    dR, dV, dP = integrate(a, w, s, dt, ti + td, tj + td, bi, th)
    T = tj - ti
    eR = so3_log(dR.T @ Ri.T @ Rj)
    eV = Ri.T @ (vj - vi - GRAVITY * T) - dV
    eP = Ri.T @ (pj - pi - vi * T - 0.5 * GRAVITY * T * T) - dP
    return L9 @ jnp.concatenate([eR, eV, eP])


def bias_walk(dbi, dbj, bi0, bj0, L6):
    """Whitened random-walk residual between consecutive keyframe biases."""
    return L6 @ ((bj0 + dbj) - (bi0 + dbi))


# Batched residual and Jacobian functions. The first arguments are the perturbations (evaluated at zero).
reprojection_r = jax.jit(jax.vmap(reprojection, in_axes=(0, 0, None, 0, 0, 0, None, 0, 0, None, None, None)))
reprojection_J = jax.jit(jax.vmap(jax.jacfwd(reprojection, argnums=(0, 1, 2)),
                                  in_axes=(0, 0, None, 0, 0, 0, None, 0, 0, None, None, None)))
_imu_axes = (0, 0, None, 0, 0, 0, 0, 0, 0, 0, None, 0, 0, 0, 0, 0, 0, 0)
imu_r = jax.jit(jax.vmap(imu, in_axes=_imu_axes))
imu_J = jax.jit(jax.vmap(jax.jacfwd(imu, argnums=(0, 1, 2)), in_axes=_imu_axes))
bias_r = jax.jit(jax.vmap(bias_walk))
bias_J = jax.jit(jax.vmap(jax.jacfwd(bias_walk, argnums=(0, 1))))
