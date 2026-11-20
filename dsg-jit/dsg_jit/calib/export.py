# Copyright (c) 2025.
# This file is part of DSG-JIT, released under the Business Source License 1.1.
"""
Load an ORB-SLAM3 estimation-problem export into solver-ready arrays.

The export is written by ``System::SaveEstimationProblem`` in the OASIS ORB-SLAM3 fork (branch
``phase1-export``): keyframes, map points, observations, IMU preintegration and the raw IMU samples behind
it, as CSV (optionally gzipped). This module selects one map, re-indexes everything densely, and builds the
IMU sample windows that let the IMU factor be re-integrated under a changed calibration, including a
time offset that moves the window edges.

Conventions
-----------
* Poses are ``Twb`` (body to world) for the state; observations are in the rectified left camera, whose
  pose is ``Twb @ Tbc``. ``u_right < 0`` marks a monocular observation.
* An IMU sample k covers ``[s_k, s_k + dt_k)`` in the IMU time base. Interval r integrates the samples
  whose (shifted) time falls in ``[t_i + td, t_j + td)``; with ``td = 0`` this is exactly ORB-SLAM3's set.
"""

from __future__ import annotations

import gzip
import os
from dataclasses import dataclass, field

import numpy as np


def _open(path: str):
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        path += ".gz"
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def read_csv(path: str) -> dict[str, np.ndarray]:
    """CSV with a header line -> dict of float64 columns."""
    with _open(path) as f:
        names = f.readline().strip().split(",")
        data = np.loadtxt(f, delimiter=",", ndmin=2)
    if data.size == 0:
        data = np.zeros((0, len(names)))
    return {n: data[:, i] for i, n in enumerate(names)}


def read_meta(path: str) -> dict:
    meta: dict = {"maps": []}
    with _open(path) as f:
        for line in f:
            k, *v = line.split()
            if k == "map":
                d = dict(zip(v[1::2], v[2::2]))
                meta["maps"].append({"id": int(v[0]), **{a: int(b) for a, b in d.items()}})
            elif len(v) == 1:
                meta[k] = float(v[0])
            else:
                meta[k] = np.array(v, dtype=float)
    for k in ("Tbc", "Tlr_unrectified"):
        if k in meta:
            meta[k] = meta[k].reshape(4, 4)
    return meta


def quat_to_R(q: np.ndarray) -> np.ndarray:
    """(..., 4) quaternions as x, y, z, w -> (..., 3, 3)."""
    x, y, z, w = np.moveaxis(q, -1, 0)
    R = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                  2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                  2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)
    return R.reshape(q.shape[:-1] + (3, 3))


@dataclass
class Problem:
    """One map of an export, densely indexed. All arrays are numpy float64 / int64."""

    # camera (rectified left) and calibration as ORB-SLAM3 used it
    fx: float
    fy: float
    cx: float
    cy: float
    bf: float
    Tbc: np.ndarray                      # (4, 4) body <- camera, includes the rectification rotation
    imu_cov: np.ndarray                  # (6,) continuous-time noise variances, gyro then accel
    # keyframes
    kf_id: np.ndarray                    # (K,) ORB-SLAM3 ids
    t: np.ndarray                        # (K,) seconds
    Rwb: np.ndarray                      # (K, 3, 3)
    pwb: np.ndarray                      # (K, 3)
    v: np.ndarray                        # (K, 3)
    bias: np.ndarray                     # (K, 6) gyro then accel
    # map points
    mp_id: np.ndarray                    # (P,)
    X: np.ndarray                        # (P, 3)
    # observations
    obs_kf: np.ndarray                   # (N,) keyframe index
    obs_mp: np.ndarray                   # (N,) point index
    obs_uv: np.ndarray                   # (N, 2)
    obs_ur: np.ndarray                   # (N,) right-image column, < 0 for monocular
    obs_sigma2: np.ndarray               # (N,) pixel variance of the keypoint's pyramid level
    # IMU intervals (one per keyframe with a valid predecessor)
    imu_i: np.ndarray                    # (M,) keyframe index of the interval start
    imu_j: np.ndarray                    # (M,) keyframe index of the interval end
    imu_dt: np.ndarray                   # (M,)
    imu_C: np.ndarray                    # (M, 15, 15) ORB-SLAM3 covariance: dR, dV, dP, bg walk, ba walk
    imu_b0: np.ndarray                   # (M, 6) bias the preintegration was computed with
    imu_dR: np.ndarray                   # (M, 3, 3) ORB-SLAM3's own preintegration (for checks)
    imu_dV: np.ndarray                   # (M, 3)
    imu_dP: np.ndarray                   # (M, 3)
    # padded sample windows per interval (built by build_windows)
    win_a: np.ndarray = field(default=None)    # (M, L, 3)
    win_w: np.ndarray = field(default=None)    # (M, L, 3)
    win_s: np.ndarray = field(default=None)    # (M, L) sample start time
    win_dt: np.ndarray = field(default=None)   # (M, L) sample length, 0 for padding
    map_id: int = -1
    meta: dict = field(default_factory=dict)

    @property
    def n_kf(self) -> int:
        return len(self.kf_id)

    @property
    def n_mp(self) -> int:
        return len(self.mp_id)


def load_problem(export_dir: str, map_id: int | None = None, pad: int = 12) -> Problem:
    """Load one map (default: the largest, the one ORB-SLAM3's trajectory file holds).

    :param pad: samples added on each side of every IMU window, so a time offset of up to
        ``pad / imu_rate`` seconds can move the window edges.
    """
    meta = read_meta(os.path.join(export_dir, "meta.txt"))
    mid = int(meta["largest_map"]) if map_id is None else int(map_id)
    kf = read_csv(os.path.join(export_dir, "keyframes.csv"))
    mp = read_csv(os.path.join(export_dir, "mappoints.csv"))
    ob = read_csv(os.path.join(export_dir, "observations.csv"))
    pre = read_csv(os.path.join(export_dir, "imu_preint.csv"))
    raw = read_csv(os.path.join(export_dir, "imu_meas.csv"))

    ks = np.flatnonzero(kf["map_id"] == mid)
    ks = ks[np.argsort(kf["t"][ks])]
    kidx = {int(k): i for i, k in enumerate(kf["kf_id"][ks])}
    q = lambda p: np.stack([kf[f"{p}_{a}"][ks] for a in "xyzw"], -1)
    tr = lambda p: np.stack([kf[f"{p}_{a}"][ks] for a in "xyz"], -1)
    Rwb, pwb = quat_to_R(q("qwb")), tr("twb")

    # observations of this map whose keyframe and point both exist
    om = ob["map_id"] == mid
    okf = np.array([kidx.get(int(k), -1) for k in ob["kf_id"][om]])
    pm = np.flatnonzero(mp["map_id"] == mid)
    pidx = {int(p): i for i, p in enumerate(mp["mp_id"][pm])}
    omp = np.array([pidx.get(int(p), -1) for p in ob["mp_id"][om]])
    keep = (okf >= 0) & (omp >= 0) & (ob["cam"][om] == 0)
    sig2 = np.asarray(meta["level_sigma2"])[ob["octave"][om][keep].astype(int)]

    # IMU intervals of this map with both keyframes present
    rm = np.flatnonzero(pre["map_id"] == mid)
    ii = np.array([kidx.get(int(k), -1) for k in pre["prev_kf_id"][rm]])
    jj = np.array([kidx.get(int(k), -1) for k in pre["kf_id"][rm]])
    ok = (ii >= 0) & (jj >= 0)
    rm, ii, jj = rm[ok], ii[ok], jj[ok]
    order = np.argsort(kf["t"][ks][jj])
    rm, ii, jj = rm[order], ii[order], jj[order]
    col = lambda name, n: np.stack([pre[f"{name}{k}"][rm] for k in range(n)], -1)

    P = Problem(
        fx=meta["fx"], fy=meta["fy"], cx=meta["cx"], cy=meta["cy"], bf=meta["bf"],
        Tbc=np.asarray(meta["Tbc"]), imu_cov=np.asarray(meta["imu_cov"]),
        kf_id=kf["kf_id"][ks].astype(np.int64), t=kf["t"][ks], Rwb=Rwb, pwb=pwb, v=tr("v"),
        bias=np.stack([kf[c][ks] for c in ("bg_x", "bg_y", "bg_z", "ba_x", "ba_y", "ba_z")], -1),
        mp_id=mp["mp_id"][pm].astype(np.int64), X=np.stack([mp[a][pm] for a in "xyz"], -1),
        obs_kf=okf[keep], obs_mp=omp[keep],
        obs_uv=np.stack([ob["u"][om][keep], ob["v"][om][keep]], -1), obs_ur=ob["u_right"][om][keep],
        obs_sigma2=sig2,
        imu_i=ii, imu_j=jj, imu_dt=pre["dt"][rm], imu_C=col("C", 225).reshape(-1, 15, 15),
        imu_b0=np.stack([pre[c][rm] for c in ("bg_x", "bg_y", "bg_z", "ba_x", "ba_y", "ba_z")], -1),
        imu_dR=col("dR", 9).reshape(-1, 3, 3), imu_dV=col("dV", 3), imu_dP=col("dP", 3),
        map_id=mid, meta=meta,
    )
    _build_windows(P, raw, pre["map_id"][rm], pre["kf_id"][rm], pad)
    return P


def _build_windows(P: Problem, raw: dict, map_ids: np.ndarray, kf_ids: np.ndarray, pad: int) -> None:
    """Concatenate the intervals' samples (in time order) into one stream and cut a padded window per interval."""
    key = raw["map_id"] * 1e9 + raw["kf_id"]
    order = np.lexsort((raw["k"], key))
    key_s = key[order]
    A, W, S, DT, spans = [], [], [], [], []
    n0 = 0
    for r, (m, k) in enumerate(zip(map_ids, kf_ids)):
        lo, hi = np.searchsorted(key_s, m * 1e9 + k), np.searchsorted(key_s, m * 1e9 + k, side="right")
        idx = order[lo:hi]
        dt = raw["dt"][idx]
        s = P.t[P.imu_i[r]] + np.concatenate([[0.0], np.cumsum(dt)[:-1]])
        A.append(np.stack([raw[f"a_{c}"][idx] for c in "xyz"], -1))
        W.append(np.stack([raw[f"w_{c}"][idx] for c in "xyz"], -1))
        S.append(s)
        DT.append(dt)
        spans.append((n0, n0 + len(idx)))
        n0 += len(idx)
    M = len(spans)
    if M == 0:
        P.win_a = P.win_w = np.zeros((0, 1, 3))
        P.win_s = P.win_dt = np.zeros((0, 1))
        return
    A, W, S, DT = np.concatenate(A), np.concatenate(W), np.concatenate(S), np.concatenate(DT)
    L = (max(b - a for a, b in spans) if spans else 0) + 2 * pad
    P.win_a, P.win_w = np.zeros((M, L, 3)), np.zeros((M, L, 3))
    P.win_s, P.win_dt = np.zeros((M, L)), np.zeros((M, L))
    for r, (a, b) in enumerate(spans):
        lo, hi = max(a - pad, 0), min(b + pad, n0)
        n = hi - lo
        P.win_a[r, :n], P.win_w[r, :n] = A[lo:hi], W[lo:hi]
        P.win_s[r, :n], P.win_dt[r, :n] = S[lo:hi], DT[lo:hi]
        P.win_s[r, n:] = S[hi - 1] + DT[hi - 1] if n else 0.0   # padding: zero-length samples after the end
