#!/usr/bin/env python3
"""Absolute Trajectory Error against EuRoC ground truth.

Deliberately self-contained rather than reusing ORB_SLAM3/evaluation/
evaluate_ate_scale.py, which is Python-2 era and expects TUM-format ground
truth. EuRoC ships ASL CSV, which differs in two ways that silently corrupt
results if missed:
  * column order is  ts, px,py,pz, qw,qx,qy,qz   (TUM is qx,qy,qz,qw)
  * timestamps are integer nanoseconds

Alignment is SE(3) with scale FIXED AT 1. For stereo-inertial, metric scale is
observable, so solving for scale would hide genuine scale error. --with-scale
is available only as a diagnostic.
"""
import argparse, sys
import numpy as np


def load_tum(path):
    """ORB-SLAM3 output: ts tx ty tz qx qy qz qw (ts in ns)."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 8:
                continue
            rows.append([float(x) for x in p[:8]])
    a = np.asarray(rows)
    return a[:, 0], a[:, 1:4]


def load_euroc_gt(path):
    """EuRoC ASL: ts, px,py,pz, qw,qx,qy,qz, ..."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split(",")
            if len(p) < 8:
                continue
            rows.append([float(p[0])] + [float(x) for x in p[1:4]])
    a = np.asarray(rows)
    return a[:, 0], a[:, 1:4]


def associate(ts_a, ts_b, max_diff_ns):
    """Nearest-neighbour timestamp association (both sorted, ns)."""
    idx = np.searchsorted(ts_b, ts_a)
    idx = np.clip(idx, 1, len(ts_b) - 1)
    left, right = ts_b[idx - 1], ts_b[idx]
    pick = np.where(np.abs(ts_a - left) < np.abs(ts_a - right), idx - 1, idx)
    keep = np.abs(ts_b[pick] - ts_a) <= max_diff_ns
    return np.nonzero(keep)[0], pick[keep]


def umeyama(src, dst, with_scale=False):
    """Least-squares similarity transform mapping src -> dst."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    cov = (D.T @ S) / len(src)
    U, d, Vt = np.linalg.svd(cov)
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:   # reflection guard
        W[2, 2] = -1
    R = U @ W @ Vt
    s = (np.trace(np.diag(d) @ W) / ((S ** 2).sum() / len(src))) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("estimate"); ap.add_argument("groundtruth")
    ap.add_argument("--max-diff-ms", type=float, default=20.0)
    ap.add_argument("--with-scale", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    ts_e, xyz_e = load_tum(a.estimate)
    ts_g, xyz_g = load_euroc_gt(a.groundtruth)
    o = np.argsort(ts_g); ts_g, xyz_g = ts_g[o], xyz_g[o]

    ie, ig = associate(ts_e, ts_g, a.max_diff_ms * 1e6)
    if len(ie) < 10:
        print(f"ERROR: only {len(ie)} associations", file=sys.stderr); sys.exit(1)

    s, R, t = umeyama(xyz_e[ie], xyz_g[ig], a.with_scale)
    err = np.linalg.norm((s * (R @ xyz_e[ie].T).T + t) - xyz_g[ig], axis=1)

    rmse = float(np.sqrt((err ** 2).mean()))
    if a.quiet:
        print(f"{rmse:.6f}")
    else:
        print(f"estimate    : {a.estimate}")
        print(f"associated  : {len(ie)} / {len(ts_e)} est poses")
        print(f"scale       : {s:.6f}" + ("" if a.with_scale else "  (fixed)"))
        print(f"ATE RMSE    : {rmse:.6f} m")
        print(f"  mean/med  : {err.mean():.6f} / {np.median(err):.6f} m")
        print(f"  max       : {err.max():.6f} m")


if __name__ == "__main__":
    main()
