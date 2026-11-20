#!/usr/bin/env python3
"""Phase 1.4: apply an estimated calibration correction to a (perturbed) EuRoC-format sequence.

  python3 calib_correct.py --src <perturbed seq> --dst <out> --estimate <run>/calib_estimate.json [--mode oracle]

The estimate's theta is the correction on top of what ORB-SLAM3 ran with (dsg_jit/calib/factors.py):
  IMU:     true = M^-1 meas per sensor (M = I + diag(s) + offdiag(m)); stamps t - td
  config:  IMU.T_b_c1 rotation Exp(rbc) R, translation t + tbc
Biases are left to ORB-SLAM3. Images are symlinked, as in calib_perturb.py. Writes config.yaml and
CORRECTION_MANIFEST.json (the theta applied).
"""
import argparse, json, os, re, shutil, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from calib_perturb import LINK, read_mat, write_mat  # noqa: E402

OFF = ((0, 0, 1, 1, 2, 2), (1, 2, 0, 2, 0, 1))


def so3_exp(w):
    th = np.linalg.norm(w)
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    if th < 1e-12:
        return np.eye(3) + K
    return np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th ** 2 * K @ K


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--estimate", required=True)
    ap.add_argument("--mode", default="oracle", choices=["oracle", "blind"])
    ap.add_argument("--key", default=None, help="oracle entry to apply (default: the first)")
    ap.add_argument("--yaml", default=None, help="config to correct (default: <src>/config.yaml)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    est = json.load(open(a.estimate))
    if a.mode == "blind":
        d = est["blind"]
    else:
        d = est["oracle"][a.key] if a.key else next(iter(est["oracle"].values()))
    th = np.asarray(d["theta"])
    if os.path.exists(a.dst):
        if not a.force:
            sys.exit(f"{a.dst} exists (use --force)")
        shutil.rmtree(a.dst)
    sm, dm = os.path.join(a.src, "mav0"), os.path.join(a.dst, "mav0")
    os.makedirs(os.path.join(dm, "imu0"))
    for e in LINK:
        if os.path.exists(os.path.join(sm, e)):
            os.symlink(os.path.realpath(os.path.join(sm, e)), os.path.join(dm, e))

    imu = np.loadtxt(os.path.join(sm, "imu0", "data.csv"), delimiter=",", comments="#", dtype=np.float64)
    ts = np.loadtxt(os.path.join(sm, "imu0", "data.csv"), delimiter=",", comments="#", usecols=0, dtype=np.int64)
    Mg, Ma = np.eye(3) + np.diag(th[6:9]), np.eye(3) + np.diag(th[15:18])
    Mg[OFF], Ma[OFF] = Mg[OFF] + th[9:15], Ma[OFF] + th[18:24]
    w = np.linalg.solve(Mg, imu[:, 1:4].T).T
    acc = np.linalg.solve(Ma, imu[:, 4:7].T).T
    ts = ts - int(round(th[24] * 1e9))
    with open(os.path.join(dm, "imu0", "data.csv"), "w") as f:
        f.write("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
                "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n")
        for t, g, x in zip(ts, w, acc):
            f.write(f"{t}," + ",".join(f"{v:.17g}" for v in np.concatenate([g, x])) + "\n")

    cfg = open(a.yaml or os.path.join(a.src, "config.yaml")).read()
    T, _ = read_mat(cfg, "IMU.T_b_c1")
    T[:3, :3] = so3_exp(th[0:3]) @ T[:3, :3]
    T[:3, 3] = T[:3, 3] + th[3:6]
    cfg = write_mat(cfg, "IMU.T_b_c1", T)
    open(os.path.join(a.dst, "config.yaml"), "w").write(cfg)
    json.dump({"estimate": os.path.abspath(a.estimate), "mode": a.mode, "free": d["free"], "theta": th.tolist(),
               "source": os.path.abspath(a.src)}, open(os.path.join(a.dst, "CORRECTION_MANIFEST.json"), "w"), indent=1)
    print(f"{a.dst}: corrected ({a.mode}, free {d['free']})")


if __name__ == "__main__":
    main()
