#!/usr/bin/env python3
"""Replay test for the Phase 1 estimator: inject the Phase 0 perturbations into a clean export, offline.

  python3 calib_replay.py <run_dir>/export [...] [--tiers 2 3] [--seeds 0 1 2] [--out replay.json]

For each arm/tier/seed, the perturbation is drawn exactly as calib_perturb.py draws it (same tiers, same
seeded directions) and applied to the export instead of the sequence: IMU samples are scaled, misaligned or
re-stamped; Tbc is rotated or shifted the way the perturbed config would have it after rectification. The
estimator (oracle free set) then runs from the exported states, and its estimate minus the clean estimate
(same export, same free set) is compared with the truth from calib_reestimate.truth_from_manifest.

What this does NOT test: ORB-SLAM3 tracking with the wrong calibration. In the real sweep the exported
states were optimized under the wrong calibration (and may be worse or fragmented); here they are the clean
states. It isolates the estimator and the truth conversions from that.
"""
import argparse, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import calib_perturb as cp  # noqa: E402
from calib_reestimate import ORACLE, OFF, truth_from_manifest  # noqa: E402
from dsg_jit.calib import CalibBA, THETA_GROUPS, load_problem  # noqa: E402

# EuRoC cam0 extrinsics (mav0/cam0/sensor.yaml T_BS), unrectified; the export's Tbc is this times the
# rectification rotation, so only R_true is needed to express a config-side rotation in export coordinates.
EUROC_T_BS = np.array([[0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
                       [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
                       [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
                       [0, 0, 0, 1.0]])
UNITS = {"rbc": np.degrees(1), "tbc": 100, "sg": 100, "sa": 100, "mg": np.degrees(1), "ma": np.degrees(1), "td": 1000}


def cfg_text(T):
    rows = ",\n         ".join(",".join(f"{v:.15f}" for v in T[r]) for r in range(4))
    return f"IMU.T_b_c1: !!opencv-matrix\n   rows: 4\n   cols: 4\n   dt: f\n   data: [{rows}]\n"


def draw(arm, level, seed):
    """The manifest calib_perturb.py would write, and the perturbed T_b_c1."""
    unit_name, mags = cp.TIERS[arm]
    mag = mags[level - 1]
    rng = np.random.default_rng([seed, cp.ARM_ID["isf" if arm == "isfs" else arm]])
    man, T = {"arm": arm, "level": level, "seed": seed, "magnitude": mag}, EUROC_T_BS.copy()
    if arm == "tbc":
        d = cp.unit(rng); T[:3, 3] += d * mag / 100.0; man["direction"] = d.tolist()
    elif arm == "rbc":
        ax = cp.unit(rng); T[:3, :3] = T[:3, :3] @ cp.rotvec(ax, np.radians(mag)); man["axis"] = ax.tolist()
    elif arm == "toff":
        man["sign"] = 1 if seed % 2 == 0 else -1
    elif arm == "isf":
        man["scale_errors"] = (rng.choice([-1.0, 1.0], size=6) * mag / 100.0).tolist()
    elif arm == "isfs":
        man["scale_errors"] = (rng.choice([-1.0, 1.0], size=6) * np.repeat(mag, 3) / 100.0).tolist()
    elif arm == "imis":
        m = np.radians(mag)
        for key in ("gyro_skew", "accel_skew"):
            man[key] = (rng.choice([-1.0, 1.0], size=6) * m).tolist()
    return man, T


def apply(P, man, T):
    arm = man["arm"]
    if arm == "toff":
        P.win_s = P.win_s + man["sign"] * man["magnitude"] / 1000.0
    elif arm in ("isf", "isfs"):
        s = np.asarray(man["scale_errors"])
        P.win_w, P.win_a = P.win_w * (1 + s[:3]), P.win_a * (1 + s[3:])
    elif arm == "imis":
        Sg, Sa = np.eye(3), np.eye(3)
        Sg[OFF], Sa[OFF] = man["gyro_skew"], man["accel_skew"]
        P.win_w, P.win_a = P.win_w @ Sg.T, P.win_a @ Sa.T
    elif arm in ("rbc", "tbc"):
        Rt = EUROC_T_BS[:3, :3]
        P.Tbc = P.Tbc.copy()
        P.Tbc[:3, :3] = (T[:3, :3] @ Rt.T) @ P.Tbc[:3, :3]          # = R_cfg R_rect
        P.Tbc[:3, 3] = P.Tbc[:3, 3] + (T[:3, 3] - EUROC_T_BS[:3, 3])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exports", nargs="+")
    ap.add_argument("--arms", nargs="+", default=["toff", "rbc", "tbc", "isf", "isfs", "imis"])
    ap.add_argument("--tiers", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = []
    for exp in a.exports:
        name = os.path.basename(os.path.dirname(os.path.normpath(exp)))
        ref = {}
        for arm in a.arms:
            P = load_problem(exp)
            ref[arm] = CalibBA(P, free=ORACLE[arm]).solve(max_iter=40).theta
        for arm in a.arms:
            for lv in a.tiers:
                for sd in a.seeds:
                    man, T = draw(arm, lv, sd)
                    truth, _ = truth_from_manifest(man, cfg_text(T))
                    P = load_problem(exp)
                    apply(P, man, T)
                    r = CalibBA(P, free=ORACLE[arm]).solve(max_iter=40)
                    est = r.theta - ref[arm]
                    row = {"export": name, "arm": arm, "tier": lv, "seed": sd, "iterations": r.iterations}
                    msg = []
                    for g in ORACLE[arm]:
                        k = THETA_GROUPS[g]
                        e = (est[k] - truth[k]) * UNITS[g]
                        row[g] = {"truth": (truth[k] * UNITS[g]).tolist(), "error": e.tolist()}
                        msg.append(f"{g} truth {np.abs(truth[k]*UNITS[g]).max():.3g} err {np.abs(e).max():.3g}")
                    rows.append(row)
                    print(f"{name} {arm} L{lv} s{sd}: " + "; ".join(msg), flush=True)
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
