#!/usr/bin/env python3
"""Phase 1.3: re-estimate the calibration from an exported ORB-SLAM3 map (dsg_jit.calib) for one run.

  python3 calib_reestimate.py <run_dir> [<run_dir> ...] [--modes oracle blind ident] [--force]

Each run directory holds ``export/`` (OASIS_EXPORT=1 run), ``record.json`` and, for perturbed runs,
``LOWLIGHT_MANIFEST.json`` + ``config.yaml`` (copied by phase2_suite.py). Writes ``calib_estimate.json``:

  oracle   free parameters = the perturbed group (the monitor of Phase 2 is assumed to name it);
           clean runs are solved once per group, as the reference for every arm
  blind    free = sg, sa, rbc, tbc, td on every run (nothing known about what is wrong)
  ident    marginal Fisher information of every parameter at the plain-BA solution (Phase 1.2)

``truth`` is the injected error expressed as the correction theta the estimator should find, in its own
coordinates (see dsg_jit/calib/factors.py): sg/sa/mg/ma relative, rbc rad (body frame), tbc m, td s.
TUM-VI runs (xtv<room>_<cal|nos|raw>_s<seed>) take their truth from the published IMU calibration
(tumvi_imu_variants.py): nos/raw data are M^-1 x, so the correction is M_model = M^-1.
The stereo-rotation arm (rst) changes the rectification, which the model does not represent; it is run in
oracle mode with rbc+tbc free to see what the estimator absorbs, and has no truth.
"""
import argparse, json, os, re, sys, time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from dsg_jit.calib import CalibBA, THETA_GROUPS, THETA_NAMES, N_THETA, load_problem  # noqa: E402

ORACLE = {"toff": ["td"], "isf": ["sg", "sa"], "isfs": ["sg", "sa"], "imis": ["sg", "mg", "sa", "ma"],
          "rbc": ["rbc"], "tbc": ["tbc"], "rst": ["rbc", "tbc"]}
ORACLE_KEY = {"tumvi_nos": "imis", "tumvi_raw": "imis"}         # same free set as imis
OFF = ((0, 0, 1, 1, 2, 2), (1, 2, 0, 2, 0, 1))
TUMVI_MA = np.array([[1.00422, 0, 0], [-7.82123e-05, 1.00136, 0], [-0.0097745, -0.000976476, 0.970467]])
TUMVI_MG = np.array([[0.943611, 0.00148681, 0.000824366], [0.000369694, 1.09413, -0.00273521],
                     [-0.00175252, 0.00834754, 1.01588]])
BLIND = ["sg", "sa", "rbc", "tbc", "td"]


def read_mat(text, key):
    m = re.search(rf"{re.escape(key)}:\s*!!opencv-matrix.*?data:\s*\[(.*?)\]", text, re.S)
    return np.array([float(x) for x in m.group(1).replace("\n", " ").split(",")]).reshape(4, 4)


def rotvec(axis, ang):
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def so3_log(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
    return v if th < 1e-12 else v * th / np.sin(th)


def intrinsics_theta(Sg, Sa):
    """meas = S true per sensor -> theta (sg, mg, sa, ma), with Tbc unchanged."""
    th = np.zeros(N_THETA)
    th[THETA_GROUPS["sg"]], th[THETA_GROUPS["mg"]] = np.diag(Sg) - 1, Sg[OFF]
    th[THETA_GROUPS["sa"]], th[THETA_GROUPS["ma"]] = np.diag(Sa) - 1, Sa[OFF]
    return th


def truth_theta(run_dir):
    """Injected error -> the correction theta that undoes it (None for clean or unmodeled arms)."""
    m = re.match(r"xtv\d+_(cal|nos|raw)_s\d+$", os.path.basename(os.path.normpath(run_dir)))
    if m:
        if m.group(1) == "cal":
            return None, "clean"
        return intrinsics_theta(np.linalg.inv(TUMVI_MG), np.linalg.inv(TUMVI_MA)), "tumvi_" + m.group(1)
    mf = os.path.join(run_dir, "LOWLIGHT_MANIFEST.json")
    if not os.path.exists(mf):
        return None, "clean"
    cfg = os.path.join(run_dir, "config.yaml")
    return truth_from_manifest(json.load(open(mf)), open(cfg).read() if os.path.exists(cfg) else None)


def truth_from_manifest(man, cfg_text):
    """calib_perturb.py manifest (+ the run's perturbed config text, for rbc) -> (theta, arm)."""
    arm, mag = man["arm"], man["magnitude"]
    th = np.zeros(N_THETA)
    if arm == "toff":                       # IMU stamps moved by sign*mag ms -> td = +shift
        th[THETA_GROUPS["td"]] = man["sign"] * mag / 1000.0
    elif arm in ("isf", "isfs"):            # meas = (1 + s) true, gyro then accel
        s = np.asarray(man["scale_errors"])
        th[THETA_GROUPS["sg"]], th[THETA_GROUPS["sa"]] = s[:3], s[3:]
    elif arm == "rbc":                      # config R_bc = R_true Rot(axis, mag): correction = -R_true axis mag
        Rcfg = read_mat(cfg_text, "IMU.T_b_c1")[:3, :3]
        ax, ang = np.asarray(man["axis"]), np.radians(mag)
        Rtrue = Rcfg @ rotvec(ax, ang).T
        th[THETA_GROUPS["rbc"]] = so3_log(Rtrue @ Rcfg.T)
    elif arm == "tbc":                      # config t_bc = t_true + d mag
        th[THETA_GROUPS["tbc"]] = -np.asarray(man["direction"]) * mag / 100.0
    elif arm == "imis":                     # meas = S true, S = I + off, per sensor
        Sg, Sa = np.eye(3), np.eye(3)
        Sg[OFF], Sa[OFF] = man["gyro_skew"], man["accel_skew"]
        th = intrinsics_theta(Sg, Sa)
    else:
        return None, arm
    return th, arm


def run_one(run_dir, modes, force=False, oracle_key=None):
    out = os.path.join(run_dir, "calib_estimate.json")
    if os.path.exists(out) and not force:
        return json.load(open(out))
    exp = os.path.join(run_dir, "export")
    rec = {"run": os.path.basename(os.path.normpath(run_dir)), "theta_names": THETA_NAMES}
    if not os.path.exists(os.path.join(exp, "meta.txt")) and not os.path.exists(os.path.join(exp, "meta.txt.gz")):
        rec["error"] = "no export"
        json.dump(rec, open(out, "w"))
        return rec
    truth, arm = truth_theta(run_dir)
    rec["arm"], rec["truth"] = arm, None if truth is None else truth.tolist()
    try:
        P = load_problem(exp)
    except Exception as e:                  # noqa: BLE001 -- record and move on
        rec["error"] = f"load: {e}"
        json.dump(rec, open(out, "w"))
        return rec
    rec.update(n_kf=P.n_kf, n_mp=P.n_mp, n_obs=int(len(P.obs_kf)), n_imu=int(len(P.imu_i)),
               span_s=float(P.t[-1] - P.t[0]) if P.n_kf else 0.0, imu_init=bool(len(P.imu_i) > 0))
    if P.n_kf < 10:
        rec["error"] = "map too small"
        json.dump(rec, open(out, "w"))
        return rec

    def solve(groups):
        ba = CalibBA(P, free=groups)
        r = ba.solve(max_iter=40)
        d = {"free": groups, "theta": r.theta.tolist(), "cost0": r.cost0, "cost": r.cost,
             "iterations": r.iterations, "seconds": round(r.seconds, 2), "parts": ba.evaluate()[1],
             "imu_gated": int(ba.imu_gated.sum())}
        if groups:
            try:
                I, idx = ba.marginal_information(groups)
                d["crlb"] = dict(zip([THETA_NAMES[k] for k in idx], np.sqrt(np.diag(np.linalg.inv(I))).tolist()))
            except np.linalg.LinAlgError:
                d["crlb"] = None
        return d, ba

    t0 = time.time()
    if "ident" in modes:
        d, ba = solve([])
        I, idx = ba.marginal_information()
        dI = np.sqrt(np.clip(np.diag(I), 1e-300, None))
        rec["ident"] = {"plain_ba": d, "info": I.tolist(),
                        "crlb_all": np.sqrt(np.clip(np.diag(np.linalg.pinv(I)), 0, None)).tolist(),
                        "eig_normalized": np.linalg.eigvalsh(I / np.outer(dI, dI)).tolist()}
    if "oracle" in modes:
        if oracle_key:                       # e.g. an already-corrected run: no manifest, but we know the arm
            groups_list = [(oracle_key, ORACLE[oracle_key])]
        elif arm == "clean":
            groups_list = list(ORACLE.items())
        else:
            groups_list = [(ORACLE_KEY.get(arm, arm), ORACLE.get(ORACLE_KEY.get(arm, arm)))]
        rec["oracle"] = {}
        for key, groups in groups_list:
            if groups and key not in rec["oracle"]:
                rec["oracle"][key] = solve(groups)[0]
    if "blind" in modes:
        rec["blind"] = solve(BLIND)[0]
    rec["seconds"] = round(time.time() - t0, 1)
    json.dump(rec, open(out, "w"))
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--modes", nargs="+", default=["oracle", "blind", "ident"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--oracle-key", choices=sorted(ORACLE), default=None,
                    help="solve only this oracle group (for runs without a perturbation manifest)")
    a = ap.parse_args()
    for r in a.runs:
        t = time.time()
        rec = run_one(r, a.modes, a.force, a.oracle_key)
        msg = rec.get("error") or f"{rec.get('arm')} K {rec.get('n_kf')}"
        if "oracle" in rec and rec.get("truth") is not None:
            d = next(iter(rec["oracle"].values()))
            k = sorted({i for g in d["free"] for i in THETA_GROUPS[g]})
            msg += "  est-truth " + " ".join(f"{THETA_NAMES[i]} {d['theta'][i] - rec['truth'][i]:+.2e}" for i in k)
        print(f"{time.strftime('%H:%M:%S')} {os.path.basename(os.path.normpath(r))}: {msg} ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
