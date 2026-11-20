#!/usr/bin/env python3
"""Load and check an ORB-SLAM3 estimation-problem export (Phase 1.1).

The export is written by System::SaveEstimationProblem in the OASIS fork (branch phase1-export) when a run
is started with OASIS_EXPORT=1 (see run_oasis.sh). It holds, per map: keyframes (Tcw, Twb, velocity, bias),
map points, keyframe -> map point observations, and the IMU preintegration between consecutive keyframes
together with the raw samples behind it.

  python3 orb_export.py <run_dir>/export [--map largest|all|<id>]

Checks, each against what ORB-SLAM3 itself optimized (so a correct export must pass on a clean run):
  reprojection   pixel residual of every observation under the exported poses and intrinsics
  imu residual   preintegration residual (rotation, velocity, position) under the exported states
  reintegration  re-integrating the raw samples with the preintegration bias reproduces dR, dV, dP;
                 this is what calibration re-estimation does with a changed IMU scale or misalignment
Exit status 1 if a check fails its threshold.
"""
import argparse, gzip, os, sys
import numpy as np

GRAVITY = np.array([0.0, 0.0, -9.81])


def _open(path):
    """path, or path.gz when the export has been compressed (np.loadtxt reads .gz itself)."""
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        path += ".gz"
    return path, (gzip.open(path, "rt") if path.endswith(".gz") else open(path))


def read_csv(path):
    path, f = _open(path)
    with f:
        names = f.readline().strip().split(",")
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if data.size == 0:
        data = np.zeros((0, len(names)))
    return {n: data[:, i] for i, n in enumerate(names)}, data, names


def read_meta(path):
    meta = {"maps": []}
    for line in _open(path)[1]:
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


def quat_to_R(q):  # q (..., 4) as x, y, z, w
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


def so3_exp(w):
    th = np.linalg.norm(w)
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    if th < 1e-5:
        return np.eye(3) + K + 0.5 * K @ K
    return np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th ** 2 * K @ K


def so3_log(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
    return v if th < 1e-5 else v * th / np.sin(th)


def load(export_dir):
    """Everything in the export as numpy arrays, keyed by name. Rows of observations etc. keep their map id."""
    E = {"meta": read_meta(os.path.join(export_dir, "meta.txt"))}
    for name in ("keyframes", "mappoints", "observations", "imu_preint", "imu_meas"):
        E[name], _, _ = read_csv(os.path.join(export_dir, name + ".csv"))
    kf = E["keyframes"]
    E["Rcw"] = quat_to_R(np.stack([kf[f"qcw_{a}"] for a in "xyzw"], -1))
    E["tcw"] = np.stack([kf[f"tcw_{a}"] for a in "xyz"], -1)
    E["Rwb"] = quat_to_R(np.stack([kf[f"qwb_{a}"] for a in "xyzw"], -1))
    E["pwb"] = np.stack([kf[f"twb_{a}"] for a in "xyz"], -1)
    E["v"] = np.stack([kf[f"v_{a}"] for a in "xyz"], -1)
    E["bias"] = np.stack([kf[k] for k in ("bg_x", "bg_y", "bg_z", "ba_x", "ba_y", "ba_z")], -1)
    return E


def select_map(E, which):
    m = E["meta"]
    if which == "largest":
        return int(m["largest_map"])
    return None if which == "all" else int(which)


def reprojection(E, map_id):
    m, kf, mp, ob = E["meta"], E["keyframes"], E["mappoints"], E["observations"]
    sel = (ob["map_id"] == map_id) if map_id is not None else np.ones(len(ob["kf_id"]), bool)
    kidx = {(a, b): i for i, (a, b) in enumerate(zip(kf["map_id"], kf["kf_id"]))}
    pidx = {(a, b): i for i, (a, b) in enumerate(zip(mp["map_id"], mp["mp_id"]))}
    rows = [(kidx.get((a, b)), pidx.get((a, c))) for a, b, c in zip(ob["map_id"][sel], ob["kf_id"][sel], ob["mp_id"][sel])]
    ok = np.array([i is not None and j is not None for i, j in rows])
    ki = np.array([i for (i, j), o in zip(rows, ok) if o], int)
    pi = np.array([j for (i, j), o in zip(rows, ok) if o], int)
    X = np.stack([mp[a] for a in "xyz"], -1)[pi]
    Pc = np.einsum("nij,nj->ni", E["Rcw"][ki], X) + E["tcw"][ki]
    z = Pc[:, 2]
    u = m["fx"] * Pc[:, 0] / z + m["cx"]
    v = m["fy"] * Pc[:, 1] / z + m["cy"]
    uo, vo, ur = ob["u"][sel][ok], ob["v"][sel][ok], ob["u_right"][sel][ok]
    e = np.hypot(u - uo, v - vo)
    st = ur >= 0
    er = np.abs((u - m["bf"] / z) - ur)[st]
    sig2 = m["level_sigma2"][ob["octave"][sel][ok].astype(int)]
    chi2 = ((u - uo) ** 2 + (v - vo) ** 2) / sig2
    return {"n_obs": int(ok.sum()), "n_unmatched": int((~ok).sum()), "behind_camera": int((z <= 0).sum()),
            "px_median": float(np.median(e)), "px_rms": float(np.sqrt(np.mean(e ** 2))),
            "px_p95": float(np.percentile(e, 95)), "chi2_mono_outliers": float(np.mean(chi2 > 5.991)),
            "stereo_frac": float(st.mean()), "u_right_px_median": float(np.median(er)) if st.any() else None}


def imu_residuals(E, map_id):
    kf, pre = E["keyframes"], E["imu_preint"]
    kidx = {(a, b): i for i, (a, b) in enumerate(zip(kf["map_id"], kf["kf_id"]))}
    eR, eV, eP = [], [], []
    for r in range(len(pre["kf_id"])):
        mid = pre["map_id"][r]
        if map_id is not None and mid != map_id:
            continue
        i, j = kidx.get((mid, pre["prev_kf_id"][r])), kidx.get((mid, pre["kf_id"][r]))
        if i is None or j is None:
            continue
        g = lambda name, n=9: np.array([pre[f"{name}{k}"][r] for k in range(n)])
        dR, dV, dP = g("dR").reshape(3, 3), g("dV", 3), g("dP", 3)
        JRg, JVg, JVa, JPg, JPa = (g(n).reshape(3, 3) for n in ("JRg", "JVg", "JVa", "JPg", "JPa"))
        b0 = np.array([pre[k][r] for k in ("bg_x", "bg_y", "bg_z", "ba_x", "ba_y", "ba_z")])
        db = E["bias"][i] - b0  # ORB-SLAM3 evaluates the edge at the bias of the first keyframe
        dt = pre["dt"][r]
        Ri, Rj, vi, vj, pi_, pj = E["Rwb"][i], E["Rwb"][j], E["v"][i], E["v"][j], E["pwb"][i], E["pwb"][j]
        dRc = dR @ so3_exp(JRg @ db[:3])
        eR.append(so3_log(dRc.T @ Ri.T @ Rj))
        eV.append(Ri.T @ (vj - vi - GRAVITY * dt) - (dV + JVg @ db[:3] + JVa @ db[3:]))
        eP.append(Ri.T @ (pj - pi_ - vi * dt - 0.5 * GRAVITY * dt * dt) - (dP + JPg @ db[:3] + JPa @ db[3:]))
    if not eR:
        return {"n_intervals": 0}
    eR, eV, eP = (np.linalg.norm(np.array(a), axis=1) for a in (eR, eV, eP))
    return {"n_intervals": len(eR), "rot_deg_median": float(np.degrees(np.median(eR))),
            "vel_mps_median": float(np.median(eV)), "pos_m_median": float(np.median(eP)),
            "rot_deg_p95": float(np.degrees(np.percentile(eR, 95))), "pos_m_p95": float(np.percentile(eP, 95))}


def integrate(a, w, dt, bias):
    """ORB-SLAM3's IMU::Preintegrated::IntegrateNewMeasurement, without covariance: dP and dV are updated
    with the previous dR, then dR."""
    dR, dV, dP = np.eye(3), np.zeros(3), np.zeros(3)
    for ak, wk, tk in zip(a, w, dt):
        acc = ak - bias[3:]
        dP = dP + dV * tk + 0.5 * dR @ acc * tk * tk
        dV = dV + dR @ acc * tk
        dR = dR @ so3_exp((wk - bias[:3]) * tk)
    return dR, dV, dP


def reintegration(E, map_id, max_intervals=200):
    pre, raw = E["imu_preint"], E["imu_meas"]
    key = raw["map_id"] * 1e9 + raw["kf_id"]
    order = np.argsort(key, kind="stable")
    keys_sorted = key[order]
    worst = np.zeros(3)
    dt_err = 0.0
    n = 0
    for r in range(len(pre["kf_id"])):
        if map_id is not None and pre["map_id"][r] != map_id:
            continue
        k = pre["map_id"][r] * 1e9 + pre["kf_id"][r]
        lo, hi = np.searchsorted(keys_sorted, k), np.searchsorted(keys_sorted, k, side="right")
        idx = order[lo:hi]
        idx = idx[np.argsort(raw["k"][idx])]
        a = np.stack([raw[f"a_{c}"][idx] for c in "xyz"], -1)
        w = np.stack([raw[f"w_{c}"][idx] for c in "xyz"], -1)
        b0 = np.array([pre[c][r] for c in ("bg_x", "bg_y", "bg_z", "ba_x", "ba_y", "ba_z")])
        dR, dV, dP = integrate(a, w, raw["dt"][idx], b0)
        g = lambda name, n=9: np.array([pre[f"{name}{j}"][r] for j in range(n)])
        worst = np.maximum(worst, [np.degrees(np.linalg.norm(so3_log(g("dR").reshape(3, 3).T @ dR))),
                                   np.linalg.norm(dV - g("dV", 3)), np.linalg.norm(dP - g("dP", 3))])
        dt_err = max(dt_err, abs(raw["dt"][idx].sum() - pre["dt"][r]))
        n += 1
        if n >= max_intervals:
            break
    return {"n_checked": n, "max_rot_deg": float(worst[0]), "max_vel_mps": float(worst[1]),
            "max_pos_m": float(worst[2]), "max_dt_s": float(dt_err)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export_dir")
    ap.add_argument("--map", default="largest", help="largest (default), all, or a map id")
    a = ap.parse_args()
    E = load(a.export_dir)
    m = E["meta"]
    mid = select_map(E, a.map)
    print(f"maps: " + ", ".join(f"{d['id']} ({d['kfs']} kf, {d['mps']} mp, imu_init {d['imu_init']}, ba2 {d['inertial_ba2']})"
                                for d in m["maps"]) + f"; checking map {a.map if mid is None else mid}")
    print(f"camera: fx {m['fx']:.3f} fy {m['fy']:.3f} cx {m['cx']:.3f} cy {m['cy']:.3f} bf {m['bf']:.3f} "
          f"(baseline {m['b']:.4f} m), rectified {bool(m.get('need_to_rectify', 0))}")
    kf = E["keyframes"]
    sel = kf["map_id"] == mid if mid is not None else np.ones(len(kf["kf_id"]), bool)
    print(f"keyframes {sel.sum()}  span {np.ptp(kf['t'][sel]):.1f} s  with imu {int(kf['has_imu'][sel].sum())}")
    rp, ir, ri = reprojection(E, mid), imu_residuals(E, mid), reintegration(E, mid)
    for name, d in (("reprojection", rp), ("imu residual", ir), ("reintegration", ri)):
        print(f"{name:14s} " + "  ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in d.items()))
    fails = []
    if rp["px_median"] > 2.0:
        fails.append("reprojection median > 2 px")
    if ri["n_checked"] and (ri["max_rot_deg"] > 1e-2 or ri["max_vel_mps"] > 1e-3 or ri["max_pos_m"] > 1e-3):
        fails.append("reintegration does not reproduce the preintegration")
    if ir.get("n_intervals") and ir["rot_deg_median"] > 0.5:
        fails.append("imu rotation residual median > 0.5 deg")
    print("PASS" if not fails else "FAIL: " + "; ".join(fails))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
