#!/usr/bin/env python3
"""Re-score SLAM runs on the real EuRoC sequences (MH03-05).

phase2_eval_run.py's validity rule (tracked >= 90% of frames) was tuned on
MH01, where stereo-inertial initialisation costs < 1 s. On MH03-05 the active
map is reset during IMU initialisation and the saved trajectory starts 11-21 s
in, in EVERY arm, with no gaps afterwards -- so that rule marks every run
invalid for a reason unrelated to darkness. Start-up delay also differs between
runs, so full-trajectory ATEs cover different stretches of flight and are not
comparable. This script therefore reports, per run:

  start_s        first pose relative to the first image
  gap_s          total time in mid-sequence gaps > 0.2 s (real tracking loss)
  ate_common_m   ATE (SE(3), scale fixed) on a window common to ALL runs of
                 the sequence: [latest start + 0.5 s, end]
  rpe_dark_m /   mean translational drift over 1 s windows inside that common
  rpe_bright_m   window, split by the worst equivalent darkness level of the
                 window's frames (phase0_real_dark per-frame CSV): dark = any
                 frame >= --dark-level. Drift is the error of the 1 s
                 displacement after the global alignment, so it measures local
                 tracking quality where the darkness actually is.

    rescore_real.py --tag rMH04 --gt .../data.csv --levels MH_04_difficult_per_frame.csv
"""
import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_ate import load_tum, load_euroc_gt, associate, umeyama  # noqa: E402


def frame_levels(path):
    """Worst-camera equivalent level per timestamp."""
    import csv
    lv = {}
    for r in csv.DictReader(open(path)):
        t, v = int(r["ts"]), float(r["eq_level"])
        if np.isfinite(v):
            lv[t] = max(lv.get(t, -1.0), v)
    ts = np.array(sorted(lv), dtype=np.int64)
    return ts, np.array([lv[t] for t in ts])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True); ap.add_argument("--gt", required=True)
    ap.add_argument("--levels", required=True, help="phase0_real_dark per-frame CSV for this sequence")
    ap.add_argument("--images", required=True, help="cam0/data dir (first image time)")
    ap.add_argument("--results", default="/data/tkocher/oasis_results")
    ap.add_argument("--dark-level", type=float, default=3.0)
    ap.add_argument("--rpe-dt", type=float, default=1.0)
    a = ap.parse_args()

    t0 = min(int(os.path.basename(f)[:-4]) for f in glob.glob(os.path.join(a.images, "*.png")))
    ts_g, xyz_g = load_euroc_gt(a.gt); o = np.argsort(ts_g); ts_g, xyz_g = ts_g[o], xyz_g[o]
    lts, lvl = frame_levels(a.levels)

    runs = []
    for d in sorted(glob.glob(os.path.join(a.results, f"{a.tag}_L*"))):
        rec_p, tr = os.path.join(d, "record.json"), glob.glob(os.path.join(d, "f_*.txt"))
        if not (os.path.exists(rec_p) and tr):
            continue
        ts, xyz = load_tum(tr[0])
        runs.append((json.load(open(rec_p)), ts, xyz))
    t_common = max(ts[0] for _, ts, _ in runs) + 0.5e9

    for rec, ts, xyz in runs:
        g = np.diff(ts) / 1e9
        out = dict(run_id=rec["run_id"], arm=rec["arm"], seed=rec["seed"],
                   start_s=float((ts[0] - t0) / 1e9), gap_s=float(g[g > 0.2].sum()),
                   common_start_s=float((t_common - t0) / 1e9))
        keep = ts >= t_common
        ie, ig = associate(ts[keep], ts_g, 20e6)
        pe, pg, te = xyz[keep][ie], xyz_g[ig], ts[keep][ie]
        s, R, t = umeyama(pe, pg)
        pa = (R @ pe.T).T + t
        out["ate_common_m"] = float(np.sqrt((np.linalg.norm(pa - pg, axis=1) ** 2).mean()))
        # 1 s relative displacement error, labelled by darkness of the frames it spans
        j = np.searchsorted(te, te + a.rpe_dt * 1e9)
        ok = j < len(te)
        i, j = np.nonzero(ok)[0], j[ok]
        err = np.linalg.norm((pa[j] - pa[i]) - (pg[j] - pg[i]), axis=1)
        dark = np.array([lvl[(lts >= te[x]) & (lts <= te[y])].max(initial=0) >= a.dark_level for x, y in zip(i, j)])
        out["rpe_dark_m"] = float(err[dark].mean()) if dark.any() else None
        out["rpe_bright_m"] = float(err[~dark].mean()) if (~dark).any() else None
        out["n_dark_windows"] = int(dark.sum())
        print(json.dumps(out))


if __name__ == "__main__":
    main()
