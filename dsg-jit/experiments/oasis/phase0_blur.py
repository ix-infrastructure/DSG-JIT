#!/usr/bin/env python3
"""How much REAL motion blur is in a EuRoC sequence, and does it cost features?

EuRoC does not publish exposure times, so blur length in pixels cannot be
computed directly. Two things can be measured per frame:

  blur RATE (px/s), predicted from the IMU: the gyro (mean over the frame's
      +/-25 ms) rotated into the camera frame; image-plane speed from rotation
      is f*|w_xy| at the centre plus r*|w_z| from roll at the rms radius r.
      Translation is ignored (it needs depth), so this is a lower bound.
      Blur length = rate * exposure; at 1-20 ms exposure, 500 px/s is
      0.5-10 px.
  SHARPNESS, two ways: the Crete et al. (2007) no-reference blur metric
      (0 = sharp, 1 = fully blurred) and the FAST-20 corner count (what
      ORB-SLAM3 actually consumes). Both are taken RELATIVE to the median of
      the surrounding +/-1 s of frames, which roughly cancels scene content
      (frames a second apart see a similar view).

If real blur matters, relative sharpness falls as blur rate rises. Spearman
correlation and per-rate-bin medians are reported, plus the share of frames in
each rate bin.
"""
import argparse, csv, glob, json, os, re
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import cv2


def crete(img):
    """Crete et al. 2007 perceptual blur metric (higher = blurrier)."""
    x = img.astype(np.float64)
    bv = cv2.blur(x, (1, 9)); bh = cv2.blur(x, (9, 1))
    dfv, dfh = np.abs(np.diff(x, axis=0)), np.abs(np.diff(x, axis=1))
    dbv, dbh = np.abs(np.diff(bv, axis=0)), np.abs(np.diff(bh, axis=1))
    vv, vh = np.maximum(0, dfv - dbv), np.maximum(0, dfh - dbh)
    sv, sh = dfv.sum(), dfh.sum()
    return float(max((sv - vv.sum()) / max(sv, 1e-9), (sh - vh.sum()) / max(sh, 1e-9)))


_FAST = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)


def frame_stats(f):
    x = cv2.imread(f, 0)
    return crete(x), len(_FAST.detect(x))


def rel(v, w):
    """v / centred rolling median over +/-w samples."""
    out = np.empty_like(v)
    for i in range(len(v)):
        m = np.median(v[max(0, i - w):i + w + 1])
        out[i] = v[i] / m if m > 0 else np.nan
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--yaml", required=True, help="ORB-SLAM3 YAML (IMU.T_b_c1, Camera1.fx)")
    ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--window-s", type=float, default=1.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    y = open(a.yaml).read()
    f = float(re.search(r"Camera1\.fx:\s*([\d.]+)", y).group(1))
    m = re.search(r"IMU\.T_b_c1:.*?data:\s*\[(.*?)\]", y, re.S)
    R_bc = np.array([float(v) for v in m.group(1).replace("\n", " ").split(",")]).reshape(4, 4)[:3, :3]
    r_rms = np.sqrt((752 ** 2 + 480 ** 2) / 12.0)                       # rms distance from image centre
    summary = {}
    for seq in a.seqs:
        name = os.path.basename(os.path.normpath(seq))
        fs = sorted(glob.glob(os.path.join(seq, "mav0", "cam0", "data", "*.png")))
        ts = np.array([int(os.path.basename(p)[:-4]) for p in fs])
        imu = np.loadtxt(os.path.join(seq, "mav0", "imu0", "data.csv"), delimiter=",", comments="#")
        its = np.loadtxt(os.path.join(seq, "mav0", "imu0", "data.csv"), delimiter=",", comments="#",
                         usecols=0, dtype=np.int64)
        w_cam = imu[:, 1:4] @ R_bc                                     # (R_bc^T w)^T = w^T R_bc
        rate = np.empty(len(ts))
        for i, t in enumerate(ts):
            sel = np.abs(its - t) <= 25_000_000
            w = w_cam[sel].mean(0) if sel.any() else np.full(3, np.nan)
            rate[i] = np.hypot(f * np.hypot(w[0], w[1]), r_rms * abs(w[2]))
        with ProcessPoolExecutor(a.workers) as ex:
            st = np.array(list(ex.map(frame_stats, fs, chunksize=16)))
        hw = int(round(a.window_s * 20))
        rc, rf = rel(st[:, 0], hw), rel(st[:, 1].astype(float), hw)
        with open(os.path.join(a.out, f"{name}_blur_per_frame.csv"), "w", newline="") as fh:
            wr = csv.writer(fh); wr.writerow(["ts", "blur_rate_px_s", "crete", "fast20", "rel_crete", "rel_fast20"])
            wr.writerows(zip(ts, rate, st[:, 0], st[:, 1].astype(int), rc, rf))
        from scipy.stats import spearmanr
        ok = np.isfinite(rate) & np.isfinite(rc) & np.isfinite(rf)
        bins = [0, 250, 500, 1000, 2000, np.inf]
        tab = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            s = ok & (rate >= lo) & (rate < hi)
            tab.append(dict(rate_px_s=f"{lo}-{hi}", frac=float(s.mean()),
                            rel_crete_median=float(np.median(rc[s])) if s.any() else None,
                            rel_fast_median=float(np.median(rf[s])) if s.any() else None))
        summary[name] = dict(frames=len(ts), rate_p50=float(np.nanmedian(rate)), rate_p95=float(np.nanpercentile(rate, 95)),
                             rate_max=float(np.nanmax(rate)),
                             spearman_rate_relcrete=float(spearmanr(rate[ok], rc[ok])[0]),
                             spearman_rate_relfast=float(spearmanr(rate[ok], rf[ok])[0]), bins=tab)
        print(name, json.dumps(summary[name]), flush=True)
    json.dump(summary, open(os.path.join(a.out, "blur_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
