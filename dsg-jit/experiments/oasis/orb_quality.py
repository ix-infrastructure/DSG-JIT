#!/usr/bin/env python3
"""Are extra ORB corners on crushed night frames usable, or quantization artefacts?

Lowering ORB-SLAM3's FAST threshold (arm C1) on 4Seasons night made every run
fail, although it roughly doubled the corner count. Hypothesis: at ~10 grey
levels, a low threshold fires on 1-DN quantization contours, which are not
repeatable across views. Per sampled frame t this measures what tracking needs:

  n_kp        ORB keypoints in the left image
  stereo_ok   left keypoints with a right match on the same rectified row
              (Hamming <= 50, |dy| <= 2 px, 0 < disparity <= 150 px)
  temp_inl    RANSAC fundamental-matrix inliers between left(t) and left(t+k),
              after a 0.8 ratio test
and the ratios stereo_ok / n_kp and temp_inl / n_kp.

OpenCV ORB is not ORB-SLAM3's extractor (no per-cell grid, no fallback
threshold), so absolute counts differ; the comparison between conditions is
the point.

    orb_quality.py --day <euroc-layout dir> --night <dir> --out q.json
"""
import argparse, glob, json, os
import numpy as np
import cv2


def gain_c2s(x, m_ref=76.6, gain_sat=240.0, gain_max=16.0):
    """Saturation-safe gain of real_correct_sequence.py (single image)."""
    v = x[x < 250]
    g = min(m_ref / max(x.mean(), 1e-3), gain_sat / max(np.percentile(v, 99), 1.0))
    return np.clip(x.astype(np.float64) * float(np.clip(g, 1.0, gain_max)), 0, 255).astype(np.uint8)


TRANSFORMS = {"raw": lambda x: x, "gain": gain_c2s,
              "clahe": lambda x: cv2.createCLAHE(3.0, (8, 8)).apply(x)}


def measure(L, R, L2, th):
    orb = cv2.ORB_create(nfeatures=1200, scaleFactor=1.2, nlevels=8, fastThreshold=th, edgeThreshold=19)
    kL, dL = orb.detectAndCompute(L, None)
    kR, dR = orb.detectAndCompute(R, None)
    k2, d2 = orb.detectAndCompute(L2, None)
    n = len(kL)
    out = dict(n_kp=n, stereo_ok=0, temp_inl=0)
    if n < 8 or dR is None or d2 is None:
        return out
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    s = 0
    for m in bf.match(dL, dR):
        (xl, yl), (xr, yr) = kL[m.queryIdx].pt, kR[m.trainIdx].pt
        s += m.distance <= 50 and abs(yl - yr) <= 2 and 0 < xl - xr <= 150
    out["stereo_ok"] = int(s)
    knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(dL, d2, k=2)
    good = [p[0] for p in knn if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
    if len(good) >= 8:
        a = np.float32([kL[m.queryIdx].pt for m in good]); b = np.float32([k2[m.trainIdx].pt for m in good])
        _, mask = cv2.findFundamentalMat(a, b, cv2.FM_RANSAC, 1.0, 0.999)
        out["temp_inl"] = int(mask.sum()) if mask is not None else 0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day", required=True); ap.add_argument("--night", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=150, help="sampled frames per sequence")
    ap.add_argument("--k", type=int, default=3, help="temporal gap in frames")
    ap.add_argument("--conds", nargs="+", default=["day:raw:20", "night:raw:20", "night:raw:5", "night:raw:2",
                                                    "night:gain:20", "night:gain:5", "night:clahe:20"],
                    help="seq:transform:fast_threshold")
    a = ap.parse_args()
    conds = [tuple(c.split(":")[:2]) + (int(c.split(":")[2]),) for c in a.conds]
    res = {}
    for seq, tf, th in conds:
        root = a.day if seq == "day" else a.night
        fs = sorted(glob.glob(os.path.join(root, "mav0", "cam0", "data", "*.png")))
        idx = np.linspace(len(fs) * 0.05, len(fs) * 0.95 - a.k, a.n).astype(int)        # skip start/stop standstill
        rows = []
        for i in idx:
            f = os.path.basename(fs[i])
            L = cv2.imread(fs[i], 0); R = cv2.imread(os.path.join(root, "mav0", "cam1", "data", f), 0); L2 = cv2.imread(fs[i + a.k], 0)
            if R is None:
                continue
            T = TRANSFORMS[tf]
            rows.append(measure(T(L), T(R), T(L2), th))
        n = np.array([r["n_kp"] for r in rows], float)
        s = np.array([r["stereo_ok"] for r in rows], float); t = np.array([r["temp_inl"] for r in rows], float)
        key = f"{seq} {tf} th{th}"
        res[key] = dict(frames=len(rows), n_kp=float(np.median(n)), stereo_ok=float(np.median(s)), temp_inl=float(np.median(t)),
                        stereo_ratio=float(np.median(s / np.maximum(n, 1))), temp_ratio=float(np.median(t / np.maximum(n, 1))),
                        temp_inl_p10=float(np.percentile(t, 10)))
        print(f"{key:18} " + " ".join(f"{k}={v:.3g}" for k, v in res[key].items()), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
