#!/usr/bin/env python3
"""Why does C2s (saturation-safe gain) lose the map on old_town_3 night near frame 9300 in every seed?
Per frame in a window around the break: raw and gained brightness, applied gain, saturation, ORB
keypoints at the YAML's FAST thresholds (20, fallback 7), and left-right stereo matches (row-constrained)."""
import glob, sys
import numpy as np, cv2

S = "/data/tkocher/datasets/4seasons/old_town_3_night_euroc/mav0"
G = "/data/tkocher/datasets/4seasons/gen/old_town_3_night_C2s/mav0"
lo, hi, step = (int(v) for v in (sys.argv[1:4] if len(sys.argv) > 3 else (8400, 10200, 20)))
f0 = sorted(glob.glob(f"{S}/cam0/data/*.png"))
orb20, orb7 = cv2.ORB_create(1200, fastThreshold=20), cv2.ORB_create(1200, fastThreshold=7)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def stereo(L, R, orb):
    kl, dl = orb.detectAndCompute(L, None); kr, dr = orb.detectAndCompute(R, None)
    if dl is None or dr is None:
        return len(kl), 0
    m = [x for x in bf.match(dl, dr) if abs(kl[x.queryIdx].pt[1] - kr[x.trainIdx].pt[1]) < 2
         and 0 < kl[x.queryIdx].pt[0] - kr[x.trainIdx].pt[0] < 120]
    return len(kl), len(m)


print("frame  raw_mean raw_p99 | gain  c2s_mean sat%  | raw kp20 st20 kp7 st7 | c2s kp20 st20 kp7 st7")
for k in range(lo, hi, step):
    n = f0[k].split("/")[-1]
    L, R = cv2.imread(f"{S}/cam0/data/{n}", 0), cv2.imread(f"{S}/cam1/data/{n}", 0)
    gL, gR = cv2.imread(f"{G}/cam0/data/{n}", 0), cv2.imread(f"{G}/cam1/data/{n}", 0)
    r = stereo(L, R, orb20) + stereo(L, R, orb7)
    c = stereo(gL, gR, orb20) + stereo(gL, gR, orb7)
    print(f"{k:5d}  {L.mean():7.1f} {np.percentile(L, 99):6.0f} | {gL.mean() / max(L.mean(), 1e-3):4.1f}  {gL.mean():7.1f} "
          f"{100 * (gL >= 250).mean():5.2f} | {r[0]:4d} {r[1]:4d} {r[2]:4d} {r[3]:4d} | {c[0]:4d} {c[1]:4d} {c[2]:4d} {c[3]:4d}")
