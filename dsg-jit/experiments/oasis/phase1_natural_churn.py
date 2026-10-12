#!/usr/bin/env python3
"""Natural detector churn: how much do ORB features change between two REAL
exposures of the same still scene?

Phase 1 compares each darkened frame to its own clean frame, which differs only
by the noise darkening ADDS. A real camera, though, never produces the same
features twice: consecutive frames of a static scene already differ by their
own independent sensor noise. Without this baseline, Phase 1's mild-darkening
numbers (e.g. repeatability 0.91 at L1) would read as degradation when they may
be smaller than what the camera does to itself every frame.

Uses consecutive frame pairs from a stationary segment (MH01 19.5-30.4 s:
<1 mm translation, 0.29 deg rotation over the whole window, i.e. ~0.01 px of
motion between consecutive frames) and the SAME metric functions as Phase 1:
frame k plays "clean", frame k+1 plays "corrupted".
"""
import argparse, glob, json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase1_detector_study as p1  # noqa: E402


def pair(job):
    (aL, aR), (bL, bR) = job
    cfg, orb, sf = p1.G["cfg"], p1.G["orb"], p1.G["cfg"]["sf"]
    A = {c: p1.rect(cv2.imread(f, 0), p1.G["ml" if c == "L" else "mr"]) for c, f in (("L", aL), ("R", aR))}
    B = {c: p1.rect(cv2.imread(f, 0), p1.G["ml" if c == "L" else "mr"]) for c, f in (("L", bL), ("R", bR))}
    ea = {c: p1.extract(A[c], cfg["ini"], cfg["mn"], cfg, orb) for c in A}
    eb = {c: p1.extract(B[c], cfg["ini"], cfg["mn"], cfg, orb) for c in B}
    Ma = p1.stereo_matches(ea["L"][:3], ea["R"][:3], sf, p1.G["fx"])
    Mb = p1.stereo_matches(eb["L"][:3], eb["R"][:3], sf, p1.G["fx"])
    out = {"stereo_correct_ratio": p1.correct_matches(Mb, Ma) / max(len(Ma), 1)}
    rep, dle = [], []
    for c in ("L", "R"):
        cxy, coc, cdes, _ = ea[c]
        xy, oc, _, _ = eb[c]
        r, _ = p1.repeat_stats(cxy, coc, xy, oc, sf); rep.append(r)
        dd, has = p1.desc_at(B[c], cxy, coc, [], orb, sf)
        hd = p1.hamming(cdes[has], dd[has])
        dle.append(float((hd <= 50).mean()))
    out.update(repeat=float(np.mean(rep)), desc_le50=float(np.mean(dle)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", required=True); ap.add_argument("--yaml", required=True)
    ap.add_argument("--still-ns", nargs=2, type=int, required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    cfg = p1.load_cfg(a.yaml); maps = p1.rect_maps(cfg)
    fl = sorted(glob.glob(os.path.join(a.seq, "mav0", "cam0", "data", "*.png")))
    fl = [f for f in fl if a.still_ns[0] <= int(os.path.basename(f)[:-4]) <= a.still_ns[1]]
    fr = [os.path.join(a.seq, "mav0", "cam1", "data", os.path.basename(f)) for f in fl]
    frames = list(zip(fl, fr))
    jobs = [(frames[k], frames[k + 1]) for k in range(len(frames) - 1)]
    with ProcessPoolExecutor(a.workers, initializer=p1._init, initargs=(cfg, maps, a)) as ex:
        rows = list(ex.map(pair, jobs, chunksize=4))
    res = {k: {"mean": float(np.mean([r[k] for r in rows])), "p10": float(np.percentile([r[k] for r in rows], 10)),
               "p90": float(np.percentile([r[k] for r in rows], 90))} for k in rows[0]}
    res["pairs"] = len(rows)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "natural_churn.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
