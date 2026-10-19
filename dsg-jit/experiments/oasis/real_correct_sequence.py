#!/usr/bin/env python3
"""Apply the Phase 2 correction arms to a REAL (not synthetically darkened)
EuRoC sequence, for the real-low-light check of the Phase 2 result.

Phase 2 knew the darkening exactly: the gain undid a known attenuation and NLM
was given the model's restored noise. Real footage gives neither, so both are
estimated per frame from the image alone -- which is also what a deployed
pre-SLAM corrector would have to do:

  C2  gain only:   g = clip(m_ref / m, 1, gain_max), m = mean of the stereo
                   pair, ONE gain for both cameras (per-camera gains would
                   create a left/right brightness mismatch the stereo matcher
                   would see). Frames already at/above m_ref are untouched.
  C4  gain + NLM:  as C2, then cv2.fastNlMeansDenoising with h = g * sigma_hat,
                   sigma_hat the per-camera single-frame noise estimate from
                   phase0_real_dark.noise_stats (same estimator that placed the
                   frames on the severity ladder), search window 21 as Phase 2.

  C2s, C4s  the same with a SATURATION-SAFE gain, g additionally capped at
                   gain_sat / p99 of the pair's NOT-yet-clipped pixels (light
                   sources already at 255 cannot get worse, and including them
                   would forbid any gain at all on exactly the darkest frames). Real dark EuRoC frames are not
                   uniformly dim: the darkest MH04 frames are black regions next
                   to bright windows and lit strips, so a mean-based gain of ~14
                   clips exactly the regions that carry features. Synthetic
                   darkening (Phase 2) is uniform, which is why the plain gain
                   was adequate there. Both variants run so the comparison
                   with the pre-registered arms is kept.

  C5  CLAHE:       cv2.createCLAHE(clipLimit, 8x8 tiles) on the raw image, the
                   same parameters for both cameras. Added for 4Seasons night,
                   where frames are dark but nearly noise-free (sigma ~0.2 DN,
                   ~10 grey levels in use): the loss is local contrast and bit
                   depth, not noise, so a global gain or a denoiser has little
                   to act on while ORB's FAST threshold (20) sees no corners.

C0 (no correction) needs no generation: run SLAM on the source directly.
Output layout matches lowlight_sequence.py so phase2_suite/run_oasis consume it.
"""
import argparse, glob, json, os, shutil, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1_detector_study import to_u8, nlm      # noqa: E402
from phase0_real_dark import noise_stats          # noqa: E402

LINK = ["imu0", "state_groundtruth_estimate0", "leica0", "body.yaml"]
_A = {}


def _init(a):
    _A["a"] = a


def correct_pair(job):
    fL, fR, dL, dR = job
    a = _A["a"]
    L, R = cv2.imread(fL, 0), cv2.imread(fR, 0)
    m = 0.5 * (L.mean() + R.mean())
    g = a.m_ref / max(m, 1e-3)
    if a.arm.endswith("s"):
        v = np.concatenate([L.ravel(), R.ravel()])
        v = v[v < 250]                                   # already-clipped light sources cannot get worse
        g = min(g, a.gain_sat / max(np.percentile(v, 99.0), 1.0)) if v.size else 1.0
    g = float(np.clip(g, 1.0, a.gain_max))
    out, hs = [], []
    if a.arm == "C5":
        cl = cv2.createCLAHE(clipLimit=a.clahe_clip, tileGridSize=(8, 8))
        out = [cl.apply(L), cl.apply(R)]
        g = float(np.mean(out[0]) / max(L.mean(), 1e-3))       # effective gain, for the manifest
    for img in ((L, R) if a.arm != "C5" else ()):
        y = to_u8(img.astype(np.float64) * g)
        if a.arm.startswith("C4"):
            h = g * noise_stats(img)[0]
            h = h if np.isfinite(h) else g
            y = nlm(y, h, 21); hs.append(h)
        out.append(y)
    cv2.imwrite(dL, out[0], [cv2.IMWRITE_PNG_COMPRESSION, 1])
    cv2.imwrite(dR, out[1], [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return g, (float(np.mean(hs)) if hs else 0.0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True); p.add_argument("--dst", required=True)
    p.add_argument("--yaml", required=True, help="base ORB-SLAM3 YAML (copied unchanged)")
    p.add_argument("--arm", required=True, choices=["C2", "C4", "C2s", "C4s", "C5"])
    p.add_argument("--m-ref", type=float, required=True, help="target mean DN (MH01 median frame mean)")
    p.add_argument("--gain-max", type=float, default=16.0)
    p.add_argument("--gain-sat", type=float, default=240.0, help="C2s/C4s: p99 of unclipped pixels may be raised to at most this DN")
    p.add_argument("--clahe-clip", type=float, default=3.0, help="C5: CLAHE clip limit")
    p.add_argument("--workers", type=int, default=8); p.add_argument("--force", action="store_true")
    # accepted for interface compatibility with lowlight_sequence.py; unused (no randomness here)
    p.add_argument("--level", type=int, default=0); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    if os.path.exists(a.dst):
        if not a.force:
            sys.exit(f"{a.dst} exists (use --force)")
        shutil.rmtree(a.dst)
    sm, dm = os.path.join(a.src, "mav0"), os.path.join(a.dst, "mav0")
    for cam in ("cam0", "cam1"):
        os.makedirs(os.path.join(dm, cam, "data"))
        for f in ("data.csv", "sensor.yaml"):
            if os.path.exists(os.path.join(sm, cam, f)):
                shutil.copy2(os.path.join(sm, cam, f), os.path.join(dm, cam, f))
    for e in LINK:
        if os.path.exists(os.path.join(sm, e)):
            os.symlink(os.path.abspath(os.path.join(sm, e)), os.path.join(dm, e))

    fl = sorted(glob.glob(os.path.join(sm, "cam0", "data", "*.png")))
    names = {os.path.basename(f) for f in glob.glob(os.path.join(sm, "cam1", "data", "*.png"))}
    jobs = [(f, os.path.join(sm, "cam1", "data", os.path.basename(f)),
             os.path.join(dm, "cam0", "data", os.path.basename(f)),
             os.path.join(dm, "cam1", "data", os.path.basename(f)))
            for f in fl if os.path.basename(f) in names]
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(a,)) as ex:
        res = list(ex.map(correct_pair, jobs, chunksize=32))
    shutil.copy2(a.yaml, os.path.join(a.dst, "config.yaml"))
    g = np.array([r[0] for r in res])
    man = dict(source=os.path.abspath(a.src), arm=a.arm, m_ref=a.m_ref, clahe_clip=a.clahe_clip if a.arm == "C5" else None, gain_max=a.gain_max, frames=len(jobs),
               gain_median=float(np.median(g)), gain_p90=float(np.percentile(g, 90)),
               frac_gained=float((g > 1.01).mean()),
               nlm_h_median=float(np.median([r[1] for r in res])) if a.arm.startswith("C4") else None)
    json.dump(man, open(os.path.join(a.dst, "LOWLIGHT_MANIFEST.json"), "w"), indent=1)
    print(f"{a.dst}: {a.arm}  gain median {man['gain_median']:.2f} p90 {man['gain_p90']:.2f}  "
          f"gained {man['frac_gained']:.0%}  frames={len(jobs)}")


if __name__ == "__main__":
    main()
