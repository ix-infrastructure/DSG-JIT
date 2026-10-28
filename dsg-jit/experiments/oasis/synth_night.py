#!/usr/bin/env python3
"""Synthetic night from a 4Seasons day recording, calibrated to the real night (Phase 0.5).

The Phase 1/2 darkening model (phase1_detector_study.darken) is shot + read
noise around a linear attenuation. Real 4Seasons night is different: frames
are nearly noise-free (sigma ~0.2 DN) and the loss is a black floor plus
8-bit quantization (mean 4-14 DN, ~10 grey levels in use). The model here is
exactly that:

    y = clip(round(k (x - b) + sigma n), 0, 255),   n ~ N(0, 1)

  k, b   one scale and black offset for the whole sequence (the night camera
         ran at a fixed 21 ms exposure), fitted by matching pooled intensity
         quantiles of sampled day frames to sampled night frames: on the
         quantiles where the night is above the clip floor, night_q = k day_q - k b
         is a straight line, so k is its slope and b its intercept / -k
  sigma  median single-frame noise of the night sample (phase0_real_dark.noise_stats)

With --blur-exp-ms the day frame is first motion-blurred to the night
exposure: day frames are ~0.1 ms, night 21 ms (times.txt), so at 30 Hz the
night shutter integrates f = (t_night - t_day) / frame_dt ~ 0.63 of the
frame-to-frame motion. The motion is dense Farneback flow to the next frame
(previous frame for the last one), and the blur is the mean of backward warps
at offsets s * flow, s in [-f/2, f/2] (exposure centred on the timestamp),
with enough samples for <= 1 px steps (2-24). Blur is applied before
darkening, as in the scene irradiance.

Noise is seeded per (seed, frame, camera). Output layout matches
real_correct_sequence.py, so the gain arms run on top of it unchanged and
run_oasis.sh consumes it; GT, IMU and stamps are the day recording's.

    synth_night.py --day old_town_1_day --night old_town_4_night --dst gen/synth_night
"""
import argparse, glob, json, os, shutil, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase0_real_dark import noise_stats          # noqa: E402

LINK = ["imu0", "state_groundtruth_estimate0", "leica0", "body.yaml"]
P = np.arange(1, 100)
_A = {}


def sample(seq, n):
    fs = []
    for cam in ("cam0", "cam1"):
        c = sorted(glob.glob(os.path.join(seq, "mav0", cam, "data", "*.png")))
        fs += c[:: max(len(c) // n, 1)][:n]
    return [cv2.imread(f, 0) for f in fs]


def stats(imgs):
    v = np.concatenate([i.ravel() for i in imgs]).astype(np.float64)
    return dict(mean=float(v.mean()), median=float(np.median(v)), p95=float(np.percentile(v, 95)),
                frac_lt10=float((v < 10).mean()), levels_p1_p99=int(np.percentile(v, 99) - np.percentile(v, 1)) + 1)


def _init(a):
    _A.update(a=a)


def motion_blur(x, nxt, f):
    """Mean of backward warps of x along its flow to nxt, over a fraction f of the motion."""
    flow = cv2.calcOpticalFlowFarneback(x, nxt, None, 0.5, 4, 21, 3, 5, 1.1, 0)
    n = int(np.clip(np.ceil(f * np.abs(flow).max()), 2, 24))
    gx, gy = np.meshgrid(np.arange(x.shape[1], dtype=np.float32), np.arange(x.shape[0], dtype=np.float32))
    acc = np.zeros(x.shape, np.float64)
    for s in np.linspace(-f / 2, f / 2, n, dtype=np.float32):
        acc += cv2.remap(x, gx + s * flow[..., 0], gy + s * flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return acc / n


def darken_one(job):
    fi, ci, src, nxt, sign, dst = job
    a = _A["a"]
    x = cv2.imread(src, 0)
    if a.blur_f > 0:
        x = motion_blur(x, cv2.imread(nxt, 0), sign * a.blur_f)
    x = x.astype(np.float64)
    n = np.random.default_rng([a.seed, fi, ci]).standard_normal(x.shape)
    y = np.clip(np.round(a.k * (x - a.b) + a.sigma * n), 0, 255).astype(np.uint8)
    cv2.imwrite(dst, y, [cv2.IMWRITE_PNG_COMPRESSION, 1])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--day", required=True); p.add_argument("--night", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--sample", type=int, default=150, help="frames sampled per camera for the fit")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=12); p.add_argument("--force", action="store_true")
    p.add_argument("--fit-only", action="store_true", help="print the fit and the predicted statistics, write nothing")
    p.add_argument("--blur-exp-ms", type=float, default=0.0, help="night exposure to blur to (4Seasons night: 21); 0 = no blur")
    p.add_argument("--day-exp-ms", type=float, default=0.12, help="day exposure (old_town_1 median 0.12 ms)")
    p.add_argument("--frame-dt-ms", type=float, default=1000 / 30)
    a = p.parse_args()
    a.blur_f = max(a.blur_exp_ms - a.day_exp_ms, 0.0) / a.frame_dt_ms if a.blur_exp_ms > 0 else 0.0

    day, night = sample(a.day, a.sample), sample(a.night, a.sample)
    dq = np.percentile(np.concatenate([i.ravel() for i in day]), P)
    nq = np.percentile(np.concatenate([i.ravel() for i in night]), P)
    use = nq >= 1.0                                         # above the clip floor
    slope, icpt = np.polyfit(dq[use], nq[use], 1)
    a.k, a.b = float(slope), float(-icpt / slope)
    sig = [noise_stats(i)[0] for i in night]
    a.sigma = float(np.nanmedian(sig))
    print(f"fit: k {a.k:.4f}  b {a.b:.1f} DN  sigma {a.sigma:.3f} DN  ({use.sum()} quantiles above floor)", flush=True)
    if a.fit_only:
        pred = [np.clip(np.round(a.k * (i - a.b)), 0, 255) for i in day]
        for k, v in (("real night", stats(night)), ("predicted", stats(pred)), ("day", stats(day))):
            print(f"{k:11s} " + "  ".join(f"{s} {x:.3g}" for s, x in v.items()))
        return

    if os.path.exists(a.dst):
        if not a.force:
            sys.exit(f"{a.dst} exists (use --force)")
        shutil.rmtree(a.dst)
    sm, dm = os.path.join(a.day, "mav0"), os.path.join(a.dst, "mav0")
    jobs = []
    for ci, cam in enumerate(("cam0", "cam1")):
        os.makedirs(os.path.join(dm, cam, "data"))
        fs = sorted(glob.glob(os.path.join(sm, cam, "data", "*.png")))
        for fi, f in enumerate(fs):
            nxt, sign = (fs[fi + 1], 1.0) if fi + 1 < len(fs) else (fs[fi - 1], -1.0)   # last frame: flow to previous, reversed
            jobs.append((fi, ci, f, nxt, sign, os.path.join(dm, cam, "data", os.path.basename(f))))
    for e in LINK:
        if os.path.exists(os.path.join(sm, e)):
            os.symlink(os.path.realpath(os.path.join(sm, e)), os.path.join(dm, e))
    for f in ("config.yaml", "stamps.txt"):
        shutil.copy2(os.path.join(a.day, f), os.path.join(a.dst, f))
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(a,)) as ex:
        list(ex.map(darken_one, jobs, chunksize=64))

    syn = sample(a.dst, a.sample)
    man = dict(source=os.path.abspath(a.day), night_reference=os.path.abspath(a.night), model="clip(round(k(x-b)+sigma n))",
               k=a.k, b=a.b, sigma=a.sigma, seed=a.seed, blur_exp_ms=a.blur_exp_ms, blur_fraction=a.blur_f, frames=len(jobs) // 2,
               stats_day=stats(day), stats_real_night=stats(night), stats_synthetic=stats(syn),
               quantile_fit_rms_dn=float(np.sqrt(np.mean((np.clip(a.k * (dq - a.b), 0, 255) - nq) ** 2))))
    json.dump(man, open(os.path.join(a.dst, "LOWLIGHT_MANIFEST.json"), "w"), indent=1)
    for k in ("stats_real_night", "stats_synthetic", "stats_day"):
        print(f"{k:17s} " + "  ".join(f"{s} {v:.3g}" for s, v in man[k].items()))
    print(f"{a.dst}: {man['frames']} stereo frames, quantile fit rms {man['quantile_fit_rms_dn']:.2f} DN")


if __name__ == "__main__":
    main()
