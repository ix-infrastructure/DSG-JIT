#!/usr/bin/env python3
"""Where do REAL dark frames sit on the synthetic L0-L6 severity ladder?

Phase 2 darkens a well-lit sequence (MH01) synthetically. Before trusting that
on real low light we need to know which synthetic level, if any, real dark
footage from the SAME camera corresponds to. The quantity that matters to the
SLAM front end is the noise left after brightness is restored (that is what
Phase 1 found drives descriptor ties and corner loss), so each frame is
summarised by its

    equivalent restored noise  s_eq = sigma_hat * (m_ref / m)

i.e. the noise a gain-only correction (arm C2) would leave if it brought the
frame's mean m up to m_ref, the median frame mean of the well-lit reference.

sigma_hat is a single-frame noise estimate (Immerkaer 1996: mean absolute
response of the 3x3 kernel [1 -2 1; -2 4 -2; 1 -2 1], scaled by
sqrt(pi/2)/6), taken only on FLAT pixels (low gradient after a sigma=2 blur,
not clipped). Texture leaks into any single-frame estimator, so the ladder is
not taken from the model's restored_sigma(): it is CALIBRATED by running the
identical estimator on the reference sequence darkened at L0..L6 with the
exact Phase 2 darkening code (arm C0 output, seed 0). Real frames are then
assigned a fractional equivalent level by interpolating log(s_eq) on that
ladder, so estimator bias cancels to first order (same camera, same hall).

A second check tests the noise MODEL itself: sigma_hat is compared with the
native-gain photon-transfer prediction sqrt(K*m_flat + c) measured in Phase 0.
A ratio well above the reference's own ratio means the camera raised analog
gain on those frames -- information the dataset does not record.
"""
import argparse, csv, glob, json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1_detector_study import darken, quant, to_u8  # noqa: E402

KERN = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float64)


def noise_stats(img, flat_q=0.5):
    """(sigma_hat, mean of flat pixels, frame mean, frac<20) for a uint8 frame."""
    x = img.astype(np.float64)
    g = cv2.GaussianBlur(x, (0, 0), 2.0)
    gm = np.hypot(cv2.Sobel(g, cv2.CV_64F, 1, 0), cv2.Sobel(g, cv2.CV_64F, 0, 1))
    r = np.abs(cv2.filter2D(x, -1, KERN, borderType=cv2.BORDER_REFLECT))
    ok = (x > 2) & (x < 250)
    ok[:2, :] = ok[-2:, :] = ok[:, :2] = ok[:, -2:] = False
    flat = ok & (gm <= np.quantile(gm[ok], flat_q)) if ok.sum() > 1000 else ok
    sig = float(np.sqrt(np.pi / 2) / 6 * r[flat].mean()) if flat.sum() > 500 else float("nan")
    return sig, float(x[flat].mean()) if flat.any() else float("nan"), float(x.mean()), float((x < 20).mean())


def _real(job):
    ts, cam, f = job
    s, mf, m, lt20 = noise_stats(cv2.imread(f, 0))
    return dict(ts=ts, cam=cam, sigma=s, m_flat=mf, mean=m, frac_lt20=lt20)


def _ref(job):
    fi, ci, f, lv, a = job
    raw = cv2.imread(f, 0)
    y, att = darken(raw, lv, np.random.default_rng([0, fi, lv, ci]), a["K"], a["read_e"], a["gain_max"])
    s, mf, m, _ = noise_stats(to_u8(quant(y, 1.0)))
    return dict(level=lv, sigma=s, m_flat=mf, mean=m, clean_mean=float(raw.mean()))


def frames(seq, stride):
    out = []
    for ci, cam in enumerate(("cam0", "cam1")):
        fs = sorted(glob.glob(os.path.join(seq, "mav0", cam, "data", "*.png")))[::stride]
        out += [(int(os.path.basename(f)[:-4]), ci, f) for f in fs]
    return out


def runs_above(mask):
    """Lengths of consecutive True runs."""
    out, n = [], 0
    for v in mask:
        if v: n += 1
        elif n: out.append(n); n = 0
    if n: out.append(n)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="well-lit reference sequence (MH01)")
    ap.add_argument("--seqs", nargs="+", required=True, help="real sequences to place on the ladder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-stride", type=int, default=20)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6])
    ap.add_argument("--K", type=float, default=0.0186); ap.add_argument("--read-e", type=float, default=30.0)
    ap.add_argument("--gain-max", type=float, default=4.0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pa = dict(K=a.K, read_e=a.read_e, gain_max=a.gain_max)
    ptc = lambda m: np.sqrt(a.K * np.maximum(m, 0) + (a.K * a.read_e) ** 2 + 1 / 12)  # native-gain model, DN

    with ProcessPoolExecutor(a.workers) as ex:
        ref_fr = frames(a.ref, a.ref_stride)
        ref = list(ex.map(_ref, [(fi, ci, f, lv, pa) for fi, (_, ci, f) in enumerate(ref_fr) for lv in a.levels],
                          chunksize=8))
        m_ref = float(np.median([r["clean_mean"] for r in ref if r["level"] == 0]))
        ladder = {}
        for lv in a.levels:
            rr = [r for r in ref if r["level"] == lv and np.isfinite(r["sigma"])]
            se = np.array([r["sigma"] * r["clean_mean"] / max(r["mean"], 1e-6) for r in rr])
            ladder[lv] = dict(s_eq_median=float(np.median(se)), s_eq_p10=float(np.percentile(se, 10)),
                              s_eq_p90=float(np.percentile(se, 90)),
                              ptc_ratio=float(np.median([r["sigma"] / ptc(r["m_flat"]) for r in rr])),
                              mean_dn=float(np.median([r["mean"] for r in rr])))
        lv_arr = np.array(a.levels, float)
        ls_arr = np.log([ladder[l]["s_eq_median"] for l in a.levels])
        summary = dict(m_ref=m_ref, ladder=ladder, sequences={})

        for seq in a.seqs:
            name = os.path.basename(os.path.normpath(seq))
            rows = list(ex.map(_real, frames(seq, a.stride), chunksize=16))
            for r in rows:
                r["s_eq"] = r["sigma"] * m_ref / max(r["mean"], 1e-6)
                r["eq_level"] = float(np.interp(np.log(r["s_eq"]), ls_arr, lv_arr)) if np.isfinite(r["s_eq"]) else float("nan")
                r["ptc_ratio"] = r["sigma"] / ptc(r["m_flat"])
                if np.log(r["s_eq"]) > ls_arr[-1]:
                    r["eq_level"] = float(lv_arr[-1] + 1)       # beyond the ladder: flag as > top level
            with open(os.path.join(a.out, f"{name}_per_frame.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
            # A stereo frame is as bad as its worse camera.
            by_ts = {}
            for r in rows:
                by_ts.setdefault(r["ts"], []).append(r["eq_level"])
            ts = sorted(by_ts)
            worst = np.array([max(by_ts[t]) for t in ts])
            dt = np.median(np.diff(ts)) / 1e9 if len(ts) > 1 else float("nan")
            s = dict(frames=len(ts), frame_dt_s=float(dt),
                     eq_level_median=float(np.nanmedian(worst)), eq_level_p90=float(np.nanpercentile(worst, 90)),
                     eq_level_max=float(np.nanmax(worst)),
                     mean_dn_median=float(np.median([r["mean"] for r in rows])),
                     ptc_ratio_median=float(np.nanmedian([r["ptc_ratio"] for r in rows])),
                     ptc_ratio_p90=float(np.nanpercentile([r["ptc_ratio"] for r in rows], 90)))
            for thr in (2, 3, 4, 5, 6):
                m = worst >= thr
                rl = runs_above(m)
                s[f"frac_ge_L{thr}"] = float(m.mean())
                s[f"longest_run_ge_L{thr}_s"] = float(max(rl) * dt * a.stride) if rl else 0.0
            summary["sequences"][name] = s
            print(name, json.dumps(s), flush=True)

    json.dump(summary, open(os.path.join(a.out, "real_dark_summary.json"), "w"), indent=1)
    print("ladder", json.dumps(ladder, indent=1))


if __name__ == "__main__":
    main()
