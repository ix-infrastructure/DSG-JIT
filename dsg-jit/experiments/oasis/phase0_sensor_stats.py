#!/usr/bin/env python3
"""Phase 0 of the low-light experiment: measure the starting point from the data.

Two measurements, both taken from the dataset itself rather than assumed:

A. EXPOSURE STATISTICS over every frame of both cameras -- mean DN, percentiles,
   fraction of pixels below the FAST threshold (20) and its fallback (7),
   saturation fraction, distinct grey levels, and the left/right brightness
   ratio (EuRoC ran auto-exposure independently per camera). This says how dark
   the sequence already is and how much headroom exists before darkening.

B. PHOTON TRANSFER from a stationary segment. In a still scene the only thing
   that changes a pixel between consecutive frames is sensor noise. Following
   the EMVA 1288 two-frame method:
       d = f[k+1] - f[k],   mu = (f[k] + f[k+1]) / 2,   sigma^2 = var(d) / 2
   Differencing cancels fixed-pattern noise, and consecutive frames are ~50 ms
   apart so residual camera motion is negligible. A linear Poisson-Gaussian
   sensor gives sigma^2 = K * mu + c: the slope K is the conversion gain
   (DN per electron) and the intercept carries the read noise. CURVATURE in that
   relationship would reveal companding or a tone curve -- the linearity
   assumption the whole darkening simulation rests on.

   Two guards against things that are not sensor noise:
     * per-frame gain normalisation (rescale f[k+1] to f[k]'s mean) removes
       global brightness changes from auto-exposure drift or light flicker;
     * only FLAT pixels (low local gradient) are used, so slow sub-pixel camera
       motion cannot masquerade as noise at edges.
   Variance per intensity bin uses a robust (MAD-based) estimator, so a few
   pixels disturbed by motion or occlusion do not dominate a bin.

   The intercept c = sigma_read^2 + 1/12 - K * mu_dark is degenerate with the
   unknown black level mu_dark; read noise is therefore reported under stated
   black-level assumptions rather than as a single number.

A secondary, upper-bound estimate of fixed-pattern noise is also produced from
the temporal-mean image (spatial residual after high-pass, flat pixels only).
Scene texture leaks into it, so it is an UPPER bound on DSNU/PRNU.

Writes JSON + NPZ only (no plotting dependency); plot with
phase0_plot.py on a machine with matplotlib.
"""
import argparse, csv, glob, json, os
import numpy as np
import cv2


def list_frames(cam_dir):
    fs = sorted(glob.glob(os.path.join(cam_dir, "data", "*.png")))
    ts = np.array([int(os.path.basename(f)[:-4]) for f in fs], dtype=np.int64)
    return fs, ts


def read(f):
    return cv2.imread(f, cv2.IMREAD_UNCHANGED)


# ------------------------------------------------------------ A: exposure --

def exposure_stats(seq, out_dir):
    f0, t0 = list_frames(os.path.join(seq, "mav0", "cam0"))
    f1, t1 = list_frames(os.path.join(seq, "mav0", "cam1"))
    common = np.intersect1d(t0, t1)
    i0 = {t: f for t, f in zip(t0, f0)}
    i1 = {t: f for t, f in zip(t1, f1)}
    rows = []
    for t in common:
        rec = {"ts": int(t)}
        for cam, idx in (("cam0", i0), ("cam1", i1)):
            im = read(idx[t])
            flat = im.ravel()
            rec[f"{cam}_mean"] = float(flat.mean())
            p5, p50, p95 = np.percentile(flat, [5, 50, 95])
            rec[f"{cam}_p5"], rec[f"{cam}_p50"], rec[f"{cam}_p95"] = float(p5), float(p50), float(p95)
            rec[f"{cam}_frac_lt20"] = float((flat < 20).mean())
            rec[f"{cam}_frac_lt7"] = float((flat < 7).mean())
            rec[f"{cam}_frac_sat"] = float((flat >= 250).mean())
            rec[f"{cam}_levels"] = int(np.unique(flat).size)
        rec["lr_ratio"] = rec["cam1_mean"] / max(rec["cam0_mean"], 1e-6)
        rows.append(rec)
    with open(os.path.join(out_dir, "exposure_per_frame.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    def summ(key):
        v = np.array([r[key] for r in rows])
        return {"mean": float(v.mean()), "p5": float(np.percentile(v, 5)),
                "median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
                "min": float(v.min()), "max": float(v.max())}
    keys = [k for k in rows[0] if k != "ts"]
    return {"n_frames": len(rows), "summary": {k: summ(k) for k in keys}}


# ----------------------------------------------------- B: photon transfer --

def robust_var(x):
    """Outlier-trimmed PLAIN variance.

    A MAD-based variance is NOT used: on integer-valued data with sigma ~1 DN
    the MAD itself is quantized (it can only take values like 0, 0.5, 1), which
    biases the low-intensity end of the photon-transfer curve and fakes a
    nonlinearity. A plain variance obeys Sheppard's correction (observed =
    true + 1/12 per frame), which the read-noise formula already subtracts.
    MAD is used only to set a generous trim limit for motion outliers."""
    med = np.median(x)
    s = 1.4826 * np.median(np.abs(x - med))
    keep = np.abs(x - med) <= max(5.0 * s, 3.0)
    return float(x[keep].var())


def photon_transfer(seq, cam, t_lo, t_hi, out_dir, grad_max, bin_w, min_count):
    fs, ts = list_frames(os.path.join(seq, "mav0", cam))
    sel = [(t, f) for t, f in zip(ts, fs) if t_lo <= t <= t_hi]
    frames = np.stack([read(f).astype(np.float64) for _, f in sel])
    n = len(frames)
    means = frames.reshape(n, -1).mean(1)

    # Global brightness stability inside the window (auto-exposure / flicker).
    rel = means / means.mean() - 1.0
    stability = {"frames": n, "frame_mean_dn": float(means.mean()),
                 "frame_mean_rel_std_pct": float(rel.std() * 100),
                 "frame_mean_rel_maxdev_pct": float(np.abs(rel).max() * 100)}

    # Flat-pixel mask from the temporal mean image.
    mean_img = frames.mean(0)
    sm = cv2.GaussianBlur(mean_img, (5, 5), 1.0)
    gx = cv2.Sobel(sm, cv2.CV_64F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(sm, cv2.CV_64F, 0, 1, ksize=3) / 8.0
    grad = np.hypot(gx, gy)                                   # DN per pixel
    flat = (grad < grad_max) & (mean_img > 2) & (mean_img < 250)

    # EMVA two-frame differences, gain-normalised.
    mus, ds = [], []
    for k in range(n - 1):
        a, b = frames[k], frames[k + 1] * (means[k] / means[k + 1])
        mus.append(((a + b) / 2)[flat]); ds.append((b - a)[flat])
    mu = np.concatenate(mus); d = np.concatenate(ds)

    edges = np.arange(0, 256 + bin_w, bin_w)
    centers, var_rb, var_raw, counts = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (mu >= lo) & (mu < hi)
        c = int(m.sum())
        if c < min_count:
            continue
        centers.append(float(mu[m].mean()))
        var_rb.append(float(robust_var(d[m]) / 2.0))
        var_raw.append(float(d[m].var() / 2.0))
        counts.append(c)
    centers, var_rb, var_raw, counts = map(np.array, (centers, var_rb, var_raw, counts))

    # Affine fit (the linear-sensor model), weighted by sqrt(count).
    wts = np.sqrt(counts)
    A = np.vstack([centers, np.ones_like(centers)]).T
    K, c0 = np.linalg.lstsq(A * wts[:, None], var_rb * wts, rcond=None)[0]
    pred = K * centers + c0
    ss_res = float(np.sum(wts**2 * (var_rb - pred) ** 2))
    ss_tot = float(np.sum(wts**2 * (var_rb - np.average(var_rb, weights=wts**2)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Linearity: quadratic term, and slope of the lower vs upper half.
    Aq = np.vstack([centers**2, centers, np.ones_like(centers)]).T
    qa, qb, qc = np.linalg.lstsq(Aq * wts[:, None], var_rb * wts, rcond=None)[0]
    mid = np.median(centers)
    def slope(sel):
        if sel.sum() < 3: return float("nan")
        return float(np.polyfit(centers[sel], var_rb[sel], 1, w=wts[sel])[0])
    k_lo, k_hi = slope(centers <= mid), slope(centers > mid)

    # Read noise under black-level assumptions: c0 = sr^2 + 1/12 - K*mu_dark.
    read_noise = {}
    for mu_dark in (0.0, 4.0, 8.0, 16.0):
        sr2 = c0 + K * mu_dark - 1.0 / 12.0
        sr_dn = float(np.sqrt(sr2)) if sr2 > 0 else float("nan")
        read_noise[f"mu_dark_{mu_dark:g}"] = {
            "read_noise_dn": sr_dn,
            "read_noise_e": sr_dn / K if (K > 0 and sr_dn == sr_dn) else float("nan")}

    # FPN upper bound: spatial high-pass residual of the temporal-mean image.
    hp = mean_img - cv2.GaussianBlur(mean_img, (0, 0), 3.0)
    fpn_c, fpn_s = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = flat & (mean_img >= lo) & (mean_img < hi)
        if m.sum() < min_count:
            continue
        # subtract the temporal noise that survives averaging n frames
        tn = (K * mean_img[m].mean() + c0) / n
        fpn_c.append(float(mean_img[m].mean()))
        fpn_s.append(float(np.sqrt(max(robust_var(hp[m]) - tn, 0.0))))
    fpn_c, fpn_s = np.array(fpn_c), np.array(fpn_s)
    if len(fpn_c) >= 3:
        # sigma_fpn^2 = DSNU^2 + (PRNU * mu)^2
        Af = np.vstack([fpn_c**2, np.ones_like(fpn_c)]).T
        pr2, ds2 = np.linalg.lstsq(Af, fpn_s**2, rcond=None)[0]
        fpn = {"prnu_frac_upper": float(np.sqrt(max(pr2, 0))),
               "dsnu_dn_upper": float(np.sqrt(max(ds2, 0)))}
    else:
        fpn = {}

    np.savez(os.path.join(out_dir, f"ptc_{cam}.npz"), centers=centers, var_robust=var_rb,
             var_raw=var_raw, counts=counts, fpn_centers=fpn_c, fpn_sigma=fpn_s,
             frame_means=means, grad=grad.astype(np.float32), flat=flat)

    return {
        "window_ns": [int(t_lo), int(t_hi)], "stability": stability,
        "flat_pixel_fraction": float(flat.mean()), "grad_max_dn_per_px": grad_max,
        "bins_used": int(len(centers)), "samples": int(len(d)),
        "intensity_range_dn": [float(centers.min()), float(centers.max())],
        "affine_fit": {"K_dn_per_e": float(K), "intercept_dn2": float(c0), "r2": float(r2)},
        "linearity": {"quadratic_coef": float(qa), "slope_lower_half": k_lo,
                      "slope_upper_half": k_hi,
                      "upper_over_lower": (k_hi / k_lo) if k_lo == k_lo and k_lo != 0 else float("nan")},
        "full_scale_electrons_if_linear": float(255.0 / K) if K > 0 else float("nan"),
        "read_noise_by_black_level": read_noise,
        "noise_at_mid_grey_dn": float(np.sqrt(K * 128 + c0)) if K * 128 + c0 > 0 else float("nan"),
        "fpn_upper_bound": fpn,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", required=True, help="sequence dir containing mav0/")
    ap.add_argument("--still-ns", nargs=2, type=int, required=True,
                    help="start/end timestamps (ns) of a stationary segment")
    ap.add_argument("--out", required=True)
    ap.add_argument("--grad-max", type=float, default=1.5, help="flat-pixel gradient cap, DN/px")
    ap.add_argument("--bin-width", type=float, default=4.0)
    ap.add_argument("--min-count", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    result = {"sequence": os.path.abspath(a.seq)}
    result["exposure"] = exposure_stats(a.seq, a.out)
    result["photon_transfer"] = {
        cam: photon_transfer(a.seq, cam, *a.still_ns, a.out, a.grad_max, a.bin_width, a.min_count)
        for cam in ("cam0", "cam1")}
    with open(os.path.join(a.out, "phase0_summary.json"), "w") as f:
        json.dump(result, f, indent=1)

    s = result["exposure"]["summary"]
    print(f"frames: {result['exposure']['n_frames']}")
    for cam in ("cam0", "cam1"):
        print(f"{cam}: mean DN median {s[cam+'_mean']['median']:.1f} (p5 {s[cam+'_mean']['p5']:.1f}, "
              f"p95 {s[cam+'_mean']['p95']:.1f});  px<20 median {s[cam+'_frac_lt20']['median']*100:.1f}%;  "
              f"levels median {s[cam+'_levels']['median']:.0f}")
    print(f"L/R brightness ratio: median {s['lr_ratio']['median']:.3f} "
          f"(p5 {s['lr_ratio']['p5']:.3f}, p95 {s['lr_ratio']['p95']:.3f})")
    for cam, p in result["photon_transfer"].items():
        f_, l_ = p["affine_fit"], p["linearity"]
        print(f"{cam} PTC: K={f_['K_dn_per_e']:.4f} DN/e-  c={f_['intercept_dn2']:.3f} DN^2  R2={f_['r2']:.3f}  "
              f"upper/lower slope={l_['upper_over_lower']:.2f}  sigma@128={p['noise_at_mid_grey_dn']:.2f} DN  "
              f"[{p['bins_used']} bins, {p['intensity_range_dn'][0]:.0f}-{p['intensity_range_dn'][1]:.0f} DN, "
              f"window mean drift {p['stability']['frame_mean_rel_maxdev_pct']:.2f}%]")


if __name__ == "__main__":
    main()
