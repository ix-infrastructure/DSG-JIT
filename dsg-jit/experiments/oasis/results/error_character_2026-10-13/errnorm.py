#!/usr/bin/env python3
"""Is SLAM trajectory error Gaussian, and is it a first-order Gauss-Markov process?

errchar.py showed the error is strongly autocorrelated (tau 1.5-14 s), so a
normality test on raw 20 Hz samples is invalid (n is inflated ~30-280x). Here:

NORMALITY on decorrelated samples. Per run and axis the error series is
standardised, then subsampled every 2*tau (tau = 1/e decorrelation time of that
series), and the subsamples are pooled over runs and axes. Tests: Shapiro-Wilk
and D'Agostino K^2; plus skewness, excess kurtosis.

GAUSS-MARKOV (first-order, = AR(1) at 20 Hz). Per series, fit x_k = phi x_(k-1) + w_k:
  tau_gm = -dt / ln(phi),  sigma = stationary std.
A first-order GM model is adequate if
  (a) innovations w_k are white: Ljung-Box Q on lags 1..20 (p reported, median over series);
  (b) the model ACF exp(-lag/tau_gm) matches the empirical ACF at 1 s and 5 s;
  (c) AR(2) does not fit much better: delta-BIC = BIC(AR1) - BIC(AR2) (> 10 = strong
      evidence for second order, e.g. integrated drift / "velocity" in the error).

For degraded runs the series is the ADDED error e_pert(t) - mean clean e(t)
(as errchar.py); for clean and real-dark runs, e(t) itself.
"""
import sys, glob, os, json
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.expanduser("~/repos/oasis-experiments"))
from eval_ate import load_tum, load_euroc_gt, associate, umeyama  # noqa: E402

R, E, DT = "/data/tkocher/oasis_results", "/data/tkocher/datasets/euroc", 0.05


def err(run, gts):
    ts, xyz = load_tum(glob.glob(f"{R}/{run}/f_*.txt")[0])
    tg, xg = gts
    ie, ig = associate(ts, tg, 20e6); pe, pg = xyz[ie], xg[ig]
    _, Rm, t = umeyama(pe, pg)
    return ts[ie], (Rm @ pe.T).T + t - pg


def grid(ts, e, T):
    return np.stack([np.interp(T, ts, e[:, k], left=np.nan, right=np.nan) for k in range(3)], 1)


def acf(x, lag):
    return np.corrcoef(x[:-lag], x[lag:])[0, 1]


def ljung_box(w, m=20):
    n = len(w); w = w - w.mean(); d = (w ** 2).sum()
    q = n * (n + 2) * sum((np.dot(w[:-k], w[k:]) / d) ** 2 / (n - k) for k in range(1, m + 1))
    return stats.chi2.sf(q, m)


def series_stats(x):
    """x: 1-D error series (no NaN), length >= 200."""
    x = x - x.mean()
    # decorrelation time (1/e) and decorrelated subsample
    tau_lag = next((l for l in range(1, len(x) // 3) if acf(x, l) < np.exp(-1)), len(x) // 3)
    sub = (x / x.std())[::max(1, 2 * tau_lag)]
    # AR(1) and AR(2) by least squares
    X1 = x[:-1]; y1 = x[1:]
    phi = float(np.dot(X1, y1) / np.dot(X1, X1)); w1 = y1 - phi * X1
    X2 = np.stack([x[1:-1], x[:-2]], 1); y2 = x[2:]
    c2, *_ = np.linalg.lstsq(X2, y2, rcond=None); w2 = y2 - X2 @ c2
    n = len(y2)
    bic = lambda w, k: n * np.log((w[-n:] ** 2).mean()) + k * np.log(n)
    tau_gm = -DT / np.log(phi) if 0 < phi < 1 else np.inf
    model = lambda s: np.exp(-s / tau_gm) if np.isfinite(tau_gm) else 1.0
    return dict(sub=sub, tau_1e_s=tau_lag * DT, phi=phi, tau_gm_s=tau_gm, sigma_mm=1e3 * x.std(),
                lb_p=ljung_box(w1), acf1s=(acf(x, 20), model(1.0)), acf5s=(acf(x, 100) if len(x) > 300 else np.nan, model(5.0)),
                dbic=float(bic(w1, 1) - bic(w2, 2)), innov_kurt=float(stats.kurtosis(w1)))


def summarise(name, series):
    rows = [series_stats(s) for s in series if len(s) >= 200]
    pool = np.concatenate([r["sub"] for r in rows])
    sw = stats.shapiro(pool).pvalue if 3 <= len(pool) <= 5000 else np.nan
    k2 = stats.normaltest(pool).pvalue if len(pool) >= 20 else np.nan
    med = lambda k: float(np.nanmedian([r[k] for r in rows]))
    a1 = np.array([r["acf1s"] for r in rows]); a5 = np.array([r["acf5s"] for r in rows])
    out = dict(name=name, series=len(rows), n_decorr=len(pool), shapiro_p=float(sw), dagostino_p=float(k2),
               skew=float(stats.skew(pool)), ex_kurt=float(stats.kurtosis(pool)),
               tau_gm_s=med("tau_gm_s"), sigma_mm=med("sigma_mm"), frac_white_innov=float(np.mean([r["lb_p"] > 0.05 for r in rows])),
               acf1s_emp=float(np.nanmedian(a1[:, 0])), acf1s_gm=float(np.nanmedian(a1[:, 1])),
               acf5s_emp=float(np.nanmedian(a5[:, 0])), acf5s_gm=float(np.nanmedian(a5[:, 1])),
               dbic_ar2=med("dbic"), innov_kurt=med("innov_kurt"))
    print(f"{name:22} n={out['n_decorr']:4d} SW p={out['shapiro_p']:.3f} K2 p={out['dagostino_p']:.3f} skew={out['skew']:+.2f} "
          f"kurt={out['ex_kurt']:+.2f} | GM tau={out['tau_gm_s']:.1f}s sigma={out['sigma_mm']:.0f}mm "
          f"ACF1s {out['acf1s_emp']:.2f}/{out['acf1s_gm']:.2f} ACF5s {out['acf5s_emp']:.2f}/{out['acf5s_gm']:.2f} "
          f"white-innov {out['frac_white_innov']:.0%} dBIC(AR2) {out['dbic_ar2']:.0f} innov-kurt {out['innov_kurt']:+.1f}", flush=True)
    return out


def axes(G):
    out = []
    for g in G:
        m = ~np.isnan(g).any(1)
        out += [g[m, k] for k in range(3)]
    return out


def main():
    res = []
    for seq, gtn, tag in [("MH03", "MH_03_medium", "cMH03"), ("MH01", "MH_01_easy", "cMH01")]:
        tg, xg = load_euroc_gt(f"{E}/{gtn}/mav0/state_groundtruth_estimate0/data.csv"); o = np.argsort(tg); gts = (tg[o], xg[o])
        T = np.arange(tg[o][0], tg[o][-1], DT * 1e9)
        cl = [grid(*err(f"{tag}_L0_clean_r{i:02d}", gts), T) for i in range(10)]
        clm = np.nanmean(np.stack(cl), 0)
        print(f"== {seq}")
        res.append(summarise(f"{seq} clean", axes(cl)))
        for arm in ["toff", "rst", "isf", "imis", "rbc", "tbc"]:
            for L in (1, 2, 3):
                G = []
                for s in range(10):
                    run = f"{tag}_L{L}_{arm}_s{s:02d}"
                    if json.load(open(f"{R}/{run}/record.json"))["valid"]:
                        G.append(grid(*err(run, gts), T) - clm)
                if G:
                    res.append(summarise(f"{seq} {arm} L{L} added", axes(G)))
    print("== real dark (raw error, no clean counterpart)")
    for seq, gtn in [("MH04", "MH_04_difficult"), ("MH05", "MH_05_difficult")]:
        tg, xg = load_euroc_gt(f"{E}/{gtn}/mav0/state_groundtruth_estimate0/data.csv"); o = np.argsort(tg); gts = (tg[o], xg[o])
        T = np.arange(tg[o][0], tg[o][-1], DT * 1e9)
        for arm in ["C0", "C2s"]:
            G = [grid(*err(os.path.basename(d), gts), T) for d in sorted(glob.glob(f"{R}/r{seq}_L0_{arm}_s*")) if glob.glob(f"{d}/f_*.txt")]
            if G:
                res.append(summarise(f"{seq} real {arm}", axes(G)))
    json.dump(res, open(f"{R}/errnorm.json", "w"), indent=1)


if __name__ == "__main__":
    main()
