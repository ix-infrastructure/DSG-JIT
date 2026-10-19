"""Is the trajectory error that degradations add white Gaussian noise?
For each run: align (SE3) to GT, error e(t) (3D). Report
  ac1_e    lag-1 (0.05 s) autocorrelation of e  (white noise ~0)
  tau_s    lag where autocorr of e falls below 0.5 (s)
  lf_frac  share of error power below 0.1 Hz
  kurt     excess kurtosis of e (Gaussian 0)
  xseed    correlation of e(t) between runs of SAME config, different seeds (random ~0, systematic ~1)
For degraded configs the ADDED error d(t)=e_pert(t)-mean_clean_e(t) is analysed the same way."""
import sys, glob, os, json, numpy as np
sys.path.insert(0, os.path.expanduser("~/repos/oasis-experiments"))
from eval_ate import load_tum, load_euroc_gt, associate, umeyama
from scipy.stats import kurtosis
R = "/data/tkocher/oasis_results"
E = "/data/tkocher/datasets/euroc"
def err(run, gts):
    ts, xyz = load_tum(glob.glob(f"{R}/{run}/f_*.txt")[0])
    tg, xg = gts
    ie, ig = associate(ts, tg, 20e6); pe, pg = xyz[ie], xg[ig]
    s, Rm, t = umeyama(pe, pg); return ts[ie], (Rm @ pe.T).T + t - pg
def grid(ts, e, T):  # resample onto common 20 Hz grid
    return np.stack([np.interp(T, ts, e[:, k], left=np.nan, right=np.nan) for k in range(3)], 1)
def stats(E3):
    x = E3 - np.nanmean(E3, 0); x = x[~np.isnan(x).any(1)]
    ac = lambda l: np.mean([np.corrcoef(x[:-l, k], x[l:, k])[0, 1] for k in range(3)])
    tau = next((l * 0.05 for l in range(1, 2000) if ac(l) < 0.5), np.inf)
    P = np.abs(np.fft.rfft(x, axis=0)) ** 2; f = np.fft.rfftfreq(len(x), 0.05)
    return dict(ac1=round(ac(1), 3), tau_s=round(tau, 1), lf=round(P[(f > 0) & (f < 0.1)].sum() / P[f > 0].sum(), 3),
                kurt=round(float(np.mean(kurtosis(x, axis=0))), 2), rms_mm=round(1e3 * np.sqrt((x ** 2).sum(1).mean()), 1))
def xseed(G):
    c = []
    for i in range(len(G)):
        for j in range(i + 1, len(G)):
            a, b = G[i], G[j]; m = ~(np.isnan(a).any(1) | np.isnan(b).any(1))
            c.append(np.mean([np.corrcoef(a[m, k], b[m, k])[0, 1] for k in range(3)]))
    return round(float(np.median(c)), 2)
for seq, gtn, tag in [("MH03", "MH_03_medium", "cMH03"), ("MH01", "MH_01_easy", "cMH01")]:
    tg, xg = load_euroc_gt(f"{E}/{gtn}/mav0/state_groundtruth_estimate0/data.csv"); o = np.argsort(tg); gts = (tg[o], xg[o])
    T = np.arange(tg[o][0], tg[o][-1], 50e6)
    cl = [grid(*err(f"{tag}_L0_clean_r{i:02d}", gts), T) for i in range(10)]
    clm = np.nanmean(np.stack(cl), 0)
    print(f"== {seq}")
    print("clean        ", json.dumps(stats(np.vstack([c for c in cl[:1]]))), "xseed", xseed(cl))
    for arm, L, seeds in [("toff", 2, range(0, 10, 2)), ("toff", 3, range(0, 10, 2)), ("rst", 1, range(10)), ("isf", 3, range(10)), ("imis", 3, range(10)), ("rbc", 3, range(10))]:
        G = []
        for s in seeds:
            run = f"{tag}_L{L}_{arm}_s{s:02d}"
            if json.load(open(f"{R}/{run}/record.json"))["valid"]: G.append(grid(*err(run, gts), T))
        D = [g - clm for g in G]
        st = [stats(d) for d in D]
        agg = {k: round(float(np.median([x[k] for x in st])), 3) for k in st[0]}
        print(f"{arm} L{L} added", json.dumps(agg), "xseed(e)", xseed(G))
