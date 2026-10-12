#!/usr/bin/env python3
"""What does sensor-level noise look like once it reaches the trajectory?

The prior SE(3) denoising experiments injected i.i.d. zero-mean Gaussian noise
directly into relative-pose edges. The question for the raw-sensor arms is
whether noise injected UPSTREAM of SLAM arrives at the trajectory in that same
form. So the primary quantity here is the RELATIVE (edge) pose error between
consecutive frames -- exactly what the denoiser consumes -- characterised by:

    bias       mean of each component        (a smoother cannot remove it)
    scale      std of each component         (compared run-by-run to clean)
    shape      skew, excess kurtosis, QQ     (is it Gaussian?)
    whiteness  lag-1 autocorrelation         (is it independent across edges?)

plus absolute trajectory error (ATE) against the clean nondeterminism floor,
and the autocorrelation of absolute error (white vs. drift).

The unit of replication is the RUN, not the edge: thousands of edges from one
run are not independent evidence. Between-condition comparisons are therefore
made on per-run statistics.

Usage:
  analyze_error_shape.py --results <dir containing run dirs> --gt <EuRoC GT csv>
                         --out <dir> [--pattern 'fpn_*'] [--clean clean]
"""
import argparse, glob, json, os, re
import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------- geometry --

def quat_xyzw_to_R(q):
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q.T
    return np.stack([
        np.stack([1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)], -1),
        np.stack([2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)], -1),
        np.stack([2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)], -1),
    ], axis=1)


def slerp(q0, q1, a):
    dot = np.sum(q0 * q1, axis=1)
    q1 = np.where(dot[:, None] < 0, -q1, q1)
    dot = np.abs(dot)
    out = np.empty_like(q0)
    lin = dot > 0.9995
    out[lin] = q0[lin] + a[lin, None] * (q1[lin] - q0[lin])
    th = np.arccos(np.clip(dot[~lin], -1, 1))
    s = np.sin(th)
    out[~lin] = (np.sin((1 - a[~lin]) * th)[:, None] * q0[~lin]
                 + np.sin(a[~lin] * th)[:, None] * q1[~lin]) / s[:, None]
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def so3_log(R):
    tr = np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)
    th = np.arccos(tr)
    v = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], 1)
    s = np.sin(th)
    k = np.where(s > 1e-9, th / (2 * np.maximum(s, 1e-12)), 0.5)
    return v * k[:, None]


def umeyama_se3(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((dst - mu_d).T @ (src - mu_s))
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1
    R = U @ W @ Vt
    return R, mu_d - R @ mu_s

# ---------------------------------------------------------------------- io --

def load_est(path):
    """ORB-SLAM3 TUM-style output: ts[ns] tx ty tz qx qy qz qw (body frame)."""
    a = np.loadtxt(path, ndmin=2)
    return a[:, 0], quat_xyzw_to_R(a[:, 4:8]), a[:, 1:4]


def load_gt(path):
    """EuRoC state_groundtruth_estimate0: ts[ns], p xyz, q wxyz, ... (body frame)."""
    a = np.loadtxt(path, delimiter=",", comments="#", ndmin=2)
    a = a[np.argsort(a[:, 0])]
    q_xyzw = a[:, [5, 6, 7, 4]]
    return a[:, 0], q_xyzw, a[:, 1:4]


def interp_gt(ts_g, q_g, p_g, ts_q, max_gap_ns=20e6):
    """Linear position + slerp rotation at query times. Ground truth is 200 Hz
    and frames are 20 Hz; nearest-neighbour association would leave up to
    2.5 ms of timing error, ~1 mm at MH01 speeds -- the same order as the
    per-edge errors being measured -- so we interpolate instead."""
    i = np.searchsorted(ts_g, ts_q) - 1
    ok = (i >= 0) & (i < len(ts_g) - 1)
    i = np.clip(i, 0, len(ts_g) - 2)
    gap = ts_g[i + 1] - ts_g[i]
    ok &= gap <= max_gap_ns
    a = np.clip((ts_q - ts_g[i]) / np.maximum(gap, 1), 0, 1)
    p = p_g[i] + a[:, None] * (p_g[i + 1] - p_g[i])
    q = slerp(q_g[i], q_g[i + 1], a)
    return ok, quat_xyzw_to_R(q), p

# ----------------------------------------------------------------- metrics --

def analyze_run(est_path, gt):
    ts, Re, te = load_est(est_path)
    ok, Rg, tg = interp_gt(*gt, ts)
    ts, Re, te, Rg, tg = ts[ok], Re[ok], te[ok], Rg[ok], tg[ok]

    Ra, ta = umeyama_se3(te, tg)
    abs_err = (Ra @ te.T).T + ta - tg
    ate = float(np.sqrt((np.linalg.norm(abs_err, axis=1) ** 2).mean()))

    # Relative errors over consecutive frames; skip gaps (lost frames).
    dt = np.diff(ts)
    keep = dt < 75e6
    dRg = np.einsum("nji,njk->nik", Rg[:-1], Rg[1:])
    dRe = np.einsum("nji,njk->nik", Re[:-1], Re[1:])
    dtg = np.einsum("nji,nj->ni", Rg[:-1], tg[1:] - tg[:-1])
    dte = np.einsum("nji,nj->ni", Re[:-1], te[1:] - te[:-1])
    eR = np.einsum("nji,njk->nik", dRg, dRe)
    rel_t = np.einsum("nji,nj->ni", dRg, dte - dtg)[keep]
    rel_r = so3_log(eR)[keep]
    return {"ts": ts, "ate": ate, "abs_err": abs_err, "rel_t": rel_t, "rel_r": rel_r,
            "n_poses": int(len(ts)), "n_edges": int(keep.sum())}


def acf(x, max_lag):
    x = x - x.mean()
    d = (x * x).sum()
    return np.array([1.0] + [(x[:-k] * x[k:]).sum() / d for k in range(1, max_lag + 1)])


def corr_time(a):
    """First lag at which the ACF falls below 1/e -- frames of 'memory'."""
    below = np.nonzero(a < 1 / np.e)[0]
    return int(below[0]) if len(below) else len(a)


def shape_stats(x):
    return {"mean": float(x.mean()), "std": float(x.std(ddof=1)),
            "skew": float(stats.skew(x)), "ex_kurt": float(stats.kurtosis(x)),
            "lag1_acf": float(acf(x, 1)[1])}

# -------------------------------------------------------------------- main --

def condition_of(name):
    n = re.sub(r"_(seed|run|rep)\d+$", "", name)
    return re.sub(r"^fpn_", "", n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pattern", default="fpn_*")
    ap.add_argument("--clean", default="clean")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    gt = load_gt(a.gt)

    runs, invalid = {}, {}
    for d in sorted(glob.glob(os.path.join(a.results, a.pattern))):
        if not os.path.isdir(d):
            continue
        cond = condition_of(os.path.basename(d))
        f = glob.glob(os.path.join(d, "f_*.txt"))
        if not f or sum(1 for _ in open(f[0])) < 100:
            invalid[cond] = invalid.get(cond, 0) + 1
            continue
        runs.setdefault(cond, []).append((os.path.basename(d), analyze_run(f[0], gt)))
    if not runs:
        raise SystemExit("no valid runs found")

    order = ([a.clean] if a.clean in runs else []) + sorted(c for c in runs if c != a.clean)
    clean = runs.get(a.clean, [])
    floor_mu = np.mean([r["ate"] for _, r in clean]) if clean else None
    floor_sd = np.std([r["ate"] for _, r in clean], ddof=1) if len(clean) > 1 else None
    axes, comps = ["x", "y", "z"], [("rel_t", "trans", 1e3, "mm"), ("rel_r", "rot", 1e3, "mrad")]
    summary = {"floor_ate_mean_m": floor_mu, "floor_ate_std_m": floor_sd, "conditions": {}}
    L = []

    # --- ATE vs floor ------------------------------------------------------
    L += ["## ATE versus the clean nondeterminism floor", "",
          "| condition | valid | ATE mean (mm) | ATE std (mm) | Δ vs clean (mm) | Δ / clean std | Mann-Whitney p |",
          "|---|---|---|---|---|---|---|"]
    for c in order:
        v = np.array([r["ate"] for _, r in runs[c]])
        n_inv = invalid.get(c, 0)
        row = {"n_valid": len(v), "n_invalid": n_inv, "ate": v.tolist()}
        if clean and c != a.clean:
            cv = np.array([r["ate"] for _, r in clean])
            p = stats.mannwhitneyu(v, cv, alternative="two-sided").pvalue if len(v) and len(cv) else float("nan")
            d = v.mean() - floor_mu
            row.update(delta_m=float(d), delta_over_floor=float(d / floor_sd) if floor_sd else None, mwu_p=float(p))
            L.append(f"| {c} | {len(v)}/{len(v)+n_inv} | {v.mean()*1e3:.2f} | {v.std(ddof=1)*1e3 if len(v)>1 else 0:.2f} "
                     f"| {d*1e3:+.2f} | {d/floor_sd:+.2f} | {p:.3f} |")
        else:
            L.append(f"| {c} | {len(v)}/{len(v)+n_inv} | {v.mean()*1e3:.2f} | {v.std(ddof=1)*1e3 if len(v)>1 else 0:.2f} | — | — | — |")
        summary["conditions"][c] = row
    L.append("")

    # --- relative error shape, per condition -------------------------------
    L += ["## Shape of the relative (edge) pose error — what the SE(3) denoiser sees", "",
          "Per-run statistics, averaged over runs. Std ratio is vs. clean; p is Mann-Whitney on per-run std.", ""]
    for key, label, scale, unit in comps:
        L += [f"### {label} ({unit})", "",
              "| condition | axis | mean | std | std ratio vs clean | std p | skew | ex. kurtosis | lag-1 ACF |",
              "|---|---|---|---|---|---|---|---|---|"]
        for c in order:
            per_run = [[shape_stats(r[key][:, j] * scale) for _, r in runs[c]] for j in range(3)]
            summary["conditions"][c].setdefault("rel", {})[label] = per_run
            for j, ax in enumerate(axes):
                S = per_run[j]
                m = lambda k: float(np.mean([s[k] for s in S]))
                stds = np.array([s["std"] for s in S])
                if clean and c != a.clean:
                    cs = np.array([shape_stats(r[key][:, j] * scale)["std"] for _, r in clean])
                    ratio = stds.mean() / cs.mean()
                    p = stats.mannwhitneyu(stds, cs, alternative="two-sided").pvalue
                    rs, ps = f"{ratio:.3f}", f"{p:.3f}"
                else:
                    rs, ps = "—", "—"
                L.append(f"| {c} | {ax} | {m('mean'):+.4f} | {m('std'):.4f} | {rs} | {ps} "
                         f"| {m('skew'):+.2f} | {m('ex_kurt'):+.2f} | {m('lag1_acf'):+.3f} |")
        L.append("")

    # --- absolute error memory ---------------------------------------------
    L += ["## Absolute error memory (descriptive only)", "",
          "Caution: absolute error integrates edge error, so even perfectly white edge noise",
          "produces a random walk with long memory here. This column cannot distinguish white",
          "from biased edge error -- use the lag-1 ACF of the RELATIVE error for whiteness.", "",
          "| condition | mean correlation time of |abs err| (frames) | at 20 Hz (s) |", "|---|---|---|"]
    for c in order:
        ct = [corr_time(acf(np.linalg.norm(r["abs_err"], axis=1), 600)) for _, r in runs[c]]
        summary["conditions"][c]["abs_corr_time_frames"] = ct
        L.append(f"| {c} | {np.mean(ct):.0f} | {np.mean(ct)/20:.1f} |")
    L.append("")

    # --- figures -------------------------------------------------------------
    colors = {c: plt.cm.tab10(i) for i, c in enumerate(order)}

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for i, c in enumerate(order):
        v = np.array([r["ate"] for _, r in runs[c]]) * 1e3
        ax.scatter(np.full(len(v), i) + np.linspace(-.12, .12, len(v)), v, color=colors[c], zorder=3)
        ax.hlines(v.mean(), i - .25, i + .25, color="k", lw=1.5, zorder=4)
    if floor_sd:
        ax.axhspan((floor_mu - 2*floor_sd)*1e3, (floor_mu + 2*floor_sd)*1e3, color="grey", alpha=.15,
                   label="clean ±2σ (nondeterminism floor)")
        ax.legend(fontsize=8, loc="upper left")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, fontsize=8)
    ax.set_ylabel("ATE (mm)"); ax.set_title("ATE by condition (each dot = one run)", fontsize=10)
    ax.grid(axis="y", alpha=.3); fig.tight_layout(); fig.savefig(os.path.join(a.out, "ate_by_condition.png"), dpi=160)
    plt.close(fig)

    for key, label, scale, unit in comps:
        fig, axs = plt.subplots(2, 3, figsize=(11, 6.2))
        for j, axn in enumerate(axes):
            h, q = axs[0, j], axs[1, j]
            lim = None
            for c in order:
                x = np.concatenate([r[key][:, j] for _, r in runs[c]]) * scale
                lo, hi = np.percentile(x, [0.5, 99.5])
                lim = (min(lo, lim[0]), max(hi, lim[1])) if lim else (lo, hi)
            bins = np.linspace(*lim, 70)
            for c in order:
                x = np.concatenate([r[key][:, j] for _, r in runs[c]]) * scale
                h.hist(x, bins=bins, density=True, histtype="step", lw=1.3, color=colors[c], label=c)
                g = np.linspace(*lim, 300)
                h.plot(g, stats.norm.pdf(g, x.mean(), x.std()), ls=":", lw=1, color=colors[c])
                (osm, osr), _ = stats.probplot(x, dist="norm")
                sel = np.linspace(0, len(osm) - 1, 400).astype(int)
                q.plot(osm[sel], (osr[sel] - x.mean()) / x.std(), ".", ms=2.5, color=colors[c], label=c)
            q.plot([-4, 4], [-4, 4], "k--", lw=.8)
            h.set_title(f"{label} error, {axn} ({unit})", fontsize=9)
            q.set_title(f"QQ vs. normal, {axn} (standardised)", fontsize=9)
            q.set_xlim(-4.2, 4.2); q.set_ylim(-8, 8); q.grid(alpha=.3); h.grid(alpha=.3)
        axs[0, 0].legend(fontsize=7); axs[1, 0].set_ylabel("sample quantile"); axs[0, 0].set_ylabel("density")
        fig.suptitle(f"Relative {label} error per edge — solid: histogram, dotted: Gaussian with same mean/std", fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, f"rel_{label}_shape.png"), dpi=160); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    for c in order:
        A = np.mean([acf(np.linalg.norm(r["abs_err"], axis=1), 400) for _, r in runs[c]], axis=0)
        axs[0].plot(np.arange(len(A)) / 20, A, color=colors[c], label=c)
        B = np.mean([acf(r["rel_t"][:, 0], 40) for _, r in runs[c]], axis=0)
        axs[1].plot(np.arange(len(B)), B, ".-", color=colors[c], label=c, ms=3)
    axs[0].axhline(1/np.e, ls=":", c="k", lw=.8); axs[0].set_xlabel("lag (s)")
    axs[0].set_title("ACF of |absolute error| — long memory = drift", fontsize=9)
    axs[1].axhline(0, c="k", lw=.6); axs[1].set_xlabel("lag (frames)")
    axs[1].set_title("ACF of relative trans. error (x) — white noise ⇒ 0 for lag ≥ 1", fontsize=9)
    for x in axs: x.grid(alpha=.3); x.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "error_autocorrelation.png"), dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 3.6))
    for c in order:
        for k, (_, r) in enumerate(runs[c]):
            ax.plot((r["ts"] - r["ts"][0]) / 1e9, np.linalg.norm(r["abs_err"], axis=1) * 1e3,
                    color=colors[c], lw=.7, alpha=.75, label=c if k == 0 else None)
    ax.set_xlabel("time (s)"); ax.set_ylabel("|abs error| (mm)"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    ax.set_title("Absolute position error over the sequence (each line = one run)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "abs_error_timeseries.png"), dpi=160); plt.close(fig)

    with open(os.path.join(a.out, "summary.md"), "w") as f:
        f.write("# Trajectory-level error shape\n\n" + "\n".join(L) + "\n")
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("\n".join(L))
    print(f"\nfigures + summary written to {a.out}")


if __name__ == "__main__":
    main()
