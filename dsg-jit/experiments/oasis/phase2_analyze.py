#!/usr/bin/env python3
"""Phase 2 analysis — written BEFORE the results were seen, so the decision
rules cannot be tuned to the outcome.

Pre-registered (see the low-light design in reports/Upgraded ORB SLAM3
correction pipeline.md, and the go/no-go table agreed before Phase 2 finished):

  Composite outcome per run. A run that is invalid (no trajectory, or tracked
  < 90% of the sequence) ranks WORSE than every valid run; invalid runs are
  ordered among themselves by tracked fraction. ATE alone is not used, because
  ATE is computed only on tracked poses and can look normal while a quarter of
  the sequence is missing.

  Paired tests. Every arm at a given (level, seed) saw the same darkened noise
  realization, so arms are compared with a Wilcoxon signed-rank test paired on
  seed, on the composite score.

  PRIMARY contrast:   C4 (gain + denoise) vs C2 (gain only) at L5.
  Secondary families: C4 vs C2 at every level; C2 vs C1 at every level;
                      each corrected arm vs C0. Holm-corrected within family.
  Valid-run rates:    Fisher exact test per pair.
  Realistic-regime null (L1, L2): two one-sided Welch tests (TOST) on valid-run
                      ATE against the clean runs, equivalence margin +/-5 mm.

  Decision table:
    GO          C4 beats C2 (Holm p < 0.05) on the composite at some level in
                L3-L5, AND L1-L2 are equivalent to clean.
    LUT ONLY    C4 ~ C2 everywhere, but C2 beats C1 or C0 -> brightness LUT.
    NO-GO       C4 worse than C2, or nothing beats C0.
"""
import argparse, json, os
from collections import defaultdict
import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MIN_TRACKED = 0.9


def composite(r, big):
    if r.get("valid") and r.get("ate_m") is not None:
        return r["ate_m"]
    return big + (1.0 - r.get("frac_tracked", 0.0))


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i])); adj[i] = run
    return adj


def wilcox(a, b):
    d = np.asarray(a) - np.asarray(b)
    if np.all(d == 0) or len(d) < 2:
        return 1.0, 0.0
    res = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return float(res.pvalue), float(np.median(d))


def tost(x, y, margin):
    """Equivalence of means within +/-margin (Welch). Returns max one-sided p."""
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    p_lo = stats.ttest_ind(np.asarray(x) + margin, y, equal_var=False, alternative="greater").pvalue
    p_hi = stats.ttest_ind(np.asarray(x) - margin, y, equal_var=False, alternative="less").pvalue
    return float(max(p_lo, p_hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl"); ap.add_argument("--out", required=True)
    ap.add_argument("--margin-mm", type=float, default=5.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = [json.loads(l) for l in open(a.jsonl) if l.strip()]
    clean = [r for r in rows if r["arm"] == "clean"]
    noisy = [r for r in rows if r["arm"] != "clean"]
    valid_ates = [r["ate_m"] for r in rows if r.get("valid") and r.get("ate_m") is not None]
    big = 10.0 * max(valid_ates) if valid_ates else 10.0
    cell = defaultdict(dict)                      # (level, arm) -> {seed: row}
    for r in noisy:
        cell[(r["level"], r["arm"])][r["seed"]] = r
    levels = sorted({k[0] for k in cell})
    L = []

    # ---- clean floor ---------------------------------------------------------
    cv = [r["ate_m"] * 1000 for r in clean if r.get("valid")]
    floor_mu, floor_sd = (np.mean(cv), np.std(cv, ddof=1)) if len(cv) > 1 else (np.nan, np.nan)
    L += ["# Phase 2 results — low-light SLAM grid (MH01)", "",
          f"Runs: {len(noisy)} darkened + {len(clean)} clean. "
          f"Clean floor: {len(cv)}/{len(clean)} valid, ATE {floor_mu:.1f} ± {floor_sd:.1f} mm.", ""]
    if clean:
        lc = [r.get("load_start", np.nan) for r in clean if r.get("valid")]
        if len(lc) > 3 and np.nanstd(lc) > 0:
            rho, p = stats.spearmanr(lc, cv)
            L += [f"Machine-load check on clean runs: Spearman(load, ATE) = {rho:+.2f} (p = {p:.2f}).", ""]

    # ---- per-cell table --------------------------------------------------------
    L += ["## Per-cell outcomes", "",
          "| level | arm | valid | median tracked | median ATE valid (mm) | mean resets |", "|---|---|---|---|---|---|"]
    for (lv, arm) in sorted(cell):
        v = list(cell[(lv, arm)].values())
        ok = [r for r in v if r.get("valid")]
        med_ate = np.median([r["ate_m"] for r in ok]) * 1000 if ok else float("nan")
        rs = np.mean([r.get("reset_bad_imu", 0) + r.get("reset_imu_not_init", 0) for r in v])
        L.append(f"| L{lv} | {arm} | {len(ok)}/{len(v)} | {np.median([r.get('frac_tracked', 0) for r in v]):.3f} "
                 f"| {med_ate:.1f} | {rs:.1f} |")
    L.append("")

    # ---- paired contrasts --------------------------------------------------------
    def contrast_family(A, B, title):
        out, ps, lines = [], [], []
        for lv in levels:
            if (lv, A) not in cell or (lv, B) not in cell:
                continue
            seeds = sorted(set(cell[(lv, A)]) & set(cell[(lv, B)]))
            if len(seeds) < 2:
                continue
            xa = [composite(cell[(lv, A)][s], big) for s in seeds]
            xb = [composite(cell[(lv, B)][s], big) for s in seeds]
            p, md = wilcox(xa, xb)
            va = sum(cell[(lv, A)][s].get("valid", False) for s in seeds)
            vb = sum(cell[(lv, B)][s].get("valid", False) for s in seeds)
            pf = stats.fisher_exact([[va, len(seeds) - va], [vb, len(seeds) - vb]])[1]
            out.append((lv, p, md, va, vb, len(seeds), pf)); ps.append(p)
        adj = holm(np.array(ps)) if ps else []
        lines += [f"### {title}", "",
                  "| level | n | valid A/B | median Δ composite (A−B) | Wilcoxon p | Holm p | Fisher p (valid) |",
                  "|---|---|---|---|---|---|---|"]
        res = []
        for (lv, p, md, va, vb, n, pf), pa in zip(out, adj):
            mdtxt = f"{md*1000:+.1f} mm" if abs(md) < 1 else f"{md:+.2f} (validity-driven)"
            lines.append(f"| L{lv} | {n} | {va}/{vb} | {mdtxt} | {p:.3f} | {pa:.3f} | {pf:.3f} |")
            res.append(dict(level=lv, p=p, p_holm=float(pa), median_diff=md, valid_a=va, valid_b=vb, n=n))
        return lines + [""], res

    fams = {}
    for A, B, t in (("C4", "C2", "C4 (gain + denoise) vs C2 (gain only) — negative favours denoising"),
                    ("C2", "C1", "C2 (gain) vs C1 (lowered FAST threshold) — negative favours gain"),
                    ("C2", "C0", "C2 (gain) vs C0 (none)"),
                    ("C4", "C0", "C4 (gain + denoise) vs C0 (none)"),
                    ("C1", "C0", "C1 (threshold) vs C0 (none)")):
        lines, res = contrast_family(A, B, t); L += lines; fams[f"{A}_vs_{B}"] = res

    # ---- realistic-regime null -----------------------------------------------------
    L += ["## Realistic regime (gain-compensated): equivalence to clean", "",
          f"TOST, margin ±{a.margin_mm:.0f} mm on valid-run ATE. p < 0.05 means equivalent.", "",
          "| level | arm | valid | mean ATE (mm) | Δ vs clean (mm) | TOST p |", "|---|---|---|---|---|---|"]
    null_ok = True
    for lv in (1, 2):
        for arm in ("C0", "C4"):
            if (lv, arm) not in cell:
                continue
            x = [r["ate_m"] * 1000 for r in cell[(lv, arm)].values() if r.get("valid")]
            p = tost(x, cv, a.margin_mm)
            nv = len(x); nt = len(cell[(lv, arm)])
            if arm == "C0":
                null_ok &= (p < 0.05) and nv == nt
            L.append(f"| L{lv} | {arm} | {nv}/{nt} | {np.mean(x):.1f} | {np.mean(x) - floor_mu:+.1f} | {p:.3f} |")
    L.append("")

    # ---- decision -------------------------------------------------------------------
    c4c2 = {r["level"]: r for r in fams.get("C4_vs_C2", [])}
    go_lv = [lv for lv in (3, 4, 5) if lv in c4c2 and c4c2[lv]["p_holm"] < 0.05 and c4c2[lv]["median_diff"] < 0]
    worse = [lv for lv in (3, 4, 5) if lv in c4c2 and c4c2[lv]["p_holm"] < 0.05 and c4c2[lv]["median_diff"] > 0]
    gain_helps = any(r["p_holm"] < 0.05 and r["median_diff"] < 0
                     for k in ("C2_vs_C1", "C2_vs_C0") for r in fams.get(k, []))
    prim = c4c2.get(5)
    if go_lv and null_ok:
        verdict = f"GO — denoising beats gain-only at L{', L'.join(map(str, go_lv))}, and the realistic regime is equivalent to clean."
    elif go_lv:
        # Not covered by the pre-registered table: the denoising criterion is met
        # but the realistic-regime null is not confirmed. Report both halves
        # instead of falling through to a label that denies the denoising result.
        verdict = (f"UNDETERMINED (outside pre-registered table) — denoising beats gain-only at "
                   f"L{', L'.join(map(str, go_lv))}, but L1-L2 equivalence to clean is NOT confirmed.")
    elif worse:
        verdict = f"NO-GO — denoising is significantly WORSE than gain-only at L{', L'.join(map(str, worse))}."
    elif gain_helps:
        verdict = "LUT ONLY — no denoising benefit at the trajectory level, but brightening beats the alternatives."
    else:
        verdict = "NO-GO — no correction beats doing nothing at the trajectory level."
    L = L[:2] + ["## Verdict (pre-registered decision rules)", "", f"**{verdict}**", "",
                 (f"Primary contrast, C4 vs C2 at L5: Wilcoxon p = {prim['p']:.4f} (Holm {prim['p_holm']:.4f}), "
                  f"valid {prim['valid_a']}/{prim['n']} vs {prim['valid_b']}/{prim['n']}." if prim else
                  "Primary contrast not available."), ""] + L[2:]

    open(os.path.join(a.out, "phase2_summary.md"), "w").write("\n".join(L) + "\n")
    json.dump({"verdict": verdict, "families": fams, "floor_mm": [floor_mu, floor_sd]},
              open(os.path.join(a.out, "phase2_summary.json"), "w"), indent=1)

    # ---- figures ----------------------------------------------------------------------
    colors = {"C0": "#444", "C1": "#d62728", "C2": "#1f77b4", "C4": "#2ca02c"}
    names = {"C0": "none", "C1": "lower FAST thr.", "C2": "linear gain", "C4": "gain + NLM"}
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    for arm in ("C0", "C1", "C2", "C4"):
        lv = [l for l in levels if (l, arm) in cell]
        vr = [np.mean([r.get("valid", False) for r in cell[(l, arm)].values()]) for l in lv]
        tr = [np.median([r.get("frac_tracked", 0) for r in cell[(l, arm)].values()]) for l in lv]
        at = [np.median([r["ate_m"] * 1000 for r in cell[(l, arm)].values() if r.get("valid")] or [np.nan]) for l in lv]
        for ax, y in zip(axs, (vr, tr, at)):
            ax.plot(lv, y, "o-", color=colors[arm], lw=2, label=f"{arm} {names[arm]}")
    if not np.isnan(floor_mu):
        axs[2].axhspan(floor_mu - 2 * floor_sd, floor_mu + 2 * floor_sd, color="gold", alpha=.35, label="clean ±2σ")
    for ax, t in zip(axs, ("valid-run rate (tracked ≥ 90%)", "median fraction of sequence tracked",
                           "median ATE of VALID runs (mm)")):
        ax.set_title(t, fontsize=10); ax.set_xlabel("darkness level (stops)"); ax.grid(alpha=.3)
        ax.axvspan(0.7, 2.3, color="grey", alpha=.08)
    axs[0].legend(fontsize=8)
    fig.suptitle("Phase 2 — ORB-SLAM3 stereo-inertial under simulated low light (MH01, 10 paired seeds per cell)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "phase2_overview.png"), dpi=150)
    print("\n".join(L[:8]))


if __name__ == "__main__":
    main()
