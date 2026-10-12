#!/usr/bin/env python3
"""Summarise SLAM runs on REAL low-light sequences (arms clean=C0, C2, C4, C2s, C4s).

Unlike Phase 2 there is no noise seed to pair on: every repeat of an arm sees
identical input, and run-to-run spread is ORB-SLAM3's own nondeterminism. Arms
are therefore compared unpaired: Mann-Whitney U on the composite score (same
definition as phase2_analyze) and Fisher's exact test on the
valid-run counts, Holm-corrected within each sequence.

    real_analyze.py results.jsonl [more.jsonl ...] --out DIR
Each JSONL is one sequence (the suite's --tag names it).
"""
import argparse, json, os
import numpy as np
from scipy.stats import mannwhitneyu, fisher_exact


def composite(r, big):
    """Identical to phase2_analyze.composite: ATE if valid, else a penalty above
    every valid ATE that still ranks runs by how much they tracked."""
    if r.get("valid") and r.get("ate_m") is not None:
        return r["ate_m"]
    return big + (1.0 - r.get("frac_tracked", 0.0))


def holm(ps):
    order = np.argsort(ps); out = [0.0] * len(ps); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i])); out[i] = run
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", nargs="+"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lines = ["# Real low-light sequences: C0 (no correction), C2/C4 (mean gain [+ NLM]), C2s/C4s (saturation-safe gain [+ NLM])", ""]
    for path in a.jsonl:
        recs = [json.loads(l) for l in open(path) if l.strip()]
        seq = os.path.basename(path).replace("_results.jsonl", "")
        arms = {"C0": [r for r in recs if r["arm"] == "clean"],
                **{k: [r for r in recs if r["arm"] == k] for k in ("C2", "C4", "C2s", "C4s")}}
        valid_ates = [r["ate_m"] for r in recs if r.get("valid") and r.get("ate_m") is not None]
        big = 10.0 * max(valid_ates) if valid_ates else 10.0
        comp = {k: np.array([composite(r, big) for r in v]) for k, v in arms.items()}
        lines += [f"## {seq}", "", "| arm | runs | valid | median tracked | median ATE valid (mm) | ATE IQR (mm) | mean resets |",
                  "|---|---|---|---|---|---|---|"]
        for k, v in arms.items():
            if not v:
                continue
            ate = np.array([r["ate_m"] for r in v if r.get("valid")]) * 1000
            iqr = f"{np.percentile(ate, 25):.1f}-{np.percentile(ate, 75):.1f}" if ate.size else "-"
            lines.append(f"| {k} | {len(v)} | {sum(bool(r.get('valid')) for r in v)} | "
                         f"{np.median([r.get('frac_tracked', 0) for r in v]):.3f} | "
                         f"{np.median(ate):.1f} | {iqr} | " if ate.size else
                         f"| {k} | {len(v)} | 0 | {np.median([r.get('frac_tracked', 0) for r in v]):.3f} | - | - | ")
            lines[-1] += f"{np.mean([r.get('reset_bad_imu', 0) + r.get('reset_imu_not_init', 0) for r in v]):.1f} |"
        tests = [("C4", "C2"), ("C4", "C0"), ("C2", "C0"), ("C4s", "C2s"), ("C4s", "C0"), ("C2s", "C0")]
        rows, ps = [], []
        for x, y in tests:
            if len(comp[x]) < 3 or len(comp[y]) < 3:
                continue
            p = mannwhitneyu(comp[x], comp[y], alternative="two-sided").pvalue
            vx, vy = sum(bool(r.get("valid")) for r in arms[x]), sum(bool(r.get("valid")) for r in arms[y])
            fp = fisher_exact([[vx, len(arms[x]) - vx], [vy, len(arms[y]) - vy]])[1]
            d = (np.median(comp[x]) - np.median(comp[y])) * 1000
            rows.append((x, y, d, p, fp)); ps.append(p)
        if rows:
            lines += ["", "| comparison | median Δ composite (mm, negative favours first) | Mann-Whitney p | Holm p | Fisher p (valid) |",
                      "|---|---|---|---|---|"]
            for (x, y, d, p, fp), hp in zip(rows, holm(ps)):
                lines.append(f"| {x} vs {y} | {d:+.1f} | {p:.3f} | {hp:.3f} | {fp:.3f} |")
        lines.append("")
    txt = "\n".join(lines)
    open(os.path.join(a.out, "real_summary.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
