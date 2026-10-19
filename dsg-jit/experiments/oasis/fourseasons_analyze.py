#!/usr/bin/env python3
"""Summarise 4Seasons runs (day control, night C0, night correction arms).

phase2_eval_run's `valid` looks only at coverage after the first saved pose.
ORB-SLAM3 saves only the final map, so a run that lost the map and restarted
(2-3 maps, 28-49 % of the sequence) still looked valid. Here a run is valid
only if the atlas holds ONE map and the saved trajectory covers >= --min-cov of
all frames.

    fourseasons_analyze.py [--results DIR]   (reads s4D_*/s4N_* record.json)
"""
import argparse, glob, json, os
import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="/data/tkocher/oasis_results")
    ap.add_argument("--min-cov", type=float, default=0.9)
    a = ap.parse_args()
    groups = {}
    for p in sorted(glob.glob(os.path.join(a.results, "s4[DN]_L0_*", "record.json"))):
        r = json.load(open(p))
        cond, arm = r["run_id"].split("_")[0], r["run_id"].split("_")[2]
        r["ok"] = bool(r["ate_m"] is not None and r["maps_in_atlas"] == 1 and r["frac_tracked"] >= a.min_cov)
        # maps_in_atlas is None when ORB-SLAM3 died before its shutdown summary (crash): counted invalid, shown as -1
        groups.setdefault(f"{'day' if cond == 's4D' else 'night'} {arm}", []).append(r)
    base = groups.get("night clean", [])
    print(f"{'group':14} {'n':>2} {'valid':>6}  {'ATE med (m)':>11} {'range':>17}  {'cov med':>7} {'fails med':>9} {'maps':>9}   vs night C0")
    for k, rs in groups.items():
        ok = [r for r in rs if r["ok"]]
        ate = np.array([r["ate_m"] for r in ok])
        s = f"{k:14} {len(rs):>2} {len(ok):>3}/{len(rs):<2}  "
        s += f"{np.median(ate):>11.2f} [{ate.min():7.2f}-{ate.max():7.2f}]  " if len(ate) else " " * 33
        s += f"{np.median([r['frac_tracked'] for r in rs]):>7.3f} {np.median([r['fail_track_local_map'] for r in rs]):>9.0f} {str(sorted({r['maps_in_atlas'] if r['maps_in_atlas'] is not None else -1 for r in rs})):>9}"
        if base and k != "night clean" and k.startswith("night"):
            pf = fisher_exact([[len(ok), len(rs) - len(ok)], [sum(r["ok"] for r in base), len(base) - sum(r["ok"] for r in base)]])[1]
            comp = lambda R: [r["ate_m"] if r["ok"] else 1e6 for r in R]          # invalid ranks worst
            pm = mannwhitneyu(comp(rs), comp(base)).pvalue
            s += f"   valid Fisher p={pf:.3f}, composite MW p={pm:.3f}"
        print(s)


if __name__ == "__main__":
    main()
