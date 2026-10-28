#!/usr/bin/env python3
"""Phase 0.3 summary: sourced IMU-scale tiers (isfs) vs clean and the old provisional tiers (isf).
Strict validity: one map, coverage after start >= 0.9, ATE < 10 m."""
import json, sys
import numpy as np
from scipy.stats import mannwhitneyu

for s in sys.argv[1:] or ("cMH01", "cMH03"):
    R = [json.loads(l) for l in open(f"/data/tkocher/oasis_results/{s}_results.jsonl")]
    sel = lambda arm, lv: [r for r in R if r["arm"] == arm and r["level"] == lv]
    ok = lambda r: r.get("maps_in_atlas") == 1 and r.get("frac_tracked_after_start", 0) >= 0.9 and r.get("ate_m") and r["ate_m"] < 10
    c = [r["ate_m"] * 1000 for r in sel("clean", 0) if r.get("ate_m")]
    print(f"== {s}  clean n={len(c)} median {np.median(c):.1f} mm")
    print("arm   L  mag(g/a %)   n  strict  ATE med   min-max        xclean  p_MW     maps>1  fails")
    for arm in ("isf", "isfs"):
        for lv in (1, 2, 3):
            x = sel(arm, lv)
            if not x:
                continue
            a = [r["ate_m"] * 1000 for r in x if r.get("ate_m")]
            mag = x[0].get("cfg_magnitude")
            mag = "/".join(f"{v:g}" for v in mag) if isinstance(mag, list) else f"{mag:g}"
            p = mannwhitneyu(a, c).pvalue if len(a) > 2 else float("nan")
            print(f"{arm:5s} {lv}  {mag:10s} {len(x):3d}  {sum(map(ok, x)):2d}/{len(x):<2d}  {np.median(a):8.1f}  "
                  f"[{min(a):6.0f}-{max(a):6.0f}]  {np.median(a) / np.median(c):5.1f}  {p:.5f}  "
                  f"{sum((r.get('maps_in_atlas') or 0) > 1 for r in x):5d}  {sum(r['fail_track_local_map'] for r in x):5d}")
