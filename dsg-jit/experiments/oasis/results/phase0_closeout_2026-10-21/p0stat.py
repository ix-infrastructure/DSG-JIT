#!/usr/bin/env python3
"""Phase 0.4-0.6 summary from record.json files.

  good (with GT)    one map, coverage after start >= 0.9, ATE < 1 km   (4Seasons strict rule)
  good (no GT)      one map, coverage >= 0.9                           (0.6 test-split nights)
  crash             no trajectory / no atlas line (counted as not good)
Fisher exact on good counts vs the stated reference arm."""
import glob, json, os, sys
import numpy as np
from scipy.stats import fisher_exact

O = sys.argv[1] if len(sys.argv) > 1 else "/data/tkocher/oasis_results"
GROUPS = [  # (title, prefix, reference prefix or None, has GT)
    ("0.4 visual-only stereo, real night, no correction", "s4V_L0_clean_", None, True),
    ("0.4 visual-only stereo, real night, gain C2s", "s4V_L0_C2s_", "s4V_L0_clean_", True),
    ("real night stereo-inertial, no correction (n=30, earlier)", "s4N_L0_clean_", None, True),
    ("real night stereo-inertial, gain C2s (n=30, earlier)", "s4N_L0_C2s_", "s4N_L0_clean_", True),
    ("real day stereo-inertial (earlier)", "s4D_L0_clean_", None, True),
    ("0.5 synthetic night, no correction", "s4S_L0_C0_", "s4N_L0_clean_", True),
    ("0.5 synthetic night, gain C2s", "s4S_L0_C2s_", "s4N_L0_C2s_", True),
    ("0.5b synthetic night + motion blur, no correction", "s4B_L0_C0_", "s4N_L0_clean_", True),
    ("0.5b synthetic night + motion blur, gain C2s", "s4B_L0_C2s_", "s4N_L0_C2s_", True),
    ("0.6 maximilianeum_2 night, no correction", "s4m2N_L0_clean_", None, False),
    ("0.6 maximilianeum_2 night, gain C2s", "s4m2N_L0_C2s_", "s4m2N_L0_clean_", False),
    ("0.6 old_town_3 night, no correction", "s4o3N_L0_clean_", None, False),
    ("0.6 old_town_3 night, gain C2s", "s4o3N_L0_C2s_", "s4o3N_L0_clean_", False),
]


def recs(prefix):
    out = []
    for f in sorted(glob.glob(os.path.join(O, prefix + "*", "record.json"))):
        try:
            out.append(json.load(open(f)))
        except Exception:
            pass
    return out


def good(r, gt):
    one = r.get("maps_in_atlas") == 1
    if gt:
        return one and r.get("frac_tracked_after_start", 0) >= 0.9 and r.get("ate_m") is not None and r["ate_m"] < 1000
    return one and r.get("frac_tracked", 0) >= 0.9


for title, pre, ref, gt in GROUPS:
    R = recs(pre)
    if not R:
        continue
    g = [good(r, gt) for r in R]
    crash = sum(r.get("maps_in_atlas") is None for r in R)
    ates = sorted(r["ate_m"] for r, ok in zip(R, g) if ok) if gt else []
    line = f"{title}\n    good {sum(g)}/{len(R)}  crashed {crash}"
    if gt and ates:
        line += f"  ATE(good) median {np.median(ates):.1f} m [{ates[0]:.1f}-{ates[-1]:.1f}]"
    line += f"  maps>1 {sum((r.get('maps_in_atlas') or 0) > 1 for r in R)}  fails/run {np.mean([r['fail_track_local_map'] for r in R]):.0f}"
    line += f"  cov median {np.median([r.get('frac_tracked', 0) for r in R]):.3f}"
    if ref:
        Q = recs(ref)
        if Q:
            q = sum(good(r, gt) for r in Q)
            p = fisher_exact([[sum(g), len(R) - sum(g)], [q, len(Q) - q]])[1]
            line += f"\n    vs {ref}*: {sum(g)}/{len(R)} vs {q}/{len(Q)}, Fisher p = {p:.2g}"
    print(line)
