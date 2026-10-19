import json, glob, numpy as np
from scipy.stats import mannwhitneyu, fisher_exact
G = {}
for p in glob.glob("/data/tkocher/oasis_results/s4[DN]_L0_*/record.json"):
    r = json.load(open(p)); k = r["run_id"].split("_"); G.setdefault(k[0] + "_" + k[2], []).append(r)
ok = lambda r: r["ate_m"] is not None and r["maps_in_atlas"] == 1 and r["frac_tracked"] >= 0.9 and r["ate_m"] < 1000
B = G["s4N_clean"]; fB = [r["fail_track_local_map"] for r in B]
comp = lambda X: [r["ate_m"] if ok(r) else 1e6 for r in X]
for k in ["s4D_clean", "s4N_clean", "s4N_C2s", "s4N_C5", "s4N_C1"]:
    R = G[k]; v = [r for r in R if ok(r)]; a = sorted(round(r["ate_m"], 1) for r in v); f = [r["fail_track_local_map"] for r in R]
    s = f"{k:10} good {len(v)}/{len(R)} ATE {a} med {np.median(a) if a else float('nan'):.1f} fails med {np.median(f):.0f}"
    if k != "s4N_clean":
        nb = sum(map(ok, B))
        s += (f" | vs night C0: Fisher p={fisher_exact([[len(v), len(R)-len(v)], [nb, len(B)-nb]])[1]:.3f}"
              f" composite p={mannwhitneyu(comp(R), comp(B)).pvalue:.3f} fails p={mannwhitneyu(f, fB).pvalue:.4f}")
        if v and nb: s += f" ATE(good only) p={mannwhitneyu([r['ate_m'] for r in v], [r['ate_m'] for r in B if ok(r)]).pvalue:.3f}"
    print(s)
