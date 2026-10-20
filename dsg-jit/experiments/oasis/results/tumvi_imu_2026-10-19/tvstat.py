"""TUM-VI real IMU miscalibration: cal vs nos vs raw per room. Good = 1 map, >= 90% frames tracked, ATE < 10 m."""
import json, glob, numpy as np
from scipy.stats import mannwhitneyu, fisher_exact
R = {}
for p in glob.glob("/data/tkocher/oasis_results/tv[1-6]_*_s*/record.json"):
    try: r = json.load(open(p))
    except Exception: continue
    room, cond, _ = r["run_id"].split("_"); R.setdefault((room, cond), []).append(r)
ok = lambda r: r["ate_m"] is not None and r["maps_in_atlas"] == 1 and r["frac_tracked"] >= 0.9 and r["ate_m"] < 10
comp = lambda X: [r["ate_m"] if ok(r) else 1e6 for r in X]
print(f"{'room':5} {'cond':4} {'n':>2} {'good':>5} {'ATE med mm (good)':>18} {'maps':>10} {'badimu med':>10}   vs cal")
for room in sorted({k[0] for k in R}):
    base = R.get((room, "cal"), [])
    for cond in ("cal", "nos", "raw"):
        X = R.get((room, cond), [])
        if not X: continue
        g = [r["ate_m"] * 1e3 for r in X if ok(r)]
        s = f"{room:5} {cond:4} {len(X):>2} {len(g):>2}/{len(X):<2} {np.median(g) if g else float('nan'):>18.1f} {str(sorted({r['maps_in_atlas'] if r['maps_in_atlas'] is not None else -1 for r in X})):>10} {np.median([r['reset_bad_imu'] for r in X]):>10.0f}"
        if cond != "cal" and base:
            nb = sum(map(ok, base))
            s += f"   Fisher p={fisher_exact([[len(g), len(X)-len(g)], [nb, len(base)-nb]])[1]:.3f} composite p={mannwhitneyu(comp(X), comp(base)).pvalue:.4f}"
        print(s)
for cond in ("cal", "nos", "raw"):
    X = [r for k, v in R.items() if k[1] == cond for r in v]
    g = [r["ate_m"] * 1e3 for r in X if ok(r)]
    print(f"ALL   {cond:4} {len(X):>3} good {len(g)}/{len(X)}  median ATE (good) {np.median(g) if g else float('nan'):.1f} mm")
