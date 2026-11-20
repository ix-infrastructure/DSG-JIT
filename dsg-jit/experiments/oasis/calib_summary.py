#!/usr/bin/env python3
"""Phase 1 summary: calibration recovery and identifiability from calib_estimate.json files.

  python3 calib_summary.py [--results /data/tkocher/oasis_results] [--prefixes xMH01 xMH03 xtv x4D ...]

Recovery (oracle and blind). Each sequence has its own calibration offset (clean data are not fitted best by
the nominal calibration), so an estimate is scored against the clean runs of the same sequence:
    error = (theta_hat - mean theta_hat over clean runs) - truth
on the parameters of the perturbed group, in display units (deg, cm, ms, %). "uncorrected" is |truth| in
the same metric, i.e. the error ORB-SLAM3 ran with. Targets (roadmap Phase 1): 0.05 deg rotation, 1 ms
time offset, 0.2 % IMU scale.

Identifiability: per sequence, the median Cramer-Rao bound of every parameter from the clean runs' marginal
Fisher information (all parameters free, gauge-free set), and the smallest normalized eigenvalue.
"""
import argparse, glob, json, os, re
from collections import defaultdict

import numpy as np

NAMES = None
GROUPS = {"rbc": [0, 1, 2], "tbc": [3, 4, 5], "sg": [6, 7, 8], "mg": list(range(9, 15)), "sa": [15, 16, 17],
          "ma": list(range(18, 24)), "ma_ut": [18, 19, 21], "td": [24]}
UNIT = {"rbc": (np.degrees(1), "deg"), "tbc": (100, "cm"), "sg": (100, "%"), "sa": (100, "%"),
        "mg": (np.degrees(1), "deg"), "ma": (np.degrees(1), "deg"), "ma_ut": (np.degrees(1), "deg"),
        "td": (1000, "ms")}
TARGET = {"rbc": 0.05, "td": 1.0, "sg": 0.2, "sa": 0.2}
METRIC = {  # arm -> list of (group, how): norm for vectors that are one quantity, max for per-axis errors
    "toff": [("td", "max")], "isf": [("sg", "max"), ("sa", "max")], "isfs": [("sg", "max"), ("sa", "max")],
    "rbc": [("rbc", "norm")], "tbc": [("tbc", "norm")], "imis": [("mg", "max"), ("ma", "max")],
    "tumvi_nos": [("sg", "max"), ("mg", "max"), ("sa", "max"), ("ma", "max")],
    "tumvi_raw": [("sg", "max"), ("mg", "max"), ("sa", "max"), ("ma", "max")],
    "rst": [("rbc", "norm"), ("tbc", "norm")],
}
ORACLE_KEY = {"tumvi_nos": "imis", "tumvi_raw": "imis"}


def seq_of(run):
    m = re.match(r"(xtv\d+)_", run)
    return m.group(1) if m else run.split("_")[0]


def tier_of(run):
    m = re.search(r"_L(\d)_", run)
    return int(m.group(1)) if m else 0


def measure(vec, group, how):
    v = np.asarray(vec)[GROUPS[group]] * UNIT[group][0]
    return float(np.linalg.norm(v) if how == "norm" else np.max(np.abs(v)))


def load(results, prefixes):
    recs = []
    for p in prefixes:
        for f in sorted(glob.glob(os.path.join(results, p + "*", "calib_estimate.json"))):
            try:
                r = json.load(open(f))
            except Exception:
                continue
            r["seq"], r["tier"] = seq_of(r["run"]), tier_of(r["run"])
            recs.append(r)
    return recs


def recovery(recs, mode):
    ref = defaultdict(lambda: defaultdict(list))        # seq -> key -> clean thetas
    for r in recs:
        if r.get("arm") != "clean" or mode not in r:
            continue
        if mode == "oracle":
            for key, d in r["oracle"].items():
                ref[r["seq"]][key].append(d["theta"])
        else:
            ref[r["seq"]]["blind"].append(r["blind"]["theta"])
    rows = defaultdict(list)
    for r in recs:
        arm = r.get("arm")
        if arm in (None, "clean") or mode not in r or arm not in METRIC:
            continue
        key = ORACLE_KEY.get(arm, arm) if mode == "oracle" else "blind"
        d = r["oracle"].get(key) if mode == "oracle" else r["blind"]
        if d is None or not ref[r["seq"]][key]:
            continue
        est = np.asarray(d["theta"]) - np.mean(ref[r["seq"]][key], axis=0)
        truth = np.zeros_like(est) if r.get("truth") is None else np.asarray(r["truth"])
        row = {}
        for g, how in METRIC[arm]:
            if r.get("truth") is None:              # unmodeled arm: what got absorbed
                row[g] = (None, measure(est, g, how))
            else:
                row[g] = (measure(truth, g, how), measure(est - truth, g, how))
        rows[(r["seq"], arm, r["tier"])].append(row)
    return rows


def fmt_recovery(rows, title):
    out = [f"## {title}", "", "| sequence | arm | tier | n | group | uncorrected | error median [p90] | within target |",
           "|---|---|---|---|---|---|---|---|"]
    for (seq, arm, tier) in sorted(rows):
        R = rows[(seq, arm, tier)]
        for g, _ in METRIC[arm]:
            unc = [x[g][0] for x in R]
            err = np.array([x[g][1] for x in R])
            u = "--" if unc[0] is None else f"{np.median(unc):.3g} {UNIT[g][1]}"
            tgt = TARGET.get(g)
            win = f"{np.mean(err <= tgt) * 100:.0f} % (<= {tgt} {UNIT[g][1]})" if tgt and unc[0] is not None else "--"
            out.append(f"| {seq} | {arm} | {tier} | {len(R)} | {g} | {u} | {np.median(err):.3g} [{np.percentile(err, 90):.3g}] {UNIT[g][1]} | {win} |")
    return out + [""]


def fmt_ident(recs):
    by = defaultdict(list)
    for r in recs:
        if r.get("arm") == "clean" and "ident" in r:
            by[r["seq"]].append(r)
    if not by:
        return []
    names = recs[0]["theta_names"]
    idx = sorted({k for g in ("rbc", "tbc", "sg", "mg", "sa", "ma_ut", "td") for k in GROUPS[g]})
    groups = [("rbc", "deg"), ("tbc", "cm"), ("sg", "%"), ("mg", "deg"), ("sa", "%"), ("ma_ut", "deg"), ("td", "ms")]
    out = ["## Identifiability (clean runs; median Cramer-Rao bound with every parameter free)", "",
           "| sequence | runs | keyframes | span (s) | " + " | ".join(f"{g} ({u})" for g, u in groups) + " | min eig |",
           "|---|---|---|---|" + "---|" * len(groups) + "---|"]
    for seq in sorted(by):
        R = by[seq]
        crlb = np.median([r["ident"]["crlb_all"] for r in R], axis=0)    # over idx order
        pos = {k: i for i, k in enumerate(idx)}
        cells = []
        for g, _ in groups:
            v = [crlb[pos[k]] * UNIT[g][0] for k in GROUPS[g]]
            cells.append("/".join(f"{x:.2g}" for x in v))
        mev = np.median([min(r["ident"]["eig_normalized"]) for r in R])
        out.append(f"| {seq} | {len(R)} | {np.median([r['n_kf'] for r in R]):.0f} | {np.median([r['span_s'] for r in R]):.0f} | "
                   + " | ".join(cells) + f" | {mev:.2g} |")
    return out + ["", f"Parameter order per cell: x/y/z; mg off-diagonals 01/02/10/12/20/21; ma_ut 01/02/12. "
                      f"Names: {', '.join(names[k] for k in idx)}", ""]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="/data/tkocher/oasis_results")
    ap.add_argument("--prefixes", nargs="+", default=["xMH01", "xMH02", "xMH03", "xMH04", "xMH05", "xV101", "xV103",
                                                       "xV201", "xV203", "xtv", "x4D", "p1x"])
    ap.add_argument("--out", default=None, help="write markdown here (default: print)")
    a = ap.parse_args()
    recs = load(a.results, a.prefixes)
    errs = [r for r in recs if "error" in r]
    lines = [f"# Phase 1 calibration re-estimation: {len(recs)} runs ({len(errs)} without an estimate)", ""]
    if errs:
        c = defaultdict(int)
        for r in errs:
            c[r["error"]] += 1
        lines += ["Runs without an estimate: " + ", ".join(f"{k}: {v}" for k, v in c.items()), ""]
    lines += fmt_ident(recs)
    lines += fmt_recovery(recovery(recs, "oracle"), "Recovery, oracle (free = the perturbed group)")
    lines += fmt_recovery(recovery(recs, "blind"), "Recovery, blind (free = sg, sa, rbc, tbc, td on every run)")
    text = "\n".join(lines)
    if a.out:
        open(a.out, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
