#!/usr/bin/env python3
"""Aggregate ATE across repeated runs -> the nondeterminism floor.

Any denoising effect smaller than the run-to-run spread reported here is not
measurable on this setup, regardless of how many conditions are swept.
"""
import argparse, glob, os, re, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GT_DEFAULT = ("/data/tkocher/datasets/euroc/MH_01_easy/mav0/"
              "state_groundtruth_estimate0/data.csv")


def ate_for(run_dir, gt):
    hits = glob.glob(os.path.join(run_dir, "f_*.txt"))
    if not hits:
        return None
    out = subprocess.run(
        [sys.executable, os.path.join(HERE, "eval_ate.py"), hits[0], gt, "--quiet"],
        capture_output=True, text=True)
    if out.returncode != 0:
        return None
    return float(out.stdout.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="/data/tkocher/oasis_results")
    ap.add_argument("--pattern", default="baseline_*")
    ap.add_argument("--gt", default=GT_DEFAULT)
    a = ap.parse_args()

    groups = {}
    for d in sorted(glob.glob(os.path.join(a.results_root, a.pattern))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        cfg = re.sub(r"_run\d+$", "", name)
        v = ate_for(d, a.gt)
        if v is not None:
            groups.setdefault(cfg, []).append((name, v))

    if not groups:
        print("no completed runs yet"); return

    print(f"{'config':<34}{'n':>3}{'mean':>10}{'std':>10}{'min':>10}{'max':>10}{'spread':>9}")
    print("-" * 86)
    stats = {}
    for cfg, runs in sorted(groups.items()):
        v = np.array([x for _, x in runs])
        stats[cfg] = v
        print(f"{cfg:<34}{len(v):>3}{v.mean():>10.4f}{v.std(ddof=1) if len(v)>1 else 0:>10.4f}"
              f"{v.min():>10.4f}{v.max():>10.4f}{v.max()-v.min():>9.4f}")
    print("\nper-run ATE (m):")
    for cfg, runs in sorted(groups.items()):
        print(f"  {cfg}")
        for n, v in sorted(runs):
            print(f"      {n:<40} {v:.4f}")

    keys = sorted(stats)
    if len(keys) == 2:
        a_, b_ = stats[keys[0]], stats[keys[1]]
        print(f"\nconfig difference: {keys[1]} - {keys[0]} = "
              f"{b_.mean()-a_.mean():+.4f} m")
        pooled = np.sqrt((a_.var(ddof=1)+b_.var(ddof=1))/2) if min(len(a_),len(b_))>1 else 0
        if pooled > 0:
            print(f"  pooled within-config std: {pooled:.4f} m")
            print(f"  difference / noise floor : {abs(b_.mean()-a_.mean())/pooled:.2f}x")
            if abs(b_.mean()-a_.mean()) < pooled:
                print("  -> difference is SMALLER than run-to-run noise; not distinguishable")


if __name__ == "__main__":
    main()
