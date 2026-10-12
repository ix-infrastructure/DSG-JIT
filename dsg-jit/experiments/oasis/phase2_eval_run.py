#!/usr/bin/env python3
"""Evaluate one SLAM run directory into a single JSON record.

ATE alone is misleading under degradation: it is computed only on poses that
were tracked, so a run that loses tracking for half the sequence can report a
LOWER ATE than one that tracks throughout. Every record therefore carries the
tracked fraction and the failure/reset counts alongside ATE, and a run is only
`valid` if, after its first pose, it tracked at least --min-tracked of the frames.

Counts are parsed from ORB-SLAM3's console log (slam_stdout.log), using the
exact strings printed by the fork's Tracking.cc / CellManager.cc.
"""
import argparse, glob, json, os, re, subprocess, sys
import numpy as np

PATTERNS = {
    "fail_track_local_map": r"Fail to track local map!",
    "reset_imu_not_init":   r"IMU is not or recently initialized\. Reseting active map",
    "reset_bad_imu":        r"TRACK: Reset map because local mapper set the bad imu flag",
    "timestamp_jump":       r"Timestamp jump detected",
    "new_map":              r"New Map created",
    "oasis_dropped_frame":  r"Skipped/Dropped frame",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir"); ap.add_argument("--gt", required=True)
    ap.add_argument("--n-frames", type=int, default=3682, help="frames in the sequence")
    ap.add_argument("--min-tracked", type=float, default=0.9, help="min coverage after the first pose")
    ap.add_argument("--frame-dt", type=float, default=0.05, help="camera period, s (EuRoC 20 Hz)")
    ap.add_argument("--eval-ate", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_ate.py"))
    a = ap.parse_args()

    rec = {"run_id": os.path.basename(os.path.normpath(a.run_dir))}
    log = os.path.join(a.run_dir, "slam_stdout.log")
    txt = open(log, errors="replace").read() if os.path.exists(log) else ""
    for k, pat in PATTERNS.items():
        rec[k] = len(re.findall(pat, txt))
    m = re.search(r"There are (\d+) maps in the atlas", txt)
    rec["maps_in_atlas"] = int(m.group(1)) if m else None

    # OASIS guard: did the field-of-view mask ever shrink below full frame?
    cm = os.path.join(a.run_dir, "cellManager.txt")
    if os.path.exists(cm):
        sizes = set(re.findall(r"FOV Mask: (\d+)x\d+", open(cm, errors="replace").read()))
        rec["oasis_mask_sizes"] = sorted(int(s) for s in sizes)
    else:
        rec["oasis_mask_sizes"] = []

    traj = glob.glob(os.path.join(a.run_dir, "f_*.txt"))
    n = sum(1 for l in open(traj[0]) if l.strip() and not l.startswith("#")) if traj else 0
    rec["n_poses"] = n
    rec["frac_tracked"] = n / a.n_frames
    # Coverage AFTER the first pose. On MH03-05 stereo-inertial initialisation
    # resets the map for 11-21 s in every arm (no darkness involved), so the
    # whole-sequence fraction above marks healthy runs invalid. Validity uses
    # post-start coverage; start delay and mid-run gaps are recorded separately
    # so a correction that delays initialisation is still visible.
    rec.update(start_ts=None, end_ts=None, gap_s=None, frac_tracked_after_start=0.0)
    if traj and n >= 2:
        ts = np.array([float(l.split()[0]) for l in open(traj[0]) if l.strip() and not l.startswith("#")])
        g = np.diff(ts) / 1e9
        span = (ts[-1] - ts[0]) / 1e9
        rec.update(start_ts=int(ts[0]), end_ts=int(ts[-1]), gap_s=float(g[g > 0.2].sum()),
                   frac_tracked_after_start=float(min(1.0, n / (span / a.frame_dt + 1))))
    rec["ate_m"] = None
    if traj and n >= 10:
        r = subprocess.run([sys.executable, a.eval_ate, traj[0], a.gt, "--quiet"], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            rec["ate_m"] = float(r.stdout.strip())
    rec["valid"] = bool(rec["ate_m"] is not None and rec["frac_tracked_after_start"] >= a.min_tracked)

    # A failure of the HARNESS (missing/unreadable input, container never ran)
    # is not a SLAM outcome and must never be scored as tracking failure.
    # Found after a power loss: runs died on "Failed to load image" and would
    # otherwise have been recorded as invalid SLAM runs.
    reasons = []
    if "Failed to load image" in txt:
        reasons.append("input image unreadable")
    if len(txt) < 200:
        reasons.append("SLAM produced (almost) no output")
    rec["harness_error"] = "; ".join(reasons) if reasons else None

    man = os.path.join(a.run_dir, "LOWLIGHT_MANIFEST.json")
    if os.path.exists(man):
        rec.update({f"cfg_{k}": v for k, v in json.load(open(man)).items() if k != "source"})
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
