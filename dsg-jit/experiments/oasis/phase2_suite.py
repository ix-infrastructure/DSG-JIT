#!/usr/bin/env python3
"""Phase 2 orchestrator: low-light SLAM grid on one sequence.

For each (seed, level, arm) cell: generate the darkened + corrected sequence,
run ORB-SLAM3 on it, evaluate, record, delete the imagery. Plus clean L0 runs
that re-establish the nondeterminism floor in the same batch.

Design choices, each for a stated reason:
  * Interleaved order (seed-major, cells round-robin, clean runs spread through
    the batch) so slow drift in machine load lands evenly on every arm instead
    of on whichever arm happened to run during a busy hour.
  * Bounded concurrency. This is a shared machine and ORB-SLAM3's results
    depend on thread timing; --concurrency keeps runs from starving each other.
    System load is recorded at the start and end of every run so any
    load-related effect can be checked afterwards rather than assumed away.
  * Base config defaults to stock EuRoC.yaml. EuRoC_oasis.yaml differs only by
    System.enableOasis, whose field-of-view mask is driven by wall-clock frame
    time and therefore by other users' load. phase2_eval_run.py records the
    OASIS mask sizes either way.
  * Imagery is ephemeral (regenerated bit-identically from the seed); the
    manifest, config, trajectory and SLAM log are kept.
"""
import argparse, json, os, shutil, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
LOCK = threading.Lock()


def sh(cmd, log=None):
    with open(log, "a") if log else open(os.devnull, "w") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode


def run_job(job, a):
    rid, level, arm, seed = job["run_id"], job["level"], job["arm"], job["seed"]
    out_dir = os.path.join(a.results, rid)
    if a.skip_done and os.path.exists(os.path.join(out_dir, "record.json")):
        return f"{rid}: already done"
    t0, load0 = time.time(), os.getloadavg()[0]
    if arm == "clean":
        data, cfg = a.src, os.path.basename(a.base_yaml)
    else:
        data = os.path.join(a.scratch, rid)
        rc = sh([sys.executable, os.path.join(HERE, a.gen_script), "--src", a.src, "--dst", data,
                 "--yaml", a.base_yaml, "--level", str(level), "--seed", str(seed), "--arm", arm,
                 "--workers", str(a.gen_workers), "--force"] + a.gen_args.split(),
                log=os.path.join(a.results, f"{rid}.gen.log"))
        if rc != 0:
            shutil.rmtree(data, ignore_errors=True)
            return f"{rid}: GENERATION FAILED (see {rid}.gen.log) -- will retry on resume"
        cfg = os.path.join(data, "config.yaml")
    t_gen = time.time() - t0
    sh([a.run_oasis, data, a.stamps, rid, cfg])
    t_slam = time.time() - t0 - t_gen
    if arm != "clean":
        for f in ("LOWLIGHT_MANIFEST.json", "config.yaml"):
            if os.path.exists(os.path.join(data, f)):
                shutil.copy2(os.path.join(data, f), out_dir)
        shutil.rmtree(data, ignore_errors=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, "phase2_eval_run.py"), out_dir, "--gt", a.gt,
                        "--n-frames", str(a.n_frames)],
                       capture_output=True, text=True)
    rec = json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else {"run_id": rid, "valid": False,
                                                                              "eval_error": r.stderr[-300:]}
    rec.update(level=level, arm=arm, seed=seed, gen_s=round(t_gen, 1), slam_s=round(t_slam, 1),
               load_start=round(load0, 2), load_end=round(os.getloadavg()[0], 2))
    if rec.get("harness_error") or t_slam < 5:
        # Not a SLAM outcome: leave no record.json so --skip-done retries it.
        return f"{rid}: HARNESS ERROR ({rec.get('harness_error') or 'slam ran <5 s'}) -- will retry on resume"
    with open(os.path.join(out_dir, "record.json"), "w") as f:
        json.dump(rec, f)
    with LOCK, open(a.jsonl, "a") as f:
        f.write(json.dumps(rec) + "\n")
    ate = f"{rec['ate_m']*1000:.1f} mm" if rec.get("ate_m") else "n/a"
    return (f"{time.strftime('%H:%M:%S')} {rid}: ATE {ate}  tracked {rec.get('frac_tracked', 0):.3f}  "
            f"valid={rec.get('valid')}  fails={rec.get('fail_track_local_map')}  "
            f"(gen {t_gen:.0f}s, slam {t_slam:.0f}s, load {load0:.0f}->{rec['load_end']:.0f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", nargs="+", required=True, help="LEVEL:ARM, e.g. 5:C0 5:C2")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--clean-runs", type=int, default=20)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--src", default="/data/tkocher/datasets/euroc/MH_01_easy")
    ap.add_argument("--gt", default="/data/tkocher/datasets/euroc/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv")
    ap.add_argument("--stamps", default="MH01")
    ap.add_argument("--base-yaml", default="/data/tkocher/oasis_results/_configs/EuRoC.yaml",
                    help="must live under /data so per-run copies are visible in the container")
    ap.add_argument("--scratch", default="/data/tkocher/datasets/euroc_noisy")
    ap.add_argument("--results", default="/data/tkocher/oasis_results")
    ap.add_argument("--run-oasis", default=os.path.expanduser("~/repos/oasis-experiments/run_oasis.sh"))
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--gen-workers", type=int, default=6)
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--gen-script", default="lowlight_sequence.py",
                    help="real_correct_sequence.py for real (not synthetically darkened) sequences")
    ap.add_argument("--gen-args", default="", help="extra arguments for --gen-script, e.g. '--m-ref 90'")
    ap.add_argument("--n-frames", type=int, default=3682, help="stereo frames in the sequence (MH01: 3682)")
    a = ap.parse_args()
    a.jsonl = os.path.join(a.results, f"{a.tag}_results.jsonl")

    cells = [(int(c.split(":")[0]), c.split(":")[1]) for c in a.cells]
    jobs = []
    clean_left = a.clean_runs
    per_round = -(-a.clean_runs // max(len(a.seeds), 1))           # ceil: spread clean runs
    for s in a.seeds:
        for lv, arm in cells:
            jobs.append(dict(run_id=f"{a.tag}_L{lv}_{arm}_s{s:02d}", level=lv, arm=arm, seed=s))
        for _ in range(min(per_round, clean_left)):
            k = a.clean_runs - clean_left
            jobs.append(dict(run_id=f"{a.tag}_L0_clean_r{k:02d}", level=0, arm="clean", seed=k))
            clean_left -= 1
    print(f"{len(jobs)} jobs ({len(jobs) - a.clean_runs} noisy, {a.clean_runs} clean), "
          f"concurrency {a.concurrency}, log -> {a.jsonl}", flush=True)
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        for msg in ex.map(lambda j: run_job(j, a), jobs):
            print(msg, flush=True)
    print(f"=== {a.tag} complete {time.strftime('%H:%M:%S')} ===", flush=True)


if __name__ == "__main__":
    main()
